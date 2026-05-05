from __future__ import annotations

import threading
import time
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from ember_core.core.registry import BACKEND_REGISTRY, register_backend
from ember_core.densification.families import GaussianFamilyOps
from ember_core.initialization import InitializedModel
from ember_core.training.checkpoints import (
    load_checkpoint_dir,
    save_checkpoint_dir,
)
from ember_core.training.config import (
    CheckpointExportConfig,
    LossConfig,
    OptimizationConfig,
    ParameterGroupConfig,
    ParameterTargetSpec,
    RenderPipelineSpec,
    TensorSliceSpec,
    TensorViewSpec,
    TrainingConfig,
)
from ember_core.training.protocols import TrainState
from ember_core.training.runtime import OptimizerBinding, build_optimizer_set
from ember_native_faster_gs.faster_gs.renderer import (
    FasterGSNativeRenderOptions,
)
from ember_native_faster_gs.faster_gs_depth.renderer import (
    FasterGSDepthNativeRenderOptions,
)
from ember_native_faster_gs.fastgs.renderer import FastGSNativeRenderOptions
from ember_native_faster_gs.gaussian_pop.renderer import (
    GaussianPopNativeRenderOptions,
)
from ember_splatting_training.fastergs import (
    GaussianMipSplatting3DFilter,
    active_sh_bases_for_step,
    fastergs_training_backend_options,
)
from ember_splatting_training.losses import rgb_l1_dssim_loss
from ember_splatting_training.recipes import (
    Gaussian3DGSOptimizationRecipe,
    gaussian_3dgs_optimization_config,
)
from ember_splatting_training.training_viewer import (
    TrainingViewerCancelled,
    TrainingViewerConfig,
    TrainingViewerHandle,
    TrainingViewerHook,
    create_training_viewer,
)
from torch.optim import Optimizer


def test_fastergs_active_sh_schedule() -> None:
    assert active_sh_bases_for_step(0) == 1
    assert active_sh_bases_for_step(999) == 1
    assert active_sh_bases_for_step(1000) == 4
    assert active_sh_bases_for_step(2000) == 9
    assert active_sh_bases_for_step(3000) == 16


def test_fastergs_morton_helpers_are_root_exports() -> None:
    from ember_splatting_training import morton_codes, morton_order

    assert callable(morton_codes)
    assert callable(morton_order)
    assert active_sh_bases_for_step(30_000) == 16


def test_fastergs_training_backend_options_use_state_step() -> None:
    state = TrainState(
        model=None,
        step=2_000,
        seed=0,
        device=torch.device("cpu"),
    )

    assert fastergs_training_backend_options(state) == {
        "active_sh_bases": 9,
        "clamp_output": False,
    }


def test_gaussian_mip_splatting_3d_filter_has_no_render_requirements() -> None:
    method = GaussianMipSplatting3DFilter()

    requirements = method.get_render_requirements(object())

    assert requirements.backend_options == {}


def test_rgb_l1_dssim_loss_hides_inactive_regularizer_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ember_splatting_training.losses.dssim_loss",
        lambda prediction, target, *, backend="cuda": prediction.new_tensor(
            0.0
        ),
    )
    scene = GaussianScene3D(
        center_position=torch.zeros((1, 3), dtype=torch.float32),
        log_scales=torch.zeros((1, 3), dtype=torch.float32),
        quaternion_orientation=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        logit_opacity=torch.zeros((1,), dtype=torch.float32),
        feature=torch.zeros((1, 1, 3), dtype=torch.float32),
        sh_degree=0,
    )
    state = TrainState(
        model=InitializedModel(
            scene=scene,
            modules={},
            parameters={},
        ),
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )
    batch = SimpleNamespace(
        images=torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    )
    render_output = RenderOutput(
        render=torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    )

    inactive = rgb_l1_dssim_loss(
        state,
        batch,
        render_output,
        weights={},
        lambda_opacity_regularization=0.0,
        lambda_scale_regularization=0.0,
    )
    active = rgb_l1_dssim_loss(
        state,
        batch,
        render_output,
        weights={},
        lambda_opacity_regularization=0.01,
        lambda_scale_regularization=0.01,
    )

    assert "opacity_regularization" not in inactive.metrics
    assert "scale_regularization" not in inactive.metrics
    assert "opacity_regularization" in active.metrics
    assert "scale_regularization" in active.metrics


class RecordingOptimizer(Optimizer):
    def __init__(self, params, lr: float, eps: float = 0.0) -> None:
        defaults = {"lr": lr, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure=None):
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.data.add_(parameter.grad, alpha=-group["lr"])


def scene_target(
    name: str,
    *,
    view: TensorViewSpec | None = None,
) -> ParameterTargetSpec:
    return ParameterTargetSpec(scope="scene", name=name, view=view)


def _register_unit_test_backend() -> None:
    if "unit_test_backend" in BACKEND_REGISTRY:
        return

    @register_backend(
        name="unit_test_backend",
        default_options=RenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
    )
    def render_unit_test_backend(
        scene: GaussianScene3D,
        camera: CameraState,
        *,
        return_alpha: bool = False,
        return_depth: bool = False,
        return_gaussian_impact_score: bool = False,
        return_normals: bool = False,
        return_2d_projections: bool = False,
        return_projective_intersection_transforms: bool = False,
        options: RenderOptions | None = None,
    ) -> RenderOutput:
        del (
            return_alpha,
            return_depth,
            return_gaussian_impact_score,
            return_normals,
            return_2d_projections,
            return_projective_intersection_transforms,
        )
        options = options or RenderOptions()
        color = scene.feature[:, 0, :].mean(dim=0)
        render = color.view(1, 1, 1, 3).expand(
            camera.width.shape[0],
            int(camera.height[0].item()),
            int(camera.width[0].item()),
            3,
        ) + options.background_color.view(1, 1, 1, 3)
        return RenderOutput(render=render)


def _dummy_loss(
    state: TrainState,
    batch: object,
    render_output: RenderOutput,
    *,
    weights: dict[str, float],
) -> torch.Tensor:
    del state, batch, render_output, weights
    return torch.tensor(0.0)


def _training_config(
    optimizer: str = "adam",
    optimizer_kwargs: dict[str, object] | None = None,
) -> TrainingConfig:
    _register_unit_test_backend()
    return TrainingConfig(
        render=RenderPipelineSpec(backend="unit_test_backend"),
        loss=LossConfig(target={"target": f"{__name__}._dummy_loss"}),
        optimization=OptimizationConfig(
            parameter_groups=[
                ParameterGroupConfig(
                    target=scene_target("logit_opacity"),
                    optimizer=optimizer,
                    optimizer_kwargs=dict(optimizer_kwargs or {}),
                    lr=0.1,
                )
            ]
        ),
        checkpoint=CheckpointExportConfig(output_dir=Path("unused")),
    )


def test_gaussian_3dgs_optimization_recipe_accepts_paper_aliases() -> None:
    recipe = Gaussian3DGSOptimizationRecipe.model_validate(
        {
            "optimizer": "unit.Optimizer",
            "means_lr_init": 0.1,
            "means_lr_final": 0.01,
            "means_lr_max_steps": 50,
            "center_position_lr_step_offset": 1,
            "sh_dc_lr": 0.2,
            "sh_rest_lr": 0.3,
            "opacity_lr": 0.4,
            "scale_lr": 0.5,
            "rotation_lr": 0.6,
        }
    )
    config = gaussian_3dgs_optimization_config(
        recipe,
        position_lr_scale=2.0,
        max_steps=100,
    )
    payload = recipe.model_dump(mode="json")

    assert payload["center_position_lr_init"] == 0.1
    assert "means_lr_init" not in payload
    assert [group.target.name for group in config.parameter_groups] == [
        "center_position",
        "feature",
        "feature",
        "logit_opacity",
        "log_scales",
        "quaternion_orientation",
    ]
    assert config.parameter_groups[0].lr == 0.2
    assert config.parameter_groups[0].scheduler.kwargs == {
        "final_lr": 0.02,
        "max_steps": 50,
        "step_offset": 1,
    }
    assert config.parameter_groups[3].lr == 0.4


def _state_for_scene(scene: GaussianScene3D) -> TrainState:
    return TrainState(
        model=InitializedModel(scene=scene, modules={}, parameters={}),
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )


def _trainable_scene(scene: GaussianScene3D) -> GaussianScene3D:
    updates = {}
    for field_def in fields(scene):
        value = getattr(scene, field_def.name)
        if isinstance(value, torch.Tensor):
            updates[field_def.name] = (
                value.detach().clone().requires_grad_(value.is_floating_point())
            )
    return replace(scene, **updates)


def _assert_scene_tensors_equal(
    left: GaussianScene3D,
    right: GaussianScene3D,
) -> None:
    for field_def in fields(left):
        left_value = getattr(left, field_def.name)
        right_value = getattr(right, field_def.name)
        if isinstance(left_value, torch.Tensor):
            assert torch.equal(left_value, right_value), field_def.name


def _logit_optimizer_binding(
    state: TrainState,
) -> tuple[torch.optim.Adam, OptimizerBinding]:
    optimizer = torch.optim.Adam([state.model.scene.logit_opacity], lr=0.1)
    binding = OptimizerBinding(
        target=scene_target("logit_opacity"),
        optimizer=optimizer,
        base_parameter=state.model.scene.logit_opacity,
        field_name="logit_opacity",
    )
    state.model.scene.logit_opacity.sum().backward()
    binding.step()
    binding.zero_grad()
    return optimizer, binding


def _exp_avg(optimizer: torch.optim.Optimizer) -> torch.Tensor:
    parameter = optimizer.param_groups[0]["params"][0]
    return optimizer.state[parameter]["exp_avg"]


def test_initialized_model_to_moves_buffers(cpu_scene: GaussianScene3D) -> None:
    model = InitializedModel(
        scene=cpu_scene,
        modules={},
        parameters={},
        buffers={"probe_mask": torch.ones(3, dtype=torch.float32)},
    )
    moved = model.to(torch.device("cpu"))
    assert moved.buffers["probe_mask"].device.type == "cpu"


def test_build_optimizer_set_supports_import_path_optimizer(
    cpu_scene: GaussianScene3D,
) -> None:
    state = _state_for_scene(cpu_scene)
    config = _training_config(
        optimizer=f"{__name__}.RecordingOptimizer",
        optimizer_kwargs={"eps": 1e-3},
    )
    optimizers = build_optimizer_set(state, config)
    optimizer = optimizers[0].optimizer
    assert isinstance(optimizer, RecordingOptimizer)
    assert optimizer.defaults["eps"] == pytest.approx(1e-3)


def test_checkpoint_roundtrip_preserves_buffers(
    cpu_scene: GaussianScene3D,
    tmp_path: Path,
) -> None:
    _register_unit_test_backend()
    config = _training_config()
    state = TrainState(
        model=InitializedModel(
            scene=cpu_scene,
            modules={},
            parameters={},
            buffers={"visible_mask": torch.tensor([1.0, 0.0, 1.0])},
        ),
        step=3,
        seed=0,
        device=torch.device("cpu"),
    )
    checkpoint_dir = save_checkpoint_dir(tmp_path / "checkpoint", state, config)
    loaded = load_checkpoint_dir(checkpoint_dir)
    assert torch.equal(
        loaded.model.buffers["visible_mask"],
        torch.tensor([1.0, 0.0, 1.0]),
    )


def test_gaussian_family_ops_copy_append_reorder_and_reset_state(
    cpu_scene: GaussianScene3D,
) -> None:
    scene = replace(
        cpu_scene,
        logit_opacity=cpu_scene.logit_opacity.detach()
        .clone()
        .requires_grad_(True),
    )
    state = _state_for_scene(scene)
    optimizer = torch.optim.Adam([state.model.scene.logit_opacity], lr=0.1)
    binding = OptimizerBinding(
        target=scene_target("logit_opacity"),
        optimizer=optimizer,
        base_parameter=state.model.scene.logit_opacity,
        field_name="logit_opacity",
    )

    state.model.scene.logit_opacity.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    family_ops = GaussianFamilyOps(state, [binding])
    original_count = int(state.model.scene.logit_opacity.shape[0])
    family_ops.copy_from_indices(
        torch.tensor([0]),
        torch.tensor([1]),
        field_overrides={"logit_opacity": torch.tensor([7.0])},
    )
    assert state.model.scene.logit_opacity[0].item() == pytest.approx(7.0)

    family_ops.append_from_indices(torch.tensor([1]))
    updated_parameter = optimizer.param_groups[0]["params"][0]
    updated_state = optimizer.state[updated_parameter]["exp_avg"]
    assert updated_parameter.shape[0] == original_count + 1
    assert updated_state.shape[0] == original_count + 1

    family_ops.reset_optimizer_state(torch.tensor([1]), ("logit_opacity",))
    reset_state = optimizer.state[optimizer.param_groups[0]["params"][0]][
        "exp_avg"
    ]
    assert torch.allclose(reset_state[1], torch.zeros_like(reset_state[1]))

    before_reorder = reset_state.clone()
    order = torch.arange(updated_parameter.shape[0] - 1, -1, -1)
    family_ops.reorder(order)
    reordered_state = optimizer.state[optimizer.param_groups[0]["params"][0]][
        "exp_avg"
    ]
    assert torch.equal(reordered_state, before_reorder[order])


def test_gaussian_family_ops_updates_all_bindings_for_shared_scene_field(
    cpu_scene: GaussianScene3D,
) -> None:
    scene = replace(
        cpu_scene,
        feature=cpu_scene.feature.detach().clone().requires_grad_(True),
    )
    state = _state_for_scene(scene)
    dc_optimizer = torch.optim.Adam([state.model.scene.feature], lr=0.1)
    rest_optimizer = torch.optim.Adam([state.model.scene.feature], lr=0.1)
    dc_binding = OptimizerBinding(
        target=scene_target(
            "feature",
            view=TensorViewSpec(
                slices=(TensorSliceSpec(axis=1, start=0, stop=1),)
            ),
        ),
        optimizer=dc_optimizer,
        base_parameter=state.model.scene.feature,
        field_name="feature",
    )
    rest_binding = OptimizerBinding(
        target=scene_target(
            "feature",
            view=TensorViewSpec(
                slices=(TensorSliceSpec(axis=1, start=1, stop=None),)
            ),
        ),
        optimizer=rest_optimizer,
        base_parameter=state.model.scene.feature,
        field_name="feature",
    )

    state.model.scene.feature.square().mean().backward()
    dc_binding.step()
    rest_binding.step()
    dc_binding.zero_grad()
    rest_binding.zero_grad()

    family_ops = GaussianFamilyOps(state, [dc_binding, rest_binding])
    family_ops.append_from_indices(torch.tensor([1]))

    updated_parameter = dc_optimizer.param_groups[0]["params"][0]
    dc_state = dc_optimizer.state[updated_parameter]["exp_avg"]
    rest_state = rest_optimizer.state[updated_parameter]["exp_avg"]
    assert updated_parameter.shape[0] == int(cpu_scene.feature.shape[0]) + 1
    assert dc_state.shape == updated_parameter.shape
    assert rest_state.shape == updated_parameter.shape

    dc_before_reset = dc_state.clone()
    rest_before_reset = rest_state.clone()
    family_ops.reset_optimizer_state(torch.tensor([1]), ("feature",))
    dc_reset = dc_optimizer.state[dc_optimizer.param_groups[0]["params"][0]][
        "exp_avg"
    ]
    rest_reset = rest_optimizer.state[
        rest_optimizer.param_groups[0]["params"][0]
    ]["exp_avg"]
    assert torch.allclose(
        dc_reset[1, 0, :], torch.zeros_like(dc_reset[1, 0, :])
    )
    assert torch.equal(dc_reset[1, 1:, :], dc_before_reset[1, 1:, :])
    assert torch.allclose(
        rest_reset[1, 1:, :],
        torch.zeros_like(rest_reset[1, 1:, :]),
    )
    assert torch.equal(rest_reset[1, :1, :], rest_before_reset[1, :1, :])

    before_reorder_dc = dc_reset.clone()
    before_reorder_rest = rest_reset.clone()
    order = torch.arange(updated_parameter.shape[0] - 1, -1, -1)
    family_ops.reorder(order)
    reordered_parameter = dc_optimizer.param_groups[0]["params"][0]
    reordered_dc = dc_optimizer.state[reordered_parameter]["exp_avg"]
    reordered_rest = rest_optimizer.state[reordered_parameter]["exp_avg"]
    assert torch.equal(reordered_dc, before_reorder_dc[order])
    assert torch.equal(reordered_rest, before_reorder_rest[order])


def test_gaussian_family_ops_clone_and_split_matches_separate_ops(
    cpu_scene: GaussianScene3D,
) -> None:
    old_state = _state_for_scene(_trainable_scene(cpu_scene))
    new_state = _state_for_scene(_trainable_scene(cpu_scene))
    old_optimizer, old_binding = _logit_optimizer_binding(old_state)
    new_optimizer, new_binding = _logit_optimizer_binding(new_state)
    clone_mask = torch.tensor([True, False, True])
    split_mask = torch.tensor([False, True, False])

    torch.manual_seed(123)
    old_ops = GaussianFamilyOps(old_state, [old_binding])
    old_ops.clone(clone_mask)
    padded_split_mask = torch.cat(
        [
            split_mask,
            torch.zeros(
                int(clone_mask.sum().item()),
                dtype=torch.bool,
            ),
        ]
    )
    old_ops.split(padded_split_mask, num_children=2, scale_shrink=0.625)

    torch.manual_seed(123)
    new_ops = GaussianFamilyOps(new_state, [new_binding])
    new_ops.clone_and_split(
        clone_mask,
        split_mask,
        num_children=2,
        scale_shrink=0.625,
    )

    _assert_scene_tensors_equal(old_state.model.scene, new_state.model.scene)
    assert torch.equal(_exp_avg(old_optimizer), _exp_avg(new_optimizer))


def test_gaussian_family_ops_clone_split_prune_matches_separate_ops(
    cpu_scene: GaussianScene3D,
) -> None:
    old_state = _state_for_scene(_trainable_scene(cpu_scene))
    new_state = _state_for_scene(_trainable_scene(cpu_scene))
    old_optimizer, old_binding = _logit_optimizer_binding(old_state)
    new_optimizer, new_binding = _logit_optimizer_binding(new_state)
    clone_mask = torch.tensor([True, False, True])
    split_mask = torch.tensor([False, True, False])

    def keep_fn(scene: GaussianScene3D) -> torch.Tensor:
        keep = torch.ones(scene.center_position.shape[0], dtype=torch.bool)
        keep[0] = False
        keep[-1] = False
        return keep

    torch.manual_seed(321)
    old_ops = GaussianFamilyOps(old_state, [old_binding])
    old_ops.clone(clone_mask)
    padded_split_mask = torch.cat(
        [
            split_mask,
            torch.zeros(
                int(clone_mask.sum().item()),
                dtype=torch.bool,
            ),
        ]
    )
    old_ops.split(padded_split_mask, num_children=2, scale_shrink=0.625)
    old_ops.prune(keep_fn(old_ops.scene))

    torch.manual_seed(321)
    new_ops = GaussianFamilyOps(new_state, [new_binding])
    new_ops.clone_and_split(
        clone_mask,
        split_mask,
        num_children=2,
        scale_shrink=0.625,
        prune_fn=keep_fn,
        prune_field_names=("center_position",),
    )

    _assert_scene_tensors_equal(old_state.model.scene, new_state.model.scene)
    assert torch.equal(_exp_avg(old_optimizer), _exp_avg(new_optimizer))
    assert new_state.diagnostics["metrics"]["refinement_clone_count"] == 2.0
    assert new_state.diagnostics["metrics"]["refinement_split_count"] == 1.0
    assert new_state.diagnostics["metrics"]["refinement_prune_count"] == 2.0


def test_gaussian_family_ops_fused_copy_helpers_match_separate_ops(
    cpu_scene: GaussianScene3D,
) -> None:
    old_state = _state_for_scene(_trainable_scene(cpu_scene))
    new_state = _state_for_scene(_trainable_scene(cpu_scene))
    old_optimizer, old_binding = _logit_optimizer_binding(old_state)
    new_optimizer, new_binding = _logit_optimizer_binding(new_state)
    sampled_indices = torch.tensor([1])
    dead_indices = torch.tensor([0])
    overrides = {"logit_opacity": torch.tensor([4.0])}

    old_ops = GaussianFamilyOps(old_state, [old_binding])
    old_ops.copy_from_indices(
        sampled_indices,
        sampled_indices,
        field_overrides=overrides,
    )
    old_ops.copy_from_indices(
        dead_indices,
        sampled_indices,
        field_overrides=overrides,
    )

    new_ops = GaussianFamilyOps(new_state, [new_binding])
    new_ops.copy_to_indices(
        (
            (sampled_indices, sampled_indices, overrides),
            (dead_indices, sampled_indices, overrides),
        )
    )

    _assert_scene_tensors_equal(old_state.model.scene, new_state.model.scene)
    assert torch.equal(_exp_avg(old_optimizer), _exp_avg(new_optimizer))

    old_ops.append_from_indices(sampled_indices, field_overrides=overrides)
    new_ops.copy_and_append_from_indices(
        sampled_indices,
        field_overrides=overrides,
    )

    _assert_scene_tensors_equal(old_state.model.scene, new_state.model.scene)
    assert torch.equal(_exp_avg(old_optimizer), _exp_avg(new_optimizer))


def test_splatting_training_package_exports() -> None:
    splatting_training = pytest.importorskip("ember_splatting_training")
    assert hasattr(splatting_training, "FusedAdam")
    assert hasattr(splatting_training, "GaussianMCMC")
    assert hasattr(splatting_training, "create_training_viewer")


class _FakeTrainingViewerHandle:
    def __init__(self) -> None:
        self.config = TrainingViewerConfig(
            update_every_steps=2,
            min_update_seconds=0.0,
            pause_poll_seconds=0.0,
        )
        self.viewer = object()
        self.attach_count = 0
        self.render_count = 0
        self.pause_count = 0
        self.progress_count = 0
        self.active_checks = 0

    @property
    def interaction_active(self) -> bool:
        self.active_checks += 1
        return self.active_checks == 1

    def attach_state(self, state: TrainState) -> None:
        del state
        self.attach_count += 1

    def raise_if_stop_requested(self) -> None:
        return

    def pause_for_interaction(self) -> None:
        self.pause_count += 1

    def update_progress(
        self,
        state: TrainState,
        metrics: dict[str, float],
    ) -> None:
        del state, metrics
        self.progress_count += 1

    def maybe_rerender_after_step(self, state: TrainState) -> None:
        del state
        self.render_count += 1


def test_training_viewer_hook_updates_and_renders_after_step() -> None:
    handle = _FakeTrainingViewerHandle()
    hook = TrainingViewerHook(handle)
    state = TrainState(
        model=None,
        step=2,
        seed=0,
        device=torch.device("cpu"),
    )

    hook.before_step(state)
    hook.after_step(state, {"loss": 1.0})

    assert handle.attach_count == 1
    assert handle.pause_count == 1
    assert handle.progress_count == 1
    assert handle.render_count == 1


class _RecordingViewer:
    def __init__(self) -> None:
        self.wait_values: list[bool] = []

    def rerender(self, *, wait: bool) -> None:
        self.wait_values.append(wait)


def test_training_viewer_rerender_is_nonblocking_by_default() -> None:
    viewer = _RecordingViewer()
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(),
        viewer=viewer,
    )

    handle.rerender()

    assert viewer.wait_values == [False]


def test_training_viewer_rerender_can_wait_when_configured() -> None:
    viewer = _RecordingViewer()
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(wait_for_render=True),
        viewer=viewer,
    )

    handle.rerender()

    assert viewer.wait_values == [True]


def test_training_viewer_is_noop_outside_notebook(
    monkeypatch, tmp_path: Path
) -> None:
    del tmp_path
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.mo.running_in_notebook",
        lambda: False,
    )

    handle = create_training_viewer(object(), object())

    assert handle.viewer is None
    assert handle.runtime_hooks() == ()
    assert handle.start_training(object(), object()) is False


def test_training_viewer_progress_hook_does_not_require_viewer() -> None:
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(enabled=False, show_progress=True),
        _running_in_notebook=True,
    )

    hooks = handle.runtime_hooks()

    assert len(hooks) == 1


class _FakeProgressBar:
    def __init__(self) -> None:
        self.updates: list[int] = []
        self.subtitles: list[str | None] = []

    def update(self, *, increment: int, subtitle: str | None = None) -> None:
        self.updates.append(increment)
        self.subtitles.append(subtitle)


class _FakeProgressContext:
    def __init__(self) -> None:
        self.bar = _FakeProgressBar()
        self.closed = False

    def __enter__(self) -> _FakeProgressBar:
        return self.bar

    def __exit__(self, *args: object) -> None:
        del args
        self.closed = True


def test_training_viewer_progress_updates_are_throttled(monkeypatch) -> None:
    progress_context = _FakeProgressContext()
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.mo.status.progress_bar",
        lambda **kwargs: progress_context,
    )
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(
            enabled=False,
            show_progress=True,
            progress_every_steps=10,
        ),
        _training_config=SimpleNamespace(
            runtime=SimpleNamespace(max_steps=25),
        ),
        _running_in_notebook=True,
    )

    for step in range(1, 26):
        handle.update_progress(
            TrainState(
                model=InitializedModel(
                    scene=GaussianScene3D(
                        center_position=torch.zeros((3, 3)),
                        log_scales=torch.zeros((3, 3)),
                        quaternion_orientation=torch.zeros((3, 4)),
                        logit_opacity=torch.zeros((3,)),
                        feature=torch.zeros((3, 1, 3)),
                        sh_degree=0,
                    ),
                    modules={},
                    parameters={},
                ),
                step=step,
                seed=0,
                device=torch.device("cpu"),
            ),
            {"loss": 1.0},
        )

    assert progress_context.bar.updates == [10, 10, 5]
    assert progress_context.bar.subtitles[-1] == "loss=1 | primitives=3"
    assert progress_context.closed is True


def test_training_viewer_snapshot_reports_throughput_and_eta() -> None:
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(enabled=False, show_progress=False),
        _training_config=SimpleNamespace(runtime=SimpleNamespace(max_steps=10)),
    )
    state = TrainState(
        model=InitializedModel(
            scene=GaussianScene3D(
                center_position=torch.zeros((3, 3)),
                log_scales=torch.zeros((3, 3)),
                quaternion_orientation=torch.zeros((3, 4)),
                logit_opacity=torch.zeros((3,)),
                feature=torch.zeros((3, 1, 3)),
                sh_degree=0,
            ),
            modules={},
            parameters={},
        ),
        step=1,
        seed=0,
        device=torch.device("cpu"),
    )

    handle._started_at = time.monotonic() - 2.0
    handle.update_progress(state, {"loss": 1.0})
    handle._throughput_at -= 0.5
    state.step = 3
    handle.update_progress(state, {"loss": 0.5})

    snapshot = handle.snapshot()
    assert snapshot.iterations_per_second == pytest.approx(4.0, abs=1e-3)
    assert snapshot.elapsed_seconds == pytest.approx(2.0, abs=0.1)
    assert snapshot.eta_seconds == pytest.approx(1.75, abs=1e-3)


def test_training_viewer_start_training_rejects_duplicate_runs(
    monkeypatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def fake_run_training(*args, **kwargs):
        del args, kwargs
        started.set()
        release.wait(timeout=2.0)
        return SimpleNamespace(
            state=TrainState(
                model=InitializedModel(
                    scene=GaussianScene3D(
                        center_position=torch.zeros((1, 3)),
                        log_scales=torch.zeros((1, 3)),
                        quaternion_orientation=torch.zeros((1, 4)),
                        logit_opacity=torch.zeros((1,)),
                        feature=torch.zeros((1, 1, 3)),
                        sh_degree=0,
                    ),
                    modules={},
                    parameters={},
                ),
                step=1,
                seed=0,
                device=torch.device("cpu"),
            ),
            history=[{"loss": 1.0}],
            checkpoint_dir="checkpoint",
        )

    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.run_training",
        fake_run_training,
    )
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(enabled=False, show_progress=False),
        _training_config=SimpleNamespace(runtime=SimpleNamespace(max_steps=1)),
        _running_in_notebook=True,
    )

    assert handle.start_training(object()) is True
    assert started.wait(timeout=2.0)
    assert handle.start_training(object()) is False
    release.set()
    assert handle._thread is not None
    handle._thread.join(timeout=2.0)

    snapshot = handle.snapshot()
    assert snapshot.status == "complete"
    assert snapshot.result is not None


def test_training_viewer_snapshot_records_failed_run(monkeypatch) -> None:
    def fake_run_training(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.run_training",
        fake_run_training,
    )
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(enabled=False, show_progress=False),
        _training_config=SimpleNamespace(runtime=SimpleNamespace(max_steps=1)),
        _running_in_notebook=True,
    )

    assert handle.start_training(object()) is True
    assert handle._thread is not None
    handle._thread.join(timeout=2.0)

    snapshot = handle.snapshot()
    assert snapshot.status == "failed"
    assert snapshot.error_text is not None
    assert "RuntimeError: boom" in snapshot.error_text


def test_training_viewer_request_stop_cancels_run_at_step_boundary(
    monkeypatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    state = TrainState(
        model=InitializedModel(
            scene=GaussianScene3D(
                center_position=torch.zeros((1, 3)),
                log_scales=torch.zeros((1, 3)),
                quaternion_orientation=torch.zeros((1, 4)),
                logit_opacity=torch.zeros((1,)),
                feature=torch.zeros((1, 1, 3)),
                sh_degree=0,
            ),
            modules={},
            parameters={},
        ),
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )

    def fake_run_training(*args, **kwargs):
        del args
        hooks = kwargs["runtime_hooks"]
        started.set()
        release.wait(timeout=2.0)
        with pytest.raises(TrainingViewerCancelled):
            hooks[0].before_step(state)
        raise TrainingViewerCancelled

    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.run_training",
        fake_run_training,
    )
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(enabled=False, show_progress=True),
        _training_config=SimpleNamespace(runtime=SimpleNamespace(max_steps=10)),
        _running_in_notebook=True,
    )

    assert handle.start_training(object()) is True
    assert started.wait(timeout=2.0)
    handle.request_stop()
    release.set()
    assert handle._thread is not None
    handle._thread.join(timeout=2.0)

    snapshot = handle.snapshot()
    assert snapshot.status == "cancelled"
    assert snapshot.error_text is None


def test_training_viewer_rerender_cadence_respects_step_and_time() -> None:
    viewer = _RecordingViewer()
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(
            update_every_steps=2,
            min_update_seconds=0.0,
        ),
        viewer=viewer,
    )
    state = TrainState(
        model=None,
        step=2,
        seed=0,
        device=torch.device("cpu"),
    )

    handle.maybe_rerender_after_step(state)
    handle.maybe_rerender_after_step(state)

    assert viewer.wait_values == [False]


class _ViewerCamera:
    height = torch.tensor([2])
    width = torch.tensor([2])

    def to(self, device: torch.device) -> _ViewerCamera:
        del device
        return self


def test_training_viewer_render_uses_stable_snapshot() -> None:
    scene = GaussianScene3D(
        center_position=torch.zeros((1, 3)),
        log_scales=torch.zeros((1, 3)),
        quaternion_orientation=torch.zeros((1, 4)),
        logit_opacity=torch.zeros((1,)),
        feature=torch.ones((1, 1, 3)),
        sh_degree=0,
    )
    state = TrainState(
        model=InitializedModel(
            scene=scene,
            modules={},
            parameters={},
        ),
        step=1,
        seed=0,
        device=torch.device("cpu"),
    )
    handle = TrainingViewerHandle(config=TrainingViewerConfig())
    handle._render_fn = lambda model, camera: model.scene.feature[0, 0].expand(
        2,
        2,
        3,
    )

    handle.update_render_snapshot(state)
    state.model.scene.feature.data.zero_()

    rendered = handle.render(_ViewerCamera())

    assert torch.equal(rendered, torch.ones((2, 2, 3)))


def test_training_viewer_render_falls_back_to_dc_features_for_black_frame() -> (
    None
):
    scene = GaussianScene3D(
        center_position=torch.zeros((1, 3)),
        log_scales=torch.zeros((1, 3)),
        quaternion_orientation=torch.zeros((1, 4)),
        logit_opacity=torch.zeros((1,)),
        feature=torch.tensor([[[0.8, 0.4, 0.2], [-10.0, -10.0, -10.0]]]),
        sh_degree=1,
    )
    state = TrainState(
        model=InitializedModel(
            scene=scene,
            modules={},
            parameters={},
        ),
        step=1,
        seed=0,
        device=torch.device("cpu"),
    )
    handle = TrainingViewerHandle(config=TrainingViewerConfig())

    def render_feature_sum(model, camera):
        del camera
        return model.scene.feature.sum(dim=1)[0].expand(2, 2, 3)

    handle._render_fn = render_feature_sum
    handle.update_render_snapshot(state)

    rendered = handle.render(_ViewerCamera())

    assert torch.equal(
        rendered,
        torch.tensor([0.8, 0.4, 0.2]).expand(2, 2, 3),
    )


def test_training_viewer_interaction_boost_uses_boost_cadence() -> None:
    viewer = _RecordingViewer()
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(
            update_every_steps=100,
            min_update_seconds=0.0,
            interaction_boost_seconds=3.0,
            boost_update_every_steps=1,
            boost_min_update_seconds=0.0,
        ),
        viewer=viewer,
    )
    state = TrainState(
        model=None,
        step=1,
        seed=0,
        device=torch.device("cpu"),
    )

    handle.maybe_rerender_after_step(state)
    handle.start_interaction_boost()
    handle.maybe_rerender_after_step(state)

    assert viewer.wait_values == [False]


def test_faster_gs_family_antialiasing_defaults_enabled() -> None:
    assert FasterGSNativeRenderOptions().mip_splatting_screen_filter is True
    assert FastGSNativeRenderOptions().mip_splatting_screen_filter is False
    assert FasterGSDepthNativeRenderOptions().proper_antialiasing is True
    assert GaussianPopNativeRenderOptions().proper_antialiasing is True
