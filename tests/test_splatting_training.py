from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import ember_core.data.adapters as data_adapters
import ember_splatting_training.training_viewer as training_viewer_module
import pytest
import torch
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
    Scene,
)
from ember_core.core.families import GAUSSIAN
from ember_core.core.products import SCREEN_SPACE_DENSIFICATION_SIGNALS
from ember_core.core.registry import BACKEND_REGISTRY, register_backend
from ember_core.data import DatasetFrame, PreparedFrameSample
from ember_core.densification.contracts import DensificationContext, Schedule
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
from ember_core.viewer import Marimo3DVViewerConfig
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
    GaussianFastGS,
    GaussianMipSplatting3DFilter,
    active_sh_bases_for_step,
    fastergs_training_backend_options,
)
from ember_splatting_training.losses import rgb_l1_dssim_loss
from ember_splatting_training.recipes import (
    Gaussian3DGSOptimizationRecipe,
    gaussian_3dgs_optimization_config,
)
from ember_splatting_training.stages import GAUSSIAN_ACCUMULATE_SCREEN_STATS
from ember_splatting_training.training_viewer import (
    TrainingPreparationSnapshot,
    TrainingViewerCancelled,
    TrainingViewerConfig,
    TrainingViewerErrorMap,
    TrainingViewerHandle,
    TrainingViewerHook,
    TrainingViewInspectorConfig,
    TrainingViewMapSpec,
    TrainingViserViewerConfig,
    create_training_preparation,
    create_training_run,
    create_training_viewer,
    training_preparation_outputs,
    viridis_error_map,
)
from ember_splatting_training.typed_recipes import FastGSDensificationRecipe
from torch.optim import Optimizer


def test_fastgs_typed_recipe_builds_existing_method() -> None:
    recipe = FastGSDensificationRecipe(importance_threshold=5.0)
    method = recipe.build_method()

    assert recipe.scene_family == GAUSSIAN
    assert GAUSSIAN_ACCUMULATE_SCREEN_STATS in recipe.stages()
    assert SCREEN_SPACE_DENSIFICATION_SIGNALS in recipe.products()
    assert isinstance(method, GaussianFastGS)
    assert method.importance_threshold == 5.0


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


def test_gaussian_fastgs_opacity_reset_defaults_match_paper_behavior() -> None:
    method = GaussianFastGS()

    assert method.extra_opacity_reset_iter is None
    assert method.max_reset_opacity == 0.8
    assert method.scheduled_reset_opacity == 0.01
    assert method.final_prune_mode == "fastgs"
    assert method.should_reset_opacity(500) is False
    assert method.should_reset_opacity(3_000) is True


def test_gaussian_fastgs_final_prune_mode_keeps_reference_schedule() -> None:
    method = GaussianFastGS(
        final_prune_start_iter=15_000,
        final_prune_stop_iter=30_000,
        final_prune_every=3_000,
    )

    assert method.should_final_prune(15_000) is False
    assert method.should_final_prune(18_000) is True
    assert method.should_final_prune(30_000) is False


def test_gaussian_fastgs_disabled_final_prune_skips_background_prune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = GaussianScene3D(
        center_position=torch.zeros((2, 3), dtype=torch.float32),
        log_scales=torch.zeros((2, 3), dtype=torch.float32),
        quaternion_orientation=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        logit_opacity=torch.tensor(
            [0.01, 0.5],
            dtype=torch.float32,
        ).logit(),
        feature=torch.zeros((2, 1, 3), dtype=torch.float32),
        sh_degree=0,
    )
    method = GaussianFastGS(
        final_prune_mode="disabled",
        final_prune_start_iter=15_000,
        final_prune_stop_iter=30_000,
        final_prune_every=3_000,
    )
    pruned: list[torch.Tensor] = []

    class _FamilyOps:
        def __init__(self, scene: GaussianScene3D) -> None:
            self.scene = scene

        def prune(self, keep_mask: torch.Tensor) -> None:
            pruned.append(keep_mask.clone())

        def reset_opacity(self, _max_opacity: float) -> None:
            raise AssertionError(
                "disabled final prune should not reset opacity"
            )

    def fail_compute_pruning_score(_context: object) -> torch.Tensor:
        raise AssertionError(
            "disabled final prune should not compute VCP scores"
        )

    monkeypatch.setattr(
        method,
        "compute_pruning_score",
        fail_compute_pruning_score,
    )
    method.family_ops = _FamilyOps(scene)  # type: ignore[assignment]
    context = SimpleNamespace(
        state=SimpleNamespace(model=SimpleNamespace(scene=scene)),
        step=17_999,
    )

    method.pre_optimizer_step(context)  # type: ignore[arg-type]

    assert pruned == []
    assert method.should_final_prune(18_000) is False


def test_gaussian_fastgs_final_prune_keeps_fastgs_opacity_and_vcp_policy() -> (
    None
):
    scene = GaussianScene3D(
        center_position=torch.zeros((3, 3), dtype=torch.float32),
        log_scales=torch.zeros((3, 3), dtype=torch.float32),
        quaternion_orientation=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        logit_opacity=torch.tensor(
            [0.01, 0.5, 0.5],
            dtype=torch.float32,
        ).logit(),
        feature=torch.zeros((3, 1, 3), dtype=torch.float32),
        sh_degree=0,
    )
    method = GaussianFastGS(final_prune_opacity_threshold=0.1)
    keep_masks: list[torch.Tensor] = []

    class _FamilyOps:
        def __init__(self, scene: GaussianScene3D) -> None:
            self.scene = scene

        def prune(self, keep_mask: torch.Tensor) -> None:
            keep_masks.append(keep_mask.clone())

    method.family_ops = _FamilyOps(scene)  # type: ignore[assignment]

    method.final_prune(torch.tensor([0.0, 0.95, 0.2], dtype=torch.float32))

    assert len(keep_masks) == 1
    torch.testing.assert_close(
        keep_masks[0],
        torch.tensor([False, False, True]),
    )


def test_gaussian_fastgs_refinement_prunes_sampled_candidates_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = GaussianFastGS(
        camera_extent=1.0,
        opacity_reset_every=3_000,
        prune_opacity_threshold=0.005,
        max_reset_opacity=0.8,
    )
    scene = GaussianScene3D(
        center_position=torch.zeros((4, 3), dtype=torch.float32),
        log_scales=torch.tensor(
            [
                [-3.0, -3.0, -3.0],
                [-3.0, -3.0, -3.0],
                [-1.6094, -1.6094, -1.6094],
                [-3.0, -3.0, -3.0],
            ],
            dtype=torch.float32,
        ),
        quaternion_orientation=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        logit_opacity=torch.tensor(
            [0.001, 0.5, 0.5, 0.5],
            dtype=torch.float32,
        ).logit(),
        feature=torch.zeros((4, 1, 3), dtype=torch.float32),
        sh_degree=0,
    )
    method.visible_count = torch.ones(4, dtype=torch.float32)
    method.clone_grad_sum = torch.zeros(4, dtype=torch.float32)
    method.split_grad_sum = torch.zeros(4, dtype=torch.float32)
    method.max_screen_radii = torch.tensor(
        [0.0, 0.0, 25.0, 0.0],
        dtype=torch.float32,
    )
    sampled_mask = torch.tensor([True, False, False, False])
    captured: dict[str, torch.Tensor] = {}

    def sample_prune_mask(
        prune_mask: torch.Tensor,
        pruning_score: torch.Tensor,
    ) -> torch.Tensor:
        del pruning_score
        captured["prune_candidate_mask"] = prune_mask.clone()
        return sampled_mask.clone()

    class _FamilyOps:
        def __init__(self, scene: GaussianScene3D) -> None:
            self.scene = scene

        def clone_and_split(self, *_args: object, **kwargs: object) -> None:
            prune_fn = kwargs["prune_fn"]
            assert callable(prune_fn)
            captured["keep_mask"] = prune_fn(self.scene)

        def reset_opacity(self, max_opacity: float) -> None:
            captured["max_reset_opacity"] = torch.tensor(max_opacity)

    monkeypatch.setattr(
        method, "_sample_refinement_prune_mask", sample_prune_mask
    )
    monkeypatch.setattr(
        method,
        "compute_fastgs_scores",
        lambda _context, *, densify: (
            torch.zeros(4, dtype=torch.float32),
            torch.zeros(4, dtype=torch.float32),
        ),
    )
    method.family_ops = _FamilyOps(scene)  # type: ignore[assignment]
    context = SimpleNamespace(
        state=SimpleNamespace(diagnostics={}),
        runtime=None,
    )

    method.adaptive_density_control(context, scene, step=3_001)

    torch.testing.assert_close(
        captured["prune_candidate_mask"],
        torch.tensor([True, False, True, False]),
    )
    torch.testing.assert_close(
        captured["keep_mask"],
        torch.tensor([False, True, True, False]),
    )
    assert float(captured["max_reset_opacity"].item()) == pytest.approx(0.8)
    assert (
        context.state.diagnostics["metrics"][
            "refinement_fastgs_prune_candidate_count"
        ]
        == 2.0
    )
    assert (
        context.state.diagnostics["metrics"][
            "refinement_fastgs_sampled_prune_count"
        ]
        == 1.0
    )


def test_gaussian_fastgs_scores_use_floor_counts_and_l1_ssim_probe_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = GaussianFastGS(probe_view_count=2)
    scene = GaussianScene3D(
        center_position=torch.zeros((2, 3), dtype=torch.float32),
        log_scales=torch.zeros((2, 3), dtype=torch.float32),
        quaternion_orientation=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        logit_opacity=torch.zeros((2,), dtype=torch.float32),
        feature=torch.zeros((2, 1, 3), dtype=torch.float32),
        sh_degree=0,
    )
    monkeypatch.setattr(
        "ember_splatting_training.losses.ssim_score",
        lambda _prediction, _target: torch.tensor(0.5),
    )

    class _ProbeOutput:
        def __init__(self, render: torch.Tensor) -> None:
            self.render = render

    class _Sample:
        def __init__(self, target: float) -> None:
            self.camera = object()
            self.image = torch.full((2, 2, 3), target, dtype=torch.float32)

    class _Attribution:
        def __init__(self) -> None:
            self.values = iter(
                (
                    torch.tensor([3.0, 0.0]),
                    torch.tensor([0.0, 4.0]),
                )
            )

        def attribute_metric_map(self, *_args, **_kwargs) -> torch.Tensor:
            return next(self.values)

    class _Runtime:
        render_options = None

        def __init__(self) -> None:
            self.views = (_Sample(1.0), _Sample(0.25))
            self.renders = iter(
                (
                    torch.zeros((1, 2, 2, 3), dtype=torch.float32),
                    torch.zeros((1, 2, 2, 3), dtype=torch.float32),
                )
            )
            self.attribution = _Attribution()

        def sample_views(self, _count: int):
            return self.views

        def render_raw(self, *_args):
            return _ProbeOutput(next(self.renders))

        def resolve_trait(self, _trait_type):
            return self.attribution

    importance_score, pruning_score = method.compute_fastgs_scores(
        DensificationContext(
            state=SimpleNamespace(
                model=InitializedModel(
                    scene=scene,
                    modules={},
                    parameters={},
                )
            ),
            batch=None,
            render_output=None,
            loss_result=None,
            step=0,
            optimizers=(),
            runtime=_Runtime(),
        ),
        densify=True,
    )

    torch.testing.assert_close(
        importance_score,
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        pruning_score,
        torch.tensor([1.0, 0.0]),
    )


def test_gaussian_fastgs_photometric_loss_casts_half_probes_for_ssim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = GaussianFastGS()
    seen_dtypes: list[tuple[torch.dtype, torch.dtype]] = []

    def fake_ssim_score(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        seen_dtypes.append((prediction.dtype, target.dtype))
        return torch.tensor(0.75, dtype=prediction.dtype)

    monkeypatch.setattr(
        "ember_splatting_training.losses.ssim_score",
        fake_ssim_score,
    )

    loss = method._photometric_loss(
        torch.zeros((2, 2, 3), dtype=torch.float16),
        torch.ones((2, 2, 3), dtype=torch.float16),
    )

    assert loss.dtype == torch.float32
    assert seen_dtypes == [(torch.float32, torch.float32)]
    assert loss.item() == pytest.approx(0.85)


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
    for name in scene.parameter_field_names:
        value = getattr(scene, name)
        if isinstance(value, torch.Tensor):
            updates[name] = (
                value.detach().clone().requires_grad_(value.is_floating_point())
            )
    trainable_scene = scene.detached_copy()
    trainable_scene.replace_fields_(**updates)
    return trainable_scene


def _assert_scene_tensors_equal(
    left: GaussianScene3D,
    right: GaussianScene3D,
) -> None:
    for name in left.field_names:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if isinstance(left_value, torch.Tensor):
            assert torch.equal(left_value, right_value), name


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
    scene = cpu_scene.detached_copy()
    scene.replace_fields_(
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
    scene = cpu_scene.detached_copy()
    scene.replace_fields_(
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


def test_gaussian_mcmc_appends_from_post_relocation_scene(
    cpu_scene: GaussianScene3D,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ember_splatting_training.densification import mcmc as mcmc_module

    scene = _trainable_scene(cpu_scene)
    scene.replace_fields_(
        logit_opacity=torch.tensor([-20.0, 2.0, 1.0], requires_grad=True),
    )
    state = _state_for_scene(scene)
    method = mcmc_module.GaussianMCMC(
        schedule=Schedule(frequency=1),
        min_opacity=0.005,
        cap_growth_factor=2.0,
        inject_position_noise=False,
    )
    family_ops = GaussianFamilyOps(state, [])
    method.bind(state, [], family_ops)
    multinomial_calls = 0

    def fake_relocation_adjustment(
        old_opacities: torch.Tensor,
        old_scales: torch.Tensor,
        n_samples_per_primitive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del n_samples_per_primitive
        return old_opacities.clamp_min(0.8)[:, None], old_scales

    def fake_multinomial(
        weights: torch.Tensor,
        num_samples: int,
        replacement: bool,
    ) -> torch.Tensor:
        nonlocal multinomial_calls
        del replacement
        multinomial_calls += 1
        if multinomial_calls == 1:
            return torch.zeros(num_samples, dtype=torch.long)
        assert weights[0] > 0.79
        return torch.arange(num_samples, dtype=torch.long) % weights.numel()

    monkeypatch.setattr(
        mcmc_module,
        "relocation_adjustment",
        fake_relocation_adjustment,
    )
    monkeypatch.setattr(mcmc_module.torch, "multinomial", fake_multinomial)

    method.post_optimizer_step(
        DensificationContext(
            state=state,
            batch=None,
            render_output=None,
            loss_result=None,
            step=0,
            optimizers=[],
        )
    )

    assert multinomial_calls == 2
    assert state.model.scene.logit_opacity.shape == (6,)
    assert torch.sigmoid(state.model.scene.logit_opacity[0]) > 0.79


def test_splatting_training_package_exports() -> None:
    splatting_training = pytest.importorskip("ember_splatting_training")
    assert hasattr(splatting_training, "FusedAdam")
    assert hasattr(splatting_training, "GaussianMCMC")
    assert hasattr(splatting_training, "create_training_preparation")
    assert hasattr(splatting_training, "create_training_run")
    assert hasattr(splatting_training, "create_training_view_inspector")
    assert hasattr(splatting_training, "create_training_viewer")
    assert hasattr(splatting_training, "render_training_preparation_status")
    assert hasattr(splatting_training, "training_inspector_spinner")
    assert hasattr(splatting_training, "training_preparation_outputs")


def test_training_viewer_default_snapshot_cadence_is_100() -> None:
    assert TrainingViewerConfig().update_every_steps == 100


def test_training_view_inspector_config_validates_l1_range() -> None:
    config = TrainingViewInspectorConfig()

    assert config.l1_range == (0.0, 0.25)
    with pytest.raises(ValueError, match="l1_range must lie"):
        TrainingViewInspectorConfig(l1_range=(0.0, 2.0))


class _NoValueElement:
    def __init__(self, element_id: str, frontend_value: object, value: object):
        self._id = element_id
        self._value_frontend = frontend_value
        self._converted_value = value
        self.registered_parent = None
        self.registered_key = None

    @property
    def value(self) -> object:
        raise AssertionError("TrainingViewInspector must not read child .value")

    def _convert_value(self, value: object) -> object:
        del value
        return self._converted_value

    def _register_as_view(self, *, parent: object, key: str) -> None:
        self.registered_parent = parent
        self.registered_key = key


def test_training_view_inspector_initializes_without_child_value_reads() -> (
    None
):
    validation = _NoValueElement("validation", ["Validation"], "val:0:0")
    show_validation = _NoValueElement("show-validation", 0, 0)
    training = _NoValueElement("training", ["Training"], "train:0:0")
    show_training = _NoValueElement("show-training", 0, 0)
    l1_range = _NoValueElement("l1-range", [0.0, 0.25], [0.0, 0.25])
    controls = training_viewer_module.TrainingViewInspectorControls(
        view=object(),
        validation_view_selector=validation,
        show_validation_view_button=show_validation,
        training_view_selector=training,
        show_training_view_button=show_training,
        l1_range_slider=l1_range,
    )
    inspector = training_viewer_module.TrainingViewInspector(
        config=TrainingViewInspectorConfig(),
        controls=controls,
        elements={
            training_viewer_module._INSPECTOR_VALIDATION_VIEW_KEY: validation,
            training_viewer_module._INSPECTOR_SHOW_VALIDATION_KEY: show_validation,
            training_viewer_module._INSPECTOR_TRAINING_VIEW_KEY: training,
            training_viewer_module._INSPECTOR_SHOW_TRAINING_KEY: show_training,
            training_viewer_module._INSPECTOR_L1_RANGE_KEY: l1_range,
        },
    )

    catalog = SimpleNamespace(view_ref_by_key=lambda key: key)

    assert inspector.selected_view_ref(catalog) == "val:0:0"
    assert inspector.l1_value_range() == (0.0, 0.25)
    assert validation.registered_parent is inspector

    inspector._convert_value(
        {
            training_viewer_module._INSPECTOR_SHOW_TRAINING_KEY: 1,
        }
    )

    assert inspector.selected_view_ref(catalog) == "train:0:0"


def test_training_preparation_status_uses_marimo_spinner() -> None:
    status_view = training_viewer_module.render_training_preparation_status(
        TrainingPreparationSnapshot(
            status="loading_scene",
            elapsed_seconds=12.5,
        )
    )

    assert status_view.__class__.__name__ == "spinner"
    assert status_view.title == "Preparing training inspector"
    assert status_view.subtitle == "Loading scene... elapsed 12s"


def test_training_preparation_status_uses_progress_bar_for_progress() -> None:
    status_view = training_viewer_module.render_training_preparation_status(
        TrainingPreparationSnapshot(
            status="preparing_views",
            elapsed_seconds=12.5,
            progress_label="Materializing prepared dataset",
            progress_current=2,
            progress_total=4,
        )
    )

    assert status_view.__class__.__name__ == "ProgressBar"
    assert status_view.title == "Preparing training inspector"
    assert (
        status_view.subtitle
        == "Materializing prepared dataset: 2/4 | elapsed 12s"
    )
    assert status_view.current == 2
    assert status_view.total == 4


def test_training_preparation_handle_runs_inline_and_publishes_snapshot() -> (
    None
):
    scene_record = object()
    frame_dataset = object()
    frame_view_catalog = SimpleNamespace(training_dataset=frame_dataset)
    handle, snapshot = create_training_preparation(
        load_scene=lambda: scene_record,
        prepare_frame_view_catalog=lambda scene: frame_view_catalog,
    )

    assert handle.start(wait=True) is True
    assert handle.start(wait=True) is False

    published = snapshot()
    assert published.status == "ready"
    assert published.scene_record is scene_record
    assert published.frame_view_catalog is frame_view_catalog
    assert published.frame_dataset is frame_dataset
    assert published.elapsed_seconds is not None


def test_training_preparation_handle_publishes_materialization_progress() -> (
    None
):
    scene_record = object()
    frame_dataset = object()
    frame_view_catalog = SimpleNamespace(training_dataset=frame_dataset)

    def prepare_frame_view_catalog(scene: object) -> object:
        assert scene is scene_record
        data_adapters._report_materialization_progress(
            label="Materializing prepared dataset",
            current=1,
            total=2,
        )
        return frame_view_catalog

    handle, snapshot = create_training_preparation(
        load_scene=lambda: scene_record,
        prepare_frame_view_catalog=prepare_frame_view_catalog,
    )

    assert handle.start(wait=True) is True

    published = snapshot()
    assert published.status == "ready"
    assert published.progress_label == "Materializing prepared dataset"
    assert published.progress_current == 1
    assert published.progress_total == 2


def test_training_preparation_outputs_route_load_errors() -> None:
    snapshot = TrainingPreparationSnapshot(
        status="failed",
        error_text="load exploded",
        error_phase="loading_scene",
    )

    scene_error, scene_record, frame_dataset, dataset_error, catalog = (
        training_preparation_outputs(snapshot)
    )

    assert isinstance(scene_error, RuntimeError)
    assert "load exploded" in str(scene_error)
    assert scene_record is None
    assert frame_dataset is None
    assert dataset_error is None
    assert catalog is None


def test_training_preparation_outputs_route_view_errors() -> None:
    scene_record = object()
    snapshot = TrainingPreparationSnapshot(
        status="failed",
        scene_record=scene_record,
        error_text="views exploded",
        error_phase="preparing_views",
    )

    scene_error, returned_scene, frame_dataset, dataset_error, catalog = (
        training_preparation_outputs(snapshot)
    )

    assert scene_error is None
    assert returned_scene is scene_record
    assert frame_dataset is None
    assert isinstance(dataset_error, RuntimeError)
    assert "views exploded" in str(dataset_error)
    assert catalog is None


def test_training_view_inspector_groups_images_in_two_columns() -> None:
    assert training_viewer_module._two_column_rows([1, 2, 3, 4, 5]) == [
        (1, 2),
        (3, 4),
        (5,),
    ]


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


class _CountingFrameDataset:
    def __init__(self, *, raw_camera: CameraState, sample_camera: CameraState):
        self.indices = [0]
        self.camera_stream = SimpleNamespace(
            frames=[SimpleNamespace(camera_index=0)],
            camera=raw_camera,
        )
        self.sample_camera = sample_camera
        self.len_count = 0
        self.getitem_count = 0
        self.prepared_camera_count = 0

    def __len__(self) -> int:
        self.len_count += 1
        return 1

    def __getitem__(self, index: int) -> SimpleNamespace:
        self.getitem_count += 1
        assert index == 0
        return SimpleNamespace(camera=self.sample_camera)

    def prepared_camera(self, index: int) -> CameraState:
        self.prepared_camera_count += 1
        assert index == 0
        return self.sample_camera


class _NonIndexableFrameDataset:
    def __len__(self) -> int:
        raise AssertionError("dataset should not be measured")

    def __getitem__(self, index: int) -> SimpleNamespace:
        del index
        raise AssertionError("dataset should not be indexed")


def _viewer_test_camera(width: int, height: int) -> CameraState:
    return CameraState(
        width=torch.tensor([width], dtype=torch.int64),
        height=torch.tensor([height], dtype=torch.int64),
        fov_degrees=torch.tensor([50.0], dtype=torch.float32),
        cam_to_world=torch.eye(4, dtype=torch.float32)[None],
        camera_convention="opencv",
    )


def _training_config_stub() -> SimpleNamespace:
    return SimpleNamespace(runtime=SimpleNamespace(max_steps=1))


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


def test_training_viewer_snaps_to_prepared_camera() -> None:
    class CameraRecordingViewer:
        def __init__(self) -> None:
            self.camera = None
            self.wait = None

        def set_camera_state(self, camera, *, wait: bool = False) -> None:
            self.camera = camera
            self.wait = wait

    viewer = CameraRecordingViewer()
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(wait_for_render=True),
        viewer=viewer,
    )

    snapped = handle.snap_to_camera(_viewer_test_camera(320, 180))

    assert snapped is True
    assert viewer.camera is not None
    assert viewer.camera.width == 320
    assert viewer.camera.height == 180
    assert viewer.wait is True


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


def test_training_run_creates_noninteractive_notebook_handle(
    monkeypatch,
) -> None:
    training_viewer_module._ACTIVE_TRAINING_VIEWERS.clear()
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.mo.running_in_notebook",
        lambda: True,
    )
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.launch_viewer",
        lambda *args, **kwargs: pytest.fail("viewer should not launch"),
    )

    handle = create_training_run(object(), _training_config_stub())

    try:
        assert handle.viewer is None
        assert len(handle.runtime_hooks()) == 1
    finally:
        handle.close(join_timeout=0.0)


def test_training_viewer_uses_prepared_dataset_camera(monkeypatch) -> None:
    raw_camera = _viewer_test_camera(1024, 768)
    prepared_camera = _viewer_test_camera(128, 96)
    frame_dataset = _CountingFrameDataset(
        raw_camera=raw_camera,
        sample_camera=prepared_camera,
    )
    captured: dict[str, object] = {}

    class FakeViewer:
        interaction_active = False

        def __init__(self) -> None:
            self.closed = False

        def rerender(self, *, wait: bool = False) -> None:
            del wait

        def close(self) -> None:
            self.closed = True

    def fake_launch_viewer(
        render_fn,
        *,
        state,
        marimo_3dv_config=None,
    ):
        del render_fn
        captured["state"] = state
        captured["marimo_3dv_config"] = marimo_3dv_config
        return FakeViewer()

    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.mo.running_in_notebook",
        lambda: True,
    )
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.launch_viewer",
        fake_launch_viewer,
    )

    handle = create_training_viewer(frame_dataset, _training_config_stub())

    initial_camera = captured["state"].camera
    assert int(initial_camera.width[0]) == 128
    assert int(initial_camera.height[0]) == 96
    assert isinstance(
        captured["marimo_3dv_config"],
        Marimo3DVViewerConfig,
    )
    assert frame_dataset.len_count == 1
    assert frame_dataset.prepared_camera_count == 1
    assert frame_dataset.getitem_count == 0
    handle.close()


def test_training_viewer_initial_camera_overrides_dataset(monkeypatch) -> None:
    explicit_camera = _viewer_test_camera(320, 180)
    frame_dataset = _NonIndexableFrameDataset()
    captured: dict[str, object] = {}

    class FakeViewer:
        interaction_active = False

        def rerender(self, *, wait: bool = False) -> None:
            del wait

        def close(self) -> None:
            pass

    def fake_launch_viewer(
        render_fn,
        *,
        state,
        marimo_3dv_config=None,
    ):
        del render_fn
        captured["state"] = state
        captured["marimo_3dv_config"] = marimo_3dv_config
        return FakeViewer()

    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.mo.running_in_notebook",
        lambda: True,
    )
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.launch_viewer",
        fake_launch_viewer,
    )

    handle = create_training_viewer(
        frame_dataset,
        _training_config_stub(),
        config=TrainingViewerConfig(
            marimo_3dv=Marimo3DVViewerConfig(
                interactive_quality=30,
                interactive_max_side=640,
                interactive_max_fps=4.0,
            )
        ),
        initial_camera=explicit_camera,
    )

    initial_camera = captured["state"].camera
    assert int(initial_camera.width[0]) == 320
    assert int(initial_camera.height[0]) == 180
    assert isinstance(
        captured["marimo_3dv_config"],
        Marimo3DVViewerConfig,
    )
    assert captured["marimo_3dv_config"].interactive_quality == 30
    assert captured["marimo_3dv_config"].interactive_max_side == 640
    assert captured["marimo_3dv_config"].interactive_max_fps == 4.0
    handle.close()


def test_training_viewer_reuses_running_notebook_handle(monkeypatch) -> None:
    class FakeViewer:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    training_viewer_module._ACTIVE_TRAINING_VIEWERS.clear()
    viewer = FakeViewer()
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(),
        viewer=viewer,
        _running_in_notebook=True,
        _training_config=_training_config_stub(),
    )
    handle._status = "running"
    training_viewer_module._register_training_viewer(handle)
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.mo.running_in_notebook",
        lambda: True,
    )
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.launch_viewer",
        lambda *args, **kwargs: pytest.fail(
            "running notebook handles should be reused"
        ),
    )

    reused = create_training_viewer(object(), _training_config_stub())

    assert reused is handle
    assert viewer.closed is False
    handle.close()


def test_training_viewer_replaces_idle_notebook_handle(monkeypatch) -> None:
    class FakeViewer:
        def __init__(self) -> None:
            self.closed = False

        def rerender(self, *, wait: bool = False) -> None:
            del wait

        def close(self) -> None:
            self.closed = True

    training_viewer_module._ACTIVE_TRAINING_VIEWERS.clear()
    idle_viewer = FakeViewer()
    idle_handle = TrainingViewerHandle(
        config=TrainingViewerConfig(),
        viewer=idle_viewer,
        _running_in_notebook=True,
        _training_config=_training_config_stub(),
    )
    training_viewer_module._register_training_viewer(idle_handle)
    launched_viewer = FakeViewer()
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.mo.running_in_notebook",
        lambda: True,
    )
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.launch_viewer",
        lambda *args, **kwargs: launched_viewer,
    )

    handle = create_training_viewer(
        _NonIndexableFrameDataset(),
        _training_config_stub(),
        initial_camera=_viewer_test_camera(320, 180),
    )

    assert handle is not idle_handle
    assert idle_viewer.closed is True
    assert handle.viewer is launched_viewer
    handle.close()


def test_training_viewer_disabled_does_not_index_dataset(monkeypatch) -> None:
    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.mo.running_in_notebook",
        lambda: True,
    )

    handle = create_training_viewer(
        _NonIndexableFrameDataset(),
        _training_config_stub(),
        config=TrainingViewerConfig(enabled=False),
    )

    assert handle.viewer is None
    handle.close()


def test_training_viewer_rejects_empty_dataset(monkeypatch) -> None:
    class EmptyFrameDataset:
        def __len__(self) -> int:
            return 0

    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.mo.running_in_notebook",
        lambda: True,
    )

    with pytest.raises(ValueError, match="non-empty frame dataset"):
        create_training_viewer(EmptyFrameDataset(), _training_config_stub())


def test_training_viewer_can_launch_viser_backend(monkeypatch) -> None:
    import marimo_viser

    frame_dataset = _CountingFrameDataset(
        raw_camera=_viewer_test_camera(800, 600),
        sample_camera=_viewer_test_camera(80, 60),
    )
    captured: dict[str, object] = {}

    class FakeViserViewer:
        interaction_active = False

        def __init__(self, render_fn, *, state, **kwargs) -> None:
            del render_fn
            captured["state"] = state
            captured["kwargs"] = kwargs
            self.closed = False

        def rerender(self, *, wait: bool = False) -> None:
            del wait

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "ember_splatting_training.training_viewer.mo.running_in_notebook",
        lambda: True,
    )
    monkeypatch.setattr(marimo_viser, "ViserViewer", FakeViserViewer)

    handle = create_training_viewer(
        frame_dataset,
        _training_config_stub(),
        config=TrainingViewerConfig(
            viewer_backend="viser",
            viser=TrainingViserViewerConfig(port=18080, viewer_res=512),
        ),
    )

    assert isinstance(handle.viewer, FakeViserViewer)
    assert captured["state"].camera.camera_convention == "opencv"
    assert int(captured["state"].camera.width[0]) == 80
    assert int(captured["state"].camera.height[0]) == 60
    assert frame_dataset.prepared_camera_count == 1
    assert frame_dataset.getitem_count == 0
    assert captured["kwargs"]["mode"] == "training"
    assert captured["kwargs"]["server_config"].port == 18080
    assert captured["kwargs"]["render_config"].viewer_res == 512
    handle.close()


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


def test_training_viewer_snapshot_reports_scene_num_primitives() -> None:
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(enabled=False, show_progress=False),
        _training_config=SimpleNamespace(runtime=SimpleNamespace(max_steps=10)),
    )
    state = TrainState(
        model=InitializedModel(
            scene=_NumPrimitiveScene(7),
            modules={},
            parameters={},
        ),
        step=1,
        seed=0,
        device=torch.device("cpu"),
    )

    handle.update_progress(state, {"loss": 1.0})

    snapshot = handle.snapshot()
    assert snapshot.primitive_count == 7


class _NumPrimitiveScene(Scene):
    def __init__(self, num_primitives: int) -> None:
        super().__init__()
        self.num_primitives = num_primitives

    @property
    def scene_family(self):
        return GAUSSIAN

    def _validate(self) -> None:
        pass


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


def test_training_run_publishes_render_snapshots_without_viewer() -> None:
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
        step=2,
        seed=0,
        device=torch.device("cpu"),
    )
    handle = TrainingViewerHandle(
        config=TrainingViewerConfig(
            update_every_steps=2,
            min_update_seconds=1000.0,
        ),
        _training_config=SimpleNamespace(runtime=SimpleNamespace(max_steps=2)),
        _running_in_notebook=True,
    )

    handle.maybe_rerender_after_step(state)

    assert handle.snapshot().render_step == 2


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


def test_training_viewer_error_map_uses_latest_render_snapshot() -> None:
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
    sample = PreparedFrameSample(
        frame=DatasetFrame(
            frame_id="frame_0",
            sensor_id="camera",
            camera_index=0,
            width=2,
            height=2,
            timestamp_us=0,
        ),
        image=torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
                [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        camera=_viewer_test_camera(2, 2),
    )
    handle = TrainingViewerHandle(config=TrainingViewerConfig())
    handle._render_fn = lambda model, camera: model.scene.feature[0, 0].expand(
        2,
        2,
        3,
    )

    handle.update_render_snapshot(state)
    state.model.scene.feature.data.zero_()
    error_map = handle.render_view_error_map(
        sample,
        quantile=1.0,
        value_range=(0.0, 1.0),
    )

    assert isinstance(error_map, TrainingViewerErrorMap)
    assert error_map.available is True
    assert error_map.image.shape == (2, 2, 3)
    assert error_map.image.dtype == torch.uint8
    torch.testing.assert_close(
        error_map.error,
        torch.tensor([[1.0, 0.75], [0.5, 0.0]]),
    )
    assert error_map.max_error == pytest.approx(1.0)
    assert error_map.mean_error == pytest.approx(0.5625)


def test_training_view_inspection_uses_one_render_for_builtin_and_custom_maps() -> (
    None
):
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
        step=5,
        seed=0,
        device=torch.device("cpu"),
    )
    sample = PreparedFrameSample(
        frame=DatasetFrame(
            frame_id="frame_0",
            sensor_id="camera",
            camera_index=0,
            width=2,
            height=2,
            timestamp_us=0,
        ),
        image=torch.zeros((2, 2, 3), dtype=torch.float32),
        camera=_viewer_test_camera(2, 2),
    )
    render_count = 0

    def render_once(model, camera):
        nonlocal render_count
        del camera
        render_count += 1
        return model.scene.feature[0, 0].expand(2, 2, 3)

    handle = TrainingViewerHandle(config=TrainingViewerConfig())
    handle._render_fn = render_once
    handle.update_render_snapshot(state)
    l1_doubled = TrainingViewMapSpec(
        key="l1x2",
        label="L1 x2",
        fn=lambda context: context.l1_error * 2.0,
        value_range=(0.0, 2.0),
    )
    prediction_rgb = TrainingViewMapSpec(
        key="prediction_rgb",
        label="Prediction RGB",
        fn=lambda context: context.prediction,
        color="rgb",
    )

    inspection = handle.inspect_view(
        sample,
        value_range=(0.0, 1.0),
        map_specs=(l1_doubled, prediction_rgb),
    )
    cached = handle.inspect_view(
        sample,
        value_range=(0.0, 1.0),
        map_specs=(l1_doubled, prediction_rgb),
    )

    assert render_count == 1
    assert cached is inspection
    assert inspection.available is True
    assert inspection.render_step == 5
    assert inspection.target_image.shape == (2, 2, 3)
    assert inspection.prediction_image.dtype == torch.uint8
    torch.testing.assert_close(inspection.l1_error, torch.ones((2, 2)))
    assert inspection.l1_mean == pytest.approx(1.0)
    assert [result.key for result in inspection.maps] == [
        "l1x2",
        "prediction_rgb",
    ]
    assert inspection.maps[0].mean_value == pytest.approx(2.0)
    assert inspection.maps[1].values is None


def test_viridis_error_map_clips_to_explicit_range() -> None:
    colors = viridis_error_map(
        torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32),
        value_range=(0.25, 0.75),
    )
    endpoints = viridis_error_map(
        torch.tensor([[0.25, 0.75]], dtype=torch.float32),
        value_range=(0.25, 0.75),
    )

    assert torch.equal(colors[0, 0], endpoints[0, 0])
    assert torch.equal(colors[0, 2], endpoints[0, 1])
    assert colors.dtype == torch.uint8


def test_training_viewer_error_map_returns_placeholder_before_snapshot() -> (
    None
):
    sample = PreparedFrameSample(
        frame=DatasetFrame(
            frame_id="frame_0",
            sensor_id="camera",
            camera_index=0,
            width=2,
            height=2,
            timestamp_us=0,
        ),
        image=torch.zeros((2, 2, 3), dtype=torch.float32),
        camera=_viewer_test_camera(2, 2),
    )
    handle = TrainingViewerHandle(config=TrainingViewerConfig())

    error_map = handle.render_view_error_map(sample)

    assert error_map.available is False
    assert error_map.image.shape == (2, 2, 3)
    assert torch.equal(
        error_map.image,
        torch.full((2, 2, 3), 245, dtype=torch.uint8),
    )
    assert torch.equal(error_map.error, torch.zeros((2, 2)))


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
