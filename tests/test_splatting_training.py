from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
from ember_native_faster_gs.gaussian_pop.renderer import (
    GaussianPopNativeRenderOptions,
)
from ember_splatting_training.fastergs import (
    GaussianMipSplattingAntialiasing,
    active_sh_bases_for_step,
    fastergs_training_backend_options,
)
from ember_splatting_training.recipes import (
    Gaussian3DGSOptimizationRecipe,
    gaussian_3dgs_optimization_config,
)
from torch.optim import Optimizer


def test_fastergs_active_sh_schedule() -> None:
    assert active_sh_bases_for_step(0) == 1
    assert active_sh_bases_for_step(999) == 1
    assert active_sh_bases_for_step(1000) == 4
    assert active_sh_bases_for_step(2000) == 9
    assert active_sh_bases_for_step(3000) == 16
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


def test_gaussian_mip_splatting_antialiasing_requests_proper_aa() -> None:
    method = GaussianMipSplattingAntialiasing()

    requirements = method.get_render_requirements(object())

    assert requirements.backend_options == {"proper_antialiasing": True}


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
    assert [
        group.target.name for group in config.parameter_groups
    ] == [
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


def test_splatting_training_package_exports() -> None:
    splatting_training = pytest.importorskip("ember_splatting_training")
    assert hasattr(splatting_training, "FusedAdam")
    assert hasattr(splatting_training, "GaussianMCMC")


def test_faster_gs_family_antialiasing_defaults_enabled() -> None:
    assert FasterGSNativeRenderOptions().proper_antialiasing is True
    assert FasterGSDepthNativeRenderOptions().proper_antialiasing is True
    assert GaussianPopNativeRenderOptions().proper_antialiasing is True
