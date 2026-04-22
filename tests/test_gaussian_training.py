from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from splatkit.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from splatkit.core.registry import BACKEND_REGISTRY, register_backend
from splatkit.densification.families import GaussianFamilyOps
from splatkit.initialization import InitializedModel
from splatkit.training.checkpoints import (
    load_checkpoint_dir,
    save_checkpoint_dir,
)
from splatkit.training.config import (
    CheckpointExportConfig,
    LossConfig,
    OptimizationConfig,
    ParameterGroupConfig,
    RenderPipelineSpec,
    TrainingConfig,
)
from splatkit.training.protocols import TrainState
from splatkit.training.runtime import OptimizerBinding, build_optimizer_set
from splatkit_native_faster_gs.faster_gs.renderer import (
    FasterGSNativeRenderOptions,
)
from splatkit_native_faster_gs.faster_gs_depth.renderer import (
    FasterGSDepthNativeRenderOptions,
)
from splatkit_native_faster_gs.gaussian_pop.renderer import (
    GaussianPopNativeRenderOptions,
)
from torch.optim import Optimizer


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
            options,
        )
        color = scene.feature[:, 0, :].mean(dim=0)
        render = color.view(1, 1, 1, 3).expand(
            camera.width.shape[0],
            int(camera.height[0].item()),
            int(camera.width[0].item()),
            3,
        )
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
                    selector="scene.logit_opacity",
                    optimizer=optimizer,
                    optimizer_kwargs=dict(optimizer_kwargs or {}),
                    lr=0.1,
                )
            ]
        ),
        checkpoint=CheckpointExportConfig(output_dir=Path("unused")),
    )


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
        selector="scene.logit_opacity",
        optimizer=optimizer,
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


def test_gaussian_training_package_exports() -> None:
    gaussian_training = pytest.importorskip("splatkit_gaussian_training")
    assert hasattr(gaussian_training, "FusedAdam")
    assert hasattr(gaussian_training, "GaussianMCMC")


def test_faster_gs_family_antialiasing_defaults_enabled() -> None:
    assert FasterGSNativeRenderOptions().proper_antialiasing is True
    assert FasterGSDepthNativeRenderOptions().proper_antialiasing is True
    assert GaussianPopNativeRenderOptions().proper_antialiasing is True
