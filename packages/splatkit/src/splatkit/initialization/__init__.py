"""Initialization helpers for declarative training pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor, nn

from splatkit.core.contracts import GaussianScene3D, Scene
from splatkit.data.contracts import PointCloudState, SceneRecord


@beartype
@dataclass(frozen=True)
class InitializedModel:
    """Learnable payload returned by an initializer."""

    scene: Scene
    modules: dict[str, nn.Module]
    parameters: dict[str, nn.Parameter]
    buffers: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> InitializedModel:
        """Move the initialized model to a device."""
        moved_modules = {
            name: module.to(device) for name, module in self.modules.items()
        }
        moved_parameters = {
            name: nn.Parameter(
                parameter.detach().to(device),
                requires_grad=parameter.requires_grad,
            )
            for name, parameter in self.parameters.items()
        }
        moved_buffers = {
            name: buffer.detach().to(device)
            for name, buffer in self.buffers.items()
        }
        return replace(
            self,
            scene=self.scene.to(device),
            modules=moved_modules,
            parameters=moved_parameters,
            buffers=moved_buffers,
        )


def _build_sh_features(
    point_cloud: PointCloudState,
    *,
    sh_degree: int,
    default_color: tuple[float, float, float],
) -> Float[Tensor, "num_points sh_coeffs 3"]:
    num_points = int(point_cloud.points.shape[0])
    sh_coeffs = (sh_degree + 1) ** 2
    feature = torch.zeros((num_points, sh_coeffs, 3), dtype=torch.float32)
    if point_cloud.colors is not None:
        feature[:, 0, :] = point_cloud.colors.to(torch.float32)
    else:
        feature[:, 0, :] = torch.tensor(default_color, dtype=torch.float32)
    return feature


def _require_point_cloud(
    scene_record: SceneRecord,
    explicit_point_cloud: PointCloudState | None,
) -> PointCloudState:
    point_cloud = explicit_point_cloud or scene_record.point_cloud
    if point_cloud is None:
        raise ValueError(
            "SFM initialization requires a point cloud either on the scene "
            "record or passed explicitly."
        )
    return point_cloud


def initialize_gaussian_scene_from_scene_record(
    scene_record: SceneRecord,
    *,
    sh_degree: int = 0,
    initial_scale: float = 0.01,
    initial_opacity: float = 0.1,
    default_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    point_cloud: PointCloudState | None = None,
) -> GaussianScene3D:
    """Build a GaussianScene3D from an SfM point cloud."""
    resolved_point_cloud = _require_point_cloud(scene_record, point_cloud)
    centers = resolved_point_cloud.points.to(torch.float32)
    num_points = int(centers.shape[0])
    log_scales = torch.full(
        (num_points, 3),
        fill_value=float(torch.log(torch.tensor(initial_scale)).item()),
        dtype=torch.float32,
    )
    quaternion_orientation = torch.zeros((num_points, 4), dtype=torch.float32)
    quaternion_orientation[:, 0] = 1.0
    opacity = torch.full((num_points,), initial_opacity, dtype=torch.float32)
    opacity = opacity.clamp(1e-5, 1.0 - 1e-5)
    logit_opacity = torch.logit(opacity)
    feature = _build_sh_features(
        resolved_point_cloud,
        sh_degree=sh_degree,
        default_color=default_color,
    )
    return GaussianScene3D(
        center_position=centers.requires_grad_(True),
        log_scales=log_scales.requires_grad_(True),
        quaternion_orientation=quaternion_orientation.requires_grad_(True),
        logit_opacity=logit_opacity.requires_grad_(True),
        feature=feature.requires_grad_(True),
        sh_degree=sh_degree,
    )


def initialize_gaussian_model_from_scene_record(
    scene_record: SceneRecord,
    *,
    modules: dict[str, nn.Module] | None = None,
    parameters: dict[str, nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    sh_degree: int = 0,
    initial_scale: float = 0.01,
    initial_opacity: float = 0.1,
    default_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    point_cloud: PointCloudState | None = None,
) -> InitializedModel:
    """Build a default Gaussian training payload from scene geometry."""
    scene = initialize_gaussian_scene_from_scene_record(
        scene_record,
        sh_degree=sh_degree,
        initial_scale=initial_scale,
        initial_opacity=initial_opacity,
        default_color=default_color,
        point_cloud=point_cloud,
    )
    return InitializedModel(
        scene=scene,
        modules=dict(modules or {}),
        parameters=dict(parameters or {}),
        buffers=dict(buffers or {}),
        metadata=dict(metadata or {}),
    )


__all__ = [
    "InitializedModel",
    "initialize_gaussian_model_from_scene_record",
    "initialize_gaussian_scene_from_scene_record",
]
