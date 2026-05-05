"""Compatibility exports for initialization helpers."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from ember_core.core.contracts import GaussianScene3D
from ember_core.data.contracts import PointCloudState, SceneRecord
from ember_core.initialization import (
    InitializedModel,
    initialize_gaussian_model_from_scene_record,
    initialize_gaussian_scene_from_scene_record,
)


def initialize_gaussian_scene_from_point_cloud(
    scene_record: SceneRecord,
    *,
    sh_degree: int = 0,
    initial_scale: float = 0.01,
    initial_opacity: float = 0.1,
    default_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    point_cloud: PointCloudState | None = None,
) -> GaussianScene3D:
    """Build a GaussianScene3D from an SfM point cloud on a scene record."""
    return initialize_gaussian_scene_from_scene_record(
        scene_record,
        sh_degree=sh_degree,
        initial_scale=initial_scale,
        initial_opacity=initial_opacity,
        default_color=default_color,
        point_cloud=point_cloud,
    )


def initialize_gaussian_model_from_dataset(
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
    """Compatibility wrapper for scene-record Gaussian initialization."""
    return initialize_gaussian_model_from_scene_record(
        scene_record,
        modules=modules,
        parameters=parameters,
        buffers=buffers,
        metadata=metadata,
        sh_degree=sh_degree,
        initial_scale=initial_scale,
        initial_opacity=initial_opacity,
        default_color=default_color,
        point_cloud=point_cloud,
    )


__all__ = [
    "InitializedModel",
    "initialize_gaussian_model_from_dataset",
    "initialize_gaussian_model_from_scene_record",
    "initialize_gaussian_scene_from_point_cloud",
    "initialize_gaussian_scene_from_scene_record",
]
