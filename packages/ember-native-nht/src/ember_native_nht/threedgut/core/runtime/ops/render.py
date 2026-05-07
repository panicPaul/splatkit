"""Composed render stage for the native NHT runtime."""

from __future__ import annotations

from typing import Literal

import torch
from ember_native_nht.threedgut.core.runtime.ops._common import CameraModelName
from ember_native_nht.threedgut.core.runtime.ops.intersect import (
    intersect,
)
from ember_native_nht.threedgut.core.runtime.ops.project import project
from ember_native_nht.threedgut.core.runtime.ops.rasterize import (
    rasterize_depth,
    rasterize_features,
)
from ember_native_nht.threedgut.core.runtime.types import RenderResult
from torch import Tensor


def _broadcast_features(
    features: Tensor,
    *,
    num_cameras: int,
) -> Tensor:
    """Broadcast per-Gaussian features to per-camera rasterization input."""
    return features.unsqueeze(0).expand(num_cameras, -1, -1).contiguous()


def _broadcast_opacities(
    opacities: Tensor,
    *,
    num_cameras: int,
) -> Tensor:
    """Broadcast per-Gaussian opacities to per-camera rasterization input."""
    return opacities.unsqueeze(0).expand(num_cameras, -1).contiguous()


def render(
    *,
    center_positions: Tensor,
    quaternions: Tensor,
    scales: Tensor,
    opacities: Tensor,
    features: Tensor,
    world_to_camera_matrices: Tensor,
    camera_intrinsics: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int = 16,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    render_mode: Literal["RGB", "RGB+D", "RGB+ED"] = "RGB+ED",
    near_plane: float = 0.01,
    far_plane: float = 1.0e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    camera_model: CameraModelName = "pinhole",
    center_ray_mode: bool = False,
    ray_direction_scale: float = 3.0,
) -> RenderResult:
    """Run the full staged NHT rasterization pipeline."""
    if center_positions.ndim != 2 or center_positions.shape[-1] != 3:
        raise ValueError(
            "Expected center_positions with shape (num_splats, 3), got "
            f"{tuple(center_positions.shape)}."
        )
    if features.ndim != 2:
        raise ValueError(
            "Expected features with shape (num_splats, feature_dim), got "
            f"{tuple(features.shape)}."
        )
    if camera_model not in {"pinhole", "ortho", "fisheye", "ftheta"}:
        raise ValueError(f"Unsupported NHT camera model: {camera_model!r}.")

    num_cameras = int(world_to_camera_matrices.shape[0])
    calculate_compensations = rasterize_mode == "antialiased"
    projection = project(
        center_positions=center_positions,
        quaternions=quaternions,
        scales=scales,
        opacities=opacities,
        world_to_camera_matrices=world_to_camera_matrices,
        camera_intrinsics=camera_intrinsics,
        image_width=image_width,
        image_height=image_height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        calculate_compensations=calculate_compensations,
        camera_model=camera_model,
    )

    tiled_opacities = _broadcast_opacities(opacities, num_cameras=num_cameras)
    if calculate_compensations and projection.compensations is not None:
        tiled_opacities = tiled_opacities * projection.compensations

    intersections = intersect(
        projected_means=projection.projected_means,
        radii=projection.radii,
        primitive_depths=projection.primitive_depths,
        num_cameras=num_cameras,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )

    feature_result = rasterize_features(
        center_positions=center_positions,
        quaternions=quaternions,
        scales=scales,
        features=_broadcast_features(features, num_cameras=num_cameras),
        opacities=tiled_opacities,
        world_to_camera_matrices=world_to_camera_matrices,
        camera_intrinsics=camera_intrinsics,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        tile_offsets=intersections.tile_offsets,
        flattened_gaussian_ids=intersections.flattened_gaussian_ids,
        camera_model=camera_model,
        center_ray_mode=center_ray_mode,
        ray_direction_scale=ray_direction_scale,
    )
    rendered_features = feature_result.features

    if render_mode in {"RGB+D", "RGB+ED"}:
        # Depth is evaluated by the native eval3d path using projected depths as
        # the per-Gaussian scalar payload.
        depth_result = rasterize_depth(
            center_positions=center_positions,
            quaternions=quaternions,
            scales=scales,
            depth_features=projection.primitive_depths[..., None],
            opacities=tiled_opacities,
            world_to_camera_matrices=world_to_camera_matrices,
            camera_intrinsics=camera_intrinsics,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            tile_offsets=intersections.tile_offsets,
            flattened_gaussian_ids=intersections.flattened_gaussian_ids,
            camera_model=camera_model,
        )
        rendered_depths = depth_result.depths
        if render_mode == "RGB+ED":
            rendered_depths = rendered_depths / feature_result.alphas.clamp_min(
                1.0e-10
            )
        rendered_features = torch.cat(
            (rendered_features, rendered_depths), dim=-1
        )

    return RenderResult(
        renders=rendered_features,
        alphas=feature_result.alphas,
        projection=projection,
        intersections=intersections,
    )
