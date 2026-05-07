"""Compatibility wrapper over the staged native NHT runtime."""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from ember_native_nht.threedgut.core.runtime import render


def rasterization_nht(
    *,
    means: Float[Tensor, " num_splats 3"],
    quats: Float[Tensor, " num_splats 4"],
    scales: Float[Tensor, " num_splats 3"],
    opacities: Float[Tensor, " num_splats"],
    colors: Float[Tensor, " num_splats feature_dim"],
    viewmats: Float[Tensor, " num_cams 4 4"],
    Ks: Float[Tensor, " num_cams 3 3"],
    width: int,
    height: int,
    tile_size: int = 16,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    render_mode: Literal["RGB", "RGB+D", "RGB+ED"] = "RGB+ED",
    near_plane: float = 0.01,
    far_plane: float = 1.0e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
    center_ray_mode: bool = False,
    ray_dir_scale: float = 3.0,
) -> tuple[
    Float[Tensor, " num_cams height width channels"],
    Float[Tensor, " num_cams height width 1"],
    dict[str, Tensor],
]:
    """Render NHT features with the staged native pipeline."""
    render_result = render(
        center_positions=means,
        quaternions=quats,
        scales=scales,
        opacities=opacities,
        features=colors,
        world_to_camera_matrices=viewmats,
        camera_intrinsics=Ks,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        rasterize_mode=rasterize_mode,
        render_mode=render_mode,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        eps2d=eps2d,
        camera_model=camera_model,
        center_ray_mode=center_ray_mode,
        ray_direction_scale=ray_dir_scale,
    )
    metadata = {
        "camera_width": torch.as_tensor(width, device=means.device),
        "camera_height": torch.as_tensor(height, device=means.device),
        "tiles_per_gauss": render_result.intersections.tiles_per_gaussian,
        "isect_ids": render_result.intersections.tile_intersection_ids,
        "flatten_ids": render_result.intersections.flattened_gaussian_ids,
        "isect_offsets": render_result.intersections.tile_offsets,
        "radii": render_result.projection.radii,
        "means2d": render_result.projection.projected_means,
        "depths": render_result.projection.primitive_depths,
        "conics": render_result.projection.conics,
    }
    return render_result.renders, render_result.alphas, metadata
