"""Public staged runtime API for the SVRaster native backend."""

from __future__ import annotations

import torch
from torch import Tensor

from ember_native_svraster.core.runtime import utils
from ember_native_svraster.core.runtime.gather import (
    gather_triinterp_feat_params,
    gather_triinterp_geo_params,
)
from ember_native_svraster.core.runtime.ops import (
    preprocess_op,
    rasterize_op,
    sh_eval_op,
)
from ember_native_svraster.core.runtime.packing import (
    parse_preprocess_outputs,
    parse_rasterize_outputs,
)
from ember_native_svraster.core.runtime.types import (
    PreprocessResult,
    RasterizeResult,
)


def preprocess(
    *,
    image_width: int,
    image_height: int,
    tanfovx: float,
    tanfovy: float,
    cx: float,
    cy: float,
    world_to_camera: Tensor,
    camera_to_world: Tensor,
    near: float,
    octree_paths: Tensor,
    voxel_centers: Tensor,
    voxel_lengths: Tensor,
) -> PreprocessResult:
    """Run the native preprocess stage."""
    return parse_preprocess_outputs(
        preprocess_op(
            image_width,
            image_height,
            tanfovx,
            tanfovy,
            cx,
            cy,
            world_to_camera,
            camera_to_world,
            near,
            octree_paths,
            voxel_centers,
            voxel_lengths,
        )
    )


def sh_eval(
    *,
    active_sh_degree: int,
    voxel_centers: Tensor,
    camera_position: Tensor,
    sh0: Tensor,
    shs: Tensor,
    indices: Tensor | None = None,
) -> Tensor:
    """Evaluate SVRaster spherical harmonics into per-voxel RGB values."""
    if indices is None:
        indices = torch.empty(
            (0,),
            device=voxel_centers.device,
            dtype=torch.int64,
        )
    return sh_eval_op(
        active_sh_degree,
        indices,
        voxel_centers,
        camera_position,
        sh0,
        shs,
    )


def rasterize(
    *,
    samples_per_voxel: int,
    image_width: int,
    image_height: int,
    tanfovx: float,
    tanfovy: float,
    cx: float,
    cy: float,
    world_to_camera: Tensor,
    camera_to_world: Tensor,
    background_color: float,
    return_depth: bool,
    return_normal: bool,
    track_max_weight: bool,
    sort_rank_max_level: int,
    octree_paths: Tensor,
    voxel_centers: Tensor,
    voxel_lengths: Tensor,
    voxel_geometries: Tensor,
    voxel_colors: Tensor,
    subdivision_priority: Tensor,
    geometry_buffer: Tensor,
    color_concentration_weight: float = 0.0,
    ascending_weight: float = 0.0,
    distortion_weight: float = 0.0,
    ground_truth_color: Tensor | None = None,
    debug: bool = False,
) -> RasterizeResult:
    """Run the native rasterize stage."""
    if ground_truth_color is None:
        ground_truth_color = torch.empty(
            (0,),
            dtype=voxel_colors.dtype,
            device=voxel_colors.device,
        )
    return parse_rasterize_outputs(
        rasterize_op(
            samples_per_voxel,
            image_width,
            image_height,
            tanfovx,
            tanfovy,
            cx,
            cy,
            world_to_camera,
            camera_to_world,
            background_color,
            return_depth,
            return_normal,
            track_max_weight,
            sort_rank_max_level,
            color_concentration_weight,
            ascending_weight,
            distortion_weight,
            ground_truth_color,
            debug,
            octree_paths,
            voxel_centers,
            voxel_lengths,
            voxel_geometries,
            voxel_colors,
            subdivision_priority,
            geometry_buffer,
        )
    ).result


def render(
    *,
    active_sh_degree: int,
    image_width: int,
    image_height: int,
    tanfovx: float,
    tanfovy: float,
    cx: float,
    cy: float,
    world_to_camera: Tensor,
    camera_to_world: Tensor,
    near: float,
    background_color: float,
    octree_paths: Tensor,
    voxel_centers: Tensor,
    voxel_lengths: Tensor,
    voxel_geometries: Tensor,
    sh0: Tensor,
    shs: Tensor,
    return_depth: bool,
    return_normal: bool = False,
    track_max_weight: bool = False,
    samples_per_voxel: int = 1,
    sort_rank_max_level: int | None = None,
    subdivision_priority: Tensor | None = None,
    color_concentration_weight: float = 0.0,
    ascending_weight: float = 0.0,
    distortion_weight: float = 0.0,
    ground_truth_color: Tensor | None = None,
    debug: bool = False,
    color_mode: str = "sh",
) -> RasterizeResult:
    """Run the full staged SVRaster render path."""
    preprocess_result = preprocess(
        image_width=image_width,
        image_height=image_height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=cx,
        cy=cy,
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        near=near,
        octree_paths=octree_paths,
        voxel_centers=voxel_centers,
        voxel_lengths=voxel_lengths,
    )
    if color_mode == "dontcare":
        voxel_colors = torch.zeros(
            (int(octree_paths.numel()), 3),
            dtype=sh0.dtype,
            device=sh0.device,
        )
    elif color_mode == "sh":
        voxel_colors = sh_eval(
            active_sh_degree=active_sh_degree,
            voxel_centers=voxel_centers,
            camera_position=camera_to_world[:3, 3],
            sh0=sh0,
            shs=shs,
        )
    else:
        raise ValueError(f"Unsupported SVRaster color mode: {color_mode!r}.")
    if subdivision_priority is None:
        subdivision_priority = torch.ones(
            (int(octree_paths.numel()), 1),
            dtype=sh0.dtype,
            device=sh0.device,
        )
    if sort_rank_max_level is None:
        sort_rank_max_level = utils.max_num_levels()
    return rasterize(
        samples_per_voxel=samples_per_voxel,
        image_width=image_width,
        image_height=image_height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=cx,
        cy=cy,
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        background_color=background_color,
        return_depth=return_depth,
        return_normal=return_normal,
        track_max_weight=track_max_weight,
        sort_rank_max_level=sort_rank_max_level,
        color_concentration_weight=color_concentration_weight,
        ascending_weight=ascending_weight,
        distortion_weight=distortion_weight,
        ground_truth_color=ground_truth_color,
        debug=debug,
        octree_paths=octree_paths,
        voxel_centers=voxel_centers,
        voxel_lengths=voxel_lengths,
        voxel_geometries=voxel_geometries,
        voxel_colors=voxel_colors,
        subdivision_priority=subdivision_priority,
        geometry_buffer=preprocess_result.geom_buffer,
    )


__all__ = [
    "PreprocessResult",
    "RasterizeResult",
    "gather_triinterp_feat_params",
    "gather_triinterp_geo_params",
    "preprocess",
    "rasterize",
    "render",
    "sh_eval",
    "utils",
]
