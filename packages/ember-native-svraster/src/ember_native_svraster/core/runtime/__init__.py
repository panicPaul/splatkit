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
    w2c_matrix: Tensor,
    c2w_matrix: Tensor,
    near: float,
    octree_paths: Tensor,
    vox_centers: Tensor,
    vox_lengths: Tensor,
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
            w2c_matrix,
            c2w_matrix,
            near,
            octree_paths,
            vox_centers,
            vox_lengths,
        )
    )


def sh_eval(
    *,
    active_sh_degree: int,
    vox_centers: Tensor,
    cam_pos: Tensor,
    sh0: Tensor,
    shs: Tensor,
    indices: Tensor | None = None,
) -> Tensor:
    """Evaluate SVRaster spherical harmonics into per-voxel RGB values."""
    if indices is None:
        indices = torch.empty(
            (0,),
            device=vox_centers.device,
            dtype=torch.int64,
        )
    return sh_eval_op(
        active_sh_degree,
        indices,
        vox_centers,
        cam_pos,
        sh0,
        shs,
    )


def rasterize(
    *,
    n_samp_per_vox: int,
    image_width: int,
    image_height: int,
    tanfovx: float,
    tanfovy: float,
    cx: float,
    cy: float,
    w2c_matrix: Tensor,
    c2w_matrix: Tensor,
    bg_color: float,
    need_depth: bool,
    need_normal: bool,
    track_max_w: bool,
    octree_paths: Tensor,
    vox_centers: Tensor,
    vox_lengths: Tensor,
    geos: Tensor,
    rgbs: Tensor,
    subdivision_priority: Tensor,
    geom_buffer: Tensor,
) -> RasterizeResult:
    """Run the native rasterize stage."""
    return parse_rasterize_outputs(
        rasterize_op(
            n_samp_per_vox,
            image_width,
            image_height,
            tanfovx,
            tanfovy,
            cx,
            cy,
            w2c_matrix,
            c2w_matrix,
            bg_color,
            need_depth,
            need_normal,
            track_max_w,
            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,
            subdivision_priority,
            geom_buffer,
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
    w2c_matrix: Tensor,
    c2w_matrix: Tensor,
    near: float,
    bg_color: float,
    octree_paths: Tensor,
    vox_centers: Tensor,
    vox_lengths: Tensor,
    geos: Tensor,
    sh0: Tensor,
    shs: Tensor,
    need_depth: bool,
    need_normal: bool = False,
    track_max_w: bool = False,
) -> RasterizeResult:
    """Run the full staged SVRaster render path."""
    preprocess_result = preprocess(
        image_width=image_width,
        image_height=image_height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=cx,
        cy=cy,
        w2c_matrix=w2c_matrix,
        c2w_matrix=c2w_matrix,
        near=near,
        octree_paths=octree_paths,
        vox_centers=vox_centers,
        vox_lengths=vox_lengths,
    )
    rgbs = sh_eval(
        active_sh_degree=active_sh_degree,
        vox_centers=vox_centers,
        cam_pos=c2w_matrix[:3, 3],
        sh0=sh0,
        shs=shs,
    )
    subdivision_priority = torch.ones(
        (int(octree_paths.numel()), 1),
        dtype=sh0.dtype,
        device=sh0.device,
    )
    return rasterize(
        n_samp_per_vox=1,
        image_width=image_width,
        image_height=image_height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=cx,
        cy=cy,
        w2c_matrix=w2c_matrix,
        c2w_matrix=c2w_matrix,
        bg_color=bg_color,
        need_depth=need_depth,
        need_normal=need_normal,
        track_max_w=track_max_w,
        octree_paths=octree_paths,
        vox_centers=vox_centers,
        vox_lengths=vox_lengths,
        geos=geos,
        rgbs=rgbs,
        subdivision_priority=subdivision_priority,
        geom_buffer=preprocess_result.geom_buffer,
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
