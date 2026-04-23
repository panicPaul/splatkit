"""Rasterize-stage custom ops for the SVRaster native runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from splatkit_native_svraster.core.runtime.ops._common import backend
from splatkit_native_svraster.core.runtime.packing import (
    parse_rasterize_outputs,
)


@torch.library.custom_op("svraster::rasterize", mutates_args=())
def rasterize_op(
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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run the native rasterize stage."""
    del subdivision_priority
    (
        num_rendered,
        binning_buffer,
        image_buffer,
        color,
        depth,
        normal,
        transmittance,
        max_weight,
    ) = backend().rasterize_voxels(
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
        False,
        need_normal,
        track_max_w,
        octree_paths,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,
        geom_buffer,
        False,
    )
    return (
        torch.tensor(
            [num_rendered],
            device=octree_paths.device,
            dtype=torch.int32,
        ),
        binning_buffer,
        image_buffer,
        color,
        depth,
        normal,
        transmittance,
        max_weight,
    )


@rasterize_op.register_fake
def _rasterize_fake(
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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        n_samp_per_vox,
        tanfovx,
        tanfovy,
        cx,
        cy,
        w2c_matrix,
        c2w_matrix,
        bg_color,
        vox_centers,
        vox_lengths,
        rgbs,
        subdivision_priority,
        geom_buffer,
    )
    device = octree_paths.device
    dtype = geos.dtype
    num_voxels = int(geos.shape[0])
    empty = torch.empty((0,), device=device, dtype=dtype)
    return (
        torch.empty((1,), device=device, dtype=torch.int32),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((3, image_height, image_width), device=device, dtype=dtype),
        (
            torch.empty((3, image_height, image_width), device=device, dtype=dtype)
            if need_depth
            else empty
        ),
        (
            torch.empty((3, image_height, image_width), device=device, dtype=dtype)
            if need_normal
            else empty
        ),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        (
            torch.empty((num_voxels, 1), device=device, dtype=dtype)
            if track_max_w
            else empty
        ),
    )


def _rasterize_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    parsed = parse_rasterize_outputs(output)
    ctx.n_samp_per_vox = inputs[0]
    ctx.image_width = inputs[1]
    ctx.image_height = inputs[2]
    ctx.tanfovx = inputs[3]
    ctx.tanfovy = inputs[4]
    ctx.cx = inputs[5]
    ctx.cy = inputs[6]
    ctx.bg_color = inputs[9]
    ctx.need_depth = inputs[10]
    ctx.need_normal = inputs[11]
    ctx.save_for_backward(
        parsed.num_rendered,
        inputs[7],
        inputs[8],
        inputs[13],
        inputs[14],
        inputs[15],
        inputs[16],
        inputs[17],
        inputs[18],
        inputs[19],
        parsed.binning_buffer,
        parsed.image_buffer,
        parsed.result.transmittance,
        parsed.result.depth,
        parsed.result.normal,
    )


def _rasterize_backward(
    ctx: Any,
    grad_num_rendered: Tensor,
    grad_binning_buffer: Tensor,
    grad_image_buffer: Tensor,
    grad_color: Tensor,
    grad_depth: Tensor,
    grad_normal: Tensor,
    grad_transmittance: Tensor,
    grad_max_weight: Tensor,
) -> tuple[Tensor | None, ...]:
    del (
        grad_num_rendered,
        grad_binning_buffer,
        grad_image_buffer,
        grad_max_weight,
    )
    (
        num_rendered,
        w2c_matrix,
        c2w_matrix,
        octree_paths,
        vox_centers,
        vox_lengths,
        geos,
        rgbs,
        _subdivision_priority,
        geom_buffer,
        binning_buffer,
        image_buffer,
        transmittance,
        depth,
        normal,
    ) = ctx.saved_tensors
    grad_geos, grad_rgbs, grad_subdivision_priority = (
        backend().rasterize_voxels_backward(
            int(num_rendered.item()),
            ctx.n_samp_per_vox,
            ctx.image_width,
            ctx.image_height,
            ctx.tanfovx,
            ctx.tanfovy,
            ctx.cx,
            ctx.cy,
            w2c_matrix,
            c2w_matrix,
            ctx.bg_color,
            octree_paths,
            vox_centers,
            vox_lengths,
            geos,
            rgbs,
            geom_buffer,
            binning_buffer,
            image_buffer,
            transmittance,
            grad_color,
            grad_depth,
            grad_normal,
            grad_transmittance,
            0.0,
            torch.empty(0, device=rgbs.device, dtype=rgbs.dtype),
            0.0,
            0.0,
            ctx.need_depth,
            ctx.need_normal,
            depth,
            normal,
            False,
        )
    )
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        grad_geos,
        grad_rgbs,
        grad_subdivision_priority,
        None,
    )


rasterize_op.register_autograd(
    _rasterize_backward,
    setup_context=_rasterize_setup_context,
)
