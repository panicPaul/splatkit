"""Rasterize-stage custom ops for the SVRaster native runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ember_native_svraster.core.runtime.ops._common import backend
from ember_native_svraster.core.runtime.packing import (
    parse_rasterize_outputs,
)


@torch.library.custom_op("svraster::rasterize", mutates_args=())
def rasterize_op(
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
    color_concentration_weight: float,
    ascending_weight: float,
    distortion_weight: float,
    ground_truth_color: Tensor,
    debug: bool,
    octree_paths: Tensor,
    voxel_centers: Tensor,
    voxel_lengths: Tensor,
    voxel_geometries: Tensor,
    voxel_colors: Tensor,
    subdivision_priority: Tensor,
    geometry_buffer: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run the native rasterize stage."""
    del (
        subdivision_priority,
        color_concentration_weight,
        ascending_weight,
        ground_truth_color,
    )
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
        distortion_weight > 0.0,
        return_normal,
        track_max_weight,
        octree_paths,
        voxel_centers,
        voxel_lengths,
        voxel_geometries,
        voxel_colors,
        geometry_buffer,
        debug,
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
    color_concentration_weight: float,
    ascending_weight: float,
    distortion_weight: float,
    ground_truth_color: Tensor,
    debug: bool,
    octree_paths: Tensor,
    voxel_centers: Tensor,
    voxel_lengths: Tensor,
    voxel_geometries: Tensor,
    voxel_colors: Tensor,
    subdivision_priority: Tensor,
    geometry_buffer: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        samples_per_voxel,
        tanfovx,
        tanfovy,
        cx,
        cy,
        world_to_camera,
        camera_to_world,
        background_color,
        voxel_centers,
        voxel_lengths,
        voxel_colors,
        subdivision_priority,
        geometry_buffer,
        color_concentration_weight,
        ascending_weight,
        ground_truth_color,
        debug,
    )
    device = octree_paths.device
    dtype = voxel_geometries.dtype
    num_voxels = int(voxel_geometries.shape[0])
    empty = torch.empty((0,), device=device, dtype=dtype)
    return (
        torch.empty((1,), device=device, dtype=torch.int32),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((0,), device=device, dtype=torch.uint8),
        torch.empty((3, image_height, image_width), device=device, dtype=dtype),
        (
            torch.empty(
                (3, image_height, image_width), device=device, dtype=dtype
            )
            if return_depth or distortion_weight > 0.0
            else empty
        ),
        (
            torch.empty(
                (3, image_height, image_width), device=device, dtype=dtype
            )
            if return_normal
            else empty
        ),
        torch.empty((1, image_height, image_width), device=device, dtype=dtype),
        (
            torch.empty((num_voxels, 1), device=device, dtype=dtype)
            if track_max_weight
            else empty
        ),
    )


def _rasterize_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    parsed = parse_rasterize_outputs(output)
    ctx.samples_per_voxel = inputs[0]
    ctx.image_width = inputs[1]
    ctx.image_height = inputs[2]
    ctx.tanfovx = inputs[3]
    ctx.tanfovy = inputs[4]
    ctx.cx = inputs[5]
    ctx.cy = inputs[6]
    ctx.background_color = inputs[9]
    ctx.return_depth = inputs[10]
    ctx.return_normal = inputs[11]
    ctx.color_concentration_weight = inputs[13]
    ctx.ascending_weight = inputs[14]
    ctx.distortion_weight = inputs[15]
    ctx.debug = inputs[17]
    ctx.save_for_backward(
        parsed.num_rendered,
        inputs[7],
        inputs[8],
        inputs[16],
        inputs[18],
        inputs[19],
        inputs[20],
        inputs[21],
        inputs[22],
        inputs[23],
        inputs[24],
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
        world_to_camera,
        camera_to_world,
        ground_truth_color,
        octree_paths,
        voxel_centers,
        voxel_lengths,
        voxel_geometries,
        voxel_colors,
        _subdivision_priority,
        geometry_buffer,
        binning_buffer,
        image_buffer,
        transmittance,
        depth,
        normal,
    ) = ctx.saved_tensors
    grad_geos, grad_rgbs, grad_subdivision_priority = (
        backend().rasterize_voxels_backward(
            int(num_rendered.item()),
            ctx.samples_per_voxel,
            ctx.image_width,
            ctx.image_height,
            ctx.tanfovx,
            ctx.tanfovy,
            ctx.cx,
            ctx.cy,
            world_to_camera,
            camera_to_world,
            ctx.background_color,
            octree_paths,
            voxel_centers,
            voxel_lengths,
            voxel_geometries,
            voxel_colors,
            geometry_buffer,
            binning_buffer,
            image_buffer,
            transmittance,
            grad_color,
            grad_depth,
            grad_normal,
            grad_transmittance,
            ctx.color_concentration_weight,
            ground_truth_color,
            ctx.ascending_weight,
            ctx.distortion_weight,
            ctx.return_depth,
            ctx.return_normal,
            depth,
            normal,
            ctx.debug,
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
