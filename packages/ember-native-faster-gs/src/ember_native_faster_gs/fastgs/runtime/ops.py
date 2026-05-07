"""FastGS-specific composed custom ops."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ember_native_faster_gs.faster_gs.reuse.factories import (
    register_blend_family,
    register_render_family,
)
from ember_native_faster_gs.faster_gs.runtime.ops._common import (
    TILE_HEIGHT,
    TILE_WIDTH,
    requires_grad,
)
from ember_native_faster_gs.faster_gs.runtime.ops.preprocess import (
    _preprocess_bwd_fake,
    preprocess_bwd_op,
)
from ember_native_faster_gs.faster_gs.runtime.packing import (
    pack_render_outputs,
    parse_blend_outputs,
    parse_render_outputs,
)
from ember_native_faster_gs.fastgs.runtime._extension import load_extension

RenderOpOutput = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


def backend() -> Any:
    """Return the loaded FastGS native extension."""
    return load_extension()


@torch.library.custom_op("fastgs::blend_fwd", mutates_args=())
def blend_fwd_op(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    mip_splatting_screen_filter: bool,
    image_width: int,
    image_height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Low-level native FastGS blend forward op."""
    return backend().blend_fwd(
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        mip_splatting_screen_filter,
        image_width,
        image_height,
    )


@blend_fwd_op.register_fake
def _blend_fwd_fake(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    mip_splatting_screen_filter: bool,
    image_width: int,
    image_height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        instance_primitive_indices,
        tile_bucket_offsets,
        bucket_count,
        conic_opacity,
        colors_rgb,
        bg_color,
        mip_splatting_screen_filter,
    )
    device = projected_means.device
    dtype = projected_means.dtype
    tile_count = int(tile_instance_ranges.shape[0])
    tile_pixels = tile_count * TILE_WIDTH * TILE_HEIGHT
    return (
        torch.empty((3, image_height, image_width), device=device, dtype=dtype),
        torch.empty((tile_pixels,), device=device, dtype=dtype),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((tile_pixels,), device=device, dtype=torch.int32),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty(
            (tile_count * TILE_WIDTH * TILE_HEIGHT, 4),
            device=device,
            dtype=dtype,
        ),
    )


@torch.library.custom_op("fastgs::blend_metric_counts_fwd", mutates_args=())
def blend_metric_counts_fwd_op(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    metric_map: Tensor,
    mip_splatting_screen_filter: bool,
    image_width: int,
    image_height: int,
) -> Tensor:
    """Low-level native FastGS metric-count attribution op."""
    return backend().blend_metric_counts_fwd(
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        metric_map,
        mip_splatting_screen_filter,
        image_width,
        image_height,
    )


@blend_metric_counts_fwd_op.register_fake
def _blend_metric_counts_fwd_fake(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    metric_map: Tensor,
    mip_splatting_screen_filter: bool,
    image_width: int,
    image_height: int,
) -> Tensor:
    del (
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        conic_opacity,
        colors_rgb,
        bg_color,
        metric_map,
        mip_splatting_screen_filter,
        image_width,
        image_height,
    )
    return torch.empty(
        (projected_means.shape[0],),
        device=projected_means.device,
        dtype=torch.int32,
    )


@torch.library.custom_op("fastgs::blend_bwd", mutates_args=())
def blend_bwd_op(
    grad_image: Tensor,
    image: Tensor,
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    tile_final_transmittances: Tensor,
    tile_max_n_processed: Tensor,
    tile_n_processed: Tensor,
    bucket_tile_index: Tensor,
    bucket_color_transmittance: Tensor,
    mip_splatting_screen_filter: bool,
    image_width: int,
    image_height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Low-level native FastGS blend backward op."""
    return backend().blend_bwd(
        grad_image,
        image,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
        mip_splatting_screen_filter,
        image_width,
        image_height,
    )


@blend_bwd_op.register_fake
def _blend_bwd_fake(
    grad_image: Tensor,
    image: Tensor,
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    tile_final_transmittances: Tensor,
    tile_max_n_processed: Tensor,
    tile_n_processed: Tensor,
    bucket_tile_index: Tensor,
    bucket_color_transmittance: Tensor,
    mip_splatting_screen_filter: bool,
    image_width: int,
    image_height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del (
        grad_image,
        image,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bg_color,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
        mip_splatting_screen_filter,
        image_width,
        image_height,
    )
    return (
        torch.empty_like(projected_means),
        torch.empty_like(projected_means),
        torch.empty_like(conic_opacity),
        torch.empty_like(colors_rgb),
    )


def _blend_impl(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    mip_splatting_screen_filter: bool,
    image_width: int,
    image_height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Autograd-enabled FastGS blend op."""
    return blend_fwd_op(
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        mip_splatting_screen_filter,
        image_width,
        image_height,
    )


def _blend_fake(
    *args: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fake implementation for the autograd FastGS blend op."""
    return _blend_fwd_fake(*args)


def _blend_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    blend_result = parse_blend_outputs(output)
    ctx.save_for_backward(
        blend_result.image,
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        blend_result.tile_final_transmittances,
        blend_result.tile_max_n_processed,
        blend_result.tile_n_processed,
        blend_result.bucket_tile_index,
        blend_result.bucket_color_transmittance,
    )
    ctx.mip_splatting_screen_filter = inputs[8]
    ctx.image_width = inputs[9]
    ctx.image_height = inputs[10]


def _blend_backward(
    ctx: Any,
    grad_image: Tensor,
    grad_tile_final_transmittances: Tensor,
    grad_tile_max_n_processed: Tensor,
    grad_tile_n_processed: Tensor,
    grad_bucket_tile_index: Tensor,
    grad_bucket_color_transmittance: Tensor,
) -> tuple[Tensor | None, ...]:
    del (
        grad_tile_final_transmittances,
        grad_tile_max_n_processed,
        grad_tile_n_processed,
        grad_bucket_tile_index,
        grad_bucket_color_transmittance,
    )
    grad_projected_means, _, grad_conic_opacity, grad_colors_rgb = blend_bwd_op(
        grad_image,
        *ctx.saved_tensors,
        ctx.mip_splatting_screen_filter,
        ctx.image_width,
        ctx.image_height,
    )
    return (
        None,
        None,
        None,
        None,
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        None,
        None,
        None,
        None,
    )


blend_op = register_blend_family(
    op_name="fastgs::blend",
    forward_impl=_blend_impl,
    fake_impl=_blend_fake,
    setup_context=_blend_setup_context,
    backward_impl=_blend_backward,
)


@torch.library.custom_op("fastgs::preprocess_fwd", mutates_args=())
def preprocess_fwd_op(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    near_plane: float,
    far_plane: float,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    mip_splatting_screen_filter: bool,
    active_sh_bases: int,
    compact_box_scale: float,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    """Low-level native FastGS preprocess forward op."""
    return backend().preprocess_fwd(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        world_2_camera,
        camera_position,
        near_plane,
        far_plane,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        mip_splatting_screen_filter,
        active_sh_bases,
        compact_box_scale,
    )


def update_densification_radii_fwd(
    num_touched_tiles: Tensor,
    conic_opacity: Tensor,
    densification_info: Tensor,
) -> None:
    """Update FastGS max screen radii from AA-adjusted conics."""
    backend().update_densification_radii_fwd(
        num_touched_tiles,
        conic_opacity,
        densification_info,
    )


@preprocess_fwd_op.register_fake
def _preprocess_fwd_fake(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    near_plane: float,
    far_plane: float,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    mip_splatting_screen_filter: bool,
    active_sh_bases: int,
    compact_box_scale: float,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    del (
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        world_2_camera,
        camera_position,
        near_plane,
        far_plane,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        mip_splatting_screen_filter,
        active_sh_bases,
        compact_box_scale,
    )
    num_primitives = int(center_positions.shape[0])
    device = center_positions.device
    dtype = center_positions.dtype
    return (
        torch.empty((num_primitives, 2), device=device, dtype=dtype),
        torch.empty((num_primitives, 4), device=device, dtype=dtype),
        torch.empty((num_primitives, 3), device=device, dtype=dtype),
        torch.empty((num_primitives,), device=device, dtype=dtype),
        torch.empty((num_primitives,), device=device, dtype=torch.int32),
        torch.empty((num_primitives,), device=device, dtype=torch.int32),
        torch.empty((num_primitives,), device=device, dtype=torch.int32),
        torch.empty((num_primitives, 4), device=device, dtype=torch.uint16),
        torch.empty((1,), device=device, dtype=torch.int32),
        torch.empty((1,), device=device, dtype=torch.int32),
    )


@torch.library.custom_op("fastgs::sort_fwd", mutates_args=())
def sort_fwd_op(
    depth_keys: Tensor,
    primitive_indices: Tensor,
    num_touched_tiles: Tensor,
    screen_bounds: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    visible_count: Tensor,
    instance_count: Tensor,
    width: int,
    height: int,
    compact_box_scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Low-level native FastGS sort forward op."""
    return backend().sort_fwd(
        depth_keys,
        primitive_indices,
        num_touched_tiles,
        screen_bounds,
        projected_means,
        conic_opacity,
        visible_count,
        instance_count,
        width,
        height,
        compact_box_scale,
    )


@sort_fwd_op.register_fake
def _sort_fwd_fake(
    depth_keys: Tensor,
    primitive_indices: Tensor,
    num_touched_tiles: Tensor,
    screen_bounds: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    visible_count: Tensor,
    instance_count: Tensor,
    width: int,
    height: int,
    compact_box_scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del (
        depth_keys,
        num_touched_tiles,
        screen_bounds,
        projected_means,
        conic_opacity,
        visible_count,
        instance_count,
        compact_box_scale,
    )
    device = primitive_indices.device
    tile_count = ((width + TILE_WIDTH - 1) // TILE_WIDTH) * (
        (height + TILE_HEIGHT - 1) // TILE_HEIGHT
    )
    return (
        torch.empty_like(primitive_indices),
        torch.empty((tile_count, 2), device=device, dtype=torch.int32),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((1,), device=device, dtype=torch.int32),
    )


@torch.library.custom_op("fastgs::render_fwd", mutates_args=())
def render_fwd_op(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    near_plane: float,
    far_plane: float,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    bg_color: Tensor,
    mip_splatting_screen_filter: bool,
    active_sh_bases: int,
    compact_box_scale: float,
) -> RenderOpOutput:
    """Low-level composed FastGS render forward op."""
    preprocess_outputs = preprocess_fwd_op(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        world_2_camera,
        camera_position,
        near_plane,
        far_plane,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        mip_splatting_screen_filter,
        active_sh_bases,
        compact_box_scale,
    )
    sort_outputs = sort_fwd_op(
        preprocess_outputs[4],
        preprocess_outputs[5],
        preprocess_outputs[6],
        preprocess_outputs[7],
        preprocess_outputs[0],
        preprocess_outputs[1],
        preprocess_outputs[8],
        preprocess_outputs[9],
        width,
        height,
        compact_box_scale,
    )
    blend_outputs = blend_fwd_op(
        sort_outputs[0],
        sort_outputs[1],
        sort_outputs[2],
        sort_outputs[3],
        preprocess_outputs[0],
        preprocess_outputs[1],
        preprocess_outputs[2],
        bg_color,
        mip_splatting_screen_filter,
        width,
        height,
    )
    return pack_render_outputs(
        preprocess_outputs=preprocess_outputs,
        sort_outputs=sort_outputs,
        blend_outputs=blend_outputs,
    )


@render_fwd_op.register_fake
def _render_fwd_fake(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    near_plane: float,
    far_plane: float,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    bg_color: Tensor,
    mip_splatting_screen_filter: bool,
    active_sh_bases: int,
    compact_box_scale: float,
) -> RenderOpOutput:
    preprocess_outputs = _preprocess_fwd_fake(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        world_2_camera,
        camera_position,
        near_plane,
        far_plane,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        mip_splatting_screen_filter,
        active_sh_bases,
        compact_box_scale,
    )
    sort_outputs = _sort_fwd_fake(
        preprocess_outputs[4],
        preprocess_outputs[5],
        preprocess_outputs[6],
        preprocess_outputs[7],
        preprocess_outputs[0],
        preprocess_outputs[1],
        preprocess_outputs[8],
        preprocess_outputs[9],
        width,
        height,
        compact_box_scale,
    )
    blend_outputs = _blend_fwd_fake(
        sort_outputs[0],
        sort_outputs[1],
        sort_outputs[2],
        sort_outputs[3],
        preprocess_outputs[0],
        preprocess_outputs[1],
        preprocess_outputs[2],
        bg_color,
        mip_splatting_screen_filter,
        width,
        height,
    )
    return pack_render_outputs(
        preprocess_outputs=preprocess_outputs,
        sort_outputs=sort_outputs,
        blend_outputs=blend_outputs,
    )


@torch.library.custom_op(
    "fastgs::render_bwd",
    mutates_args=("densification_info",),
)
def render_bwd_op(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    num_touched_tiles: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    image: Tensor,
    bg_color: Tensor,
    tile_final_transmittances: Tensor,
    tile_max_n_processed: Tensor,
    tile_n_processed: Tensor,
    bucket_tile_index: Tensor,
    bucket_color_transmittance: Tensor,
    densification_info: Tensor,
    grad_image: Tensor,
    mip_splatting_screen_filter: bool,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Low-level composed FastGS render backward op."""
    (
        grad_projected_means,
        grad_projected_means_abs,
        grad_conic_opacity,
        grad_colors_rgb,
    ) = blend_bwd_op(
        grad_image,
        image,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
        mip_splatting_screen_filter,
        width,
        height,
    )
    if densification_info.ndim == 2 and densification_info.shape[0] >= 4:
        grad_projected_means_abs_ndc = 0.5 * torch.stack(
            (
                grad_projected_means_abs[:, 0] * width,
                grad_projected_means_abs[:, 1] * height,
            ),
            dim=1,
        )
        densification_info[2].add_(grad_projected_means_abs_ndc.norm(dim=-1))
    if densification_info.ndim == 2 and densification_info.shape[0] >= 2:
        core_densification_info = densification_info[:2]
    else:
        core_densification_info = densification_info
    grads = preprocess_bwd_op(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        world_2_camera,
        camera_position,
        num_touched_tiles,
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        torch.zeros_like(projected_means[:, 0]),
        core_densification_info,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        mip_splatting_screen_filter,
        active_sh_bases,
    )
    return grads


@render_bwd_op.register_fake
def _render_bwd_fake(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    num_touched_tiles: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    image: Tensor,
    bg_color: Tensor,
    tile_final_transmittances: Tensor,
    tile_max_n_processed: Tensor,
    tile_n_processed: Tensor,
    bucket_tile_index: Tensor,
    bucket_color_transmittance: Tensor,
    densification_info: Tensor,
    grad_image: Tensor,
    mip_splatting_screen_filter: bool,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        image,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bg_color,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
    )
    (
        grad_projected_means,
        _grad_projected_means_abs,
        grad_conic_opacity,
        grad_colors_rgb,
    ) = _blend_bwd_fake(
        grad_image,
        center_positions,
        center_positions,
        center_positions,
        center_positions,
        projected_means,
        conic_opacity,
        colors_rgb,
        center_positions,
        center_positions,
        center_positions,
        center_positions,
        center_positions,
        center_positions,
        mip_splatting_screen_filter,
        width,
        height,
    )
    return _preprocess_bwd_fake(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        world_2_camera,
        camera_position,
        num_touched_tiles,
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        torch.zeros_like(projected_means[:, 0]),
        densification_info,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        mip_splatting_screen_filter,
        active_sh_bases,
    )


def _render_impl(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    near_plane: float,
    far_plane: float,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    bg_color: Tensor,
    mip_splatting_screen_filter: bool,
    active_sh_bases: int,
    compact_box_scale: float,
    densification_info: Tensor,
) -> RenderOpOutput:
    """Autograd-enabled full FastGS render op."""
    del densification_info
    return render_fwd_op(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        world_2_camera,
        camera_position,
        near_plane,
        far_plane,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        bg_color,
        mip_splatting_screen_filter,
        active_sh_bases,
        compact_box_scale,
    )


def _render_fake(*args: Any) -> RenderOpOutput:
    """Fake implementation for the autograd FastGS render op."""
    return _render_fwd_fake(*args[:-1])


def _render_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    if not requires_grad(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
    ):
        ctx.has_context = False
        return
    render_result = parse_render_outputs(output)
    ctx.has_context = True
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        render_result.preprocess.num_touched_tiles,
        render_result.preprocess.projected_means,
        render_result.preprocess.conic_opacity,
        render_result.preprocess.colors_rgb,
        render_result.sort.instance_primitive_indices,
        render_result.sort.tile_instance_ranges,
        render_result.sort.tile_bucket_offsets,
        render_result.image,
        inputs[16],
        render_result.blend.tile_final_transmittances,
        render_result.blend.tile_max_n_processed,
        render_result.blend.tile_n_processed,
        render_result.blend.bucket_tile_index,
        render_result.blend.bucket_color_transmittance,
        inputs[20],
    )
    ctx.width = inputs[10]
    ctx.height = inputs[11]
    ctx.focal_x = inputs[12]
    ctx.focal_y = inputs[13]
    ctx.center_x = inputs[14]
    ctx.center_y = inputs[15]
    ctx.mip_splatting_screen_filter = inputs[17]
    ctx.active_sh_bases = inputs[18]


def _render_backward(
    ctx: Any,
    grad_image: Tensor,
    *grad_aux: Tensor,
) -> tuple[Tensor | None, ...]:
    del grad_aux
    if not ctx.has_context:
        return (None,) * 21
    grads = render_bwd_op(
        *ctx.saved_tensors,
        grad_image,
        ctx.mip_splatting_screen_filter,
        ctx.width,
        ctx.height,
        ctx.focal_x,
        ctx.focal_y,
        ctx.center_x,
        ctx.center_y,
        ctx.active_sh_bases,
    )
    return (*grads, *(None,) * 15)


render_op = register_render_family(
    op_name="fastgs::render",
    forward_impl=_render_impl,
    fake_impl=_render_fake,
    setup_context=_render_setup_context,
    backward_impl=_render_backward,
)
