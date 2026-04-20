"""Registered torch custom ops for the FasterGS native runtime."""

from __future__ import annotations

import itertools
import weakref
from typing import Any

import torch
from torch import Tensor

from splatkit_native_backends.faster_gs_native.runtime._extension import (
    load_extension,
)

_TILE_WIDTH = 16
_TILE_HEIGHT = 16
_BLOCK_SIZE_BLEND = _TILE_WIDTH * _TILE_HEIGHT
_RENDER_CONTEXTS: dict[int, tuple[Tensor, ...]] = {}
_RENDER_OUTPUT_TO_CONTEXT: dict[int, int] = {}
_RENDER_CONTEXT_KEYS = itertools.count(1)


def _backend() -> Any:
    """Return the loaded native rasterization extension."""
    return load_extension()


def _cleanup_render_context(context_key: int, output_id: int) -> None:
    """Release staged render context once the output tensor dies."""
    _RENDER_CONTEXTS.pop(context_key, None)
    _RENDER_OUTPUT_TO_CONTEXT.pop(output_id, None)


def _render_requires_grad(*tensors: Tensor) -> bool:
    """Return whether the composed render op needs backward state."""
    return any(tensor.requires_grad for tensor in tensors)


@torch.library.custom_op("faster_gs_native::preprocess_fwd", mutates_args=())
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
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Low-level native preprocess forward op."""
    return _backend().preprocess_fwd(
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
        proper_antialiasing,
        active_sh_bases,
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
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        proper_antialiasing,
        active_sh_bases,
    )
    num_primitives = int(center_positions.shape[0])
    device = center_positions.device
    dtype = center_positions.dtype
    return (
        torch.empty((num_primitives, 2), device=device, dtype=dtype),
        torch.empty((num_primitives, 4), device=device, dtype=dtype),
        torch.empty((num_primitives, 3), device=device, dtype=dtype),
        torch.empty((num_primitives,), device=device, dtype=torch.int32),
        torch.empty((num_primitives,), device=device, dtype=torch.int32),
        torch.empty((num_primitives,), device=device, dtype=torch.int32),
        torch.empty((num_primitives, 4), device=device, dtype=torch.uint16),
        torch.empty((1,), device=device, dtype=torch.int32),
        torch.empty((1,), device=device, dtype=torch.int32),
    )


@torch.library.custom_op("faster_gs_native::preprocess_bwd", mutates_args=())
def preprocess_bwd_op(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    num_touched_tiles: Tensor,
    grad_projected_means: Tensor,
    grad_conic_opacity: Tensor,
    grad_colors_rgb: Tensor,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Low-level native preprocess backward op."""
    del sh_coefficients_0
    return _backend().preprocess_bwd(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_rest,
        world_2_camera,
        camera_position,
        num_touched_tiles,
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing,
        active_sh_bases,
    )


@preprocess_bwd_op.register_fake
def _preprocess_bwd_fake(
    center_positions: Tensor,
    log_scales: Tensor,
    unnormalized_rotations: Tensor,
    opacities: Tensor,
    sh_coefficients_0: Tensor,
    sh_coefficients_rest: Tensor,
    world_2_camera: Tensor,
    camera_position: Tensor,
    num_touched_tiles: Tensor,
    grad_projected_means: Tensor,
    grad_conic_opacity: Tensor,
    grad_colors_rgb: Tensor,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        world_2_camera,
        camera_position,
        num_touched_tiles,
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing,
        active_sh_bases,
    )
    return (
        torch.empty_like(center_positions),
        torch.empty_like(log_scales),
        torch.empty_like(unnormalized_rotations),
        torch.empty_like(opacities),
        torch.empty_like(sh_coefficients_0),
        torch.empty_like(sh_coefficients_rest),
    )


@torch.library.custom_op("faster_gs_native::preprocess", mutates_args=())
def preprocess_op(
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
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Autograd-enabled preprocess op."""
    return preprocess_fwd_op(
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
        proper_antialiasing,
        active_sh_bases,
    )


@preprocess_op.register_fake
def _preprocess_fake(
    *args: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fake implementation for the autograd preprocess op."""
    return _preprocess_fwd_fake(*args)


def _preprocess_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        output[5],
    )
    ctx.width = inputs[10]
    ctx.height = inputs[11]
    ctx.focal_x = inputs[12]
    ctx.focal_y = inputs[13]
    ctx.center_x = inputs[14]
    ctx.center_y = inputs[15]
    ctx.proper_antialiasing = inputs[16]
    ctx.active_sh_bases = inputs[17]


def _preprocess_backward(
    ctx: Any,
    grad_projected_means: Tensor,
    grad_conic_opacity: Tensor,
    grad_colors_rgb: Tensor,
    grad_depth_keys: Tensor,
    grad_primitive_indices: Tensor,
    grad_num_touched_tiles: Tensor,
    grad_screen_bounds: Tensor,
    grad_visible_count: Tensor,
    grad_instance_count: Tensor,
) -> tuple[Tensor | None, ...]:
    del (
        grad_depth_keys,
        grad_primitive_indices,
        grad_num_touched_tiles,
        grad_screen_bounds,
        grad_visible_count,
        grad_instance_count,
    )
    grads = preprocess_bwd_op(
        *ctx.saved_tensors,
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        ctx.width,
        ctx.height,
        ctx.focal_x,
        ctx.focal_y,
        ctx.center_x,
        ctx.center_y,
        ctx.proper_antialiasing,
        ctx.active_sh_bases,
    )
    return (*grads, *(None,) * 12)


preprocess_op.register_autograd(
    _preprocess_backward,
    setup_context=_preprocess_setup_context,
)


@torch.library.custom_op("faster_gs_native::sort_fwd", mutates_args=())
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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Low-level native sort forward op."""
    return _backend().sort_fwd(
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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del (
        depth_keys,
        num_touched_tiles,
        screen_bounds,
        projected_means,
        conic_opacity,
        visible_count,
        instance_count,
    )
    device = primitive_indices.device
    tile_count = ((width + _TILE_WIDTH - 1) // _TILE_WIDTH) * (
        (height + _TILE_HEIGHT - 1) // _TILE_HEIGHT
    )
    return (
        torch.empty_like(primitive_indices),
        torch.empty((tile_count, 2), device=device, dtype=torch.int32),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((1,), device=device, dtype=torch.int32),
    )


@torch.library.custom_op("faster_gs_native::sort", mutates_args=())
def sort_op(
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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Public non-differentiable sort op."""
    return sort_fwd_op(
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
    )


@sort_op.register_fake
def _sort_fake(*args: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fake implementation for the public sort op."""
    return _sort_fwd_fake(*args)


@torch.library.custom_op("faster_gs_native::blend_fwd", mutates_args=())
def blend_fwd_op(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Low-level native blend forward op."""
    return _backend().blend_fwd(
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        proper_antialiasing,
        width,
        height,
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
    proper_antialiasing: bool,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        instance_primitive_indices,
        tile_bucket_offsets,
        bucket_count,
        conic_opacity,
        colors_rgb,
        bg_color,
        proper_antialiasing,
    )
    device = projected_means.device
    dtype = projected_means.dtype
    tile_count = int(tile_instance_ranges.shape[0])
    tile_pixels = tile_count * _BLOCK_SIZE_BLEND
    return (
        torch.empty((3, height, width), device=device, dtype=dtype),
        torch.empty((tile_pixels,), device=device, dtype=dtype),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((tile_pixels,), device=device, dtype=torch.int32),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((tile_count * _BLOCK_SIZE_BLEND, 4), device=device, dtype=dtype),
    )


@torch.library.custom_op("faster_gs_native::blend_bwd", mutates_args=())
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
    proper_antialiasing: bool,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Low-level native blend backward op."""
    return _backend().blend_bwd(
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
        proper_antialiasing,
        width,
        height,
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
    proper_antialiasing: bool,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor]:
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
        proper_antialiasing,
        width,
        height,
    )
    return (
        torch.empty_like(projected_means),
        torch.empty_like(conic_opacity),
        torch.empty_like(colors_rgb),
    )


@torch.library.custom_op("faster_gs_native::blend", mutates_args=())
def blend_op(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Autograd-enabled blend op."""
    return blend_fwd_op(
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        proper_antialiasing,
        width,
        height,
    )


@blend_op.register_fake
def _blend_fake(*args: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fake implementation for the autograd blend op."""
    return _blend_fwd_fake(*args)


def _blend_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    ctx.save_for_backward(
        output[0],
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        output[1],
        output[2],
        output[3],
        output[4],
        output[5],
    )
    ctx.proper_antialiasing = inputs[8]
    ctx.width = inputs[9]
    ctx.height = inputs[10]


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
    grad_projected_means, grad_conic_opacity, grad_colors_rgb = blend_bwd_op(
        grad_image,
        *ctx.saved_tensors,
        ctx.proper_antialiasing,
        ctx.width,
        ctx.height,
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


blend_op.register_autograd(
    _blend_backward,
    setup_context=_blend_setup_context,
)


@torch.library.custom_op("faster_gs_native::render_fwd", mutates_args=())
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
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[Tensor]:
    """Low-level composed render forward op."""
    preprocess_result = preprocess_fwd_op(
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
        proper_antialiasing,
        active_sh_bases,
    )
    sort_result = sort_fwd_op(
        preprocess_result[3],
        preprocess_result[4],
        preprocess_result[5],
        preprocess_result[6],
        preprocess_result[0],
        preprocess_result[1],
        preprocess_result[7],
        preprocess_result[8],
        width,
        height,
    )
    blend_result = blend_fwd_op(
        sort_result[0],
        sort_result[1],
        sort_result[2],
        sort_result[3],
        preprocess_result[0],
        preprocess_result[1],
        preprocess_result[2],
        bg_color,
        proper_antialiasing,
        width,
        height,
    )
    image = blend_result[0]
    if _render_requires_grad(
        center_positions,
        log_scales,
        unnormalized_rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
    ):
        context_key = int(next(_RENDER_CONTEXT_KEYS))
        output_id = id(image)
        _RENDER_CONTEXTS[context_key] = (
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            preprocess_result[5],
            preprocess_result[0],
            preprocess_result[1],
            preprocess_result[2],
            sort_result[0],
            sort_result[1],
            sort_result[2],
            bg_color,
            blend_result[1],
            blend_result[2],
            blend_result[3],
            blend_result[4],
            blend_result[5],
        )
        _RENDER_OUTPUT_TO_CONTEXT[output_id] = context_key
        weakref.finalize(image, _cleanup_render_context, context_key, output_id)
    return (image,)


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
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[Tensor]:
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
        focal_x,
        focal_y,
        center_x,
        center_y,
        bg_color,
        proper_antialiasing,
        active_sh_bases,
    )
    return (
        torch.empty(
            (3, height, width),
            device=center_positions.device,
            dtype=center_positions.dtype,
        ),
    )


@torch.library.custom_op("faster_gs_native::render_bwd", mutates_args=())
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
    grad_image: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Low-level composed render backward op."""
    grad_projected_means, grad_conic_opacity, grad_colors_rgb = blend_bwd_op(
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
        proper_antialiasing,
        width,
        height,
    )
    return preprocess_bwd_op(
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
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing,
        active_sh_bases,
    )


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
    grad_image: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    active_sh_bases: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        world_2_camera,
        camera_position,
        num_touched_tiles,
        projected_means,
        conic_opacity,
        colors_rgb,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        image,
        bg_color,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
        grad_image,
        proper_antialiasing,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        active_sh_bases,
    )
    return (
        torch.empty_like(center_positions),
        torch.empty_like(log_scales),
        torch.empty_like(unnormalized_rotations),
        torch.empty_like(opacities),
        torch.empty_like(sh_coefficients_0),
        torch.empty_like(sh_coefficients_rest),
    )


@torch.library.custom_op("faster_gs_native::render", mutates_args=())
def render_op(
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
    proper_antialiasing: bool,
    active_sh_bases: int,
) -> tuple[Tensor]:
    """Autograd-enabled full render op."""
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
        proper_antialiasing,
        active_sh_bases,
    )


@render_op.register_fake
def _render_fake(*args: Any) -> tuple[Tensor]:
    """Fake implementation for the autograd render op."""
    return _render_fwd_fake(*args)


def _render_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor],
) -> None:
    context_key = _RENDER_OUTPUT_TO_CONTEXT.pop(id(output[0]), -1)
    if context_key < 0:
        ctx.has_context = False
        return
    saved = _RENDER_CONTEXTS.pop(context_key, None)
    if saved is None:
        raise RuntimeError("Missing render forward context for faster_gs_native::render.")
    ctx.has_context = True
    ctx.save_for_backward(
        saved[0],
        saved[1],
        saved[2],
        saved[3],
        saved[4],
        saved[5],
        saved[6],
        saved[7],
        saved[8],
        saved[9],
        saved[10],
        saved[11],
        saved[12],
        saved[13],
        saved[14],
        output[0],
        saved[15],
        saved[16],
        saved[17],
        saved[18],
        saved[19],
        saved[20],
    )
    ctx.width = inputs[10]
    ctx.height = inputs[11]
    ctx.focal_x = inputs[12]
    ctx.focal_y = inputs[13]
    ctx.center_x = inputs[14]
    ctx.center_y = inputs[15]
    ctx.proper_antialiasing = inputs[17]
    ctx.active_sh_bases = inputs[18]


def _render_backward(
    ctx: Any,
    grad_image: Tensor,
) -> tuple[Tensor | None, ...]:
    if not ctx.has_context:
        raise RuntimeError("Missing saved render context for faster_gs_native::render.")
    grads = render_bwd_op(
        *ctx.saved_tensors,
        grad_image,
        ctx.proper_antialiasing,
        ctx.width,
        ctx.height,
        ctx.focal_x,
        ctx.focal_y,
        ctx.center_x,
        ctx.center_y,
        ctx.active_sh_bases,
    )
    return (*grads, *(None,) * 13)


render_op.register_autograd(
    _render_backward,
    setup_context=_render_setup_context,
)
