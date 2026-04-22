"""Blend-stage custom ops for the FasterGS native runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from splatkit_native_backends.faster_gs_native.runtime.ops._common import (
    BLOCK_SIZE_BLEND,
    backend,
)
from splatkit_native_backends.faster_gs_native.runtime.packing import (
    parse_blend_outputs,
)


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
    return backend().blend_fwd(
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
    tile_pixels = tile_count * BLOCK_SIZE_BLEND
    return (
        torch.empty((3, height, width), device=device, dtype=dtype),
        torch.empty((tile_pixels,), device=device, dtype=dtype),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((tile_pixels,), device=device, dtype=torch.int32),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((tile_count * BLOCK_SIZE_BLEND, 4), device=device, dtype=dtype),
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

