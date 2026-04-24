"""Blend-stage custom ops for the FasterGS Mojo runtime."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from splatkit_native_faster_gs.faster_gs.runtime.ops.blend import (
    _blend_bwd_fake,
    _blend_fwd_fake,
    blend_bwd_op as faster_blend_bwd_op,
)
from splatkit_native_faster_gs.faster_gs.runtime.packing import parse_blend_outputs
from splatkit_native_faster_gs_mojo.core.runtime.ops._common import (
    BLOCK_SIZE_BLEND,
    mojo_backend,
    stable_capacity,
)


def _call_graph_blend_fwd(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    *,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    tile_count = int(tile_instance_ranges.shape[0])
    tile_pixels = tile_count * BLOCK_SIZE_BLEND
    actual_bucket_count = int(bucket_count.item())
    bucket_capacity = stable_capacity(
        (
            "blend_buckets",
            projected_means.device.type,
            projected_means.device.index,
            int(projected_means.shape[0]),
            width,
            height,
        ),
        actual_bucket_count,
    )
    outputs = (
        torch.empty((3, height, width), device=projected_means.device, dtype=projected_means.dtype),
        torch.empty((tile_pixels,), device=projected_means.device, dtype=projected_means.dtype),
        torch.empty((tile_count,), device=projected_means.device, dtype=torch.int32),
        torch.empty((tile_pixels,), device=projected_means.device, dtype=torch.int32),
        torch.empty((bucket_capacity,), device=projected_means.device, dtype=torch.int32),
        torch.empty(
            (bucket_capacity * BLOCK_SIZE_BLEND, 4),
            device=projected_means.device,
            dtype=projected_means.dtype,
        ),
    )
    mojo_backend().blend_fwd(
        *outputs,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
    )
    return outputs


def blend_image_only(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    *,
    width: int,
    height: int,
) -> Tensor:
    """Render only the final image through the staged MAX/Mojo blend op."""
    image = torch.empty((3, height, width), device=projected_means.device, dtype=projected_means.dtype)
    mojo_backend().blend_fwd_image_only(
        image,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
    )
    return image


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
    """Run the MAX/Mojo blend forward stage."""
    del proper_antialiasing
    return _call_graph_blend_fwd(
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        width=width,
        height=height,
    )


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
    """Delegate blend backward to the FasterGS core reference."""
    return faster_blend_bwd_op(
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


def _blend_impl(
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


def _blend_fake(*args: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    """Alias the staged blend forward helper."""
    return _blend_impl(
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
