"""Depth-aware blend custom ops for the FasterGS native proof backend."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from splatkit_native_faster_gs.faster_gs_depth.runtime._extension import (
    load_extension,
)
from splatkit_native_faster_gs.faster_gs_depth.runtime.packing import (
    parse_blend_outputs,
)
from splatkit_native_faster_gs.faster_gs.reuse import (
    blend_bwd_op as core_blend_bwd_op,
)
from splatkit_native_faster_gs.faster_gs.reuse.factories import (
    register_blend_family,
)
from splatkit_native_faster_gs.faster_gs.runtime.ops._common import (
    BLOCK_SIZE_BLEND,
)


def backend() -> Any:
    """Return the loaded native depth-only extension."""
    return load_extension()


@torch.library.custom_op("faster_gs_depth::blend_fwd", mutates_args=())
def blend_fwd_op(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    primitive_depth: Tensor,
    bg_color: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Low-level depth-aware blend forward op."""
    return backend().depth_blend_fwd(
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        primitive_depth,
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
    primitive_depth: Tensor,
    bg_color: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del (
        instance_primitive_indices,
        tile_bucket_offsets,
        bucket_count,
        conic_opacity,
        colors_rgb,
        primitive_depth,
        bg_color,
        proper_antialiasing,
    )
    device = projected_means.device
    dtype = projected_means.dtype
    tile_count = int(tile_instance_ranges.shape[0])
    tile_pixels = tile_count * BLOCK_SIZE_BLEND
    return (
        torch.empty((3, height, width), device=device, dtype=dtype),
        torch.empty((height, width), device=device, dtype=dtype),
        torch.empty((tile_pixels,), device=device, dtype=dtype),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((tile_pixels,), device=device, dtype=torch.int32),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((tile_count * BLOCK_SIZE_BLEND, 4), device=device, dtype=dtype),
        torch.empty((tile_count * BLOCK_SIZE_BLEND,), device=device, dtype=dtype),
    )


@torch.library.custom_op("faster_gs_depth::blend_bwd", mutates_args=())
def blend_bwd_op(
    grad_image: Tensor,
    grad_depth: Tensor,
    image: Tensor,
    depth: Tensor,
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    primitive_depth: Tensor,
    bg_color: Tensor,
    tile_final_transmittances: Tensor,
    tile_max_n_processed: Tensor,
    tile_n_processed: Tensor,
    bucket_tile_index: Tensor,
    bucket_color_transmittance: Tensor,
    bucket_depth_prefix: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Low-level depth-aware blend backward op."""
    grad_projected_means_rgb, grad_conic_opacity_rgb, grad_colors_rgb = core_blend_bwd_op(
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
    grad_projected_means_depth, grad_conic_opacity_depth, grad_primitive_depth = backend().depth_blend_bwd(
        grad_depth,
        depth,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        projected_means,
        conic_opacity,
        primitive_depth,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
        bucket_depth_prefix,
        proper_antialiasing,
        width,
        height,
    )
    return (
        grad_projected_means_rgb + grad_projected_means_depth,
        grad_conic_opacity_rgb + grad_conic_opacity_depth,
        grad_colors_rgb,
        grad_primitive_depth,
    )


@blend_bwd_op.register_fake
def _blend_bwd_fake(
    grad_image: Tensor,
    grad_depth: Tensor,
    image: Tensor,
    depth: Tensor,
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    primitive_depth: Tensor,
    bg_color: Tensor,
    tile_final_transmittances: Tensor,
    tile_max_n_processed: Tensor,
    tile_n_processed: Tensor,
    bucket_tile_index: Tensor,
    bucket_color_transmittance: Tensor,
    bucket_depth_prefix: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del (
        grad_image,
        grad_depth,
        image,
        depth,
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bg_color,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
        bucket_depth_prefix,
        proper_antialiasing,
        width,
        height,
    )
    return (
        torch.empty_like(projected_means),
        torch.empty_like(conic_opacity),
        torch.empty_like(colors_rgb),
        torch.empty_like(primitive_depth),
    )


def _blend_impl(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    primitive_depth: Tensor,
    bg_color: Tensor,
    proper_antialiasing: bool,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Autograd-enabled depth-aware blend op."""
    return blend_fwd_op(
        instance_primitive_indices,
        tile_instance_ranges,
        tile_bucket_offsets,
        bucket_count,
        projected_means,
        conic_opacity,
        colors_rgb,
        primitive_depth,
        bg_color,
        proper_antialiasing,
        width,
        height,
    )


def _blend_fake(*args: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fake implementation for the autograd depth-aware blend op."""
    return _blend_fwd_fake(*args)


def _blend_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    blend_result = parse_blend_outputs(output)
    ctx.save_for_backward(
        blend_result.image,
        blend_result.depth,
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        blend_result.tile_final_transmittances,
        blend_result.tile_max_n_processed,
        blend_result.tile_n_processed,
        blend_result.bucket_tile_index,
        blend_result.bucket_color_transmittance,
        blend_result.bucket_depth_prefix,
    )
    ctx.proper_antialiasing = inputs[9]
    ctx.width = inputs[10]
    ctx.height = inputs[11]


def _blend_backward(
    ctx: Any,
    grad_image: Tensor,
    grad_depth: Tensor,
    *grad_aux: Tensor,
) -> tuple[Tensor | None, ...]:
    del grad_aux
    (
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        grad_primitive_depth,
    ) = blend_bwd_op(
        grad_image,
        grad_depth,
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
        grad_primitive_depth,
        None,
        None,
        None,
        None,
    )


blend_op = register_blend_family(
    op_name="faster_gs_depth::blend",
    forward_impl=_blend_impl,
    fake_impl=_blend_fake,
    setup_context=_blend_setup_context,
    backward_impl=_blend_backward,
)
