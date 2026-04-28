"""Blend-stage custom ops for the FasterGS Mojo runtime."""

from __future__ import annotations

from typing import Any

import torch
from ember_native_faster_gs.faster_gs.runtime.ops.blend import (
    _blend_fwd_fake,
)
from ember_native_faster_gs.faster_gs.runtime.ops.blend import (
    blend_bwd_op as faster_blend_bwd_op,
)
from ember_native_faster_gs.faster_gs.runtime.packing import (
    parse_blend_outputs,
)
from torch import Tensor

from ember_native_faster_gs_mojo.core.runtime.ops._common import (
    BLOCK_SIZE_BLEND,
    TILE_HEIGHT,
    TILE_WIDTH,
    mojo_backend,
    stable_capacity,
    stable_extent_capacity,
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
    width_capacity: int | None = None,
    height_capacity: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    tile_count = int(tile_instance_ranges.shape[0])
    tile_pixels = tile_count * BLOCK_SIZE_BLEND
    actual_tile_count = (width + TILE_WIDTH - 1) // TILE_WIDTH * (
        (height + TILE_HEIGHT - 1) // TILE_HEIGHT
    )
    actual_tile_pixels = actual_tile_count * BLOCK_SIZE_BLEND
    actual_bucket_count = int(bucket_count.item())
    image_width = width if width_capacity is None else width_capacity
    image_height = height if height_capacity is None else height_capacity
    image_tile_count = (
        (image_width + TILE_WIDTH - 1)
        // TILE_WIDTH
        * ((image_height + TILE_HEIGHT - 1) // TILE_HEIGHT)
    )
    if image_tile_count > tile_count:
        raise ValueError(
            "Blend image capacity requires more tiles than sort provided: "
            f"{image_tile_count} > {tile_count}."
        )
    bucket_capacity = stable_capacity(
        (
            "blend_buckets",
            projected_means.device.type,
            projected_means.device.index,
            int(projected_means.shape[0]),
        ),
        actual_bucket_count,
    )
    outputs = (
        torch.empty(
            (3, image_height, image_width),
            device=projected_means.device,
            dtype=projected_means.dtype,
        ),
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
        torch.tensor([width], device=projected_means.device, dtype=torch.int32),
        torch.tensor([height], device=projected_means.device, dtype=torch.int32),
    )
    # MAX uses the larger allocation for shape stability, but FasterGS backward
    # treats the bucket tensor length as the real bucket count.
    return (
        outputs[0][:, :height, :width],
        outputs[1][:actual_tile_pixels],
        outputs[2][:actual_tile_count],
        outputs[3][:actual_tile_pixels],
        outputs[4][:actual_bucket_count],
        outputs[5][: actual_bucket_count * BLOCK_SIZE_BLEND],
    )


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
    stable_extent: bool = False,
) -> Tensor:
    """Render only the final image through the staged MAX/Mojo blend op."""
    image_width = width
    image_height = height
    if stable_extent:
        image_width = stable_extent_capacity(
            (
                "blend_image_width",
                projected_means.device.type,
                projected_means.device.index,
            ),
            width,
            minimum=width,
        )
        image_height = stable_extent_capacity(
            (
                "blend_image_height",
                projected_means.device.type,
                projected_means.device.index,
            ),
            height,
            minimum=height,
        )
    image = torch.empty(
        (3, image_height, image_width),
        device=projected_means.device,
        dtype=projected_means.dtype,
    )
    mojo_backend().blend_fwd_image_only(
        image,
        instance_primitive_indices,
        tile_instance_ranges,
        projected_means,
        conic_opacity,
        colors_rgb,
        bg_color,
        torch.tensor([image_width], device=projected_means.device, dtype=torch.int32),
        torch.tensor([image_height], device=projected_means.device, dtype=torch.int32),
    )
    return image[:, :height, :width]


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
    width_capacity: int | None = None,
    height_capacity: int | None = None,
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
        width_capacity=width_capacity,
        height_capacity=height_capacity,
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
