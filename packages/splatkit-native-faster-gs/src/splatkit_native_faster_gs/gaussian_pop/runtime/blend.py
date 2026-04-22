"""GaussianPOP blend-stage custom ops."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from splatkit_native_faster_gs.faster_gs_depth.reuse import (
    blend_bwd_op as depth_blend_bwd_op,
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
from splatkit_native_faster_gs.gaussian_pop.runtime._extension import (
    load_extension,
)
from splatkit_native_faster_gs.gaussian_pop.runtime.types import (
    BlendResult,
)


def backend() -> Any:
    """Return the loaded native GaussianPOP extension."""
    return load_extension()


def _parse_blend_outputs(outputs: tuple[Tensor, ...]) -> BlendResult:
    depth = outputs[1] if outputs[1].numel() > 0 else None
    bucket_depth_prefix = outputs[7] if outputs[1].numel() > 0 else None
    gaussian_impact_score = outputs[8] if outputs[8].numel() > 0 else None
    return BlendResult(
        image=outputs[0],
        depth=depth,
        tile_final_transmittances=outputs[2],
        tile_max_n_processed=outputs[3],
        tile_n_processed=outputs[4],
        bucket_tile_index=outputs[5],
        bucket_color_transmittance=outputs[6],
        bucket_depth_prefix=bucket_depth_prefix,
        gaussian_impact_score=gaussian_impact_score,
    )


@torch.library.custom_op("gaussian_pop::blend_fwd", mutates_args=())
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
    return_depth: bool,
    return_gaussian_impact_score: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Low-level GaussianPOP blend forward op."""
    return backend().pop_blend_fwd(
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
        return_depth,
        return_gaussian_impact_score,
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
    return_depth: bool,
    return_gaussian_impact_score: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    depth = (
        torch.empty((height, width), device=device, dtype=dtype)
        if return_depth
        else torch.empty((0,), device=device, dtype=dtype)
    )
    bucket_depth_prefix = (
        torch.empty((tile_count * BLOCK_SIZE_BLEND,), device=device, dtype=dtype)
        if return_depth
        else torch.empty((0,), device=device, dtype=dtype)
    )
    gaussian_impact_score = (
        torch.empty((projected_means.shape[0],), device=device, dtype=dtype)
        if return_gaussian_impact_score
        else torch.empty((0,), device=device, dtype=dtype)
    )
    return (
        torch.empty((3, height, width), device=device, dtype=dtype),
        depth,
        torch.empty((tile_pixels,), device=device, dtype=dtype),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((tile_pixels,), device=device, dtype=torch.int32),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((tile_count * BLOCK_SIZE_BLEND, 4), device=device, dtype=dtype),
        bucket_depth_prefix,
        gaussian_impact_score,
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
    return_depth: bool,
    return_gaussian_impact_score: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Autograd-enabled GaussianPOP blend op."""
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
        return_depth,
        return_gaussian_impact_score,
    )


def _blend_fake(*args: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fake implementation for the autograd GaussianPOP blend op."""
    return _blend_fwd_fake(*args)


def _blend_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    blend_result = _parse_blend_outputs(output)
    ctx.proper_antialiasing = inputs[9]
    ctx.width = inputs[10]
    ctx.height = inputs[11]
    ctx.return_depth = inputs[12]
    ctx.return_gaussian_impact_score = inputs[13]
    saved = [
        blend_result.image,
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[8],
        blend_result.tile_final_transmittances,
        blend_result.tile_max_n_processed,
        blend_result.tile_n_processed,
        blend_result.bucket_tile_index,
        blend_result.bucket_color_transmittance,
    ]
    if inputs[12]:
        saved.extend(
            [
                output[1],
                inputs[7],
                output[7],
            ]
        )
    ctx.save_for_backward(*saved)


def _blend_backward(
    ctx: Any,
    grad_image: Tensor,
    grad_depth: Tensor,
    grad_tile_final_transmittances: Tensor,
    grad_tile_max_n_processed: Tensor,
    grad_tile_n_processed: Tensor,
    grad_bucket_tile_index: Tensor,
    grad_bucket_color_transmittance: Tensor,
    grad_bucket_depth_prefix: Tensor,
    grad_gaussian_impact_score: Tensor,
) -> tuple[Tensor | None, ...]:
    del (
        grad_tile_final_transmittances,
        grad_tile_max_n_processed,
        grad_tile_n_processed,
        grad_bucket_tile_index,
        grad_bucket_color_transmittance,
        grad_bucket_depth_prefix,
        grad_gaussian_impact_score,
    )
    if ctx.return_depth:
        (
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
            depth,
            primitive_depth,
            bucket_depth_prefix,
        ) = ctx.saved_tensors
        grad_projected_means_rgb, grad_conic_opacity_rgb, grad_colors_rgb = (
            core_blend_bwd_op(
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
                ctx.proper_antialiasing,
                ctx.width,
                ctx.height,
            )
        )
        (
            grad_projected_means_depth,
            grad_conic_opacity_depth,
            grad_colors_unused,
            grad_primitive_depth,
        ) = depth_blend_bwd_op(
            grad_image.new_zeros(()).expand_as(grad_image),
            grad_depth,
            image,
            depth,
            instance_primitive_indices,
            tile_instance_ranges,
            tile_bucket_offsets,
            projected_means,
            conic_opacity,
            colors_rgb,
            primitive_depth,
            bg_color,
            tile_final_transmittances,
            tile_max_n_processed,
            tile_n_processed,
            bucket_tile_index,
            bucket_color_transmittance,
            bucket_depth_prefix,
            ctx.proper_antialiasing,
            ctx.width,
            ctx.height,
        )
        del grad_colors_unused
        return (
            None,
            None,
            None,
            None,
            grad_projected_means_rgb + grad_projected_means_depth,
            grad_conic_opacity_rgb + grad_conic_opacity_depth,
            grad_colors_rgb,
            grad_primitive_depth,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    (
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
    ) = ctx.saved_tensors
    grad_projected_means, grad_conic_opacity, grad_colors_rgb = core_blend_bwd_op(
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
        None,
        None,
        None,
    )


blend_op = register_blend_family(
    op_name="gaussian_pop::blend",
    forward_impl=_blend_impl,
    fake_impl=_blend_fake,
    setup_context=_blend_setup_context,
    backward_impl=_blend_backward,
)


def blend(
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
    *,
    width: int,
    height: int,
    return_depth: bool,
    return_gaussian_impact_score: bool,
) -> BlendResult:
    """Run the backend-owned GaussianPOP blend stage."""
    result = _parse_blend_outputs(
        blend_op(
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
            return_depth,
            return_gaussian_impact_score,
        )
    )
    return BlendResult(
        image=result.image,
        tile_final_transmittances=result.tile_final_transmittances,
        tile_max_n_processed=result.tile_max_n_processed,
        tile_n_processed=result.tile_n_processed,
        bucket_tile_index=result.bucket_tile_index,
        bucket_color_transmittance=result.bucket_color_transmittance,
        bucket_depth_prefix=result.bucket_depth_prefix,
        gaussian_impact_score=(
            result.gaussian_impact_score.detach()
            if result.gaussian_impact_score is not None
            else None
        ),
        depth=result.depth,
    )
