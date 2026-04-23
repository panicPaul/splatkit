"""Blend-stage custom ops for the FasterGS Mojo runtime."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from splatkit_native_faster_gs.faster_gs.reuse.factories import (
    register_blend_family,
)
from splatkit_native_faster_gs.faster_gs.runtime.ops.blend import (
    _blend_bwd_fake,
    _blend_fwd_fake,
)
from splatkit_native_faster_gs.faster_gs.runtime.packing import (
    parse_blend_outputs,
)
from splatkit_native_faster_gs_mojo.core.runtime._mojo import (
    custom_op_library_path,
)
from splatkit_native_faster_gs_mojo.core.runtime.ops._common import (
    BLOCK_SIZE_BLEND,
)


@lru_cache(maxsize=None)
def _graph_blend_ops(
    device_index: int,
) -> tuple[Any, Any]:
    """Build symbolic MAX graph wrappers for the Mojo blend kernels."""
    from max.dtype import DType as MaxDType
    from max.experimental.torch import graph_op
    from max.graph import DeviceRef, TensorType, ops

    device = DeviceRef.GPU(device_index)
    ops_root = custom_op_library_path()

    blend_fwd_input_types = (
        TensorType(MaxDType.int32, ("instance_count",), device=device),
        TensorType(MaxDType.int32, ("tile_count", 2), device=device),
        TensorType(MaxDType.int32, ("tile_count",), device=device),
        TensorType(MaxDType.int32, (1,), device=device),
        TensorType(MaxDType.float32, ("primitive_count", 2), device=device),
        TensorType(MaxDType.float32, ("primitive_count", 4), device=device),
        TensorType(MaxDType.float32, ("primitive_count", 3), device=device),
        TensorType(MaxDType.float32, (3,), device=device),
    )
    blend_fwd_output_types = (
        TensorType(MaxDType.float32, (3, "height", "width"), device=device),
        TensorType(MaxDType.float32, ("tile_pixels",), device=device),
        TensorType(MaxDType.int32, ("tile_count",), device=device),
        TensorType(MaxDType.int32, ("tile_pixels",), device=device),
        TensorType(MaxDType.int32, ("bucket_total",), device=device),
        TensorType(
            MaxDType.float32,
            ("bucket_color_rows", 4),
            device=device,
        ),
    )

    @graph_op(
        name=f"faster_gs_mojo_blend_fwd_graph_cuda_{device_index}",
        kernel_library=ops_root,
        input_types=blend_fwd_input_types,
        output_types=blend_fwd_output_types,
    )
    def blend_fwd_graph(
        instance_primitive_indices: Any,
        tile_instance_ranges: Any,
        tile_bucket_offsets: Any,
        bucket_count: Any,
        projected_means: Any,
        conic_opacity: Any,
        colors_rgb: Any,
        bg_color: Any,
    ) -> list[Any]:
        return ops.custom(
            "blend_fwd",
            device,
            [
                instance_primitive_indices,
                tile_instance_ranges,
                tile_bucket_offsets,
                bucket_count,
                projected_means,
                conic_opacity,
                colors_rgb,
                bg_color,
            ],
            out_types=blend_fwd_output_types,
        )

    blend_bwd_input_types = (
        TensorType(MaxDType.float32, (3, "height", "width"), device=device),
        TensorType(MaxDType.float32, (3, "height", "width"), device=device),
        TensorType(MaxDType.int32, ("instance_count",), device=device),
        TensorType(MaxDType.int32, ("tile_count", 2), device=device),
        TensorType(MaxDType.int32, ("tile_count",), device=device),
        TensorType(MaxDType.float32, ("primitive_count", 2), device=device),
        TensorType(MaxDType.float32, ("primitive_count", 4), device=device),
        TensorType(MaxDType.float32, ("primitive_count", 3), device=device),
        TensorType(MaxDType.float32, (3,), device=device),
        TensorType(MaxDType.float32, ("tile_pixels",), device=device),
        TensorType(MaxDType.int32, ("tile_count",), device=device),
        TensorType(MaxDType.int32, ("tile_pixels",), device=device),
        TensorType(MaxDType.int32, ("bucket_total",), device=device),
        TensorType(
            MaxDType.float32,
            ("bucket_color_rows", 4),
            device=device,
        ),
        TensorType(MaxDType.int32, (1,), device=device),
    )
    blend_bwd_output_types = (
        TensorType(MaxDType.float32, ("primitive_count", 2), device=device),
        TensorType(MaxDType.float32, ("primitive_count", 4), device=device),
        TensorType(MaxDType.float32, ("primitive_count", 3), device=device),
    )

    @graph_op(
        name=f"faster_gs_mojo_blend_bwd_graph_cuda_{device_index}",
        kernel_library=ops_root,
        input_types=blend_bwd_input_types,
        output_types=blend_bwd_output_types,
    )
    def blend_bwd_graph(
        grad_image: Any,
        image: Any,
        instance_primitive_indices: Any,
        tile_instance_ranges: Any,
        tile_bucket_offsets: Any,
        projected_means: Any,
        conic_opacity: Any,
        colors_rgb: Any,
        bg_color: Any,
        tile_final_transmittances: Any,
        tile_max_n_processed: Any,
        tile_n_processed: Any,
        bucket_tile_index: Any,
        bucket_color_transmittance: Any,
        proper_antialiasing_flag: Any,
    ) -> list[Any]:
        return ops.custom(
            "blend_bwd",
            device,
            [
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
                proper_antialiasing_flag,
            ],
            out_types=blend_bwd_output_types,
        )

    return blend_fwd_graph, blend_bwd_graph


def _graph_blend_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError(
            "The FasterGS Mojo blend graph op currently requires CUDA tensors."
        )
    return 0 if device.index is None else int(device.index)


def _call_graph_blend_fwd(
    instance_primitive_indices: Tensor,
    tile_instance_ranges: Tensor,
    tile_bucket_offsets: Tensor,
    bucket_count: Tensor,
    projected_means: Tensor,
    conic_opacity: Tensor,
    colors_rgb: Tensor,
    bg_color: Tensor,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    blend_fwd_graph, _ = _graph_blend_ops(
        _graph_blend_device_index(projected_means.device)
    )

    tile_count = int(tile_instance_ranges.shape[0])
    bucket_total = int(bucket_count.item())
    device = projected_means.device
    dtype = projected_means.dtype
    tile_pixels = tile_count * BLOCK_SIZE_BLEND
    image = torch.empty((3, height, width), device=device, dtype=dtype)
    tile_final_transmittances = torch.empty((tile_pixels,), device=device, dtype=dtype)
    tile_max_n_processed = torch.empty((tile_count,), device=device, dtype=torch.int32)
    tile_n_processed = torch.empty((tile_pixels,), device=device, dtype=torch.int32)
    bucket_tile_index = torch.empty((bucket_total,), device=device, dtype=torch.int32)
    bucket_color_transmittance = torch.empty(
        (bucket_total * BLOCK_SIZE_BLEND, 4),
        device=device,
        dtype=dtype,
    )
    blend_fwd_graph(
        image,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
        instance_primitive_indices.detach(),
        tile_instance_ranges.detach(),
        tile_bucket_offsets.detach(),
        bucket_count.detach(),
        projected_means.detach(),
        conic_opacity.detach(),
        colors_rgb.detach(),
        bg_color.detach(),
    )
    return (
        image,
        tile_final_transmittances,
        tile_max_n_processed,
        tile_n_processed,
        bucket_tile_index,
        bucket_color_transmittance,
    )


def _call_graph_blend_bwd(
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
    del width, height
    _, blend_bwd_graph = _graph_blend_ops(
        _graph_blend_device_index(projected_means.device)
    )

    grad_projected_means = torch.empty_like(projected_means)
    grad_conic_opacity = torch.empty_like(conic_opacity)
    grad_colors_rgb = torch.empty_like(colors_rgb)
    proper_antialiasing_flag = torch.tensor(
        [int(proper_antialiasing)],
        device=projected_means.device,
        dtype=torch.int32,
    )
    blend_bwd_graph(
        grad_projected_means,
        grad_conic_opacity,
        grad_colors_rgb,
        grad_image.detach(),
        image.detach(),
        instance_primitive_indices.detach(),
        tile_instance_ranges.detach(),
        tile_bucket_offsets.detach(),
        projected_means.detach(),
        conic_opacity.detach(),
        colors_rgb.detach(),
        bg_color.detach(),
        tile_final_transmittances.detach(),
        tile_max_n_processed.detach(),
        tile_n_processed.detach(),
        bucket_tile_index.detach(),
        bucket_color_transmittance.detach(),
        proper_antialiasing_flag,
    )
    return grad_projected_means, grad_conic_opacity, grad_colors_rgb


@torch.library.custom_op("faster_gs_mojo::blend_fwd", mutates_args=())
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
    """Low-level FasterGS Mojo blend forward op."""
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
        width,
        height,
    )


@blend_fwd_op.register_fake
def _blend_fwd_fake_local(
    *args: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _blend_fwd_fake(*args)


@torch.library.custom_op("faster_gs_mojo::blend_bwd", mutates_args=())
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
    """Low-level FasterGS Mojo blend backward op."""
    return _call_graph_blend_bwd(
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
def _blend_bwd_fake_local(*args: Any) -> tuple[Tensor, Tensor, Tensor]:
    return _blend_bwd_fake(*args)


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


blend_op = register_blend_family(
    op_name="faster_gs_mojo::blend",
    forward_impl=_blend_impl,
    fake_impl=_blend_fake,
    setup_context=_blend_setup_context,
    backward_impl=_blend_backward,
)
