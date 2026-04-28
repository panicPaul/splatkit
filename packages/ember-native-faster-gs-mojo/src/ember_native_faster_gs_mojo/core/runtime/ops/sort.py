"""Sort-stage custom ops for the FasterGS Mojo runtime."""

from __future__ import annotations

import torch
from ember_native_faster_gs.faster_gs.runtime.ops.sort import _sort_fwd_fake
from torch import Tensor

from ember_native_faster_gs_mojo.core.runtime.ops._common import (
    TILE_HEIGHT,
    TILE_WIDTH,
    mojo_backend,
    stable_capacity,
    stable_multiple_capacity,
)


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
    *,
    tile_count_minimum: int = 4096,
    capacity_headroom_numerator: int = 1,
    capacity_headroom_denominator: int = 1,
    return_capacity: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run the MAX/Mojo sort forward stage."""
    if depth_keys.device.type != "cuda":
        return _sort_fwd_fake(
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

    actual_tile_count = (
        (width + TILE_WIDTH - 1)
        // TILE_WIDTH
        * ((height + TILE_HEIGHT - 1) // TILE_HEIGHT)
    )
    tile_count = stable_capacity(
        (
            "sort_tiles",
            depth_keys.device.type,
            depth_keys.device.index,
        ),
        max(actual_tile_count, tile_count_minimum),
        minimum=tile_count_minimum,
    )
    actual_visible_count = int(visible_count.item())
    visible_capacity = stable_multiple_capacity(
        (
            "sort_visible",
            depth_keys.device.type,
            depth_keys.device.index,
        ),
        actual_visible_count,
        minimum=2048,
        multiple=2048,
        headroom_numerator=capacity_headroom_numerator,
        headroom_denominator=capacity_headroom_denominator,
    )
    actual_instance_count = int(instance_count.item())
    instance_capacity = stable_multiple_capacity(
        (
            "sort_instances",
            depth_keys.device.type,
            depth_keys.device.index,
        ),
        actual_instance_count,
        minimum=8192,
        multiple=8192,
        headroom_numerator=capacity_headroom_numerator,
        headroom_denominator=capacity_headroom_denominator,
    )
    outputs = (
        torch.empty(
            (instance_capacity,),
            device=depth_keys.device,
            dtype=torch.int32,
        ),
        torch.empty(
            (tile_count, 2), device=depth_keys.device, dtype=torch.int32
        ),
        torch.empty((tile_count,), device=depth_keys.device, dtype=torch.int32),
        torch.empty((1,), device=depth_keys.device, dtype=torch.int32),
    )
    work_outputs = (
        torch.empty(
            (visible_capacity,), device=depth_keys.device, dtype=torch.uint32
        ),
        torch.empty(
            (visible_capacity,), device=depth_keys.device, dtype=torch.int32
        ),
        torch.empty(
            (visible_capacity,), device=depth_keys.device, dtype=torch.int32
        ),
        torch.empty(
            (visible_capacity,), device=depth_keys.device, dtype=torch.int32
        ),
    )
    mojo_backend().sort_fwd(
        *outputs,
        *work_outputs,
        depth_keys,
        primitive_indices,
        num_touched_tiles,
        screen_bounds,
        projected_means,
        conic_opacity,
        visible_count,
        instance_count,
        torch.tensor([width], device=depth_keys.device, dtype=torch.int32),
        torch.tensor([height], device=depth_keys.device, dtype=torch.int32),
    )
    if return_capacity:
        return outputs
    return (
        outputs[0],
        outputs[1][:actual_tile_count],
        outputs[2][:actual_tile_count],
        outputs[3],
    )


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
    """Alias the staged sort forward op."""
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
