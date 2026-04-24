"""Sort-stage custom ops for the FasterGS Mojo runtime."""

from __future__ import annotations

import torch
from torch import Tensor

from splatkit_native_faster_gs.faster_gs.runtime.ops._common import (
    TILE_HEIGHT,
    TILE_WIDTH,
)
from splatkit_native_faster_gs.faster_gs.runtime.ops.sort import _sort_fwd_fake
from splatkit_native_faster_gs_mojo.core.runtime.ops._common import (
    mojo_backend,
    stable_capacity,
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

    tile_count = (width + TILE_WIDTH - 1) // TILE_WIDTH * (
        (height + TILE_HEIGHT - 1) // TILE_HEIGHT
    )
    actual_instance_count = int(instance_count.item())
    instance_capacity = stable_capacity(
        (
            "sort_instances",
            depth_keys.device.type,
            depth_keys.device.index,
            int(depth_keys.shape[0]),
            width,
            height,
        ),
        actual_instance_count,
    )
    outputs = (
        torch.empty(
            (instance_capacity,),
            device=depth_keys.device,
            dtype=torch.int32,
        ),
        torch.empty((tile_count, 2), device=depth_keys.device, dtype=torch.int32),
        torch.empty((tile_count,), device=depth_keys.device, dtype=torch.int32),
        torch.empty((1,), device=depth_keys.device, dtype=torch.int32),
    )
    mojo_backend().sort_fwd(
        *outputs,
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
    return outputs


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
