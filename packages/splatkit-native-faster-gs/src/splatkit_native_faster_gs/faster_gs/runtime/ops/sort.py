"""Sort-stage custom ops for the FasterGS native runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from splatkit_native_faster_gs.faster_gs.runtime.ops._common import (
    TILE_HEIGHT,
    TILE_WIDTH,
    backend,
)


@torch.library.custom_op("faster_gs::sort_fwd", mutates_args=())
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
    tile_count = ((width + TILE_WIDTH - 1) // TILE_WIDTH) * (
        (height + TILE_HEIGHT - 1) // TILE_HEIGHT
    )
    return (
        torch.empty_like(primitive_indices),
        torch.empty((tile_count, 2), device=device, dtype=torch.int32),
        torch.empty((tile_count,), device=device, dtype=torch.int32),
        torch.empty((1,), device=device, dtype=torch.int32),
    )


@torch.library.custom_op("faster_gs::sort", mutates_args=())
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

