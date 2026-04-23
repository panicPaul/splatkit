"""Sort-stage custom ops for the FasterGS Mojo runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from splatkit_native_faster_gs.faster_gs.runtime.ops.sort import (
    _sort_fwd_fake,
    sort_fwd_op as faster_sort_fwd_op,
)


@torch.library.custom_op("faster_gs_mojo::sort_fwd", mutates_args=())
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
    """Delegate sort forward to the native FasterGS root backend."""
    return faster_sort_fwd_op(
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
def _sort_fwd_fake_local(
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


@torch.library.custom_op("faster_gs_mojo::sort", mutates_args=())
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
    return _sort_fwd_fake(*args)
