"""Shared helpers for FasterGS Mojo custom ops."""

from __future__ import annotations

from typing import Any

from splatkit_native_faster_gs.faster_gs.runtime.ops._common import (
    BLOCK_SIZE_BLEND,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from torch import Tensor

from splatkit_native_faster_gs_mojo.core.runtime._mojo import (
    load_custom_op_library,
)

_CAPACITY_CACHE: dict[tuple[object, ...], int] = {}

__all__ = [
    "BLOCK_SIZE_BLEND",
    "TILE_HEIGHT",
    "TILE_WIDTH",
    "mojo_backend",
    "normalize_active_sh_bases",
    "requires_grad",
    "stable_capacity",
    "stable_extent_capacity",
    "stable_multiple_capacity",
]


def mojo_backend() -> Any:
    """Return the loaded MAX/Mojo custom-op library."""
    return load_custom_op_library()


def _round_capacity(value: int) -> int:
    """Round a non-negative count to the next power-of-two capacity."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _round_multiple_capacity(value: int, multiple: int) -> int:
    """Round a non-negative count to the next fixed-multiple capacity."""
    if value <= 0:
        return multiple
    return ((value + multiple - 1) // multiple) * multiple


def _round_extent_capacity(value: int, multiple: int) -> int:
    """Round a positive image extent to a fixed-multiple capacity."""
    if value <= 0:
        return multiple
    return ((value + multiple - 1) // multiple) * multiple


def stable_capacity(
    key: tuple[object, ...],
    required: int,
    *,
    minimum: int = 1,
) -> int:
    """Return a non-shrinking capacity for MAX shape specialization."""
    capacity = max(minimum, _round_capacity(required))
    previous = _CAPACITY_CACHE.get(key)
    if previous is not None and previous > capacity:
        capacity = previous
    _CAPACITY_CACHE[key] = capacity
    return capacity


def stable_multiple_capacity(
    key: tuple[object, ...],
    required: int,
    *,
    minimum: int = 1,
    multiple: int = 8192,
    headroom_numerator: int = 1,
    headroom_denominator: int = 1,
) -> int:
    """Return a non-shrinking fixed-multiple capacity."""
    if headroom_numerator < 1:
        raise ValueError("headroom_numerator must be >= 1.")
    if headroom_denominator < 1:
        raise ValueError("headroom_denominator must be >= 1.")
    previous = _CAPACITY_CACHE.get(key)
    if previous is not None and previous >= required:
        return previous
    required_with_headroom = (
        required * headroom_numerator + headroom_denominator - 1
    ) // headroom_denominator
    capacity = max(minimum, _round_multiple_capacity(required_with_headroom, multiple))
    if previous is not None and previous > capacity:
        capacity = previous
    _CAPACITY_CACHE[key] = capacity
    return capacity


def stable_extent_capacity(
    key: tuple[object, ...],
    required: int,
    *,
    minimum: int,
    multiple: int = 256,
) -> int:
    """Return a non-shrinking fixed-multiple image extent capacity."""
    capacity = max(minimum, _round_extent_capacity(required, multiple))
    previous = _CAPACITY_CACHE.get(key)
    if previous is not None and previous > capacity:
        capacity = previous
    _CAPACITY_CACHE[key] = capacity
    return capacity


def normalize_active_sh_bases(active_sh_bases: int) -> int:
    """Return the FasterGS core SH basis-count specialization key."""
    if active_sh_bases in (1, 4, 9, 16):
        return active_sh_bases
    raise ValueError(
        "FasterGS Mojo preprocess specialization currently supports only "
        "`active_sh_bases` in {1, 4, 9, 16}."
    )


def requires_grad(*tensors: Tensor) -> bool:
    """Return whether any differentiable input needs backward state."""
    return any(tensor.requires_grad for tensor in tensors)
