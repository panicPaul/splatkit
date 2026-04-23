"""Shared helpers for FasterGS Mojo custom ops."""

from __future__ import annotations

from torch import Tensor

from splatkit_native_faster_gs.faster_gs.runtime.ops._common import (
    BLOCK_SIZE_BLEND,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from splatkit_native_faster_gs_mojo.core.runtime._mojo import (
    load_custom_op_library,
)


def mojo_backend():
    """Return the loaded MAX/Mojo custom-op library."""
    return load_custom_op_library()


def requires_grad(*tensors: Tensor) -> bool:
    """Return whether any differentiable input needs backward state."""
    return any(tensor.requires_grad for tensor in tensors)
