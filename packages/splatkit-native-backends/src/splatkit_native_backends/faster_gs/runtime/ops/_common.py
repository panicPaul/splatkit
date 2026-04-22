"""Shared helpers for FasterGS native custom ops."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from splatkit_native_backends.faster_gs.runtime._extension import (
    load_extension,
)

TILE_WIDTH = 16
TILE_HEIGHT = 16
BLOCK_SIZE_BLEND = TILE_WIDTH * TILE_HEIGHT


def backend() -> Any:
    """Return the loaded native rasterization extension."""
    return load_extension()


def requires_grad(*tensors: Tensor) -> bool:
    """Return whether any differentiable input needs backward state."""
    return any(tensor.requires_grad for tensor in tensors)

