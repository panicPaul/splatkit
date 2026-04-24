"""Shared helpers for SVRaster native custom ops."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from splatkit_native_svraster.core.runtime._extension import load_extension


def backend() -> Any:
    """Return the loaded native SVRaster extension."""
    return load_extension()


def requires_grad(*tensors: Tensor) -> bool:
    """Return whether any differentiable input needs backward state."""
    return any(tensor.requires_grad for tensor in tensors)
