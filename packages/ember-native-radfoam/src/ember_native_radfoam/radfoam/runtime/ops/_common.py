"""Shared helpers for RADFOAM native custom ops."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor

from ember_native_radfoam.radfoam.runtime._extension import load_extension


def backend() -> Any:
    """Return Ember's vendored RADFOAM bindings module."""
    return load_extension()


@lru_cache(maxsize=8)
def pipeline(sh_degree: int, dtype: torch.dtype) -> Any:
    """Create or reuse a RADFOAM tracing pipeline."""
    return backend().create_pipeline(int(sh_degree), dtype)


def empty_depth_like(rays: Tensor) -> Tensor:
    """Return an empty depth-quantile tensor aligned with ray batches."""
    return torch.empty(
        (*rays.shape[:-1], 0),
        dtype=torch.float32,
        device=rays.device,
    )


def pow2_round_up(value: int) -> int:
    """Round a positive integer up to a power of two."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()
