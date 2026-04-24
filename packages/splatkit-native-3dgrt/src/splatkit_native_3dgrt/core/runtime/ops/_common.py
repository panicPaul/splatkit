"""Shared helpers for traced native custom ops."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from splatkit_native_3dgrt.core.runtime.state import get_state


def tracer_wrapper(state_token: Tensor) -> Any:
    """Resolve a traced state token into the underlying native tracer wrapper."""
    return get_state(state_token).tracer_wrapper


__all__ = ["tracer_wrapper"]
