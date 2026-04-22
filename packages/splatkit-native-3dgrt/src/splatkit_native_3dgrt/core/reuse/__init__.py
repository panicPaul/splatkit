"""Reusable raw ops and glue helpers for traced native backends."""

from __future__ import annotations

from typing import Any

from splatkit_native_3dgrt.core.reuse.factories import (
    register_render_family,
    register_trace_family,
)

__all__ = [
    "build_acc_op",
    "destroy_acc_op",
    "register_render_family",
    "register_trace_family",
    "render_bwd_op",
    "render_fwd_op",
    "render_op",
    "trace_bwd_op",
    "trace_fwd_op",
    "trace_op",
    "update_acc_op",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the traced root raw and combined ops."""
    if name in {
        "build_acc_op",
        "destroy_acc_op",
        "render_bwd_op",
        "render_fwd_op",
        "render_op",
        "trace_bwd_op",
        "trace_fwd_op",
        "trace_op",
        "update_acc_op",
    }:
        from splatkit_native_3dgrt.core.runtime import (
            ops as runtime_ops,
        )

        return getattr(runtime_ops, name)
    raise AttributeError(name)
