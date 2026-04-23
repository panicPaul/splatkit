"""Reusable raw ops and glue helpers owned by the FasterGS Mojo core backend."""

from __future__ import annotations

from typing import Any

from splatkit_native_faster_gs.faster_gs.reuse import (
    register_blend_family,
    register_render_family,
)

__all__ = [
    "blend_bwd_op",
    "blend_fwd_op",
    "blend_op",
    "preprocess_bwd_op",
    "preprocess_fwd_op",
    "preprocess_op",
    "register_blend_family",
    "register_render_family",
    "render_bwd_op",
    "render_fwd_op",
    "render_op",
    "sort_fwd_op",
    "sort_op",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the family-local raw and combined ops."""
    if name in {
        "blend_bwd_op",
        "blend_fwd_op",
        "blend_op",
        "preprocess_bwd_op",
        "preprocess_fwd_op",
        "preprocess_op",
        "render_bwd_op",
        "render_fwd_op",
        "render_op",
        "sort_fwd_op",
        "sort_op",
    }:
        from splatkit_native_faster_gs_mojo.core.runtime import (
            ops as runtime_ops,
        )

        return getattr(runtime_ops, name)
    raise AttributeError(name)
