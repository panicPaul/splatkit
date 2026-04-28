"""Public staged runtime API for the FasterGS Mojo backend."""

from __future__ import annotations

from ember_native_faster_gs.faster_gs.runtime.packing import (
    make_render_result,
    parse_blend_outputs,
    parse_preprocess_outputs,
    parse_sort_outputs,
)
from ember_native_faster_gs.faster_gs.runtime.types import (
    BlendResult,
    PreprocessResult,
    RenderResult,
    SortResult,
)
from ember_native_faster_gs_mojo.core.runtime.ops import (
    blend_op,
    preprocess_op,
    render_op,
    sort_op,
)


def preprocess(*args, **kwargs) -> PreprocessResult:
    """Return the staged preprocess forward result."""
    return parse_preprocess_outputs(preprocess_op(*args, **kwargs))


def sort(*args, **kwargs) -> SortResult:
    """Return the staged sort forward result."""
    return parse_sort_outputs(sort_op(*args, **kwargs))


def blend(*args, **kwargs) -> BlendResult:
    """Return the staged blend result."""
    return parse_blend_outputs(blend_op(*args, **kwargs))


def render(*args, **kwargs) -> RenderResult:
    """Return the fused render result."""
    return make_render_result(render_op(*args, **kwargs))


__all__ = [
    "BlendResult",
    "PreprocessResult",
    "RenderResult",
    "SortResult",
    "blend",
    "preprocess",
    "render",
    "sort",
]
