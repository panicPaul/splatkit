"""Helpers for packing RADFOAM runtime outputs."""

from __future__ import annotations

from torch import Tensor

from ember_native_radfoam.radfoam.runtime.types import TraceResult

_TRACE_OUTPUT_COUNT = 5


def parse_trace_outputs(outputs: tuple[Tensor, ...]) -> TraceResult:
    """Convert raw trace op outputs into a structured result."""
    if len(outputs) != _TRACE_OUTPUT_COUNT:
        raise ValueError(
            "Unexpected radfoam::trace output arity: "
            f"expected {_TRACE_OUTPUT_COUNT}, got {len(outputs)}."
        )
    return TraceResult.from_tensors(*outputs)
