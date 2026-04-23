"""Helpers for packing and unpacking staged SVRaster runtime outputs."""

from __future__ import annotations

from torch import Tensor

from splatkit_native_svraster.core.runtime.types import (
    ParsedRasterizeOutputs,
    PreprocessResult,
    RasterizeResult,
)


def parse_preprocess_outputs(outputs: tuple[Tensor, ...]) -> PreprocessResult:
    """Convert raw preprocess op outputs into a structured result."""
    return PreprocessResult.from_tensors(*outputs)


def parse_rasterize_outputs(
    outputs: tuple[Tensor, ...],
) -> ParsedRasterizeOutputs:
    """Convert raw rasterize op outputs into a structured result."""
    if len(outputs) != 8:
        raise ValueError(
            "Unexpected svraster::rasterize output arity: "
            f"expected 8, got {len(outputs)}."
        )
    return ParsedRasterizeOutputs(
        num_rendered=outputs[0],
        binning_buffer=outputs[1],
        image_buffer=outputs[2],
        result=RasterizeResult(
            color=outputs[3],
            depth=outputs[4],
            normal=outputs[5],
            transmittance=outputs[6],
            max_weight=outputs[7],
        ),
    )
