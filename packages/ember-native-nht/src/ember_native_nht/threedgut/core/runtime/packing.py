"""Helpers for packing and unpacking staged NHT runtime outputs."""

from __future__ import annotations

from ember_native_nht.threedgut.core.runtime.types import (
    DepthRasterizationResult,
    FeatureRasterizationResult,
    IntersectionResult,
    ProjectionResult,
)
from torch import Tensor


def parse_projection_outputs(
    outputs: tuple[Tensor | None, ...],
) -> ProjectionResult:
    """Convert raw projection op outputs into a structured result."""
    if len(outputs) != 5:
        raise ValueError(
            "Unexpected native NHT projection output arity: "
            f"expected 5, got {len(outputs)}."
        )
    return ProjectionResult.from_tensors(*outputs)


def parse_intersection_outputs(
    outputs: tuple[Tensor, ...],
) -> IntersectionResult:
    """Convert raw intersection op outputs into a structured result."""
    if len(outputs) != 4:
        raise ValueError(
            "Unexpected native NHT intersection output arity: "
            f"expected 4, got {len(outputs)}."
        )
    return IntersectionResult.from_tensors(*outputs)


def parse_feature_rasterization_outputs(
    outputs: tuple[Tensor, Tensor],
) -> FeatureRasterizationResult:
    """Convert raw feature rasterization outputs into a structured result."""
    return FeatureRasterizationResult.from_tensors(*outputs)


def parse_depth_rasterization_outputs(
    outputs: tuple[Tensor, Tensor],
) -> DepthRasterizationResult:
    """Convert raw depth rasterization outputs into a structured result."""
    return DepthRasterizationResult.from_tensors(*outputs)
