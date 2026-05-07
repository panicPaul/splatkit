"""Compatibility runtime entry points for the native NHT 3DGUT backend."""

from ember_native_nht.threedgut.core.runtime import (
    DepthRasterizationResult,
    FeatureRasterizationResult,
    IntersectionResult,
    ProjectionResult,
    RenderResult,
    intersect,
    project,
    rasterize_depth,
    rasterize_features,
    render,
)
from ember_native_nht.threedgut.runtime.rasterization import rasterization_nht

__all__ = [
    "DepthRasterizationResult",
    "FeatureRasterizationResult",
    "IntersectionResult",
    "ProjectionResult",
    "RenderResult",
    "intersect",
    "project",
    "rasterization_nht",
    "rasterize_depth",
    "rasterize_features",
    "render",
]
