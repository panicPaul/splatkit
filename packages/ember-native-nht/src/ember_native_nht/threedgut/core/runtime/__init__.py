"""Public staged Python API for the native NHT runtime."""

from ember_native_nht.threedgut.core.runtime.ops.intersect import intersect
from ember_native_nht.threedgut.core.runtime.ops.project import project
from ember_native_nht.threedgut.core.runtime.ops.rasterize import (
    rasterize_depth,
    rasterize_features,
)
from ember_native_nht.threedgut.core.runtime.ops.render import render
from ember_native_nht.threedgut.core.runtime.types import (
    DepthRasterizationResult,
    FeatureRasterizationResult,
    IntersectionResult,
    ProjectionResult,
    RenderResult,
)

__all__ = [
    "DepthRasterizationResult",
    "FeatureRasterizationResult",
    "IntersectionResult",
    "ProjectionResult",
    "RenderResult",
    "intersect",
    "project",
    "rasterize_depth",
    "rasterize_features",
    "render",
]
