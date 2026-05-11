"""Core product keys."""

from __future__ import annotations

from typing import Final

from ember_core.core.keys import ProductKey

RGB: Final = ProductKey("core", "rgb")
DEPTH: Final = ProductKey("core", "depth")
ALPHA: Final = ProductKey("core", "alpha")
NORMALS: Final = ProductKey("core", "normals")
GAUSSIAN_IMPACT_SCORE: Final = ProductKey("core", "gaussian_impact_score")
PROJECTIONS_2D: Final = ProductKey("core", "2d_projections")
PROJECTIVE_INTERSECTION_TRANSFORMS: Final = ProductKey(
    "core",
    "projective_intersection_transforms",
)
SCREEN_SPACE_DENSIFICATION_SIGNALS: Final = ProductKey(
    "core",
    "screen_space_densification_signals",
)
