"""Gaussian/splatting product keys."""

from __future__ import annotations

from typing import Final

from ember_core.core.keys import ProductKey

GAUSSIAN_IMPORTANCE_SCORE: Final = ProductKey(
    "gaussian",
    "importance_score",
)
GAUSSIAN_MAX_SCREEN_RADIUS: Final = ProductKey(
    "gaussian",
    "max_screen_radius",
)
GAUSSIAN_METRIC_ATTRIBUTION: Final = ProductKey(
    "gaussian",
    "metric_attribution",
)
