"""Gaussian/splatting stage keys."""

from __future__ import annotations

from typing import Final

from ember_core.core.keys import StageKey

GAUSSIAN_ACCUMULATE_SCREEN_STATS: Final = StageKey(
    "gaussian",
    "accumulate_screen_stats",
)
GAUSSIAN_COMPUTE_CLONE_SPLIT_PRUNE_MASKS: Final = StageKey(
    "gaussian",
    "compute_clone_split_prune_masks",
)
GAUSSIAN_APPLY_CLONE_SPLIT_PRUNE: Final = StageKey(
    "gaussian",
    "apply_clone_split_prune",
)
GAUSSIAN_MCMC_RELOCATE_APPEND: Final = StageKey(
    "gaussian",
    "mcmc_relocate_append",
)
GAUSSIAN_MORTON_REORDER: Final = StageKey(
    "gaussian",
    "morton_reorder",
)
