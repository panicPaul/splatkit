"""Typed Gaussian/splatting recipes."""

from __future__ import annotations

from dataclasses import dataclass

from ember_core.core.families import GAUSSIAN
from ember_core.core.keys import ProductKey, SceneFamilyKey, StageKey, TraitKey
from ember_core.core.products import RGB, SCREEN_SPACE_DENSIFICATION_SIGNALS
from ember_core.densification.contracts import Schedule

from ember_splatting_training.fastergs import GaussianFastGS
from ember_splatting_training.products import GAUSSIAN_IMPORTANCE_SCORE
from ember_splatting_training.stages import (
    GAUSSIAN_ACCUMULATE_SCREEN_STATS,
    GAUSSIAN_APPLY_CLONE_SPLIT_PRUNE,
    GAUSSIAN_COMPUTE_CLONE_SPLIT_PRUNE_MASKS,
)


@dataclass(frozen=True, slots=True)
class FastGSDensificationRecipe:
    """Declarative FastGS-style Gaussian densification recipe."""

    refine_every: int = 100
    start_iter: int = 600
    stop_iter: int = 14_900
    loss_thresh: float = 0.1
    grad_threshold: float = 2e-4
    grad_abs_threshold: float = 1.2e-3
    dense_fraction: float = 0.01
    prune_opacity_threshold: float = 0.005
    opacity_reset_every: int = 3_000
    probe_view_count: int = 10
    importance_threshold: float = 5.0

    @property
    def scene_family(self) -> SceneFamilyKey:
        """Return the targeted scene family."""
        return GAUSSIAN

    @property
    def schedule(self) -> Schedule:
        """Return the refinement schedule."""
        return Schedule(
            start_iteration=self.start_iter,
            end_iteration=self.stop_iter,
            frequency=self.refine_every,
        )

    def stages(self) -> tuple[StageKey, ...]:
        """Return symbolic stages requested by this recipe."""
        return (
            GAUSSIAN_ACCUMULATE_SCREEN_STATS,
            GAUSSIAN_COMPUTE_CLONE_SPLIT_PRUNE_MASKS,
            GAUSSIAN_APPLY_CLONE_SPLIT_PRUNE,
        )

    def products(self) -> frozenset[ProductKey]:
        """Return render/training products requested by this recipe."""
        return frozenset(
            {
                RGB,
                SCREEN_SPACE_DENSIFICATION_SIGNALS,
                GAUSSIAN_IMPORTANCE_SCORE,
            }
        )

    def traits(self) -> tuple[TraitKey, ...]:
        """Return backend/runtime traits requested by this recipe."""
        return ()

    def build_method(self) -> GaussianFastGS:
        """Build the current executable FastGS densification method."""
        return GaussianFastGS(
            refine_every=self.refine_every,
            start_iter=self.start_iter,
            stop_iter=self.stop_iter,
            loss_thresh=self.loss_thresh,
            grad_threshold=self.grad_threshold,
            grad_abs_threshold=self.grad_abs_threshold,
            dense_fraction=self.dense_fraction,
            prune_opacity_threshold=self.prune_opacity_threshold,
            opacity_reset_every=self.opacity_reset_every,
            probe_view_count=self.probe_view_count,
            importance_threshold=self.importance_threshold,
        )
