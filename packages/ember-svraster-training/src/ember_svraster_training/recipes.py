"""Reusable SVRaster training recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ember_core.training import (
    CallableSpec,
    OptimizationConfig,
    ParameterGroupConfig,
    ParameterTargetSpec,
)


@dataclass(frozen=True)
class SVRasterOptimizationRecipe:
    """Canonical SVRaster optimizer group defaults."""

    optimizer: str = "ember_svraster_training.SVRasterSparseAdam"
    geo_lr: float = 0.025
    sh0_lr: float = 0.010
    shs_lr: float = 0.00025
    betas: tuple[float, float] = (0.1, 0.99)
    eps: float = 1e-15
    sparse: bool = False
    biased: bool = False
    lr_decay_checkpoints: tuple[int, ...] = (19000,)
    lr_decay_multiplier: float = 0.1


def _scene_target(name: str) -> ParameterTargetSpec:
    return ParameterTargetSpec(scope="scene", name=name)


def _resolve_recipe(
    recipe: SVRasterOptimizationRecipe | dict[str, Any] | None,
) -> SVRasterOptimizationRecipe:
    if recipe is None:
        return SVRasterOptimizationRecipe()
    if isinstance(recipe, SVRasterOptimizationRecipe):
        return recipe
    return SVRasterOptimizationRecipe(**recipe)


def svraster_parameter_groups(
    recipe: SVRasterOptimizationRecipe | dict[str, Any] | None = None,
) -> list[ParameterGroupConfig]:
    """Build canonical SVRaster scene parameter groups."""
    resolved_recipe = _resolve_recipe(recipe)
    optimizer_kwargs = {
        "betas": resolved_recipe.betas,
        "eps": resolved_recipe.eps,
        "sparse": resolved_recipe.sparse,
        "biased": resolved_recipe.biased,
    }
    scheduler = None
    if resolved_recipe.lr_decay_checkpoints:
        scheduler = CallableSpec(
            target="torch.optim.lr_scheduler.MultiStepLR",
            kwargs={
                "milestones": list(resolved_recipe.lr_decay_checkpoints),
                "gamma": resolved_recipe.lr_decay_multiplier,
            },
        )
    return [
        ParameterGroupConfig(
            target=_scene_target("geo_grid_pts"),
            optimizer=resolved_recipe.optimizer,
            lr=resolved_recipe.geo_lr,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
        ),
        ParameterGroupConfig(
            target=_scene_target("sh0"),
            optimizer=resolved_recipe.optimizer,
            lr=resolved_recipe.sh0_lr,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
        ),
        ParameterGroupConfig(
            target=_scene_target("shs"),
            optimizer=resolved_recipe.optimizer,
            lr=resolved_recipe.shs_lr,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
        ),
    ]


def svraster_optimization_config(
    recipe: SVRasterOptimizationRecipe | dict[str, Any] | None = None,
) -> OptimizationConfig:
    """Build canonical SVRaster optimization config."""
    return OptimizationConfig(
        parameter_groups=svraster_parameter_groups(recipe)
    )


__all__ = [
    "SVRasterOptimizationRecipe",
    "svraster_optimization_config",
    "svraster_parameter_groups",
]
