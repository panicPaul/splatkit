"""Reusable Gaussian splatting training recipes."""

from __future__ import annotations

from typing import Any

from ember_core.training import (
    CallableSpec,
    OptimizationConfig,
    ParameterGroupConfig,
    ParameterTargetSpec,
    TensorSliceSpec,
    TensorViewSpec,
)
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Gaussian3DGSOptimizationRecipe(BaseModel):
    """Canonical Gaussian 3DGS optimizer group defaults."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    optimizer: str = "ember_splatting_training.FusedAdam"
    center_position_lr_init: float = Field(
        default=1.6e-4,
        gt=0.0,
        validation_alias=AliasChoices(
            "center_position_lr_init",
            "means_lr_init",
        ),
    )
    center_position_lr_final: float = Field(
        default=1.6e-6,
        gt=0.0,
        validation_alias=AliasChoices(
            "center_position_lr_final",
            "means_lr_final",
        ),
    )
    center_position_lr_max_steps: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices(
            "center_position_lr_max_steps",
            "means_lr_max_steps",
        ),
    )
    center_position_lr_step_offset: int = Field(default=0, ge=0)
    sh_dc_lr: float = Field(default=2.5e-3, gt=0.0)
    sh_rest_lr: float = Field(default=1.25e-4, gt=0.0)
    logit_opacity_lr: float = Field(
        default=2.5e-2,
        gt=0.0,
        validation_alias=AliasChoices("logit_opacity_lr", "opacity_lr"),
    )
    log_scales_lr: float = Field(
        default=5e-3,
        gt=0.0,
        validation_alias=AliasChoices("log_scales_lr", "scale_lr"),
    )
    quaternion_orientation_lr: float = Field(
        default=1e-3,
        gt=0.0,
        validation_alias=AliasChoices(
            "quaternion_orientation_lr",
            "rotation_lr",
        ),
    )
    fused_adam_eps: float = Field(default=1e-15, gt=0.0)


def scene_parameter_target(
    name: str,
    *,
    view: TensorViewSpec | None = None,
) -> ParameterTargetSpec:
    """Build a scene parameter target spec."""
    return ParameterTargetSpec(scope="scene", name=name, view=view)


def sh_feature_view(*, start: int, stop: int | None = None) -> TensorViewSpec:
    """Select a contiguous SH feature coefficient range."""
    return TensorViewSpec(
        slices=(TensorSliceSpec(axis=1, start=start, stop=stop),)
    )


def gaussian_3dgs_parameter_groups(
    recipe: Gaussian3DGSOptimizationRecipe | dict[str, Any],
    *,
    position_lr_scale: float = 1.0,
    max_steps: int,
) -> list[ParameterGroupConfig]:
    """Build canonical 3DGS Gaussian parameter optimizer groups."""
    recipe = Gaussian3DGSOptimizationRecipe.model_validate(recipe)
    center_position_lr_max_steps = (
        recipe.center_position_lr_max_steps
        if recipe.center_position_lr_max_steps is not None
        else max_steps
    )
    optimizer_kwargs = {"eps": recipe.fused_adam_eps}
    return [
        ParameterGroupConfig(
            target=scene_parameter_target("center_position"),
            optimizer=recipe.optimizer,
            lr=recipe.center_position_lr_init * position_lr_scale,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=CallableSpec(
                target="ember_core.training.exponential_decay_to",
                kwargs={
                    "final_lr": (
                        recipe.center_position_lr_final * position_lr_scale
                    ),
                    "max_steps": center_position_lr_max_steps,
                    "step_offset": recipe.center_position_lr_step_offset,
                },
            ),
        ),
        ParameterGroupConfig(
            target=scene_parameter_target(
                "feature",
                view=sh_feature_view(start=0, stop=1),
            ),
            optimizer=recipe.optimizer,
            lr=recipe.sh_dc_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
        ParameterGroupConfig(
            target=scene_parameter_target(
                "feature",
                view=sh_feature_view(start=1),
            ),
            optimizer=recipe.optimizer,
            lr=recipe.sh_rest_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
        ParameterGroupConfig(
            target=scene_parameter_target("logit_opacity"),
            optimizer=recipe.optimizer,
            lr=recipe.logit_opacity_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
        ParameterGroupConfig(
            target=scene_parameter_target("log_scales"),
            optimizer=recipe.optimizer,
            lr=recipe.log_scales_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
        ParameterGroupConfig(
            target=scene_parameter_target("quaternion_orientation"),
            optimizer=recipe.optimizer,
            lr=recipe.quaternion_orientation_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
    ]


def gaussian_3dgs_optimization_config(
    recipe: Gaussian3DGSOptimizationRecipe | dict[str, Any],
    *,
    position_lr_scale: float = 1.0,
    max_steps: int,
) -> OptimizationConfig:
    """Build an OptimizationConfig from canonical Gaussian 3DGS groups."""
    return OptimizationConfig(
        parameter_groups=gaussian_3dgs_parameter_groups(
            recipe,
            position_lr_scale=position_lr_scale,
            max_steps=max_steps,
        )
    )


__all__ = [
    "Gaussian3DGSOptimizationRecipe",
    "gaussian_3dgs_optimization_config",
    "gaussian_3dgs_parameter_groups",
    "scene_parameter_target",
    "sh_feature_view",
]
