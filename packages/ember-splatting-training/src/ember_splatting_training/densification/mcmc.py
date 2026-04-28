"""Gaussian MCMC densification backed by vendored CUDA utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from ember_core.core.contracts import GaussianScene
from ember_core.densification.contracts import (
    BaseDensificationMethod,
    DensificationContext,
    Schedule,
)
from ember_core.densification.families import GaussianFamilyOps
from ember_native_faster_gs.faster_gs.training import (
    add_noise as _add_noise,
)
from ember_native_faster_gs.faster_gs.training import (
    relocation_adjustment as _relocation_adjustment,
)
from jaxtyping import Float, Int
from torch import Tensor


def relocation_adjustment(
    old_opacities: Float[Tensor, " num_splats"],
    old_scales: Float[Tensor, "num_splats spatial_dims"],
    n_samples_per_primitive: Int[Tensor, " num_splats"],
) -> tuple[Tensor, Tensor]:
    """Adjust sampled Gaussian opacity/scale according to MCMC counts."""
    return _relocation_adjustment(
        old_opacities,
        old_scales,
        n_samples_per_primitive,
    )


def add_noise(
    raw_scales: Tensor,
    raw_rotations: Tensor,
    raw_opacities: Tensor,
    means: Tensor,
    current_lr: float,
) -> None:
    """Inject CUDA noise into Gaussian means using the vendored helper."""
    _add_noise(
        raw_scales,
        raw_rotations,
        raw_opacities,
        means,
        current_lr,
    )


def _find_optimizer_binding(
    context: DensificationContext,
    *,
    scope: str,
    name: str,
) -> Any | None:
    for binding in context.optimizers:
        matches_target = getattr(binding, "matches_target", None)
        if callable(matches_target) and matches_target(scope, name):
            return binding
    return None


@dataclass
class GaussianMCMC(BaseDensificationMethod):
    """Backend-agnostic Gaussian MCMC densification."""

    schedule: Schedule = field(default_factory=lambda: Schedule(frequency=100))
    min_opacity: float = 0.005
    cap_growth_factor: float = 1.05
    cap_max: int = 1_000_000
    inject_position_noise: bool = True
    noise_lr_scale: float = 5e5
    expected_scene_families: tuple[str, ...] = ("gaussian",)
    _family_ops: GaussianFamilyOps | None = field(
        default=None, init=False, repr=False
    )

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Bind the method to a Gaussian family ops implementation."""
        del state, optimizers
        if not isinstance(family_ops, GaussianFamilyOps):
            raise TypeError(
                "GaussianMCMC requires GaussianFamilyOps for training."
            )
        self._family_ops = family_ops

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Inject noise and schedule MCMC relocation/growth updates."""
        scene = context.state.model.scene
        if not isinstance(scene, GaussianScene) or self._family_ops is None:
            return
        if self.inject_position_noise:
            self._inject_noise(context, scene)
        if not self.schedule.includes(context.step):
            return
        self._relocate_dead(scene)
        self._append_new(scene)

    def _inject_noise(
        self,
        context: DensificationContext,
        scene: GaussianScene,
    ) -> None:
        means_binding = _find_optimizer_binding(
            context,
            scope="scene",
            name="center_position",
        )
        if means_binding is None:
            return
        current_lr_getter = getattr(means_binding, "current_lr", None)
        current_lr = (
            float(current_lr_getter())
            if callable(current_lr_getter)
            else float(means_binding.optimizer.param_groups[0]["lr"])
        )
        if current_lr <= 0.0:
            return
        add_noise(
            scene.log_scales,
            scene.quaternion_orientation,
            scene.logit_opacity[:, None],
            scene.center_position,
            self.noise_lr_scale * current_lr,
        )

    def _dead_mask(self, scene: GaussianScene) -> Tensor:
        dead_mask = torch.sigmoid(scene.logit_opacity) <= self.min_opacity
        dead_mask |= scene.quaternion_orientation.square().sum(dim=1) < 1e-8
        return dead_mask

    def _adjusted_samples(
        self,
        opacities: Tensor,
        log_scales: Tensor,
        sampled_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _, inverse, counts_per_unique = sampled_indices.unique(
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        counts = counts_per_unique[inverse] + 1
        adjusted_opacities, adjusted_scales = relocation_adjustment(
            opacities[sampled_indices],
            torch.exp(log_scales[sampled_indices]),
            counts,
        )
        adjusted_opacities = adjusted_opacities.clamp(
            self.min_opacity,
            1.0 - torch.finfo(torch.float32).eps,
        ).logit()
        return adjusted_opacities, adjusted_scales.log()

    def _relocate_dead(self, scene: GaussianScene) -> None:
        dead_mask = self._dead_mask(scene)
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return
        alive_indices = torch.where(~dead_mask)[0]
        if int(alive_indices.numel()) == 0:
            return
        dead_indices = torch.where(dead_mask)[0]
        opacities = torch.sigmoid(scene.logit_opacity)
        sampled_indices = alive_indices[
            torch.multinomial(
                opacities[alive_indices], n_dead, replacement=True
            )
        ]
        adjusted_opacities, adjusted_scales = self._adjusted_samples(
            opacities,
            scene.log_scales,
            sampled_indices,
        )
        overrides = {
            "logit_opacity": adjusted_opacities,
            "log_scales": adjusted_scales,
        }
        self._family_ops.copy_from_indices(
            sampled_indices,
            sampled_indices,
            field_overrides=overrides,
        )
        self._family_ops.copy_from_indices(
            dead_indices,
            sampled_indices,
            field_overrides=overrides,
        )
        self._family_ops.reset_optimizer_state(sampled_indices)

    def _append_new(self, scene: GaussianScene) -> None:
        current_n_points = int(scene.center_position.shape[0])
        target_n = min(
            self.cap_max,
            math.floor(self.cap_growth_factor * current_n_points),
        )
        n_added = max(0, target_n - current_n_points)
        if n_added == 0:
            return
        opacities = torch.sigmoid(scene.logit_opacity)
        sampled_indices = torch.multinomial(
            opacities, n_added, replacement=True
        )
        adjusted_opacities, adjusted_scales = self._adjusted_samples(
            opacities,
            scene.log_scales,
            sampled_indices,
        )
        overrides = {
            "logit_opacity": adjusted_opacities,
            "log_scales": adjusted_scales,
        }
        self._family_ops.copy_from_indices(
            sampled_indices,
            sampled_indices,
            field_overrides=overrides,
        )
        self._family_ops.append_from_indices(
            sampled_indices,
            field_overrides=overrides,
        )
        self._family_ops.reset_optimizer_state(sampled_indices)


__all__ = [
    "GaussianMCMC",
    "add_noise",
    "relocation_adjustment",
]
