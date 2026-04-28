"""Composable densification passes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ember_core.core.contracts import GaussianScene, SparseVoxelScene
from ember_core.densification.contracts import (
    BaseDensificationComponent,
    DensificationContext,
    DensificationSignals,
    Schedule,
)
from ember_core.densification.families import FieldBehavior


def _average_signal(
    signals: DensificationSignals,
    sum_key: str,
    count_key: str,
) -> torch.Tensor | None:
    total = signals.local.get(sum_key)
    count = signals.local.get(count_key)
    if not isinstance(total, torch.Tensor) or not isinstance(count, torch.Tensor):
        return None
    return total / count.clamp_min(1.0)


@dataclass
class GaussianClonePass(BaseDensificationComponent):
    """Clone small high-gradient Gaussians."""

    schedule: Schedule
    grad_threshold: float
    relative_size_threshold: float
    field_behaviors: dict[str, FieldBehavior] = field(default_factory=dict)
    signal_sum_key: str = "image_plane_grad_norm_sum"
    signal_count_key: str = "image_plane_visible_count"
    family_ops: Any | None = field(default=None, init=False, repr=False)

    def bind(self, state: Any, optimizers: list[Any], family_ops: Any) -> None:
        del state, optimizers
        self.family_ops = family_ops

    def post_optimizer_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        if not self.schedule.includes(context.step):
            return
        scene = context.state.model.scene
        if not isinstance(scene, GaussianScene) or self.family_ops is None:
            return
        avg_grad = _average_signal(
            signals,
            self.signal_sum_key,
            self.signal_count_key,
        )
        if avg_grad is None:
            return
        scales = torch.exp(scene.log_scales).max(dim=-1).values
        scene_extent = self.family_ops.scene_extent()
        mask = torch.logical_and(
            avg_grad >= self.grad_threshold,
            scales <= self.relative_size_threshold * scene_extent,
        )
        if torch.any(mask):
            self.family_ops.clone(mask, self.field_behaviors)


@dataclass
class GaussianSplitPass(BaseDensificationComponent):
    """Split large high-gradient Gaussians."""

    schedule: Schedule
    grad_threshold: float
    relative_size_threshold: float
    num_children: int = 2
    scale_shrink: float = 0.8
    field_behaviors: dict[str, FieldBehavior] = field(default_factory=dict)
    signal_sum_key: str = "image_plane_grad_norm_sum"
    signal_count_key: str = "image_plane_visible_count"
    family_ops: Any | None = field(default=None, init=False, repr=False)

    def bind(self, state: Any, optimizers: list[Any], family_ops: Any) -> None:
        del state, optimizers
        self.family_ops = family_ops

    def post_optimizer_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        if not self.schedule.includes(context.step):
            return
        scene = context.state.model.scene
        if not isinstance(scene, GaussianScene) or self.family_ops is None:
            return
        avg_grad = _average_signal(
            signals,
            self.signal_sum_key,
            self.signal_count_key,
        )
        if avg_grad is None:
            return
        scales = torch.exp(scene.log_scales).max(dim=-1).values
        scene_extent = self.family_ops.scene_extent()
        mask = torch.logical_and(
            avg_grad >= self.grad_threshold,
            scales > self.relative_size_threshold * scene_extent,
        )
        if torch.any(mask):
            self.family_ops.split(
                mask,
                num_children=self.num_children,
                field_behaviors=self.field_behaviors,
                scale_shrink=self.scale_shrink,
            )


@dataclass
class GaussianPruneOpacityPass(BaseDensificationComponent):
    """Prune low-opacity Gaussians."""

    schedule: Schedule
    opacity_threshold: float
    family_ops: Any | None = field(default=None, init=False, repr=False)

    def bind(self, state: Any, optimizers: list[Any], family_ops: Any) -> None:
        del state, optimizers
        self.family_ops = family_ops

    def post_optimizer_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        del signals
        if not self.schedule.includes(context.step):
            return
        scene = context.state.model.scene
        if not isinstance(scene, GaussianScene) or self.family_ops is None:
            return
        self.family_ops.prune(torch.sigmoid(scene.logit_opacity) >= self.opacity_threshold)


@dataclass
class GaussianResetOpacityPass(BaseDensificationComponent):
    """Cap Gaussian opacity on a schedule."""

    schedule: Schedule
    max_post_sigmoid_opacity: float
    family_ops: Any | None = field(default=None, init=False, repr=False)

    def bind(self, state: Any, optimizers: list[Any], family_ops: Any) -> None:
        del state, optimizers
        self.family_ops = family_ops

    def post_optimizer_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        del signals
        if self.schedule.includes(context.step) and self.family_ops is not None:
            self.family_ops.reset_opacity(self.max_post_sigmoid_opacity)


@dataclass
class GaussianOpacityDecayPass(BaseDensificationComponent):
    """Decay Gaussian opacity on a schedule."""

    schedule: Schedule
    gamma: float
    family_ops: Any | None = field(default=None, init=False, repr=False)

    def bind(self, state: Any, optimizers: list[Any], family_ops: Any) -> None:
        del state, optimizers
        self.family_ops = family_ops

    def post_optimizer_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        del signals
        if self.schedule.includes(context.step) and self.family_ops is not None:
            self.family_ops.decay_opacity(self.gamma)


@dataclass
class GaussianJitterPass(BaseDensificationComponent):
    """Apply small positional jitter to Gaussians."""

    schedule: Schedule
    sigma: float
    family_ops: Any | None = field(default=None, init=False, repr=False)

    def bind(self, state: Any, optimizers: list[Any], family_ops: Any) -> None:
        del state, optimizers
        self.family_ops = family_ops

    def post_optimizer_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        del signals
        if self.schedule.includes(context.step) and self.family_ops is not None:
            self.family_ops.jitter_positions(self.sigma)


@dataclass
class SparseVoxelSubdividePass(BaseDensificationComponent):
    """Subdivide sparse voxels with high accumulated priority."""

    schedule: Schedule
    priority_threshold: float
    family_ops: Any | None = field(default=None, init=False, repr=False)

    def bind(self, state: Any, optimizers: list[Any], family_ops: Any) -> None:
        del state, optimizers
        self.family_ops = family_ops

    def post_optimizer_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        if not self.schedule.includes(context.step) or self.family_ops is None:
            return
        scene = context.state.model.scene
        if not isinstance(scene, SparseVoxelScene):
            return
        avg_priority = _average_signal(
            signals,
            "sparse_voxel_priority_sum",
            "sparse_voxel_priority_count",
        )
        if avg_priority is None:
            return
        mask = torch.logical_and(
            avg_priority >= self.priority_threshold,
            scene.octlevel.squeeze(-1) < scene.max_num_levels,
        )
        if torch.any(mask):
            self.family_ops.subdivide(mask)


@dataclass
class SparseVoxelPrunePass(BaseDensificationComponent):
    """Prune sparse voxels with low density."""

    schedule: Schedule
    density_threshold: float
    family_ops: Any | None = field(default=None, init=False, repr=False)

    def bind(self, state: Any, optimizers: list[Any], family_ops: Any) -> None:
        del state, optimizers
        self.family_ops = family_ops

    def post_optimizer_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        del signals
        if not self.schedule.includes(context.step):
            return
        scene = context.state.model.scene
        if not isinstance(scene, SparseVoxelScene) or self.family_ops is None:
            return
        self.family_ops.prune(scene.voxel_geometries.mean(dim=-1) >= self.density_threshold)


__all__ = [
    "GaussianClonePass",
    "GaussianJitterPass",
    "GaussianOpacityDecayPass",
    "GaussianPruneOpacityPass",
    "GaussianResetOpacityPass",
    "GaussianSplitPass",
    "SparseVoxelPrunePass",
    "SparseVoxelSubdividePass",
]
