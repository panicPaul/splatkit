"""Signal collectors for densification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ember_core.core.contracts import GaussianScene, SparseVoxelScene
from ember_core.densification.contracts import (
    BaseDensificationComponent,
    DensificationContext,
    DensificationRenderRequirements,
    DensificationSignals,
)


@dataclass
class ImagePlaneGradientCollector(BaseDensificationComponent):
    """Collect image-plane gradient norms from gsplat outputs."""

    use_absolute_gradients: bool = False

    def get_render_requirements(self) -> DensificationRenderRequirements:
        return DensificationRenderRequirements(
            return_2d_projections=True,
            backend_options={
                "packed": False,
                "absgrad": self.use_absolute_gradients,
            },
        )

    def pre_backward(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        del signals
        projected_means = getattr(context.render_output, "projected_means", None)
        if isinstance(projected_means, Tensor) and projected_means.requires_grad:
            projected_means.retain_grad()

    def post_backward(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        projected_means = getattr(context.render_output, "projected_means", None)
        if not isinstance(projected_means, Tensor):
            return
        gradients = None
        if self.use_absolute_gradients and hasattr(projected_means, "absgrad"):
            gradients = projected_means.absgrad
        if gradients is None:
            gradients = projected_means.grad
        if gradients is None:
            return
        visible = gradients.abs().sum(dim=-1) > 0
        grad_norm_sum = signals.local.setdefault(
            "image_plane_grad_norm_sum",
            torch.zeros(
                gradients.shape[1],
                dtype=gradients.dtype,
                device=gradients.device,
            ),
        )
        visible_count = signals.local.setdefault(
            "image_plane_visible_count",
            torch.zeros(
                gradients.shape[1],
                dtype=gradients.dtype,
                device=gradients.device,
            ),
        )
        grad_norm_sum += gradients.norm(dim=-1).sum(dim=0)
        visible_count += visible.to(gradients.dtype).sum(dim=0)


@dataclass
class PositionGradientCollector(BaseDensificationComponent):
    """Collect world-space position gradient norms."""

    key: str = "position_grad_norm_sum"

    def post_backward(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        scene = context.state.model.scene
        if not isinstance(scene, GaussianScene):
            return
        gradients = scene.center_position.grad
        if gradients is None:
            return
        grad_norm_sum = signals.local.setdefault(
            self.key,
            torch.zeros(
                gradients.shape[0],
                dtype=gradients.dtype,
                device=gradients.device,
            ),
        )
        count = signals.local.setdefault(
            f"{self.key}_count",
            torch.zeros(
                gradients.shape[0],
                dtype=gradients.dtype,
                device=gradients.device,
            ),
        )
        grad_norm_sum += gradients.norm(dim=-1)
        count += (gradients.abs().sum(dim=-1) > 0).to(gradients.dtype)


@dataclass
class SparseVoxelGradientCollector(BaseDensificationComponent):
    """Collect sparse-voxel subdivision priorities from grid gradients."""

    def post_backward(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        scene = context.state.model.scene
        if not isinstance(scene, SparseVoxelScene):
            return
        gradients = scene.geo_grid_pts.grad
        if gradients is None:
            return
        voxel_priority = gradients[scene.vox_key].abs().mean(dim=(1, 2))
        priority_sum = signals.local.setdefault(
            "sparse_voxel_priority_sum",
            torch.zeros_like(voxel_priority),
        )
        priority_count = signals.local.setdefault(
            "sparse_voxel_priority_count",
            torch.zeros_like(voxel_priority),
        )
        priority_sum += voxel_priority
        priority_count += torch.ones_like(voxel_priority)


__all__ = [
    "ImagePlaneGradientCollector",
    "PositionGradientCollector",
    "SparseVoxelGradientCollector",
]
