"""SVRaster training regularization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from ember_core.core.contracts import SparseVoxelScene
from ember_core.training.protocols import LossResult, TrainState
from ember_native_svraster.training.runtime import (
    apply_total_variation_density_grad as native_apply_total_variation_density_grad,
)
from jaxtyping import Float, Int
from torch import Tensor


def apply_total_variation_density_grad(
    *,
    grid_points: Float[Tensor, " num_grid_points channels"],
    voxel_keys: Int[Tensor, " num_voxels 8"],
    weight: float,
    voxel_size_inverse: Float[Tensor, " num_voxels 1"],
    grid_points_grad: Float[Tensor, " num_grid_points channels"],
    no_tv_s: bool = True,
    tv_sparse: bool = False,
) -> None:
    """Accumulate native SVRaster TV-density gradients in-place."""
    native_apply_total_variation_density_grad(
        grid_points=grid_points,
        voxel_keys=voxel_keys,
        weight=weight,
        voxel_size_inverse=voxel_size_inverse,
        grid_points_grad=grid_points_grad,
        no_tv_s=no_tv_s,
        tv_sparse=tv_sparse,
    )


@dataclass
class SVRasterTVDensityHook:
    """Training hook applying SVRaster TV-density gradients after backward."""

    weight: float = 1e-10
    start_step: int = 0
    end_step: int = 10000
    no_tv_s: bool = True
    tv_sparse: bool = False

    def post_backward(
        self,
        state: TrainState,
        batch: Any,
        render_output: Any,
        loss_result: LossResult,
    ) -> None:
        """Add TV-density gradients to ``geo_grid_pts.grad`` when scheduled."""
        del batch, render_output, loss_result
        if state.step < self.start_step or state.step > self.end_step:
            return
        scene = state.model.scene
        if not isinstance(scene, SparseVoxelScene):
            return
        if scene.geo_grid_pts.grad is None:
            scene.geo_grid_pts.grad = torch.zeros_like(scene.geo_grid_pts)
        apply_total_variation_density_grad(
            grid_points=scene.geo_grid_pts,
            voxel_keys=scene.vox_key,
            weight=self.weight,
            voxel_size_inverse=1.0 / scene.vox_size,
            grid_points_grad=scene.geo_grid_pts.grad,
            no_tv_s=self.no_tv_s,
            tv_sparse=self.tv_sparse,
        )


__all__ = [
    "SVRasterTVDensityHook",
    "apply_total_variation_density_grad",
]
