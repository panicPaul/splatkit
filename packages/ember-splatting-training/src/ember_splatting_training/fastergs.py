"""Reusable FasterGS paper training utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from ember_core.core.contracts import CameraState, GaussianScene
from ember_core.densification.contracts import (
    BaseDensificationMethod,
    DensificationContext,
    DensificationLifecycleContext,
    DensificationRenderRequirements,
    Schedule,
)
from ember_core.densification.families import GaussianFamilyOps
from ember_core.training.protocols import TrainState
from ember_native_faster_gs.faster_gs.training import (
    morton_order as _morton_order,
)
from ember_native_faster_gs.faster_gs.training import (
    update_3d_filter,
)
from jaxtyping import Float, Int
from torch import Tensor


def morton_order(
    center_position: Float[Tensor, " num_splats 3"],
) -> Int[Tensor, " num_splats"]:
    """Return indices that sort Gaussian centers by native Morton code."""
    return _morton_order(center_position)


def active_sh_bases_for_step(
    step: int,
    *,
    max_degree: int = 3,
    start_step: int = 1000,
    step_interval: int = 1000,
) -> int:
    """Return active SH basis count for the FasterGS training schedule."""
    if step < start_step:
        active_degree = 0
    else:
        active_degree = 1 + (step - start_step) // step_interval
    active_degree = min(max(0, active_degree), max_degree)
    return (active_degree + 1) ** 2


def fastergs_training_backend_options(
    state: TrainState,
    *,
    max_sh_degree: int = 3,
    sh_start_step: int = 1000,
    sh_step_interval: int = 1000,
    clamp_output: bool = False,
) -> dict[str, int | bool]:
    """Return per-step FasterGS render options used only during training."""
    return {
        "active_sh_bases": active_sh_bases_for_step(
            state.step,
            max_degree=max_sh_degree,
            start_step=sh_start_step,
            step_interval=sh_step_interval,
        ),
        "clamp_output": clamp_output,
    }


@dataclass
class GaussianMortonOrdering(BaseDensificationMethod):
    """Scheduled Morton ordering for Gaussian scene tensors and optimizer state."""

    schedule: Schedule = field(
        default_factory=lambda: Schedule(
            end_iteration=15_000,
            frequency=5_000,
        )
    )
    expected_scene_families: tuple[str, ...] = ("gaussian",)
    _family_ops: GaussianFamilyOps | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.schedule, dict):
            self.schedule = Schedule(**self.schedule)

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Bind Gaussian topology operations."""
        del state, optimizers
        if not isinstance(family_ops, GaussianFamilyOps):
            raise TypeError(
                "GaussianMortonOrdering requires GaussianFamilyOps."
            )
        self._family_ops = family_ops

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Apply scheduled Morton ordering after optimizer updates."""
        if self._family_ops is None or not self.schedule.includes(
            context.step + 1
        ):
            return
        self._apply_ordering()

    def before_training(self, context: DensificationLifecycleContext) -> None:
        """Apply the iteration-0 Morton ordering before the first render."""
        del context
        if self._family_ops is None or not self.schedule.includes(0):
            return
        self._apply_ordering()

    def _apply_ordering(self) -> None:
        assert self._family_ops is not None
        scene = self._family_ops.scene
        if scene.center_position.device.type != "cuda":
            raise ValueError("GaussianMortonOrdering requires CUDA tensors.")
        self._family_ops.reorder(morton_order(scene.center_position))


@dataclass
class GaussianMipSplattingAntialiasing(BaseDensificationMethod):
    """Full FasterGS/Mip-Splatting AA utility.

    This combines the rasterizer-side proper antialiasing flag with the
    optimized 3D filter from FasterGS. The filter is recomputed when topology
    edits replace scene tensors and then periodically after densification.
    """

    filter_variance: float = 0.2
    clipping_tolerance: float = 0.15
    recompute_schedule: Schedule = field(
        default_factory=lambda: Schedule(
            start_iteration=14_900,
            end_iteration=29_900,
            frequency=100,
        )
    )
    near_plane: float | None = None
    expected_scene_families: tuple[str, ...] = ("gaussian",)
    _family_ops: GaussianFamilyOps | None = field(
        default=None, init=False, repr=False
    )
    _filter_3d: Tensor | None = field(default=None, init=False, repr=False)
    _scene_data_ptr: int | None = field(default=None, init=False, repr=False)
    _distance2filter: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.recompute_schedule, dict):
            self.recompute_schedule = Schedule(**self.recompute_schedule)

    def get_render_requirements(
        self,
        state: object,
    ) -> DensificationRenderRequirements:
        """Request the rasterizer-side FasterGS antialiasing path."""
        del state
        return DensificationRenderRequirements(
            backend_options={"proper_antialiasing": True}
        )

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Bind Gaussian topology operations."""
        del state, optimizers
        if not isinstance(family_ops, GaussianFamilyOps):
            raise TypeError(
                "GaussianMipSplattingAntialiasing requires GaussianFamilyOps."
            )
        self._family_ops = family_ops

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Refresh the 3D filter and clamp log-scales after optimizer steps."""
        self._refresh_filter(context, step=context.step + 1)

    def before_training(self, context: DensificationLifecycleContext) -> None:
        """Compute the optimized 3D filter before the first training render."""
        self._refresh_filter(context, step=0)

    def _refresh_filter(
        self,
        context: DensificationContext | DensificationLifecycleContext,
        *,
        step: int,
    ) -> None:
        if self._family_ops is None or context.runtime is None:
            return
        scene = self._family_ops.scene
        if not isinstance(scene, GaussianScene):
            return
        scene_data_ptr = scene.center_position.data_ptr()
        should_recompute = (
            self._filter_3d is None
            or self._scene_data_ptr != scene_data_ptr
            or self.recompute_schedule.includes(step)
        )
        if should_recompute:
            self._filter_3d = self._compute_filter(context, scene)
            self._scene_data_ptr = scene_data_ptr
        if self._filter_3d is not None:
            self._clamp_log_scales(scene, self._filter_3d)

    def _compute_filter(
        self,
        context: DensificationContext | DensificationLifecycleContext,
        scene: GaussianScene,
    ) -> Tensor:
        views = context.runtime.all_views()
        if not views:
            raise ValueError("3D filter computation requires training views.")
        distance2filter = self._distance2filter
        if distance2filter is None:
            max_focal = 0.0
            for view in views:
                intrinsics = view.camera.get_intrinsics()[0]
                max_focal = max(
                    max_focal,
                    float(intrinsics[0, 0].item()),
                    float(intrinsics[1, 1].item()),
                )
            distance2filter = math.sqrt(self.filter_variance) / max_focal
            self._distance2filter = distance2filter

        positions = scene.center_position
        filter_3d = torch.full(
            (positions.shape[0], 1),
            fill_value=torch.finfo(torch.float32).max,
            device=positions.device,
            dtype=torch.float32,
        )
        visibility_mask = torch.zeros_like(filter_3d, dtype=torch.bool)
        for view in views:
            self._update_filter_for_camera(
                positions,
                view.camera,
                filter_3d,
                visibility_mask,
                distance2filter,
            )
        if torch.any(visibility_mask):
            fallback = filter_3d[visibility_mask].max()
            filter_3d = torch.where(visibility_mask, filter_3d, fallback)
        else:
            filter_3d.fill_(0.0)
        return filter_3d.log()

    def _update_filter_for_camera(
        self,
        positions: Tensor,
        camera: CameraState,
        filter_3d: Tensor,
        visibility_mask: Tensor,
        distance2filter: float,
    ) -> None:
        intrinsics = camera.get_intrinsics()[0]
        cam_to_world = camera.cam_to_world[0]
        near_plane = (
            self.near_plane
            if self.near_plane is not None
            else torch.finfo(positions.dtype).eps
        )
        update_3d_filter(
            positions,
            torch.linalg.inv(cam_to_world),
            filter_3d,
            visibility_mask,
            width=int(camera.width[0].item()),
            height=int(camera.height[0].item()),
            focal_x=float(intrinsics[0, 0].item()),
            focal_y=float(intrinsics[1, 1].item()),
            center_x=float(intrinsics[0, 2].item()),
            center_y=float(intrinsics[1, 2].item()),
            near_plane=float(near_plane),
            clipping_tolerance=self.clipping_tolerance,
            distance2filter=distance2filter,
        )

    def _clamp_log_scales(self, scene: GaussianScene, filter_3d: Tensor) -> None:
        assert self._family_ops is not None
        log_scales = torch.maximum(scene.log_scales, filter_3d)
        if torch.equal(log_scales, scene.log_scales):
            return
        self._family_ops.replace_fields({"log_scales": log_scales})


__all__ = [
    "GaussianMipSplattingAntialiasing",
    "GaussianMortonOrdering",
    "active_sh_bases_for_step",
    "fastergs_training_backend_options",
    "morton_order",
]
