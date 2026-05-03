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
    morton_codes as _morton_codes,
    morton_order as _morton_order,
)
from ember_native_faster_gs.faster_gs.training import (
    update_mip_splatting_3d_filter,
)
from jaxtyping import Float, Int
from torch import Tensor


def morton_order(
    center_position: Float[Tensor, " num_splats 3"],
) -> Int[Tensor, " num_splats"]:
    """Return indices that sort Gaussian centers by native Morton code."""
    return _morton_order(center_position)


def morton_codes(
    center_position: Float[Tensor, " num_splats 3"],
) -> Int[Tensor, " num_splats"]:
    """Return native Morton codes for Gaussian centers."""
    return _morton_codes(center_position)


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


@dataclass(frozen=True)
class _MipSplattingCameraSpec:
    world_to_camera_matrix: Tensor
    image_width: int
    image_height: int
    focal_length_x: float
    focal_length_y: float
    principal_point_x: float
    principal_point_y: float
    near_plane: float


@dataclass
class GaussianMipSplatting3DFilter(BaseDensificationMethod):
    """Mip-Splatting 3D filter for FasterGS training.

    This computes a per-Gaussian world-space minimum filter radius across
    training cameras and applies it as an optimized log-scale clamp. The
    screen-space Mip-Splatting filter is configured separately by the renderer.
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
    _mip_splatting_3d_filter: Tensor | None = field(
        default=None, init=False, repr=False
    )
    _scene_data_ptr: int | None = field(default=None, init=False, repr=False)
    _distance_to_filter_scale: float | None = field(
        default=None, init=False, repr=False
    )
    _camera_specs: tuple[_MipSplattingCameraSpec, ...] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.recompute_schedule, dict):
            self.recompute_schedule = Schedule(**self.recompute_schedule)

    def get_render_requirements(
        self,
        state: object,
    ) -> DensificationRenderRequirements:
        """The screen-space Mip-Splatting filter is configured separately."""
        del state
        return DensificationRenderRequirements()

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
                "GaussianMipSplatting3DFilter requires GaussianFamilyOps."
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
            self._mip_splatting_3d_filter is None
            or self._scene_data_ptr != scene_data_ptr
            or self.recompute_schedule.includes(step)
        )
        if should_recompute:
            self._mip_splatting_3d_filter = self._compute_filter(
                context, scene
            )
            self._scene_data_ptr = scene_data_ptr
        if self._mip_splatting_3d_filter is not None:
            self._clamp_log_scales(scene, self._mip_splatting_3d_filter)

    def _compute_filter(
        self,
        context: DensificationContext | DensificationLifecycleContext,
        scene: GaussianScene,
    ) -> Tensor:
        camera_specs = self._camera_specs_for_context(context, scene)
        if not camera_specs:
            raise ValueError("3D filter computation requires training views.")
        distance_to_filter_scale = self._distance_to_filter_scale
        if distance_to_filter_scale is None:
            max_focal_length = max(
                max(spec.focal_length_x, spec.focal_length_y)
                for spec in camera_specs
            )
            distance_to_filter_scale = (
                math.sqrt(self.filter_variance) / max_focal_length
            )
            self._distance_to_filter_scale = distance_to_filter_scale

        positions = scene.center_position
        mip_splatting_3d_filter = torch.full(
            (positions.shape[0], 1),
            fill_value=torch.finfo(torch.float32).max,
            device=positions.device,
            dtype=torch.float32,
        )
        visibility_mask = torch.zeros_like(
            mip_splatting_3d_filter, dtype=torch.bool
        )
        for camera_spec in camera_specs:
            update_mip_splatting_3d_filter(
                positions,
                camera_spec.world_to_camera_matrix,
                mip_splatting_3d_filter,
                visibility_mask,
                image_width=camera_spec.image_width,
                image_height=camera_spec.image_height,
                focal_length_x=camera_spec.focal_length_x,
                focal_length_y=camera_spec.focal_length_y,
                principal_point_x=camera_spec.principal_point_x,
                principal_point_y=camera_spec.principal_point_y,
                near_plane=camera_spec.near_plane,
                clipping_tolerance=self.clipping_tolerance,
                distance_to_filter_scale=distance_to_filter_scale,
            )
        fallback = mip_splatting_3d_filter[visibility_mask].max()
        mip_splatting_3d_filter = torch.where(
            visibility_mask,
            mip_splatting_3d_filter,
            fallback,
        )
        return mip_splatting_3d_filter.log()

    def _camera_specs_for_context(
        self,
        context: DensificationContext | DensificationLifecycleContext,
        scene: GaussianScene,
    ) -> tuple[_MipSplattingCameraSpec, ...]:
        if self._camera_specs is not None:
            return self._camera_specs
        assert context.runtime is not None
        if hasattr(context.runtime, "all_cameras"):
            cameras = context.runtime.all_cameras()
        else:
            cameras = tuple(view.camera for view in context.runtime.all_views())
        if not cameras:
            self._camera_specs = ()
            return self._camera_specs

        near_plane = (
            self.near_plane
            if self.near_plane is not None
            else torch.finfo(scene.center_position.dtype).eps
        )
        specs: list[_MipSplattingCameraSpec] = []
        for camera in cameras:
            specs.append(
                self._camera_spec(
                    camera,
                    device=scene.center_position.device,
                    dtype=scene.center_position.dtype,
                    near_plane=float(near_plane),
                )
            )
        self._camera_specs = tuple(specs)
        return self._camera_specs

    @staticmethod
    def _camera_spec(
        camera: CameraState,
        *,
        device: torch.device,
        dtype: torch.dtype,
        near_plane: float,
    ) -> _MipSplattingCameraSpec:
        intrinsics = camera.get_intrinsics()[0]
        camera_to_world_matrix = camera.cam_to_world[0].to(dtype=dtype)
        world_to_camera_matrix = torch.linalg.inv(camera_to_world_matrix).to(
            device=device,
            non_blocking=True,
        )
        return _MipSplattingCameraSpec(
            world_to_camera_matrix=world_to_camera_matrix.contiguous(),
            image_width=int(camera.width[0].item()),
            image_height=int(camera.height[0].item()),
            focal_length_x=float(intrinsics[0, 0].item()),
            focal_length_y=float(intrinsics[1, 1].item()),
            principal_point_x=float(intrinsics[0, 2].item()),
            principal_point_y=float(intrinsics[1, 2].item()),
            near_plane=near_plane,
        )

    def _clamp_log_scales(
        self,
        scene: GaussianScene,
        mip_splatting_3d_filter: Tensor,
    ) -> None:
        with torch.no_grad():
            scene.log_scales.clamp_min_(mip_splatting_3d_filter)


__all__ = [
    "GaussianMipSplatting3DFilter",
    "GaussianMortonOrdering",
    "active_sh_bases_for_step",
    "fastergs_training_backend_options",
    "morton_codes",
    "morton_order",
]
