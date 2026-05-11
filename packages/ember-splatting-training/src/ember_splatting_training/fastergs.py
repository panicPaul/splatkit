"""Reusable FasterGS and FastGS paper training utilities."""

from __future__ import annotations

import math
import time
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
    GaussianFastGSSignalProvider,
    GaussianMetricAttribution,
    Schedule,
)
from ember_core.densification.families import GaussianFamilyOps
from ember_core.training.protocols import TrainState
from ember_native_faster_gs.faster_gs.training import (
    morton_codes as _morton_codes,
)
from ember_native_faster_gs.faster_gs.training import (
    morton_order as _morton_order,
)
from ember_native_faster_gs.faster_gs.training import (
    update_mip_splatting_3d_filter,
)
from jaxtyping import Float, Int
from torch import Tensor


def fastgs_normalize_score(score: Tensor) -> Tensor:
    """Normalize a score tensor without synchronizing CUDA to Python."""
    min_value = torch.amin(score)
    max_value = torch.amax(score)
    denom = max_value - min_value
    normalized = (score - min_value) / denom.clamp_min(1e-12)
    return torch.where(
        denom.abs() <= 1e-12, torch.zeros_like(score), normalized
    )


def fastgs_l1_metric_map(
    predicted: Tensor,
    target: Tensor,
    loss_thresh: float,
) -> Tensor:
    """Build FastGS' normalized L1 metric map from one rendered probe image."""
    l1_map = (predicted - target).abs().mean(dim=-1)
    normalized_l1 = fastgs_normalize_score(l1_map)
    return (normalized_l1 > loss_thresh).to(torch.int32)


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
class GaussianFastGS(BaseDensificationMethod):
    """FastGS adaptive density control over backend-provided signal traits."""

    refine_every: int = 100
    start_iter: int = 600
    stop_iter: int = 14_900
    loss_thresh: float = 0.1
    grad_threshold: float = 2e-4
    grad_abs_threshold: float = 1.2e-3
    dense_fraction: float = 0.01
    prune_opacity_threshold: float = 0.005
    opacity_reset_every: int = 3_000
    extra_opacity_reset_iter: int | None = None
    max_reset_opacity: float = 0.8
    scheduled_reset_opacity: float = 0.01
    probe_view_count: int = 10
    importance_threshold: float = 5.0
    final_prune_start_iter: int = 15_000
    final_prune_stop_iter: int = 30_000
    final_prune_every: int = 3_000
    final_prune_opacity_threshold: float = 0.1
    camera_extent: float = 1.0
    request_collect_densification_info: bool = True
    expected_scene_families: tuple[str, ...] = ("gaussian",)
    family_ops: GaussianFamilyOps | None = field(
        default=None, init=False, repr=False
    )
    clone_grad_sum: Tensor | None = field(default=None, init=False, repr=False)
    split_grad_sum: Tensor | None = field(default=None, init=False, repr=False)
    visible_count: Tensor | None = field(default=None, init=False, repr=False)
    max_screen_radii: Tensor | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.refine_schedule = Schedule(
            start_iteration=self.start_iter,
            end_iteration=self.stop_iter,
            frequency=self.refine_every,
        )

    def get_render_requirements(
        self,
        state: object,
    ) -> DensificationRenderRequirements:
        """Request backend signal collection while FastGS refinement is active."""
        if not self.request_collect_densification_info:
            return DensificationRenderRequirements()
        step = int(getattr(state, "step", 0))
        return DensificationRenderRequirements(
            backend_options={
                "collect_densification_info": step + 1 < self.stop_iter
            }
        )

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Bind Gaussian topology operations."""
        del optimizers
        if state.model.scene.scene_family not in self.expected_scene_families:
            raise TypeError(
                "GaussianFastGS expects Gaussian scene families, got "
                f"{state.model.scene.scene_family!r}."
            )
        if not isinstance(family_ops, GaussianFamilyOps):
            raise TypeError("GaussianFastGS requires GaussianFamilyOps.")
        self.family_ops = family_ops
        self.reset_accumulators()

    def pre_backward(self, context: DensificationContext) -> None:
        """Allow the backend to retain any tensors needed for signal gradients."""
        provider = self.require_runtime_trait(
            context,
            GaussianFastGSSignalProvider,
        )
        provider.prepare_fastgs_signals(context)

    def post_backward(self, context: DensificationContext) -> None:
        """Accumulate backend-provided FastGS densification statistics."""
        if context.step + 1 >= self.stop_iter:
            return
        provider = self.require_runtime_trait(
            context,
            GaussianFastGSSignalProvider,
        )
        signals = provider.collect_fastgs_signals(context)
        if signals is None:
            return
        scene = context.state.model.scene
        self._ensure_buffers(
            num_splats=int(scene.center_position.shape[0]),
            dtype=scene.center_position.dtype,
            device=scene.center_position.device,
        )
        assert self.clone_grad_sum is not None
        assert self.split_grad_sum is not None
        assert self.visible_count is not None
        assert self.max_screen_radii is not None
        self.visible_count += signals.visible_count.to(
            device=self.visible_count.device,
            dtype=self.visible_count.dtype,
        )
        self.clone_grad_sum += signals.clone_grad_sum.to(
            device=self.clone_grad_sum.device,
            dtype=self.clone_grad_sum.dtype,
        )
        self.split_grad_sum += signals.split_grad_sum.to(
            device=self.split_grad_sum.device,
            dtype=self.split_grad_sum.dtype,
        )
        self.max_screen_radii = torch.maximum(
            self.max_screen_radii,
            signals.max_screen_radii.to(
                device=self.max_screen_radii.device,
                dtype=self.max_screen_radii.dtype,
            ),
        )

    def pre_optimizer_step(self, context: DensificationContext) -> None:
        """Run scheduled FastGS clone/split/prune/reset actions."""
        if self.family_ops is None:
            return
        scene = context.state.model.scene
        if not isinstance(scene, GaussianScene):
            return
        step = context.step + 1
        if self.refine_schedule.includes(step):
            self.adaptive_density_control(context, scene, step)
            self.reset_accumulators()
        if self.should_reset_opacity(step):
            self.family_ops.reset_opacity(self.scheduled_reset_opacity)
        if self.should_final_prune(step):
            self.final_prune(self.compute_pruning_score(context))

    def adaptive_density_control(
        self,
        context: DensificationContext,
        scene: GaussianScene,
        step: int,
    ) -> None:
        """Run one scheduled FastGS adaptive-density-control update."""
        if (
            self.visible_count is None
            or self.clone_grad_sum is None
            or self.split_grad_sum is None
            or self.max_screen_radii is None
        ):
            return
        assert self.family_ops is not None
        score_started_at = time.perf_counter()
        importance_score, pruning_score = self.compute_fastgs_scores(
            context,
            densify=True,
        )
        self._record_metric(
            context,
            "refinement_fastgs_score_ms",
            (time.perf_counter() - score_started_at) * 1000.0,
        )
        avg_clone_grad = self.clone_grad_sum / self.visible_count.clamp_min(1.0)
        avg_split_grad = self.split_grad_sum / self.visible_count.clamp_min(1.0)
        scales = torch.exp(scene.log_scales).max(dim=-1).values
        metric_mask = importance_score > self.importance_threshold
        clone_mask = (
            (avg_clone_grad >= self.grad_threshold)
            & (scales <= self.dense_fraction * self.camera_extent)
            & metric_mask
        )
        split_mask = (
            (avg_split_grad >= self.grad_abs_threshold)
            & (scales > self.dense_fraction * self.camera_extent)
            & metric_mask
        )
        grown_max_screen_radii = self._grown_zero_accumulator_values(
            self.max_screen_radii,
            clone_mask,
            split_mask,
            num_children=2,
        )

        prune_candidate_count = 0
        sampled_prune_count = 0

        def refinement_keep_mask(grown_scene: GaussianScene) -> Tensor:
            nonlocal prune_candidate_count, sampled_prune_count
            prune_candidate_mask = (
                torch.sigmoid(grown_scene.logit_opacity)
                < self.prune_opacity_threshold
            )
            if step > self.opacity_reset_every and (
                grown_scene.center_position.shape[0] > 0
            ):
                max_scale = torch.exp(grown_scene.log_scales).max(dim=-1).values
                prune_candidate_mask |= max_scale > 0.1 * self.camera_extent
                prune_candidate_mask |= grown_max_screen_radii > 20.0
            sampled_prune_mask = self._sample_refinement_prune_mask(
                prune_candidate_mask,
                pruning_score,
            )
            prune_candidate_count = int(prune_candidate_mask.sum().item())
            sampled_prune_count = int(sampled_prune_mask.sum().item())
            keep_mask = ~sampled_prune_mask
            keep_mask &= (
                grown_scene.quaternion_orientation.square().sum(dim=1) >= 1e-8
            )
            return keep_mask

        topology_started_at = time.perf_counter()
        self.family_ops.clone_and_split(
            clone_mask,
            split_mask,
            num_children=2,
            scale_shrink=0.625,
            prune_fn=refinement_keep_mask,
            prune_field_names=(
                "center_position",
                "logit_opacity",
                "log_scales",
                "quaternion_orientation",
            ),
        )
        self._record_metric(
            context,
            "refinement_fastgs_topology_ms",
            (time.perf_counter() - topology_started_at) * 1000.0,
        )
        self._record_metric(
            context,
            "refinement_fastgs_prune_candidate_count",
            float(prune_candidate_count),
        )
        self._record_metric(
            context,
            "refinement_fastgs_sampled_prune_count",
            float(sampled_prune_count),
        )
        self.family_ops.reset_opacity(self.max_reset_opacity)

    def compute_fastgs_scores(
        self,
        context: DensificationContext,
        *,
        densify: bool,
    ) -> tuple[Tensor, Tensor]:
        """Compute FastGS multi-view consistency scores."""
        attribution = self.require_runtime_trait(
            context,
            GaussianMetricAttribution,
        )
        if context.runtime is None:
            raise RuntimeError("GaussianFastGS scoring requires a runtime.")
        probe_views = context.runtime.sample_views(self.probe_view_count)
        scene = context.state.model.scene
        importance_sum = torch.zeros(
            int(scene.center_position.shape[0]),
            dtype=scene.center_position.dtype,
            device=scene.center_position.device,
        )
        pruning_sum = torch.zeros_like(importance_sum)
        if not probe_views:
            return importance_sum, pruning_sum
        for sample in probe_views:
            probe_output = context.runtime.render_raw(
                context.state.model,
                sample.camera,
            )
            predicted = self.probe_prediction(context, sample, probe_output)
            metric_map = fastgs_l1_metric_map(
                predicted,
                sample.image,
                self.loss_thresh,
            )
            photometric_loss = (predicted - sample.image).abs().mean()
            attributed = attribution.attribute_metric_map(
                scene,
                sample.camera,
                metric_map,
                options=context.runtime.render_options,
            )
            if densify:
                importance_sum += attributed
            pruning_sum += photometric_loss * attributed
        importance_score = importance_sum / float(len(probe_views))
        return importance_score, fastgs_normalize_score(pruning_sum)

    def probe_prediction(
        self,
        context: DensificationContext,
        sample: Any,
        probe_output: Any,
    ) -> Tensor:
        """Return the RGB probe prediction used for FastGS metric maps."""
        del context, sample
        return probe_output.render[0]

    def compute_pruning_score(self, context: DensificationContext) -> Tensor:
        """Compute only the FastGS VCP score."""
        _importance_score, pruning_score = self.compute_fastgs_scores(
            context,
            densify=False,
        )
        return pruning_score

    def final_prune(self, pruning_score: Tensor) -> None:
        """Apply FastGS final-stage VCP pruning."""
        if self.family_ops is None:
            return
        scene = self.family_ops.scene
        prune_mask = (
            torch.sigmoid(scene.logit_opacity)
            < self.final_prune_opacity_threshold
        )
        prune_mask |= pruning_score > 0.9
        self.family_ops.prune(~prune_mask)

    def should_reset_opacity(self, step: int) -> bool:
        """Return whether FastGS should apply an opacity reset."""
        scheduled = (
            step >= self.opacity_reset_every
            and step <= self.stop_iter
            and step % self.opacity_reset_every == 0
        )
        return scheduled or (
            self.extra_opacity_reset_iter is not None
            and step == self.extra_opacity_reset_iter
        )

    def should_final_prune(self, step: int) -> bool:
        """Return whether FastGS should apply final VCP pruning."""
        return (
            step > self.final_prune_start_iter
            and step < self.final_prune_stop_iter
            and step % self.final_prune_every == 0
        )

    def reset_accumulators(self) -> None:
        """Drop accumulated FastGS signal buffers."""
        self.clone_grad_sum = None
        self.split_grad_sum = None
        self.visible_count = None
        self.max_screen_radii = None

    def _ensure_buffers(
        self,
        *,
        num_splats: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (num_splats,)
        if (
            self.clone_grad_sum is None
            or tuple(self.clone_grad_sum.shape) != shape
            or self.clone_grad_sum.device != device
        ):
            self.clone_grad_sum = torch.zeros(shape, dtype=dtype, device=device)
            self.split_grad_sum = torch.zeros(shape, dtype=dtype, device=device)
            self.visible_count = torch.zeros(shape, dtype=dtype, device=device)
            self.max_screen_radii = torch.zeros(
                shape, dtype=dtype, device=device
            )

    def _grown_zero_accumulator_values(
        self,
        value: Tensor,
        clone_mask: Tensor,
        split_mask: Tensor,
        *,
        num_children: int,
    ) -> Tensor:
        clone_count = int(clone_mask.sum().item())
        split_child_count = int(split_mask.sum().item()) * num_children
        return torch.cat(
            [
                value[~split_mask],
                torch.zeros(
                    (clone_count,),
                    dtype=value.dtype,
                    device=value.device,
                ),
                torch.zeros(
                    (split_child_count,),
                    dtype=value.dtype,
                    device=value.device,
                ),
            ]
        )

    def _sample_refinement_prune_mask(
        self,
        prune_mask: Tensor,
        pruning_score: Tensor,
    ) -> Tensor:
        remove_budget = int(0.5 * int(prune_mask.sum().item()))
        if remove_budget <= 0 or pruning_score.numel() == 0:
            return torch.zeros_like(prune_mask)
        scores = 1.0 - pruning_score.reshape(-1)
        weighted_count = min(int(scores.numel()), int(prune_mask.numel()))
        padded_importance = torch.zeros(
            (int(prune_mask.numel()),),
            dtype=torch.float32,
            device=prune_mask.device,
        )
        padded_importance[:weighted_count] = 1.0 / (
            1e-6 + scores[:weighted_count].clamp_min(0.0)
        ).to(device=prune_mask.device, dtype=torch.float32)
        sampled_indices = torch.multinomial(
            padded_importance,
            remove_budget,
            replacement=False,
        )
        sampled_mask = torch.zeros_like(prune_mask)
        sampled_mask[sampled_indices] = True
        return prune_mask & sampled_mask

    def _record_metric(
        self,
        context: DensificationContext,
        name: str,
        value: float,
    ) -> None:
        diagnostics = getattr(context.state, "diagnostics", None)
        if isinstance(diagnostics, dict):
            diagnostics.setdefault("metrics", {})[name] = float(value)


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
    "GaussianFastGS",
    "GaussianMipSplatting3DFilter",
    "GaussianMortonOrdering",
    "active_sh_bases_for_step",
    "fastergs_training_backend_options",
    "fastgs_l1_metric_map",
    "fastgs_normalize_score",
    "morton_codes",
    "morton_order",
]
