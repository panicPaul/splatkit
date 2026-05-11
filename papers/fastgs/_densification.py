"""FastGS paper densification helpers.

This module intentionally keeps the notebook-local FastGS method importable so
script-mode training can resolve ``papers.fastgs.notebook.FastGSDensification``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import ember_core as ember
import torch
from ember_core.densification import (
    BaseDensificationMethod,
    DensificationContext,
    DensificationRenderRequirements,
    GaussianFamilyOps,
    GaussianMetricAttribution,
    Schedule,
)
from ember_core.training import TrainState
from jaxtyping import Float
from torch import Tensor

FastGSBackendName = Literal["adapter.fastgs", "faster_gs.fastgs"]
FastGSMetricMapBackend = Literal["eager", "compile"]

_COMPILED_FASTGS_L1_METRIC_MAP: Any | None = None


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


def compiled_fastgs_l1_metric_map(
    predicted: Tensor,
    target: Tensor,
    loss_thresh: float,
) -> Tensor:
    """Build FastGS' metric map through a cached compiled tensor helper."""
    global _COMPILED_FASTGS_L1_METRIC_MAP
    if _COMPILED_FASTGS_L1_METRIC_MAP is None:
        _COMPILED_FASTGS_L1_METRIC_MAP = torch.compile(
            fastgs_l1_metric_map,
            mode="reduce-overhead",
            fullgraph=True,
        )
    mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(mark_step_begin):
        mark_step_begin()
    return _COMPILED_FASTGS_L1_METRIC_MAP(predicted, target, loss_thresh)


@runtime_checkable
class HasFastGSDensificationInfo(Protocol):
    """Render-output trait for FastGS densification accumulators."""

    densification_info: Float[Tensor, " 4 num_splats"]


class FastGSDensification(BaseDensificationMethod):
    """Notebook-local FastGS adaptive density control."""

    expected_scene_families = ("gaussian",)

    def __init__(
        self,
        *,
        refine_every: int = 100,
        start_iter: int = 600,
        stop_iter: int = 14_900,
        backend: FastGSBackendName = "adapter.fastgs",
        loss_thresh: float = 0.1,
        grad_threshold: float = 2e-4,
        grad_abs_threshold: float = 1.2e-3,
        dense_fraction: float = 0.01,
        prune_opacity_threshold: float = 0.005,
        opacity_reset_every: int = 3_000,
        extra_opacity_reset_iter: int | None = 500,
        max_reset_opacity: float = 0.01,
        scheduled_reset_opacity: float = 0.01,
        probe_view_count: int = 10,
        importance_threshold: float = 5.0,
        metric_map_backend: FastGSMetricMapBackend = "eager",
        final_prune_start_iter: int = 15_000,
        final_prune_stop_iter: int = 30_000,
        final_prune_every: int = 3_000,
        final_prune_opacity_threshold: float = 0.1,
        camera_extent: float = 1.0,
    ) -> None:
        self.refine_schedule = Schedule(
            start_iteration=start_iter,
            end_iteration=stop_iter,
            frequency=refine_every,
        )
        self.backend = backend
        self.stop_iter = stop_iter
        self.loss_thresh = loss_thresh
        self.grad_threshold = grad_threshold
        self.grad_abs_threshold = grad_abs_threshold
        self.dense_fraction = dense_fraction
        self.prune_opacity_threshold = prune_opacity_threshold
        self.opacity_reset_every = opacity_reset_every
        self.extra_opacity_reset_iter = extra_opacity_reset_iter
        self.max_reset_opacity = max_reset_opacity
        self.scheduled_reset_opacity = scheduled_reset_opacity
        self.probe_view_count = probe_view_count
        self.importance_threshold = importance_threshold
        self.metric_map_backend = metric_map_backend
        self.final_prune_start_iter = final_prune_start_iter
        self.final_prune_stop_iter = final_prune_stop_iter
        self.final_prune_every = final_prune_every
        self.final_prune_opacity_threshold = final_prune_opacity_threshold
        self.camera_extent = float(camera_extent)
        self.family_ops: GaussianFamilyOps | None = None
        self.clone_grad_sum: Tensor | None = None
        self.split_grad_sum: Tensor | None = None
        self.visible_count: Tensor | None = None
        self.max_screen_radii: Tensor | None = None

    def get_render_requirements(
        self,
        state: TrainState,
    ) -> DensificationRenderRequirements:
        """Collect FastGS visibility accumulators while densification runs."""
        if self.backend == "adapter.fastgs":
            return DensificationRenderRequirements()
        return DensificationRenderRequirements(
            backend_options={
                "collect_densification_info": state.step + 1 < self.stop_iter
            }
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
            raise TypeError("FastGSDensification requires GaussianFamilyOps.")
        self.family_ops = family_ops

    def post_backward(self, context: DensificationContext) -> None:
        """Accumulate FastGS screen-space densification statistics."""
        if context.step + 1 >= self.stop_iter:
            return
        if self.backend == "adapter.fastgs":
            self._accumulate_adapter_gradients(context)
            return
        self._accumulate_native_densification_info(context)

    def pre_optimizer_step(self, context: DensificationContext) -> None:
        """Run scheduled clone/split/prune/reset actions before optimizer."""
        if self.family_ops is None:
            return
        scene = context.state.model.scene
        if not isinstance(scene, ember.GaussianScene):
            return
        upstream_iteration = context.step + 1
        if self.refine_schedule.includes(upstream_iteration):
            self.adaptive_density_control(context, scene, upstream_iteration)
            self.reset_accumulators()
        if self.should_reset_opacity(upstream_iteration):
            self.family_ops.reset_opacity(self.scheduled_reset_opacity)
        if self.should_final_prune(upstream_iteration):
            pruning_score = self.compute_pruning_score(context)
            self.final_prune(pruning_score)

    def adaptive_density_control(
        self,
        context: DensificationContext,
        scene: ember.GaussianScene,
        step: int,
    ) -> None:
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

        prune_sampling_ms = 0.0

        def refinement_keep_mask(grown_scene: ember.GaussianScene) -> Tensor:
            nonlocal prune_sampling_ms
            keep_mask = torch.sigmoid(grown_scene.logit_opacity) >= (
                self.prune_opacity_threshold
            )
            if step > self.opacity_reset_every and (
                grown_scene.center_position.shape[0] > 0
            ):
                max_scale = torch.exp(grown_scene.log_scales).max(dim=-1).values
                keep_mask &= max_scale <= 0.1 * self.camera_extent
                keep_mask &= grown_max_screen_radii <= 20.0
            prune_started_at = time.perf_counter()
            sampled_prune_mask = self._sample_refinement_prune_mask(
                ~keep_mask,
                pruning_score,
            )
            prune_sampling_ms += (time.perf_counter() - prune_started_at) * 1000.0
            keep_mask = ~sampled_prune_mask
            keep_mask &= grown_scene.quaternion_orientation.square().sum(dim=1) >= 1e-8
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
            "refinement_fastgs_prune_sampling_ms",
            prune_sampling_ms,
        )
        reset_started_at = time.perf_counter()
        self.family_ops.reset_opacity(self.max_reset_opacity)
        self._record_metric(
            context,
            "refinement_fastgs_opacity_reset_ms",
            (time.perf_counter() - reset_started_at) * 1000.0,
        )

    def _sample_refinement_prune_mask(
        self,
        prune_mask: Tensor,
        pruning_score: Tensor,
    ) -> Tensor:
        """Apply upstream FastGS' budgeted refinement prune sampling."""
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

    def should_reset_opacity(self, step: int) -> bool:
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
        return (
            step > self.final_prune_start_iter
            and step < self.final_prune_stop_iter
            and step % self.final_prune_every == 0
        )

    def reset_accumulators(self) -> None:
        self.clone_grad_sum = None
        self.split_grad_sum = None
        self.visible_count = None
        self.max_screen_radii = None

    def _append_zero_accumulator_values(self, count: int) -> None:
        if count <= 0:
            return
        for name in (
            "clone_grad_sum",
            "split_grad_sum",
            "visible_count",
            "max_screen_radii",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            setattr(
                self,
                name,
                torch.cat(
                    [
                        value,
                        torch.zeros(
                            (count,),
                            dtype=value.dtype,
                            device=value.device,
                        ),
                    ]
                ),
            )

    def _split_accumulator_values(
        self,
        split_mask: torch.Tensor,
        *,
        num_children: int,
    ) -> None:
        for name in (
            "clone_grad_sum",
            "split_grad_sum",
            "visible_count",
            "max_screen_radii",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            child_count = int(split_mask.sum().item()) * num_children
            setattr(
                self,
                name,
                torch.cat(
                    [
                        value[~split_mask],
                        torch.zeros(
                            (child_count,),
                            dtype=value.dtype,
                            device=value.device,
                        ),
                    ]
                ),
            )

    def _grown_zero_accumulator_values(
        self,
        value: Tensor,
        clone_mask: Tensor,
        split_mask: Tensor,
        *,
        num_children: int,
    ) -> Tensor:
        """Return accumulator values in fused clone/split output order."""
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

    def _ensure_buffers(
        self,
        *,
        num_splats: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (num_splats,)
        if self.clone_grad_sum is None or self.clone_grad_sum.shape != shape:
            self.clone_grad_sum = torch.zeros(shape, dtype=dtype, device=device)
            self.split_grad_sum = torch.zeros(shape, dtype=dtype, device=device)
            self.visible_count = torch.zeros(shape, dtype=dtype, device=device)
            self.max_screen_radii = torch.zeros(
                shape,
                dtype=dtype,
                device=device,
            )

    def _accumulate_native_densification_info(
        self,
        context: DensificationContext,
    ) -> None:
        if not isinstance(context.render_output, HasFastGSDensificationInfo):
            raise TypeError(
                "FastGSDensification requires render outputs with "
                "densification_info."
            )
        scene = context.state.model.scene
        densification_info = context.render_output.densification_info.detach()
        if densification_info.ndim != 2 or densification_info.shape[0] != 4:
            raise ValueError(
                "FastGS densification_info must have shape "
                f"(4, num_splats), got {tuple(densification_info.shape)}."
            )
        self._ensure_buffers(
            num_splats=int(scene.center_position.shape[0]),
            dtype=scene.center_position.dtype,
            device=scene.center_position.device,
        )
        assert self.clone_grad_sum is not None
        assert self.split_grad_sum is not None
        assert self.visible_count is not None
        assert self.max_screen_radii is not None
        self.visible_count += densification_info[0]
        self.clone_grad_sum += densification_info[1]
        self.split_grad_sum += densification_info[2]
        self.max_screen_radii = torch.maximum(
            self.max_screen_radii,
            densification_info[3].to(dtype=self.max_screen_radii.dtype),
        )

    def _accumulate_adapter_gradients(
        self,
        context: DensificationContext,
    ) -> None:
        output = context.render_output
        if not all(
            hasattr(output, name)
            for name in ("viewspace_points", "visibility_filter", "radii")
        ):
            raise TypeError(
                "adapter.fastgs densification requires viewspace_points, "
                "visibility_filter, and radii render outputs."
            )
        gradients = output.viewspace_points.grad
        if gradients is None:
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
        visibility = output.visibility_filter.to(gradients.dtype)
        self.clone_grad_sum += (
            gradients[..., :2].norm(dim=-1) * visibility
        ).sum(dim=0)
        self.split_grad_sum += (
            gradients[..., 2:].norm(dim=-1) * visibility
        ).sum(dim=0)
        self.visible_count += visibility.sum(dim=0)
        visible_radii = torch.where(
            output.visibility_filter,
            output.radii.to(scene.center_position.dtype),
            torch.zeros_like(output.radii, dtype=scene.center_position.dtype),
        )
        self.max_screen_radii = torch.maximum(
            self.max_screen_radii,
            visible_radii.max(dim=0).values,
        )

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
            raise RuntimeError("FastGS scoring requires a runtime.")
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

        render_ms = 0.0
        metric_map_ms = 0.0
        photometric_loss_ms = 0.0
        attribution_ms = 0.0
        for sample in probe_views:
            started_at = time.perf_counter()
            probe_output = context.runtime.render_raw(
                context.state.model,
                sample.camera,
            )
            render_ms += (time.perf_counter() - started_at) * 1000.0
            predicted = probe_output.render[0]
            target = sample.image

            started_at = time.perf_counter()
            metric_map = self._metric_map(predicted, target)
            metric_map_ms += (time.perf_counter() - started_at) * 1000.0

            started_at = time.perf_counter()
            photometric_loss = self._photometric_loss(predicted, target)
            photometric_loss_ms += (time.perf_counter() - started_at) * 1000.0

            started_at = time.perf_counter()
            attributed = attribution.attribute_metric_map(
                scene,
                sample.camera,
                metric_map,
                options=context.runtime.render_options,
            )
            attribution_ms += (time.perf_counter() - started_at) * 1000.0
            if densify:
                importance_sum += attributed
            pruning_sum += photometric_loss * attributed

        self._record_metric(context, "refinement_fastgs_probe_views", len(probe_views))
        self._record_metric(context, "refinement_fastgs_probe_render_ms", render_ms)
        self._record_metric(
            context,
            "refinement_fastgs_metric_map_ms",
            metric_map_ms,
        )
        self._record_metric(
            context,
            "refinement_fastgs_photometric_loss_ms",
            photometric_loss_ms,
        )
        self._record_metric(
            context,
            "refinement_fastgs_attribution_ms",
            attribution_ms,
        )
        importance_score = torch.div(
            importance_sum,
            float(len(probe_views)),
            rounding_mode="floor",
        )
        return importance_score, self._normalize_score(pruning_sum)

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
        if pruning_score.numel() == prune_mask.numel():
            prune_mask |= pruning_score > 0.9
        if torch.any(prune_mask):
            self.family_ops.prune(~prune_mask)

    def _normalize_score(self, score: Tensor) -> Tensor:
        return fastgs_normalize_score(score)

    def _metric_map(
        self,
        predicted: Tensor,
        target: Tensor,
    ) -> Tensor:
        if self.metric_map_backend == "compile":
            try:
                return compiled_fastgs_l1_metric_map(
                    predicted,
                    target,
                    self.loss_thresh,
                )
            except Exception:
                return fastgs_l1_metric_map(
                    predicted,
                    target,
                    self.loss_thresh,
                )
        return fastgs_l1_metric_map(
            predicted,
            target,
            self.loss_thresh,
        )

    def _photometric_loss(
        self,
        predicted: Tensor,
        target: Tensor,
    ) -> Tensor:
        l1_loss = (predicted - target).abs().mean()
        from ember_splatting_training.losses import ssim_score

        one_minus_ssim = 1.0 - ssim_score(
            predicted[None, ...],
            target[None, ...],
        )
        return 0.8 * l1_loss + 0.2 * one_minus_ssim

    def _record_metric(
        self,
        context: DensificationContext,
        name: str,
        value: float | int,
    ) -> None:
        diagnostics = getattr(context.state, "diagnostics", None)
        if not isinstance(diagnostics, dict):
            return
        diagnostics.setdefault("metrics", {})[name] = float(value)


__all__ = [
    "FastGSDensification",
    "HasFastGSDensificationInfo",
    "compiled_fastgs_l1_metric_map",
    "fastgs_l1_metric_map",
    "fastgs_normalize_score",
]
