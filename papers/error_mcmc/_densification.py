"""Error-aware MCMC densification for the Error-MCMC paper notebook."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import ember_core as ember
import torch
from ember_core.densification import (
    DensificationContext,
    GaussianMetricAttribution,
    Schedule,
)
from ember_splatting_training.densification.mcmc import GaussianMCMC
from ember_splatting_training.fastergs import (
    fastgs_l1_metric_map,
    fastgs_normalize_score,
)
from jaxtyping import Float
from torch import Tensor

ErrorMCMCScoreAggregation = Literal[
    "mean",
    "topk_mean",
    "max",
    "visibility_normalized",
]


@dataclass
class ErrorMCMC(GaussianMCMC):
    """MCMC teleportation biased toward rare-view high-error primitives."""

    loss_thresh: float = 0.06
    probe_view_count: int = 32
    score_aggregation: ErrorMCMCScoreAggregation = "topk_mean"
    score_top_k: int = 3
    opacity_floor: float = 0.05
    opacity_power: float = 0.5
    normalize_error_score: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.schedule, dict):
            self.schedule = Schedule(**self.schedule)
        if self.score_top_k < 1:
            raise ValueError("score_top_k must be at least 1.")
        if not 0.0 <= self.opacity_floor <= 1.0:
            raise ValueError("opacity_floor must be in [0, 1].")
        if self.opacity_power < 0.0:
            raise ValueError("opacity_power must be non-negative.")

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Run noise injection, error-guided relocation, and capped growth."""
        scene = context.state.model.scene
        if (
            not isinstance(scene, ember.GaussianScene)
            or self._family_ops is None
        ):
            return
        if self.inject_position_noise:
            self._inject_noise(context, scene)

        upstream_iteration = context.step + 1
        if not self.schedule.includes(upstream_iteration):
            return

        source_weights = self.source_sampling_weights(context, scene)
        dead_indices, sampled_indices = self._relocate_dead_from_weights(
            scene,
            source_weights,
        )
        scene = self._family_ops.scene
        if int(dead_indices.numel()) > 0:
            source_weights = self._weights_after_relocation(
                source_weights,
                dead_indices,
                sampled_indices,
            )
        self._append_new_from_weights(scene, source_weights)

    def source_sampling_weights(
        self,
        context: DensificationContext,
        scene: ember.GaussianScene,
    ) -> Float[Tensor, " num_splats"]:
        """Return non-negative source sampling weights for teleportation."""
        error_score = self.compute_error_score(context, scene)
        if self.normalize_error_score:
            error_score = fastgs_normalize_score(error_score)
        weights = self._opacity_weighted_error_score(scene, error_score)
        weights = self._mask_invalid_sources(scene, weights)
        weights = self._fallback_source_weights(scene, weights)
        self._record_source_metrics(context, error_score, weights)
        return weights

    def compute_error_score(
        self,
        context: DensificationContext,
        scene: ember.GaussianScene,
    ) -> Float[Tensor, " num_splats"]:
        """Compute rare-view-aware FastGS error attribution per primitive."""
        attribution = self.require_runtime_trait(
            context,
            GaussianMetricAttribution,
        )
        if context.runtime is None:
            raise RuntimeError("ErrorMCMC scoring requires a runtime.")
        probe_views = context.runtime.sample_views(self.probe_view_count)
        if not probe_views:
            return torch.zeros(
                (int(scene.center_position.shape[0]),),
                dtype=scene.center_position.dtype,
                device=scene.center_position.device,
            )

        attributed_scores: list[Tensor] = []
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
            attributed = attribution.attribute_metric_map(
                scene,
                sample.camera,
                metric_map,
                options=context.runtime.render_options,
            )
            attributed_scores.append(
                attributed.to(
                    device=scene.center_position.device,
                    dtype=scene.center_position.dtype,
                )
            )
        per_view_scores = torch.stack(attributed_scores, dim=0)
        return self.aggregate_error_scores(per_view_scores)

    def aggregate_error_scores(
        self,
        per_view_scores: Float[Tensor, " probe_views num_splats"],
    ) -> Float[Tensor, " num_splats"]:
        """Aggregate per-view attributions without washing out rare views."""
        if per_view_scores.ndim != 2:
            raise ValueError(
                "ErrorMCMC per-view scores must have shape "
                f"(probe_views, num_splats), got {tuple(per_view_scores.shape)}."
            )
        if self.score_aggregation == "mean":
            return per_view_scores.mean(dim=0)
        if self.score_aggregation == "max":
            return per_view_scores.max(dim=0).values
        if self.score_aggregation == "visibility_normalized":
            visible = per_view_scores > 0
            return per_view_scores.sum(dim=0) / visible.sum(dim=0).clamp_min(1)
        if self.score_aggregation == "topk_mean":
            k = min(self.score_top_k, int(per_view_scores.shape[0]))
            return per_view_scores.topk(k, dim=0).values.mean(dim=0)
        raise ValueError(
            f"Unsupported ErrorMCMC score aggregation {self.score_aggregation!r}."
        )

    def probe_prediction(
        self,
        context: DensificationContext,
        sample: Any,
        probe_output: Any,
    ) -> Tensor:
        """Return the RGB probe prediction used for error metric maps."""
        del context, sample
        return probe_output.render[0]

    def _opacity_weighted_error_score(
        self,
        scene: ember.GaussianScene,
        error_score: Tensor,
    ) -> Tensor:
        opacities = torch.sigmoid(scene.logit_opacity).to(
            device=error_score.device,
            dtype=error_score.dtype,
        )
        opacity_weight = opacities.clamp_min(self.opacity_floor).pow(
            self.opacity_power
        )
        weights = error_score.clamp_min(0.0) * opacity_weight
        return torch.where(
            torch.isfinite(weights), weights, torch.zeros_like(weights)
        )

    def _mask_invalid_sources(
        self,
        scene: ember.GaussianScene,
        weights: Tensor,
    ) -> Tensor:
        source_mask = ~self._dead_mask(scene).to(device=weights.device)
        return torch.where(source_mask, weights, torch.zeros_like(weights))

    def _fallback_source_weights(
        self,
        scene: ember.GaussianScene,
        weights: Tensor,
    ) -> Tensor:
        if self._has_positive_weight(weights):
            return weights
        alive_mask = ~self._dead_mask(scene).to(device=weights.device)
        opacity_weights = torch.sigmoid(scene.logit_opacity).to(
            device=weights.device,
            dtype=weights.dtype,
        )
        opacity_weights = torch.where(
            alive_mask,
            opacity_weights.clamp_min(self.opacity_floor),
            torch.zeros_like(opacity_weights),
        )
        if self._has_positive_weight(opacity_weights):
            return opacity_weights
        return alive_mask.to(dtype=weights.dtype)

    def _relocate_dead_from_weights(
        self,
        scene: ember.GaussianScene,
        source_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self._family_ops is None:
            empty = self._empty_indices(scene)
            return empty, empty
        dead_mask = self._dead_mask(scene)
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0 or not self._has_positive_weight(source_weights):
            empty = self._empty_indices(scene)
            return empty, empty

        dead_indices = torch.where(dead_mask)[0]
        sampled_indices = self._sample_source_indices(
            source_weights,
            n_dead,
        )
        opacities = torch.sigmoid(scene.logit_opacity)
        adjusted_opacities, adjusted_scales = self._adjusted_samples(
            opacities,
            scene.log_scales,
            sampled_indices,
        )
        overrides = {
            "logit_opacity": adjusted_opacities,
            "log_scales": adjusted_scales,
        }
        self._family_ops.copy_to_indices(
            (
                (sampled_indices, sampled_indices, overrides),
                (dead_indices, sampled_indices, overrides),
            )
        )
        self._family_ops.reset_optimizer_state(sampled_indices)
        return dead_indices, sampled_indices

    def _append_new_from_weights(
        self,
        scene: ember.GaussianScene,
        source_weights: Tensor,
    ) -> None:
        if self._family_ops is None:
            return
        current_n_points = int(scene.center_position.shape[0])
        target_n = min(
            self.cap_max,
            math.floor(self.cap_growth_factor * current_n_points),
        )
        n_added = max(0, target_n - current_n_points)
        if n_added == 0 or not self._has_positive_weight(source_weights):
            return

        sampled_indices = self._sample_source_indices(source_weights, n_added)
        opacities = torch.sigmoid(scene.logit_opacity)
        adjusted_opacities, adjusted_scales = self._adjusted_samples(
            opacities,
            scene.log_scales,
            sampled_indices,
        )
        overrides = {
            "logit_opacity": adjusted_opacities,
            "log_scales": adjusted_scales,
        }
        self._family_ops.copy_and_append_from_indices(
            sampled_indices,
            field_overrides=overrides,
        )
        self._family_ops.reset_optimizer_state(sampled_indices)

    def _sample_source_indices(
        self,
        source_weights: Tensor,
        count: int,
    ) -> Tensor:
        source_indices = torch.where(source_weights > 0)[0]
        sampled_local = torch.multinomial(
            source_weights[source_indices],
            count,
            replacement=True,
        )
        return source_indices[sampled_local]

    def _weights_after_relocation(
        self,
        source_weights: Tensor,
        dead_indices: Tensor,
        sampled_indices: Tensor,
    ) -> Tensor:
        relocated = source_weights.clone()
        relocated[dead_indices] = source_weights[sampled_indices]
        return relocated

    def _empty_indices(self, scene: ember.GaussianScene) -> Tensor:
        return torch.empty(
            (0,),
            dtype=torch.long,
            device=scene.center_position.device,
        )

    def _has_positive_weight(self, weights: Tensor) -> bool:
        return bool(torch.any(weights > 0).item())

    def _record_source_metrics(
        self,
        context: DensificationContext,
        error_score: Tensor,
        weights: Tensor,
    ) -> None:
        diagnostics = getattr(context.state, "diagnostics", None)
        if not isinstance(diagnostics, dict):
            return
        metrics = diagnostics.setdefault("metrics", {})
        metrics["refinement_error_mcmc_score_max"] = float(
            error_score.detach().max().item()
        )
        metrics["refinement_error_mcmc_weight_nonzero"] = float(
            (weights > 0).sum().item()
        )


__all__ = [
    "ErrorMCMC",
    "ErrorMCMCScoreAggregation",
]
