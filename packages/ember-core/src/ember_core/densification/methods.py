"""Built-in densification methods and composition helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ember_core.core.capabilities import HasScreenSpaceDensificationSignals
from ember_core.densification.collectors import (
    ImagePlaneGradientCollector,
    PositionGradientCollector,
    SparseVoxelGradientCollector,
)
from ember_core.densification.contracts import (
    BaseDensificationMethod,
    DensificationContext,
    DensificationRenderRequirements,
    DensificationSignals,
    GaussianMetricAttribution,
    Schedule,
)
from ember_core.densification.passes import (
    GaussianClonePass,
    GaussianJitterPass,
    GaussianOpacityDecayPass,
    GaussianPruneOpacityPass,
    GaussianResetOpacityPass,
    GaussianSplitPass,
    SparseVoxelPrunePass,
    SparseVoxelSubdividePass,
)


@dataclass
class ComposedDensificationMethod:
    """Composable densification method."""

    name: str
    expected_scene_families: tuple[str, ...]
    collectors: list[Any] = field(default_factory=list)
    passes: list[Any] = field(default_factory=list)
    signals: DensificationSignals = field(
        default_factory=DensificationSignals, init=False
    )

    def get_render_requirements(self) -> DensificationRenderRequirements:
        requirements = DensificationRenderRequirements()
        for component in [*self.collectors, *self.passes]:
            requirements = requirements.merge(
                component.get_render_requirements()
            )
        return requirements

    def bind(
        self,
        state: Any,
        optimizers: list[Any],
        family_ops: Any,
    ) -> None:
        family = state.model.scene.scene_family
        if (
            self.expected_scene_families
            and family not in self.expected_scene_families
        ):
            raise TypeError(
                f"{self.name} expects scene families "
                f"{self.expected_scene_families!r}, got {family!r}."
            )
        self.signals = DensificationSignals()
        for component in [*self.collectors, *self.passes]:
            component.bind(state, optimizers, family_ops)

    def pre_backward(self, context: DensificationContext) -> None:
        for component in self.collectors:
            component.pre_backward(context, self.signals)
        for component in self.passes:
            component.pre_backward(context, self.signals)

    def post_backward(self, context: DensificationContext) -> None:
        for component in self.collectors:
            component.post_backward(context, self.signals)
        for component in self.passes:
            component.post_backward(context, self.signals)

    def post_optimizer_step(self, context: DensificationContext) -> None:
        for component in self.collectors:
            component.post_optimizer_step(context, self.signals)
        for component in self.passes:
            component.post_optimizer_step(context, self.signals)

    def after_step(
        self,
        context: DensificationContext,
        metrics: dict[str, float],
    ) -> None:
        for component in self.collectors:
            component.after_step(context, self.signals, metrics)
        for component in self.passes:
            component.after_step(context, self.signals, metrics)


def compose_densification(
    *,
    family: str,
    collectors: list[Any],
    passes: list[Any],
    name: str = "ComposedDensification",
) -> ComposedDensificationMethod:
    """Compose a custom densification method in scripts/notebooks."""
    return ComposedDensificationMethod(
        name=name,
        expected_scene_families=(family,),
        collectors=collectors,
        passes=passes,
    )


@dataclass
class Vanilla3DGS(ComposedDensificationMethod):
    """Vanilla 3DGS-style densification using image-plane gradients."""

    def __init__(
        self,
        *,
        refine_every: int = 100,
        start_iter: int = 500,
        stop_iter: int = 15_000,
        grad_threshold: float = 2e-4,
        relative_size_threshold: float = 0.01,
        prune_opacity_threshold: float = 0.005,
        opacity_reset_every: int = 3_000,
        max_reset_opacity: float = 0.01,
    ) -> None:
        schedule = Schedule(
            start_iteration=start_iter,
            end_iteration=stop_iter,
            frequency=refine_every,
        )
        super().__init__(
            name="Vanilla3DGS",
            expected_scene_families=("gaussian",),
            collectors=[
                ImagePlaneGradientCollector(use_absolute_gradients=False)
            ],
            passes=[
                GaussianClonePass(
                    schedule=schedule,
                    grad_threshold=grad_threshold,
                    relative_size_threshold=relative_size_threshold,
                ),
                GaussianSplitPass(
                    schedule=schedule,
                    grad_threshold=grad_threshold,
                    relative_size_threshold=relative_size_threshold,
                ),
                GaussianPruneOpacityPass(
                    schedule=schedule,
                    opacity_threshold=prune_opacity_threshold,
                ),
                GaussianResetOpacityPass(
                    schedule=Schedule(
                        start_iteration=0,
                        end_iteration=stop_iter,
                        frequency=opacity_reset_every,
                    ),
                    max_post_sigmoid_opacity=max_reset_opacity,
                ),
            ],
        )


@dataclass
class AbsGS(ComposedDensificationMethod):
    """AbsGS-style densification using absolute image-plane gradients."""

    def __init__(self, **kwargs: Any) -> None:
        refine_every = int(kwargs.pop("refine_every", 100))
        start_iter = int(kwargs.pop("start_iter", 500))
        stop_iter = int(kwargs.pop("stop_iter", 15_000))
        grad_threshold = float(kwargs.pop("grad_threshold", 8e-4))
        relative_size_threshold = float(
            kwargs.pop("relative_size_threshold", 0.01)
        )
        prune_opacity_threshold = float(
            kwargs.pop("prune_opacity_threshold", 0.005)
        )
        opacity_reset_every = int(kwargs.pop("opacity_reset_every", 3_000))
        max_reset_opacity = float(kwargs.pop("max_reset_opacity", 0.01))
        schedule = Schedule(
            start_iteration=start_iter,
            end_iteration=stop_iter,
            frequency=refine_every,
        )
        super().__init__(
            name="AbsGS",
            expected_scene_families=("gaussian",),
            collectors=[
                ImagePlaneGradientCollector(use_absolute_gradients=True)
            ],
            passes=[
                GaussianClonePass(
                    schedule=schedule,
                    grad_threshold=grad_threshold,
                    relative_size_threshold=relative_size_threshold,
                ),
                GaussianSplitPass(
                    schedule=schedule,
                    grad_threshold=grad_threshold,
                    relative_size_threshold=relative_size_threshold,
                ),
                GaussianPruneOpacityPass(
                    schedule=schedule,
                    opacity_threshold=prune_opacity_threshold,
                ),
                GaussianResetOpacityPass(
                    schedule=Schedule(
                        start_iteration=0,
                        end_iteration=stop_iter,
                        frequency=opacity_reset_every,
                    ),
                    max_post_sigmoid_opacity=max_reset_opacity,
                ),
            ],
        )


@dataclass
class FastGS(BaseDensificationMethod):
    """Upstream-like FastGS densification over trait-based contracts."""

    refine_every: int = 100
    start_iter: int = 500
    stop_iter: int = 15_000
    loss_thresh: float = 0.1
    grad_threshold: float = 2e-4
    grad_abs_threshold: float = 1.2e-3
    dense_fraction: float = 1e-3
    prune_opacity_threshold: float = 0.005
    opacity_reset_every: int = 3_000
    probe_view_count: int = 10
    importance_threshold: float = 5.0
    expected_scene_families: tuple[str, ...] = ("gaussian",)
    family_ops: Any | None = field(default=None, init=False, repr=False)
    clone_grad_sum: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )
    split_grad_sum: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )
    grad_count: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )
    max_screen_radii: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )

    def bind(
        self,
        state: Any,
        optimizers: list[Any],
        family_ops: Any,
    ) -> None:
        del optimizers
        family = state.model.scene.scene_family
        if family not in self.expected_scene_families:
            raise TypeError(
                f"FastGS expects scene families "
                f"{self.expected_scene_families!r}, got {family!r}."
            )
        self.family_ops = family_ops
        self.clone_grad_sum = None
        self.split_grad_sum = None
        self.grad_count = None
        self.max_screen_radii = None

    def get_render_requirements(self) -> DensificationRenderRequirements:
        return DensificationRenderRequirements()

    def pre_backward(self, context: DensificationContext) -> None:
        render_output = self._require_render_output(context)
        render_output.viewspace_points.retain_grad()

    def post_backward(self, context: DensificationContext) -> None:
        if context.step > self.stop_iter:
            return
        render_output = self._require_render_output(context)
        gradients = render_output.viewspace_points.grad
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
        assert self.grad_count is not None
        assert self.max_screen_radii is not None
        visibility = render_output.visibility_filter.to(gradients.dtype)
        clone_gradients = gradients[..., :2].norm(dim=-1) * visibility
        split_gradients = gradients[..., 2:].norm(dim=-1) * visibility
        self.clone_grad_sum += clone_gradients.sum(dim=0)
        self.split_grad_sum += split_gradients.sum(dim=0)
        self.grad_count += visibility.sum(dim=0)
        visible_radii = torch.where(
            render_output.visibility_filter,
            render_output.radii.to(scene.center_position.dtype),
            torch.zeros_like(
                render_output.radii, dtype=scene.center_position.dtype
            ),
        )
        self.max_screen_radii = torch.maximum(
            self.max_screen_radii,
            visible_radii.max(dim=0).values,
        )

    def post_optimizer_step(self, context: DensificationContext) -> None:
        if self.family_ops is None:
            return
        if context.step > self.stop_iter:
            return
        if context.step != 0 and context.step % self.opacity_reset_every == 0:
            self.family_ops.reset_opacity(0.01)
        if not self._should_refine(context.step):
            return

        scene = context.state.model.scene
        self._ensure_buffers(
            num_splats=int(scene.center_position.shape[0]),
            dtype=scene.center_position.dtype,
            device=scene.center_position.device,
        )
        assert self.clone_grad_sum is not None
        assert self.split_grad_sum is not None
        assert self.grad_count is not None
        assert self.max_screen_radii is not None
        attribution = self.require_runtime_trait(
            context,
            GaussianMetricAttribution,
        )
        runtime = context.runtime
        if runtime is None:
            raise RuntimeError("FastGS requires a densification runtime.")
        probe_views = runtime.sample_views(self.probe_view_count)
        if not probe_views:
            return

        importance_sum = torch.zeros_like(self.clone_grad_sum)
        pruning_sum = torch.zeros_like(self.clone_grad_sum)
        for sample in probe_views:
            probe_output = runtime.render_raw(
                context.state.model, sample.camera
            )
            predicted = probe_output.render[0]
            l1_map = (predicted - sample.image).abs().mean(dim=-1)
            normalized_l1 = self._normalize_l1_map(l1_map)
            metric_map = (normalized_l1 > self.loss_thresh).to(torch.int32)
            attributed = attribution.attribute_metric_map(
                context.state.model.scene,
                sample.camera,
                metric_map,
                options=runtime.render_options,
            )
            weighted_l1 = (
                normalized_l1 * metric_map.to(normalized_l1.dtype)
            ).mean()
            importance_sum += attributed
            pruning_sum += weighted_l1 * attributed

        importance_score = torch.floor(importance_sum / float(len(probe_views)))
        pruning_score = self._normalize_score(pruning_sum)
        grad_count = self.grad_count.clamp_min(1.0)
        clone_grad = self.clone_grad_sum / grad_count
        split_grad = self.split_grad_sum / grad_count
        scene_extent = self.family_ops.scene_extent()
        current_scene = context.state.model.scene
        scales = torch.exp(current_scene.log_scales).max(dim=-1).values
        clone_mask = torch.logical_and(
            clone_grad >= self.grad_threshold,
            scales <= self.dense_fraction * scene_extent,
        )
        clone_mask = torch.logical_and(
            clone_mask,
            importance_score > self.importance_threshold,
        )
        split_mask = torch.logical_and(
            split_grad >= self.grad_abs_threshold,
            scales > self.dense_fraction * scene_extent,
        )
        split_mask = torch.logical_and(
            split_mask,
            importance_score > self.importance_threshold,
        )
        max_screen_radii = self.max_screen_radii
        prune_mask = (
            torch.sigmoid(current_scene.logit_opacity)
            < self.prune_opacity_threshold
        )
        if context.step > self.opacity_reset_every:
            prune_mask = torch.logical_or(
                prune_mask,
                max_screen_radii > 20.0,
            )
        prune_mask = torch.logical_or(
            prune_mask,
            pruning_score > 0.9,
        )
        world_scales = torch.exp(current_scene.log_scales).max(dim=-1).values
        prune_mask = torch.logical_or(
            prune_mask,
            world_scales > 0.1 * scene_extent,
        )
        keep_mask = ~prune_mask
        if not torch.all(keep_mask):
            self.family_ops.prune(keep_mask)
            clone_mask = clone_mask[keep_mask]
            split_mask = split_mask[keep_mask]

        added_count = 0
        if torch.any(clone_mask):
            added_count = int(clone_mask.sum().item())
            self.family_ops.clone(clone_mask)
        if torch.any(split_mask):
            padded_split_mask = split_mask
            if added_count > 0:
                padded_split_mask = torch.cat(
                    [
                        split_mask,
                        torch.zeros(
                            added_count,
                            dtype=torch.bool,
                            device=split_mask.device,
                        ),
                    ],
                    dim=0,
                )
            self.family_ops.split(padded_split_mask)

        self.family_ops.reset_opacity(0.8)
        self._reset_buffers(context.state.model.scene)

    def _require_render_output(
        self,
        context: DensificationContext,
    ) -> HasScreenSpaceDensificationSignals:
        return self.require_render_output_trait(
            context,
            HasScreenSpaceDensificationSignals,
        )

    def _ensure_buffers(
        self,
        *,
        num_splats: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        expected_shape = (num_splats,)
        if (
            self.clone_grad_sum is None
            or self.clone_grad_sum.shape != expected_shape
            or self.clone_grad_sum.device != device
            or self.clone_grad_sum.dtype != dtype
        ):
            self.clone_grad_sum = torch.zeros(
                expected_shape, device=device, dtype=dtype
            )
        if (
            self.split_grad_sum is None
            or self.split_grad_sum.shape != expected_shape
            or self.split_grad_sum.device != device
            or self.split_grad_sum.dtype != dtype
        ):
            self.split_grad_sum = torch.zeros(
                expected_shape, device=device, dtype=dtype
            )
        if (
            self.grad_count is None
            or self.grad_count.shape != expected_shape
            or self.grad_count.device != device
            or self.grad_count.dtype != dtype
        ):
            self.grad_count = torch.zeros(
                expected_shape, device=device, dtype=dtype
            )
        if (
            self.max_screen_radii is None
            or self.max_screen_radii.shape != expected_shape
            or self.max_screen_radii.device != device
            or self.max_screen_radii.dtype != dtype
        ):
            self.max_screen_radii = torch.zeros(
                expected_shape, device=device, dtype=dtype
            )

    def _reset_buffers(self, scene: Any) -> None:
        num_splats = int(scene.center_position.shape[0])
        device = scene.center_position.device
        dtype = scene.center_position.dtype
        self.clone_grad_sum = torch.zeros(
            num_splats, device=device, dtype=dtype
        )
        self.split_grad_sum = torch.zeros(
            num_splats, device=device, dtype=dtype
        )
        self.grad_count = torch.zeros(num_splats, device=device, dtype=dtype)
        self.max_screen_radii = torch.zeros(
            num_splats, device=device, dtype=dtype
        )

    def _should_refine(self, step: int) -> bool:
        if step <= self.start_iter or step > self.stop_iter:
            return False
        return step % self.refine_every == 0

    def _normalize_l1_map(self, l1_map: torch.Tensor) -> torch.Tensor:
        min_value = l1_map.min()
        max_value = l1_map.max()
        if torch.isclose(min_value, max_value):
            return torch.zeros_like(l1_map)
        return (l1_map - min_value) / (max_value - min_value)

    def _normalize_score(self, score: torch.Tensor) -> torch.Tensor:
        min_value = score.min()
        max_value = score.max()
        if torch.isclose(min_value, max_value):
            return torch.zeros_like(score)
        return (score - min_value) / (max_value - min_value)


@dataclass
class MCMC(ComposedDensificationMethod):
    """Lightweight MCMC-style refinement with opacity decay and jitter."""

    def __init__(
        self,
        *,
        jitter_every: int = 100,
        jitter_sigma: float = 1e-3,
        decay_every: int = 50,
        opacity_gamma: float = 0.99,
        prune_every: int = 100,
        prune_opacity_threshold: float = 0.005,
    ) -> None:
        super().__init__(
            name="MCMC",
            expected_scene_families=("gaussian",),
            collectors=[PositionGradientCollector()],
            passes=[
                GaussianJitterPass(
                    schedule=Schedule(frequency=jitter_every),
                    sigma=jitter_sigma,
                ),
                GaussianOpacityDecayPass(
                    schedule=Schedule(frequency=decay_every),
                    gamma=opacity_gamma,
                ),
                GaussianPruneOpacityPass(
                    schedule=Schedule(frequency=prune_every),
                    opacity_threshold=prune_opacity_threshold,
                ),
            ],
        )


@dataclass
class SVRasterAdaptive(ComposedDensificationMethod):
    """Sparse-voxel adaptive refinement using grid-gradient priorities."""

    def __init__(
        self,
        *,
        refine_every: int = 100,
        subdivide_threshold: float = 0.01,
        prune_density_threshold: float = 0.001,
    ) -> None:
        schedule = Schedule(frequency=refine_every)
        super().__init__(
            name="SVRasterAdaptive",
            expected_scene_families=("sparse_voxel",),
            collectors=[SparseVoxelGradientCollector()],
            passes=[
                SparseVoxelSubdividePass(
                    schedule=schedule,
                    priority_threshold=subdivide_threshold,
                ),
                SparseVoxelPrunePass(
                    schedule=schedule,
                    density_threshold=prune_density_threshold,
                ),
            ],
        )


__all__ = [
    "MCMC",
    "AbsGS",
    "ComposedDensificationMethod",
    "FastGS",
    "SVRasterAdaptive",
    "Vanilla3DGS",
    "compose_densification",
]
