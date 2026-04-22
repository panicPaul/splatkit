"""Built-in densification methods and composition helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from splatkit.densification.collectors import (
    ImagePlaneGradientCollector,
    PositionGradientCollector,
    SparseVoxelGradientCollector,
)
from splatkit.densification.contracts import (
    DensificationContext,
    DensificationRenderRequirements,
    DensificationSignals,
    Schedule,
)
from splatkit.densification.passes import (
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
    signals: DensificationSignals = field(default_factory=DensificationSignals, init=False)

    def get_render_requirements(self) -> DensificationRenderRequirements:
        requirements = DensificationRenderRequirements()
        for component in [*self.collectors, *self.passes]:
            requirements = requirements.merge(component.get_render_requirements())
        return requirements

    def bind(
        self,
        state: Any,
        optimizers: list[Any],
        family_ops: Any,
    ) -> None:
        family = state.model.scene.scene_family
        if self.expected_scene_families and family not in self.expected_scene_families:
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
            collectors=[ImagePlaneGradientCollector(use_absolute_gradients=False)],
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
        relative_size_threshold = float(kwargs.pop("relative_size_threshold", 0.01))
        prune_opacity_threshold = float(kwargs.pop("prune_opacity_threshold", 0.005))
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
            collectors=[ImagePlaneGradientCollector(use_absolute_gradients=True)],
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
class FastGS(ComposedDensificationMethod):
    """FastGS-style starting point using absolute image-plane gradients."""

    def __init__(self, **kwargs: Any) -> None:
        refine_every = int(kwargs.pop("refine_every", 400))
        start_iter = int(kwargs.pop("start_iter", 500))
        stop_iter = int(kwargs.pop("stop_iter", 15_000))
        grad_threshold = float(kwargs.pop("grad_threshold", 2e-4))
        relative_size_threshold = float(kwargs.pop("relative_size_threshold", 0.01))
        prune_opacity_threshold = float(kwargs.pop("prune_opacity_threshold", 0.005))
        schedule = Schedule(
            start_iteration=start_iter,
            end_iteration=stop_iter,
            frequency=refine_every,
        )
        super().__init__(
            name="FastGS",
            expected_scene_families=("gaussian",),
            collectors=[
                ImagePlaneGradientCollector(use_absolute_gradients=True),
                PositionGradientCollector(),
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
            ],
        )


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
