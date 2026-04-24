"""Core densification contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from jaxtyping import Float
from torch import Tensor

T = TypeVar("T")


@dataclass(frozen=True)
class Schedule:
    """Simple step schedule for densification actions."""

    start_iteration: int = 0
    end_iteration: int = -1
    frequency: int = 1

    def includes(self, step: int) -> bool:
        """Return whether the schedule should trigger for a step."""
        if step < self.start_iteration:
            return False
        if self.end_iteration >= 0 and step > self.end_iteration:
            return False
        if self.frequency <= 0:
            return False
        return (step - self.start_iteration) % self.frequency == 0


@dataclass(frozen=True)
class DensificationRenderRequirements:
    """Extra render requirements requested by densification."""

    return_alpha: bool = False
    return_depth: bool = False
    return_gaussian_impact_score: bool = False
    return_normals: bool = False
    return_2d_projections: bool = False
    return_projective_intersection_transforms: bool = False
    backend_options: dict[str, Any] = field(default_factory=dict)

    def merge(
        self,
        other: DensificationRenderRequirements,
    ) -> DensificationRenderRequirements:
        """Merge two requirement objects."""
        merged_options = dict(self.backend_options)
        for name, value in other.backend_options.items():
            if name in merged_options and merged_options[name] != value:
                raise ValueError(
                    f"Conflicting densification backend option {name!r}: "
                    f"{merged_options[name]!r} vs {value!r}."
                )
            merged_options[name] = value
        return DensificationRenderRequirements(
            return_alpha=self.return_alpha or other.return_alpha,
            return_depth=self.return_depth or other.return_depth,
            return_gaussian_impact_score=(
                self.return_gaussian_impact_score
                or other.return_gaussian_impact_score
            ),
            return_normals=self.return_normals or other.return_normals,
            return_2d_projections=(
                self.return_2d_projections or other.return_2d_projections
            ),
            return_projective_intersection_transforms=(
                self.return_projective_intersection_transforms
                or other.return_projective_intersection_transforms
            ),
            backend_options=merged_options,
        )


@dataclass(frozen=True)
class DensificationContext:
    """Context passed to densification lifecycle stages."""

    state: Any
    batch: Any
    render_output: Any
    loss_result: Any
    step: int
    optimizers: Sequence[Any]
    runtime: DensificationRuntime | None = None


@dataclass
class DensificationSignals:
    """Mutable signal buffers shared across collectors and passes."""

    local: dict[str, Any] = field(default_factory=dict)
    global_: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GaussianMetricAttribution(Protocol):
    """Backend-provided attribution from probe metrics to Gaussians."""

    def attribute_metric_map(
        self,
        scene: Any,
        camera: Any,
        metric_map: Any,
        *,
        options: Any | None = None,
    ) -> Float[Tensor, " num_splats"]:
        """Attribute a probe metric map to per-Gaussian weights."""


@runtime_checkable
class DensificationRuntime(Protocol):
    """Runtime services available to densification methods."""

    backend_name: str
    render_options: Any

    def sample_views(self, count: int) -> tuple[Any, ...]:
        """Sample prepared probe views from the active training dataset."""

    def render_raw(self, model: Any, camera: Any) -> Any:
        """Render through the active backend before postprocessing."""

    def resolve_trait(self, trait_type: type[T]) -> T:
        """Resolve a runtime-checkable backend trait provider."""


class DensificationCollector(Protocol):
    """Protocol for signal collectors."""

    def get_render_requirements(self) -> DensificationRenderRequirements:
        """Return extra render requirements."""

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Bind stateful resources for training."""

    def pre_backward(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        """Run before backward."""

    def post_backward(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        """Run after backward."""

    def post_optimizer_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        """Run after optimizer step."""

    def after_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
        metrics: dict[str, float],
    ) -> None:
        """Run after metrics update."""


class DensificationPass(DensificationCollector, Protocol):
    """Protocol for refinement passes."""


class DensificationMethod(Protocol):
    """Protocol for top-level densification methods."""

    expected_scene_families: tuple[str, ...]

    def get_render_requirements(self) -> DensificationRenderRequirements:
        """Return extra render requirements."""

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Bind stateful resources for training."""

    def pre_backward(self, context: DensificationContext) -> None:
        """Run before backward."""

    def post_backward(self, context: DensificationContext) -> None:
        """Run after backward."""

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Run after optimizer step."""

    def after_step(
        self,
        context: DensificationContext,
        metrics: dict[str, float],
    ) -> None:
        """Run after metrics update."""


class BaseDensificationComponent:
    """Convenience base class with no-op lifecycle hooks."""

    def get_render_requirements(self) -> DensificationRenderRequirements:
        return DensificationRenderRequirements()

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        del state, optimizers, family_ops

    def pre_backward(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        del context, signals

    def post_backward(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        del context, signals

    def post_optimizer_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
    ) -> None:
        del context, signals

    def after_step(
        self,
        context: DensificationContext,
        signals: DensificationSignals,
        metrics: dict[str, float],
    ) -> None:
        del context, signals, metrics


class BaseDensificationMethod:
    """Convenience base class with no-op lifecycle hooks for full methods."""

    expected_scene_families: tuple[str, ...] = ()

    def get_render_requirements(self) -> DensificationRenderRequirements:
        return DensificationRenderRequirements()

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        del state, optimizers, family_ops

    def pre_backward(self, context: DensificationContext) -> None:
        del context

    def post_backward(self, context: DensificationContext) -> None:
        del context

    def post_optimizer_step(self, context: DensificationContext) -> None:
        del context

    def after_step(
        self,
        context: DensificationContext,
        metrics: dict[str, float],
    ) -> None:
        del context, metrics

    def require_runtime_trait(
        self,
        context: DensificationContext,
        trait_type: type[T],
    ) -> T:
        """Resolve a required runtime trait with a precise error."""
        if context.runtime is None:
            raise RuntimeError(
                f"{type(self).__name__} requires densification runtime "
                f"trait {trait_type.__name__}, but no densification runtime "
                "was provided."
            )
        return context.runtime.resolve_trait(trait_type)

    def require_render_output_trait(
        self,
        context: DensificationContext,
        trait_type: type[T],
    ) -> T:
        """Validate that the current render output satisfies a trait."""
        try:
            if isinstance(context.render_output, trait_type):
                return context.render_output
        except TypeError as exc:
            raise TypeError(
                "Render-output trait checks require a runtime-checkable "
                f"trait protocol or concrete type, got {trait_type!r}."
            ) from exc
        raise TypeError(
            f"{type(self).__name__} requires render outputs satisfying "
            f"{trait_type.__name__}."
        )


__all__ = [
    "BaseDensificationMethod",
    "BaseDensificationComponent",
    "DensificationRuntime",
    "DensificationCollector",
    "DensificationContext",
    "DensificationMethod",
    "DensificationPass",
    "DensificationRenderRequirements",
    "DensificationSignals",
    "GaussianMetricAttribution",
    "Schedule",
]
