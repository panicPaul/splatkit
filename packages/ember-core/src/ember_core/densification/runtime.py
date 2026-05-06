"""Runtime helpers for densification."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

from ember_core.densification.contracts import (
    BaseDensificationMethod,
    DensificationContext,
    DensificationLifecycleContext,
    DensificationMethod,
    DensificationRenderRequirements,
    DensificationRuntime,
)
from ember_core.densification.families import build_family_ops


class DensificationMethodSequence(BaseDensificationMethod):
    """Run multiple densification methods as one method."""

    def __init__(self, methods: list[DensificationMethod]) -> None:
        self.methods = methods
        self.expected_scene_families = tuple(
            family
            for method in methods
            for family in getattr(method, "expected_scene_families", ())
        )

    def get_render_requirements(
        self,
        state: object,
    ) -> DensificationRenderRequirements:
        requirements = DensificationRenderRequirements()
        for method in self.methods:
            requirements = requirements.merge(
                method.get_render_requirements(state)
            )
        return requirements

    def bind(
        self, state: Any, optimizers: Sequence[Any], family_ops: Any
    ) -> None:
        for method in self.methods:
            method.bind(state, optimizers, family_ops)

    def before_training(self, context: DensificationLifecycleContext) -> None:
        for method in self.methods:
            method.before_training(context)

    def pre_backward(self, context: DensificationContext) -> None:
        for method in self.methods:
            method.pre_backward(context)

    def post_backward(self, context: DensificationContext) -> None:
        for method in self.methods:
            method.post_backward(context)

    def post_optimizer_step(self, context: DensificationContext) -> None:
        for method in self.methods:
            method.post_optimizer_step(context)

    def after_step(
        self,
        context: DensificationContext,
        metrics: dict[str, float],
    ) -> None:
        for method in self.methods:
            method.after_step(context, metrics)

    def after_training(self, context: DensificationLifecycleContext) -> None:
        for method in self.methods:
            method.after_training(context)


def _resolve_target(target: str) -> Any:
    module_name, _, attr_path = target.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError(
            f"Target {target!r} must use 'package.module.symbol' syntax."
        )
    module = import_module(module_name)
    value: Any = module
    for attribute in attr_path.split("."):
        value = getattr(value, attribute)
    return value


def build_densification(
    config: Any | None,
) -> DensificationMethod | None:
    """Instantiate an unbound densification method."""
    if config is None:
        return None
    methods: list[DensificationMethod] = list(getattr(config, "methods", ()))
    if not methods and not config.builders:
        return None
    for builder_spec in config.builders:
        if getattr(builder_spec, "context_kwargs", None):
            raise ValueError(
                "Densification builder "
                f"{builder_spec.target!r} has runtime context bindings; use "
                "build_densification_for_context instead."
            )
        builder = getattr(builder_spec, "object_ref", None)
        if builder is None:
            builder = _resolve_target(builder_spec.target)
        if not callable(builder):
            raise TypeError(
                f"Densification builder {builder_spec.target!r} is not "
                "callable."
            )
        method = builder(**builder_spec.kwargs)
        if not hasattr(method, "get_render_requirements"):
            raise TypeError(
                "Densification builder must return a densification method."
            )
        methods.append(method)
    if len(methods) == 1:
        return methods[0]
    return DensificationMethodSequence(methods)


def bind_densification(
    method: DensificationMethod | None,
    state: Any,
    optimizers: list[Any],
) -> DensificationMethod | None:
    """Bind family ops and mutable training state to a densification method."""
    if method is None:
        return None
    method.bind(state, optimizers, build_family_ops(state, optimizers))
    return method


def merge_densification_requirements(
    config: Any,
    requirements: DensificationRenderRequirements | None,
) -> Any:
    """Merge densification render requirements into training config."""
    if requirements is None:
        return config
    for attr_name in (
        "return_alpha",
        "return_depth",
        "return_gaussian_impact_score",
        "return_normals",
        "return_2d_projections",
        "return_projective_intersection_transforms",
    ):
        if getattr(requirements, attr_name):
            setattr(config.render, attr_name, True)
    for name, value in requirements.backend_options.items():
        existing = config.render.backend_options.get(name)
        if existing is not None and existing != value:
            raise ValueError(
                f"Densification requires render backend option {name!r}="
                f"{value!r}, but config already sets it to {existing!r}."
            )
        config.render.backend_options[name] = value
    return config


def make_context(
    *,
    state: Any,
    batch: Any,
    render_output: Any,
    loss_result: Any,
    optimizers: list[Any],
    runtime: DensificationRuntime | None = None,
) -> DensificationContext:
    """Create a densification context for the current step."""
    return DensificationContext(
        state=state,
        batch=batch,
        render_output=render_output,
        loss_result=loss_result,
        step=state.step,
        optimizers=optimizers,
        runtime=runtime,
    )


def make_lifecycle_context(
    *,
    state: Any,
    optimizers: list[Any],
    runtime: DensificationRuntime | None = None,
) -> DensificationLifecycleContext:
    """Create a densification context for training-level lifecycle hooks."""
    return DensificationLifecycleContext(
        state=state,
        optimizers=optimizers,
        runtime=runtime,
    )


__all__ = [
    "DensificationMethodSequence",
    "bind_densification",
    "build_densification",
    "make_context",
    "make_lifecycle_context",
    "merge_densification_requirements",
]
