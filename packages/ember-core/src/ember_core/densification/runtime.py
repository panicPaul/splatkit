"""Runtime helpers for densification."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ember_core.densification.contracts import (
    DensificationContext,
    DensificationMethod,
    DensificationRenderRequirements,
    DensificationRuntime,
)
from ember_core.densification.families import build_family_ops


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
    if config is None or config.builder is None:
        return None
    builder = _resolve_target(config.builder.target)
    if not callable(builder):
        raise TypeError(
            f"Densification builder {config.builder.target!r} is not callable."
        )
    method = builder(**config.builder.kwargs)
    if not hasattr(method, "get_render_requirements"):
        raise TypeError(
            "Densification builder must return a densification method."
        )
    return method


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


__all__ = [
    "bind_densification",
    "build_densification",
    "make_context",
    "merge_densification_requirements",
]
