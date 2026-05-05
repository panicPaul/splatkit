"""Callable and import-target resolution for training configs."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from importlib import import_module
from typing import Any

from ember_core.training.config import CallableSpec
from ember_core.training.protocols import TrainingRunContext


def resolve_target(target: str) -> Any:
    """Resolve an importable target from ``module.symbol`` syntax."""
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


def _resolve_context_value(
    context: TrainingRunContext,
    path: str,
) -> Any:
    value: Any = context
    for name in path.split("."):
        value = getattr(value, name)
    return value


def callable_kwargs(
    spec: CallableSpec,
    context: TrainingRunContext | None = None,
) -> dict[str, Any]:
    """Resolve static and runtime-bound kwargs for a callable spec."""
    kwargs = dict(spec.kwargs)
    if not spec.context_kwargs:
        return kwargs
    if context is None:
        raise ValueError(
            f"CallableSpec {spec.target!r} requires a TrainingRunContext."
        )
    for kwarg_name, context_path in spec.context_kwargs.items():
        kwargs[kwarg_name] = _resolve_context_value(context, context_path)
    return kwargs


def resolve_callable_target(spec: CallableSpec) -> Callable[..., Any]:
    """Resolve a callable spec to its underlying callable."""
    value = (
        spec.object_ref
        if spec.object_ref is not None
        else resolve_target(spec.target)
    )
    if not callable(value):
        raise TypeError(f"Resolved target {spec.target!r} is not callable.")
    return value


def resolve_callable(
    spec: CallableSpec | None,
    *,
    context: TrainingRunContext | None = None,
) -> Callable[..., Any] | None:
    """Resolve a callable spec and bind its kwargs."""
    if spec is None:
        return None
    value = resolve_callable_target(spec)
    return partial(value, **callable_kwargs(spec, context))


def instantiate_callable(
    spec: CallableSpec,
    *,
    context: TrainingRunContext | None = None,
) -> Any:
    """Instantiate a class or builder function from a callable spec."""
    value = resolve_callable_target(spec)
    return value(**callable_kwargs(spec, context))


__all__ = [
    "callable_kwargs",
    "instantiate_callable",
    "resolve_callable",
    "resolve_callable_target",
    "resolve_target",
]
