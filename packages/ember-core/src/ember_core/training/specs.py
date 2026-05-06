"""Notebook-friendly builders for declarative training specs."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ember_core.training.config import (
    CallableSpec,
    DensificationConfig,
    HookConfig,
    LossConfig,
    OptimizationConfig,
    ParameterGroupConfig,
    ParameterTargetSpec,
    TensorSliceSpec,
    TensorViewSpec,
)


@dataclass(frozen=True)
class RuntimeContextRef:
    """Reference to a runtime value available while materializing config."""

    path: str


@dataclass(frozen=True)
class _RunContextRefs:
    frame_dataset: RuntimeContextRef = RuntimeContextRef("frame_dataset")
    camera_extent: RuntimeContextRef = RuntimeContextRef("camera_extent")
    max_steps: RuntimeContextRef = RuntimeContextRef("max_steps")
    backend: RuntimeContextRef = RuntimeContextRef("backend")
    device: RuntimeContextRef = RuntimeContextRef("device")


@dataclass(frozen=True)
class _ContextRefs:
    run: _RunContextRefs = _RunContextRefs()


ctx = _ContextRefs()


def _target_name(target: str | Callable[..., Any]) -> str:
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", "")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", ""))
    return f"{module}.{qualname}" if module and qualname else repr(target)


def _validate_callable_signature(
    target: Callable[..., Any],
    kwargs: Mapping[str, Any],
    context_kwargs: Mapping[str, str],
) -> None:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return
    parameters = signature.parameters
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return
    names = set(kwargs) | set(context_kwargs)
    accepted = {
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    unsupported = sorted(names - accepted)
    if unsupported:
        unsupported_list = ", ".join(repr(name) for name in unsupported)
        raise ValueError(
            f"Callable target {_target_name(target)!r} does not accept "
            f"declared kwarg(s): {unsupported_list}."
        )


def callable_spec(
    target: str | Callable[..., Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
    context_kwargs: Mapping[str, str] | None = None,
    **inline_kwargs: Any,
) -> CallableSpec:
    """Create a callable spec from an import string or notebook-local callable."""
    resolved_kwargs = dict(kwargs or {})
    resolved_kwargs.update(inline_kwargs)
    return CallableSpec(
        target=_target_name(target),
        object_ref=None if isinstance(target, str) else target,
        kwargs=resolved_kwargs,
        context_kwargs=dict(context_kwargs or {}),
    )


def bound_callable(
    target: str | Callable[..., Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
    bind: Mapping[str, RuntimeContextRef] | None = None,
    **inline_kwargs: Any,
) -> CallableSpec:
    """Create a callable spec with explicit runtime context bindings."""
    resolved_kwargs = dict(kwargs or {})
    duplicate_static = sorted(set(resolved_kwargs) & set(inline_kwargs))
    if duplicate_static:
        duplicate_list = ", ".join(repr(name) for name in duplicate_static)
        raise ValueError(
            "Duplicate static callable kwarg(s) supplied in both kwargs and "
            f"inline kwargs: {duplicate_list}."
        )
    resolved_kwargs.update(inline_kwargs)

    context_kwargs: dict[str, str] = {}
    for kwarg_name, ref in dict(bind or {}).items():
        if not isinstance(ref, RuntimeContextRef):
            raise TypeError(
                "bound_callable bind values must be RuntimeContextRef "
                f"objects, got {type(ref).__name__} for {kwarg_name!r}."
            )
        context_kwargs[kwarg_name] = ref.path

    duplicate_bound = sorted(set(resolved_kwargs) & set(context_kwargs))
    if duplicate_bound:
        duplicate_list = ", ".join(repr(name) for name in duplicate_bound)
        raise ValueError(
            "Callable kwarg(s) cannot be supplied both statically and from "
            f"runtime context: {duplicate_list}."
        )

    if not isinstance(target, str):
        _validate_callable_signature(target, resolved_kwargs, context_kwargs)

    return callable_spec(
        target,
        kwargs=resolved_kwargs,
        context_kwargs=context_kwargs,
    )


def tensor_slice(
    axis: int,
    *,
    start: int | None = None,
    stop: int | None = None,
) -> TensorSliceSpec:
    """Declare one contiguous tensor slice for optimizer parameter views."""
    return TensorSliceSpec(axis=axis, start=start, stop=stop)


def tensor_view(*slices: TensorSliceSpec) -> TensorViewSpec:
    """Declare a structured tensor view for one optimizer target."""
    return TensorViewSpec(slices=tuple(slices))


def parameter_target(
    scope: Literal["scene", "modules", "parameters"],
    name: str,
    *,
    view: TensorViewSpec | None = None,
) -> ParameterTargetSpec:
    """Declare a trainable parameter target."""
    return ParameterTargetSpec(scope=scope, name=name, view=view)


def parameter_group(
    scope: Literal["scene", "modules", "parameters"],
    name: str,
    *,
    lr: float,
    optimizer: str = "adam",
    view: TensorViewSpec | None = None,
    scheduler: str | Callable[..., Any] | CallableSpec | None = None,
    **optimizer_settings: Any,
) -> ParameterGroupConfig:
    """Declare optimizer settings for a scene, module, or standalone parameter."""
    scheduler_spec = (
        scheduler
        if isinstance(scheduler, CallableSpec) or scheduler is None
        else callable_spec(scheduler)
    )
    known_settings = {
        "weight_decay": optimizer_settings.pop("weight_decay", 0.0),
        "betas": optimizer_settings.pop("betas", (0.9, 0.999)),
        "momentum": optimizer_settings.pop("momentum", 0.0),
    }
    return ParameterGroupConfig(
        target=parameter_target(scope, name, view=view),
        optimizer=optimizer,
        lr=lr,
        scheduler=scheduler_spec,
        optimizer_kwargs=optimizer_settings,
        **known_settings,
    )


def scene_parameter(
    name: str,
    *,
    lr: float,
    **settings: Any,
) -> ParameterGroupConfig:
    """Declare optimizer settings for a Gaussian or sparse-voxel scene field."""
    return parameter_group("scene", name, lr=lr, **settings)


def module_parameter(
    name: str,
    *,
    lr: float,
    **settings: Any,
) -> ParameterGroupConfig:
    """Declare optimizer settings for all parameters in a named module."""
    return parameter_group("modules", name, lr=lr, **settings)


def standalone_parameter(
    name: str,
    *,
    lr: float,
    **settings: Any,
) -> ParameterGroupConfig:
    """Declare optimizer settings for a standalone named parameter."""
    return parameter_group("parameters", name, lr=lr, **settings)


def optimization_config(
    *parameter_groups: ParameterGroupConfig,
    builder: str | Callable[..., Any] | CallableSpec | None = None,
    **builder_kwargs: Any,
) -> OptimizationConfig:
    """Declare either explicit parameter groups or a config builder."""
    builder_spec = (
        builder
        if isinstance(builder, CallableSpec) or builder is None
        else callable_spec(builder, **builder_kwargs)
    )
    return OptimizationConfig(
        builder=builder_spec,
        parameter_groups=list(parameter_groups),
    )


def loss_config(
    target: str | Callable[..., Any] | CallableSpec,
    *,
    weights: Mapping[str, float] | None = None,
    **kwargs: Any,
) -> LossConfig:
    """Declare a training loss callable."""
    return LossConfig(
        target=target
        if isinstance(target, CallableSpec)
        else callable_spec(target, **kwargs),
        weights=dict(weights or {}),
    )


def hooks_config(
    *builders: str | Callable[..., Any] | CallableSpec,
) -> HookConfig:
    """Declare training hook builders."""
    return HookConfig(
        builders=[
            builder
            if isinstance(builder, CallableSpec)
            else callable_spec(builder)
            for builder in builders
        ]
    )


def densification_config(
    *methods_or_builders: Any,
) -> DensificationConfig:
    """Declare densification from method instances or builder callables."""
    methods: list[Any] = []
    builders: list[CallableSpec] = []
    for item in methods_or_builders:
        if isinstance(item, CallableSpec):
            builders.append(item)
        elif hasattr(item, "get_render_requirements"):
            methods.append(item)
        elif callable(item):
            builders.append(callable_spec(item))
        else:
            raise TypeError(
                "Densification entries must be method instances, builders, "
                f"or CallableSpec objects; got {type(item).__name__}."
            )
    return DensificationConfig(methods=methods, builders=builders)


__all__ = [
    "RuntimeContextRef",
    "bound_callable",
    "callable_spec",
    "ctx",
    "densification_config",
    "hooks_config",
    "loss_config",
    "module_parameter",
    "optimization_config",
    "parameter_group",
    "parameter_target",
    "scene_parameter",
    "standalone_parameter",
    "tensor_slice",
    "tensor_view",
]
