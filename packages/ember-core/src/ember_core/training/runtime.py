"""Declarative training builders and loop."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
from importlib import import_module
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ember_core.core.registry import (
    BACKEND_REGISTRY,
    render,
    resolve_backend_trait,
)
from ember_core.data.adapters import PreparedFrameDataset, collate_frame_samples
from ember_core.data.contracts import (
    PreparedFrameBatch,
    PreparedFrameSample,
    SceneRecord,
)
from ember_core.densification.contracts import DensificationRuntime
from ember_core.densification.runtime import (
    bind_densification,
    build_densification,
    make_context,
    merge_densification_requirements,
)
from ember_core.initialization import InitializedModel
from ember_core.training.config import (
    CallableSpec,
    ParameterGroupConfig,
    ParameterTargetSpec,
    ParameterSpec,
    TensorViewSpec,
    TrainingConfig,
)
from ember_core.training.protocols import (
    LossFn,
    LossResult,
    RenderFn,
    TrainingHook,
    TrainingResult,
    TrainState,
)


class OptimizerBinding:
    """Optimizer bound to one declared parameter target."""

    def __init__(
        self,
        *,
        target: ParameterTargetSpec,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler | None = None,
        base_parameter: torch.Tensor | None = None,
        field_name: str | None = None,
        view: _BindingView | None = None,
    ) -> None:
        self.target = target
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.base_parameter = base_parameter
        self.field_name = field_name
        self.view = view
        if (
            self.view is None
            and self.target.view is not None
            and self.base_parameter is not None
        ):
            self.view = _BindingView(
                self.target_path,
                self.target.view,
                self.base_parameter,
            )

    @property
    def target_path(self) -> str:
        return f"{self.target.scope}.{self.target.name}"

    def matches_target(self, scope: str, name: str) -> bool:
        return self.target.scope == scope and self.target.name == name

    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        snapshot = self._prepare_view_step()
        self.optimizer.step()
        self._restore_view_step(snapshot)
        if self.scheduler is not None:
            self.scheduler.step()

    def replace_parameter(
        self,
        new_parameter: torch.Tensor,
        transform: Callable[[str, torch.Tensor], torch.Tensor],
    ) -> None:
        if self.base_parameter is None:
            return
        group = self.optimizer.param_groups[0]
        old_parameter = group["params"][0]
        state = self.optimizer.state.pop(old_parameter, {})
        group["params"] = [new_parameter]
        new_state: dict[Any, Any] = {}
        for key, old_value in state.items():
            if key == "step":
                new_state[key] = old_value
            elif isinstance(old_value, torch.Tensor):
                new_state[key] = transform(key, old_value)
            else:
                new_state[key] = old_value
        self.optimizer.state[new_parameter] = new_state
        self.base_parameter = new_parameter

    def reset_state_for_indices(self, indices: torch.Tensor) -> None:
        if self.base_parameter is None:
            return
        state = self.optimizer.state.get(self.base_parameter, {})
        for key, value in state.items():
            if key == "step" or not isinstance(value, torch.Tensor):
                continue
            if value.ndim == 0:
                continue
            if self.view is None:
                value[indices] = 0
                continue
            self.view.zero_state_indices(value, indices)

    def _prepare_view_step(self) -> _ViewStepSnapshot | None:
        if self.view is None:
            return None
        if self.base_parameter is None:
            raise RuntimeError(
                f"View-backed optimizer binding {self.target_path!r} has no "
                "base parameter."
            )
        grad_snapshot: torch.Tensor | None = None
        if self.base_parameter.grad is not None:
            if self.base_parameter.grad.is_sparse:
                raise RuntimeError(
                    f"View-backed optimizer binding {self.target_path!r} does "
                    "not support sparse gradients."
                )
            grad_snapshot = self.base_parameter.grad.detach().clone()
            mask = self.view.mask_for_tensor(self.base_parameter)
            self.base_parameter.grad = self.base_parameter.grad * mask
        else:
            mask = self.view.mask_for_tensor(self.base_parameter)
        state = self.optimizer.state.get(self.base_parameter, {})
        state_snapshots: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "step" or value.ndim == 0:
                continue
            if tuple(value.shape) != tuple(self.base_parameter.shape):
                raise RuntimeError(
                    f"View-backed optimizer binding {self.target_path!r} does "
                    f"not support optimizer state {key!r} with shape "
                    f"{tuple(value.shape)!r}; expected scalar or "
                    f"{tuple(self.base_parameter.shape)!r}."
                )
            state_snapshots[key] = value.detach().clone()
        return _ViewStepSnapshot(
            parameter=self.base_parameter.detach().clone(),
            state=state_snapshots,
            mask=mask,
            grad=grad_snapshot,
        )

    def _restore_view_step(self, snapshot: _ViewStepSnapshot | None) -> None:
        if snapshot is None or self.base_parameter is None:
            return
        mask = snapshot.mask
        self.base_parameter.data.copy_(
            torch.where(mask, self.base_parameter.data, snapshot.parameter)
        )
        state = self.optimizer.state.get(self.base_parameter, {})
        for key, current in state.items():
            if not isinstance(current, torch.Tensor):
                continue
            if key == "step" or current.ndim == 0:
                continue
            if tuple(current.shape) != tuple(self.base_parameter.shape):
                raise RuntimeError(
                    f"View-backed optimizer binding {self.target_path!r} does "
                    f"not support optimizer state {key!r} with shape "
                    f"{tuple(current.shape)!r}; expected scalar or "
                    f"{tuple(self.base_parameter.shape)!r}."
                )
            previous = snapshot.state.get(key)
            if previous is None:
                previous = torch.zeros_like(current)
            current.copy_(torch.where(mask, current, previous))
        if snapshot.grad is None:
            self.base_parameter.grad = None
        else:
            self.base_parameter.grad = snapshot.grad


@dataclass(frozen=True)
class _ViewStepSnapshot:
    parameter: torch.Tensor
    state: dict[str, torch.Tensor]
    mask: torch.Tensor
    grad: torch.Tensor | None


@dataclass(frozen=True)
class _ResolvedTarget:
    target: ParameterTargetSpec
    parameters: list[torch.Tensor]
    base_parameter: torch.Tensor | None = None
    field_name: str | None = None


class _BindingView:
    """Compiled contiguous tensor view for one optimizer binding."""

    def __init__(
        self,
        target_path: str,
        view: TensorViewSpec,
        tensor: torch.Tensor,
    ) -> None:
        self._target_path = target_path
        self._axes: dict[int, tuple[int, int | None]] = {}
        for slice_spec in view.slices:
            axis = slice_spec.axis
            if axis >= tensor.ndim:
                raise ValueError(
                    f"Target {target_path!r} cannot slice axis {axis}; "
                    f"tensor rank is {tensor.ndim}."
                )
            start = 0 if slice_spec.start is None else slice_spec.start
            stop = slice_spec.stop
            size = int(tensor.shape[axis])
            if start < 0 or start > size:
                raise ValueError(
                    f"Target {target_path!r} has invalid slice start "
                    f"{slice_spec.start!r} for axis {axis}."
                )
            if stop is not None and (stop < start or stop > size):
                raise ValueError(
                    f"Target {target_path!r} has invalid slice stop "
                    f"{slice_spec.stop!r} for axis {axis}."
                )
            self._axes[axis] = (start, stop)

    def overlaps(self, other: _BindingView | None, shape: torch.Size) -> bool:
        if other is None:
            return True
        for axis, size in enumerate(shape):
            start_a, stop_a = self._bounds(axis, int(size))
            start_b, stop_b = other._bounds(axis, int(size))
            if stop_a <= start_b or stop_b <= start_a:
                return False
        return True

    def mask_for_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask[self._tensor_slices(tensor.shape)] = True
        return mask

    def zero_state_indices(
        self,
        state: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        selected = torch.zeros_like(state, dtype=torch.bool)
        selected[indices] = True
        state[selected & self.mask_for_tensor(state)] = 0

    def _tensor_slices(self, shape: torch.Size) -> tuple[slice, ...]:
        return tuple(
            slice(*self._bounds(axis, int(size)))
            for axis, size in enumerate(shape)
        )

    def _bounds(self, axis: int, size: int) -> tuple[int, int]:
        start, stop = self._axes.get(axis, (0, None))
        return start, size if stop is None else stop


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


def resolve_callable(spec: CallableSpec | None) -> Callable[..., Any] | None:
    """Resolve a callable spec and bind its kwargs."""
    if spec is None:
        return None
    value = resolve_target(spec.target)
    if not callable(value):
        raise TypeError(f"Resolved target {spec.target!r} is not callable.")
    return partial(value, **spec.kwargs)


def instantiate_callable(spec: CallableSpec) -> Any:
    """Instantiate a class or builder function from a callable spec."""
    value = resolve_target(spec.target)
    if not callable(value):
        raise TypeError(f"Resolved target {spec.target!r} is not callable.")
    return value(**spec.kwargs)


def build_modules(config: TrainingConfig) -> dict[str, nn.Module]:
    """Instantiate configured auxiliary modules."""
    modules: dict[str, nn.Module] = {}
    for name, builder in config.model.modules.items():
        module = instantiate_callable(builder)
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"Module builder {builder.target!r} returned "
                f"{type(module).__name__}, expected nn.Module."
            )
        modules[name] = module
    return modules


def _build_parameter(spec: ParameterSpec) -> nn.Parameter:
    tensor = torch.empty(spec.shape, dtype=torch.float32)
    if spec.init == "zeros":
        tensor.zero_()
    elif spec.init == "ones":
        tensor.fill_(1.0)
    elif spec.init == "constant":
        tensor.fill_(spec.value)
    else:
        tensor.normal_(mean=spec.mean, std=spec.std)
    return nn.Parameter(tensor, requires_grad=spec.requires_grad)


def build_parameters(config: TrainingConfig) -> dict[str, nn.Parameter]:
    """Instantiate configured standalone parameters."""
    return {
        name: _build_parameter(spec)
        for name, spec in config.model.parameters.items()
    }


def initialize_model(
    scene_record: SceneRecord,
    config: TrainingConfig,
) -> InitializedModel:
    """Run the configured initializer."""
    initializer = resolve_callable(config.initialization.initializer)
    assert initializer is not None
    model = initializer(
        scene_record,
        modules=build_modules(config),
        parameters=build_parameters(config),
    )
    if not isinstance(model, InitializedModel):
        raise TypeError(
            "Initializer must return InitializedModel, got "
            f"{type(model).__name__}."
        )
    return model


def build_dataloader(
    frame_dataset: PreparedFrameDataset,
    config: TrainingConfig,
) -> DataLoader[PreparedFrameBatch]:
    """Build the camera-batched dataloader."""
    return _build_dataloader_from_frame_dataset(frame_dataset, config)


def _build_dataloader_from_frame_dataset(
    frame_dataset: PreparedFrameDataset,
    config: TrainingConfig,
) -> DataLoader[PreparedFrameBatch]:
    return DataLoader(
        frame_dataset,
        batch_size=config.batching.batch_size,
        shuffle=config.batching.shuffle,
        collate_fn=collate_frame_samples,
    )


def _build_backend_options(config: TrainingConfig) -> Any:
    backend = BACKEND_REGISTRY.get(config.render.backend)
    if backend is None:
        raise ValueError(
            f"Backend {config.render.backend!r} is not registered."
        )
    default_options = backend.default_options
    if not is_dataclass(default_options):
        raise TypeError(
            "Registered backend default options must be dataclass instances."
        )
    valid_option_names = {field.name for field in fields(default_options)}
    unknown_option_names = sorted(
        set(config.render.backend_options) - valid_option_names
    )
    if unknown_option_names:
        raise ValueError(
            f"Backend {config.render.backend!r} does not support render "
            f"options {unknown_option_names!r}."
        )
    resolved_options: dict[str, Any] = {}
    for field in fields(default_options):
        if field.name not in config.render.backend_options:
            continue
        value = config.render.backend_options[field.name]
        default_value = getattr(default_options, field.name)
        if isinstance(default_value, torch.Tensor) and not isinstance(
            value,
            torch.Tensor,
        ):
            value = torch.as_tensor(
                value,
                dtype=default_value.dtype,
                device=default_value.device,
            )
        resolved_options[field.name] = value
    return replace(default_options, **resolved_options)


def _validate_requested_outputs(config: TrainingConfig) -> None:
    backend = BACKEND_REGISTRY.get(config.render.backend)
    if backend is None:
        raise ValueError(
            f"Backend {config.render.backend!r} is not registered."
        )
    requested_outputs = {
        name
        for name, enabled in (
            ("alpha", config.render.return_alpha),
            ("depth", config.render.return_depth),
            ("gaussian_impact_score", config.render.return_gaussian_impact_score),
            ("normals", config.render.return_normals),
            ("2d_projections", config.render.return_2d_projections),
            (
                "projective_intersection_transforms",
                config.render.return_projective_intersection_transforms,
            ),
        )
        if enabled
    }
    unsupported_outputs = sorted(
        requested_outputs - backend.supported_outputs - {"alpha"}
    )
    if unsupported_outputs:
        raise ValueError(
            f"Backend {config.render.backend!r} does not support outputs "
            f"{unsupported_outputs!r}."
        )


def build_raw_render_fn(config: TrainingConfig) -> RenderFn:
    """Build the stateless render pipeline before postprocessing."""
    _validate_requested_outputs(config)
    feature_fn = resolve_callable(config.render.feature_fn)
    options = _build_backend_options(config)

    def raw_render_fn(model: InitializedModel, camera: Any) -> Any:
        resolved_camera = camera
        if not hasattr(resolved_camera, "cam_to_world"):
            raise TypeError("Render camera must provide cam_to_world.")
        scene = (
            model.scene
            if feature_fn is None
            else feature_fn(model, resolved_camera)
        )
        render_output = render(
            scene,
            resolved_camera,
            backend=config.render.backend,
            return_alpha=config.render.return_alpha,
            return_depth=config.render.return_depth,
            return_gaussian_impact_score=(
                config.render.return_gaussian_impact_score
            ),
            return_normals=config.render.return_normals,
            return_2d_projections=config.render.return_2d_projections,
            return_projective_intersection_transforms=(
                config.render.return_projective_intersection_transforms
            ),
            options=options,
        )
        return render_output

    return raw_render_fn


def build_render_fn(config: TrainingConfig) -> RenderFn:
    """Build the stateless render pipeline."""
    raw_render_fn = build_raw_render_fn(config)
    postprocess_fn = resolve_callable(config.render.postprocess_fn)

    def render_fn(model: InitializedModel, camera: Any) -> Any:
        render_output = raw_render_fn(model, camera)
        if postprocess_fn is None:
            return render_output
        return postprocess_fn(model, camera, render_output)

    return render_fn


class _TrainingDensificationRuntime(DensificationRuntime):
    """Typed runtime services shared with densification methods."""

    def __init__(
        self,
        *,
        backend_name: str,
        render_options: Any,
        frame_dataset: PreparedFrameDataset,
        raw_render_fn: RenderFn,
        device: torch.device,
    ) -> None:
        self.backend_name = backend_name
        self.render_options = render_options
        self._frame_dataset = frame_dataset
        self._raw_render_fn = raw_render_fn
        self._device = device

    def sample_views(self, count: int) -> tuple[PreparedFrameSample, ...]:
        if count <= 0:
            return ()
        dataset_size = len(self._frame_dataset)
        if dataset_size == 0:
            return ()
        sample_count = min(count, dataset_size)
        indices = torch.randperm(dataset_size)[:sample_count].tolist()
        return tuple(self._frame_dataset[index].to(self._device) for index in indices)

    def render_raw(self, model: InitializedModel, camera: Any) -> Any:
        return self._raw_render_fn(model, camera)

    def resolve_trait(self, trait_type: type[Any]) -> Any:
        return resolve_backend_trait(self.backend_name, trait_type)


def _normalize_loss_result(result: Any) -> LossResult:
    if isinstance(result, LossResult):
        return result
    if isinstance(result, torch.Tensor):
        return LossResult(loss=result)
    if isinstance(result, tuple) and len(result) == 2:
        loss, metrics = result
        if not isinstance(loss, torch.Tensor):
            raise TypeError("Tuple loss result must start with a Tensor.")
        return LossResult(
            loss=loss,
            metrics={
                name: float(value) for name, value in dict(metrics).items()
            },
        )
    if isinstance(result, dict):
        if "loss" not in result:
            raise ValueError("Loss dict result must contain a 'loss' entry.")
        loss = result["loss"]
        if not isinstance(loss, torch.Tensor):
            raise TypeError("Loss dict entry 'loss' must be a Tensor.")
        metrics = {
            name: float(value)
            for name, value in result.items()
            if name != "loss"
        }
        return LossResult(loss=loss, metrics=metrics)
    raise TypeError(
        "Loss function must return LossResult, Tensor, (Tensor, metrics), "
        "or a dict containing 'loss'."
    )


def build_loss_fn(config: TrainingConfig) -> LossFn:
    """Build the loss function."""
    target = resolve_callable(config.loss.target)
    assert target is not None

    def loss_fn(
        state: TrainState, batch: Any, render_output: Any
    ) -> LossResult:
        result = target(
            state,
            batch,
            render_output,
            weights=dict(config.loss.weights),
        )
        return _normalize_loss_result(result)

    return loss_fn


def build_hooks(config: TrainingConfig) -> list[TrainingHook]:
    """Instantiate training hooks."""
    hooks: list[TrainingHook] = []
    for builder in config.hooks.builders:
        hook = instantiate_callable(builder)
        hooks.append(hook)
    return hooks


def build_densification_from_config(
    config: TrainingConfig,
) -> Any | None:
    """Instantiate an unbound densification method from config."""
    return build_densification(config.densification)


def _resolve_target(
    model: InitializedModel,
    target: ParameterTargetSpec,
) -> _ResolvedTarget:
    if target.scope == "scene":
        value = getattr(model.scene, target.name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Scene target {target.name!r} resolved to "
                f"{type(value).__name__}."
            )
        return _ResolvedTarget(
            target=target,
            parameters=[value],
            base_parameter=value,
            field_name=target.name,
        )
    if target.scope == "modules":
        if target.name not in model.modules:
            raise KeyError(f"Unknown module target {target.name!r}.")
        return _ResolvedTarget(
            target=target,
            parameters=list(model.modules[target.name].parameters()),
        )
    if target.scope == "parameters":
        if target.name not in model.parameters:
            raise KeyError(f"Unknown parameter target {target.name!r}.")
        value = model.parameters[target.name]
        return _ResolvedTarget(
            target=target,
            parameters=[value],
            base_parameter=value,
            field_name=target.name,
        )
    raise ValueError(
        f"Unsupported target scope {target.scope!r}. Expected scene, modules, "
        "or parameters."
    )


def _build_optimizer(
    config: ParameterGroupConfig,
    parameters: Sequence[torch.Tensor],
) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        adam_kwargs = {
            "lr": config.lr,
            "betas": config.betas,
            "weight_decay": config.weight_decay,
            **config.optimizer_kwargs,
        }
        return torch.optim.Adam(
            list(parameters),
            **adam_kwargs,
        )
    if config.optimizer == "sgd":
        sgd_kwargs = {
            "lr": config.lr,
            "momentum": config.momentum,
            "weight_decay": config.weight_decay,
            **config.optimizer_kwargs,
        }
        return torch.optim.SGD(
            list(parameters),
            **sgd_kwargs,
        )
    optimizer_factory = resolve_target(config.optimizer)
    if not callable(optimizer_factory):
        raise TypeError(
            f"Resolved optimizer {config.optimizer!r} is not callable."
        )
    return optimizer_factory(
        list(parameters),
        lr=config.lr,
        **config.optimizer_kwargs,
    )


def _build_scheduler(
    spec: CallableSpec | None,
    optimizer: torch.optim.Optimizer,
) -> LRScheduler | None:
    if spec is None:
        return None
    scheduler_factory = resolve_target(spec.target)
    if not callable(scheduler_factory):
        raise TypeError(
            f"Resolved scheduler {spec.target!r} is not callable."
        )
    scheduler = scheduler_factory(optimizer, **spec.kwargs)
    if not isinstance(scheduler, LRScheduler):
        raise TypeError(
            f"Scheduler builder {spec.target!r} returned "
            f"{type(scheduler).__name__}, expected LRScheduler."
        )
    return scheduler


def _build_binding_view(
    target: ParameterTargetSpec,
    base_parameter: torch.Tensor | None,
) -> _BindingView | None:
    if target.view is None:
        return None
    if base_parameter is None:
        raise ValueError(
            f"Target {target.scope}.{target.name!r} cannot use a view."
        )
    return _BindingView(
        f"{target.scope}.{target.name}",
        target.view,
        base_parameter,
    )


def _reject_overlapping_bindings(
    target: ParameterTargetSpec,
    base_parameter: torch.Tensor | None,
    view: _BindingView | None,
    seen: dict[int, list[tuple[ParameterTargetSpec, _BindingView | None]]],
) -> None:
    if base_parameter is None:
        return
    base_id = id(base_parameter)
    existing = seen.setdefault(base_id, [])
    for other_target, other_view in existing:
        if view is None or other_view is None:
            raise ValueError(
                "Overlapping optimizer targets are not allowed for "
                f"{target.scope}.{target.name!r} and "
                f"{other_target.scope}.{other_target.name!r}."
            )
        if view.overlaps(other_view, base_parameter.shape):
            raise ValueError(
                "Overlapping optimizer views are not allowed for "
                f"{target.scope}.{target.name!r} and "
                f"{other_target.scope}.{other_target.name!r}."
            )
    existing.append((target, view))


def build_optimizer_set(
    state: TrainState,
    config: TrainingConfig,
) -> list[OptimizerBinding]:
    """Build one optimizer per declared parameter group."""
    if not config.optimization.parameter_groups:
        raise ValueError(
            "TrainingConfig.optimization.parameter_groups must not be empty."
        )
    optimizers: list[OptimizerBinding] = []
    seen_targets: dict[int, list[tuple[ParameterTargetSpec, _BindingView | None]]] = {}
    for group in config.optimization.parameter_groups:
        resolved = _resolve_target(state.model, group.target)
        view = _build_binding_view(group.target, resolved.base_parameter)
        _reject_overlapping_bindings(
            group.target,
            resolved.base_parameter,
            view,
            seen_targets,
        )
        parameters = (
            [resolved.base_parameter]
            if resolved.base_parameter is not None and view is not None
            else resolved.parameters
        )
        optimizer = _build_optimizer(group, parameters)
        optimizers.append(
            OptimizerBinding(
                target=group.target,
                optimizer=optimizer,
                scheduler=_build_scheduler(group.scheduler, optimizer),
                base_parameter=resolved.base_parameter,
                field_name=resolved.field_name,
                view=view,
            )
        )
    return optimizers


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def _call_pre_backward_hooks(
    hooks: Sequence[TrainingHook],
    state: TrainState,
    batch: Any,
    render_output: Any,
    loss_result: LossResult,
) -> None:
    for hook in hooks:
        pre_backward = getattr(hook, "pre_backward", None)
        if pre_backward is not None:
            pre_backward(state, batch, render_output, loss_result)


def _call_post_backward_hooks(
    hooks: Sequence[TrainingHook],
    state: TrainState,
    batch: Any,
    render_output: Any,
    loss_result: LossResult,
) -> None:
    for hook in hooks:
        post_backward = getattr(hook, "post_backward", None)
        if post_backward is not None:
            post_backward(state, batch, render_output, loss_result)


def _call_after_step_hooks(
    hooks: Sequence[TrainingHook],
    state: TrainState,
    metrics: dict[str, float],
) -> None:
    for hook in hooks:
        after_step = getattr(hook, "after_step", None)
        if after_step is not None:
            after_step(state, metrics)


def _call_post_optimizer_step_hooks(
    hooks: Sequence[TrainingHook],
    state: TrainState,
    batch: Any,
    render_output: Any,
    loss_result: LossResult,
) -> None:
    for hook in hooks:
        post_optimizer_step = getattr(hook, "post_optimizer_step", None)
        if post_optimizer_step is not None:
            post_optimizer_step(state, batch, render_output, loss_result)


def train_step(
    state: TrainState,
    batch: Any,
    *,
    render_fn: RenderFn,
    loss_fn: LossFn,
    optimizers: Sequence[OptimizerBinding],
    densification: Any | None = None,
    probe_runtime: DensificationRuntime | None = None,
    hooks: Sequence[TrainingHook] = (),
) -> dict[str, float]:
    """Run one optimization step."""
    resolved_batch = _move_batch_to_device(batch, state.device)
    for optimizer_binding in optimizers:
        optimizer_binding.zero_grad()
    render_output = render_fn(state.model, resolved_batch.camera)
    loss_result = loss_fn(state, resolved_batch, render_output)
    densification_context = None
    if densification is not None:
        densification_context = make_context(
            state=state,
            batch=resolved_batch,
            render_output=render_output,
            loss_result=loss_result,
            optimizers=list(optimizers),
            runtime=probe_runtime,
        )
        densification.pre_backward(densification_context)
    _call_pre_backward_hooks(
        hooks,
        state,
        resolved_batch,
        render_output,
        loss_result,
    )
    loss_result.loss.backward()
    if densification_context is not None:
        densification.post_backward(densification_context)
    _call_post_backward_hooks(
        hooks,
        state,
        resolved_batch,
        render_output,
        loss_result,
    )
    for optimizer_binding in optimizers:
        optimizer_binding.step()
    if densification_context is not None:
        densification.post_optimizer_step(densification_context)
    _call_post_optimizer_step_hooks(
        hooks,
        state,
        resolved_batch,
        render_output,
        loss_result,
    )
    state.step += 1
    metrics = {
        "loss": float(loss_result.loss.detach().item()),
        **loss_result.metrics,
    }
    if densification_context is not None:
        densification.after_step(
            make_context(
                state=state,
                batch=resolved_batch,
                render_output=render_output,
                loss_result=loss_result,
                optimizers=list(optimizers),
                runtime=probe_runtime,
            ),
            metrics,
        )
    _call_after_step_hooks(hooks, state, metrics)
    return metrics


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cycle(
    loader: DataLoader[PreparedFrameBatch],
) -> Iterator[PreparedFrameBatch]:
    while True:
        yield from loader


def run_training(
    frame_dataset: PreparedFrameDataset,
    config: TrainingConfig,
) -> TrainingResult:
    """Run the declarative training loop and export a checkpoint directory."""
    _set_seed(config.runtime.seed)
    device = torch.device(config.runtime.device)
    model = initialize_model(frame_dataset.scene_record, config).to(device)
    state = TrainState(
        model=model,
        step=0,
        seed=config.runtime.seed,
        device=device,
    )
    densification = build_densification_from_config(config)
    if densification is not None:
        merge_densification_requirements(
            config,
            densification.get_render_requirements(),
        )
    dataloader = _build_dataloader_from_frame_dataset(frame_dataset, config)
    raw_render_fn = build_raw_render_fn(config)
    render_fn = build_render_fn(config)
    loss_fn = build_loss_fn(config)
    hooks = build_hooks(config)
    optimizers = build_optimizer_set(state, config)
    densification = bind_densification(densification, state, optimizers)
    probe_runtime: DensificationRuntime | None = None
    if densification is not None:
        probe_runtime = _TrainingDensificationRuntime(
            backend_name=config.render.backend,
            render_options=_build_backend_options(config),
            frame_dataset=frame_dataset,
            raw_render_fn=raw_render_fn,
            device=device,
        )
    history: list[dict[str, float]] = []
    iterator = _cycle(dataloader)
    for _ in range(config.runtime.max_steps):
        history.append(
            train_step(
                state,
                next(iterator),
                render_fn=render_fn,
                loss_fn=loss_fn,
                optimizers=optimizers,
                densification=densification,
                probe_runtime=probe_runtime,
                hooks=hooks,
            )
        )
    from ember_core.training.checkpoints import save_checkpoint_dir

    checkpoint_dir = save_checkpoint_dir(
        config.checkpoint.output_dir,
        state,
        config,
        frame_dataset=frame_dataset,
    )
    return TrainingResult(
        state=state,
        history=history,
        checkpoint_dir=str(checkpoint_dir),
    )


__all__ = [
    "build_dataloader",
    "build_densification_from_config",
    "build_hooks",
    "build_loss_fn",
    "build_modules",
    "build_optimizer_set",
    "build_parameters",
    "build_render_fn",
    "initialize_model",
    "instantiate_callable",
    "resolve_callable",
    "resolve_target",
    "run_training",
    "train_step",
]
