"""Declarative training builders and loop."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, cast

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from ember_core.core.contracts import CameraState
from ember_core.core.registry import (
    BACKEND_REGISTRY,
    render_dynamic,
    resolve_backend_trait,
)
from ember_core.data.adapters import PreparedFrameDataset, collate_frame_samples
from ember_core.data.contracts import (
    PreparedFrameBatch,
    PreparedFrameSample,
    SceneRecord,
)
from ember_core.densification.contracts import (
    DensificationMethod,
    DensificationRenderRequirements,
    DensificationRuntime,
)
from ember_core.densification.runtime import (
    DensificationMethodSequence,
    bind_densification,
    build_densification,
    make_context,
    make_lifecycle_context,
)
from ember_core.initialization import InitializedModel
from ember_core.training.config import (
    CallableSpec,
    OptimizationConfig,
    ParameterGroupConfig,
    ParameterSpec,
    ParameterTargetSpec,
    TensorViewSpec,
    TrainingConfig,
)
from ember_core.training.logging import build_training_logger
from ember_core.training.profiling import (
    TrainingStepProfile,
    build_training_profiler,
)
from ember_core.training.protocols import (
    LossFn,
    LossResult,
    RenderFn,
    TrainingConfigSource,
    TrainingHook,
    TrainingResult,
    TrainingRunContext,
    TrainState,
)
from ember_core.training.resolution import (
    callable_kwargs,
    instantiate_callable,
    resolve_callable,
    resolve_callable_target,
    resolve_target,
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
    *,
    context: TrainingRunContext | None = None,
) -> InitializedModel:
    """Run the configured initializer."""
    initializer = resolve_callable(
        config.initialization.initializer,
        context=context,
    )
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
    num_workers = config.batching.num_workers
    return cast(
        "DataLoader[PreparedFrameBatch]",
        DataLoader(
            frame_dataset,
            batch_size=config.batching.batch_size,
            shuffle=config.batching.shuffle,
            num_workers=num_workers,
            persistent_workers=(
                config.batching.persistent_workers and num_workers > 0
            ),
            pin_memory=config.batching.pin_memory,
            collate_fn=collate_frame_samples,
        ),
    )


def _build_backend_options(config: TrainingConfig) -> Any:
    return resolve_backend_options(config)


def resolve_backend_options(
    config: TrainingConfig,
    *,
    updates: dict[str, Any] | None = None,
) -> Any:
    """Resolve render backend options against registered dataclass defaults."""
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
    option_updates = dict(config.render.backend_options)
    if updates is not None:
        option_updates.update(updates)
    unknown_option_names = sorted(set(option_updates) - valid_option_names)
    if unknown_option_names:
        raise ValueError(
            f"Backend {config.render.backend!r} does not support render "
            f"options {unknown_option_names!r}."
        )
    resolved_options: dict[str, Any] = {}
    for field in fields(default_options):
        if field.name not in option_updates:
            continue
        value = option_updates[field.name]
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


def _merge_backend_option_updates(
    *updates: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for update in updates:
        for name, value in update.items():
            if name in merged and merged[name] != value:
                raise ValueError(
                    f"Conflicting render backend option {name!r}: "
                    f"{merged[name]!r} vs {value!r}."
                )
            merged[name] = value
    return merged


def _validate_requested_outputs(
    config: TrainingConfig,
    requirements: DensificationRenderRequirements | None = None,
) -> None:
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
            (
                "gaussian_impact_score",
                config.render.return_gaussian_impact_score,
            ),
            ("normals", config.render.return_normals),
            ("2d_projections", config.render.return_2d_projections),
            (
                "projective_intersection_transforms",
                config.render.return_projective_intersection_transforms,
            ),
        )
        if enabled
    }
    if requirements is not None:
        requested_outputs.update(
            name
            for name, enabled in (
                ("alpha", requirements.return_alpha),
                ("depth", requirements.return_depth),
                (
                    "gaussian_impact_score",
                    requirements.return_gaussian_impact_score,
                ),
                ("normals", requirements.return_normals),
                ("2d_projections", requirements.return_2d_projections),
                (
                    "projective_intersection_transforms",
                    requirements.return_projective_intersection_transforms,
                ),
            )
            if enabled
        )
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
        render_output = render_dynamic(
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


RenderFnWithRequirements = Callable[
    [
        InitializedModel,
        CameraState,
        DensificationRenderRequirements,
        Any | None,
    ],
    Any,
]


def _training_backend_option_updates(
    training_options_fn: Callable[..., Any] | None,
    state: TrainState | None,
    batch: Any | None,
) -> dict[str, Any]:
    """Resolve per-step backend option updates."""
    if training_options_fn is None:
        return {}
    signature = inspect.signature(training_options_fn)
    parameters = signature.parameters
    accepts_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_keywords:
        updates = training_options_fn(state=state, batch=batch)
    elif "state" in parameters or "batch" in parameters:
        keyword_arguments: dict[str, Any] = {}
        if "state" in parameters:
            keyword_arguments["state"] = state
        if "batch" in parameters:
            keyword_arguments["batch"] = batch
        updates = training_options_fn(**keyword_arguments)
    else:
        updates = training_options_fn(state)
    return dict(updates)


def build_render_fn_with_requirements(
    config: TrainingConfig,
    *,
    state: TrainState | None = None,
    context: TrainingRunContext | None = None,
) -> RenderFnWithRequirements:
    """Build a render pipeline accepting per-step densification requirements."""
    _validate_requested_outputs(config)
    feature_fn = resolve_callable(config.render.feature_fn)
    postprocess_fn = resolve_callable(config.render.postprocess_fn)
    static_options = _build_backend_options(config)
    training_options_fn = resolve_callable(
        config.render.training_backend_options_builder,
        context=context,
    )

    def render_fn(
        model: InitializedModel,
        camera: CameraState,
        requirements: DensificationRenderRequirements,
        batch: Any | None = None,
    ) -> Any:
        _validate_requested_outputs(config, requirements)
        resolved_camera = camera
        if not hasattr(resolved_camera, "cam_to_world"):
            raise TypeError("Render camera must provide cam_to_world.")
        scene = (
            model.scene
            if feature_fn is None
            else feature_fn(model, resolved_camera)
        )
        training_updates = _training_backend_option_updates(
            training_options_fn,
            state,
            batch,
        )
        backend_updates = _merge_backend_option_updates(
            training_updates,
            requirements.backend_options,
        )
        options = (
            static_options
            if not backend_updates
            else resolve_backend_options(
                config,
                updates=backend_updates,
            )
        )
        render_output = render_dynamic(
            scene,
            resolved_camera,
            backend=config.render.backend,
            return_alpha=config.render.return_alpha
            or requirements.return_alpha,
            return_depth=config.render.return_depth
            or requirements.return_depth,
            return_gaussian_impact_score=(
                config.render.return_gaussian_impact_score
                or requirements.return_gaussian_impact_score
            ),
            return_normals=config.render.return_normals
            or requirements.return_normals,
            return_2d_projections=(
                config.render.return_2d_projections
                or requirements.return_2d_projections
            ),
            return_projective_intersection_transforms=(
                config.render.return_projective_intersection_transforms
                or requirements.return_projective_intersection_transforms
            ),
            options=options,
        )
        if postprocess_fn is None:
            return render_output
        return postprocess_fn(model, resolved_camera, render_output)

    return render_fn


def build_training_render_fn(
    config: TrainingConfig,
    state: TrainState,
    *,
    context: TrainingRunContext | None = None,
) -> RenderFn:
    """Build the render function used by the training loop."""
    render_with_requirements = build_render_fn_with_requirements(
        config,
        state=state,
        context=context,
    )

    def render_fn(model: InitializedModel, camera: CameraState) -> Any:
        return render_with_requirements(
            model,
            camera,
            DensificationRenderRequirements(),
            None,
        )

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
        return tuple(
            self._frame_dataset[index].to(self._device) for index in indices
        )

    def all_views(self) -> tuple[PreparedFrameSample, ...]:
        return tuple(
            self._frame_dataset[index].to(self._device)
            for index in range(len(self._frame_dataset))
        )

    def all_cameras(self) -> tuple[Any, ...]:
        return self._frame_dataset.prepared_cameras()

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
) -> DensificationMethod | None:
    """Instantiate an unbound densification method from config."""
    return build_densification(config.densification)


def build_densification_for_context(
    config: TrainingConfig,
    *,
    context: TrainingRunContext,
) -> DensificationMethod | None:
    """Instantiate densification with runtime-bound kwargs resolved."""
    if config.densification is None or (
        not config.densification.builders and not config.densification.methods
    ):
        return None
    methods: list[DensificationMethod] = list(config.densification.methods)
    for builder_spec in config.densification.builders:
        builder = resolve_callable_target(builder_spec)
        method = builder(**callable_kwargs(builder_spec, context))
        if not hasattr(method, "get_render_requirements"):
            raise TypeError(
                "Densification builder must return a densification method."
            )
        methods.append(method)
    if len(methods) == 1:
        return methods[0]
    return DensificationMethodSequence(methods)


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
        return torch.optim.Adam(
            list(parameters),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            list(parameters),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs,
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
    *,
    context: TrainingRunContext | None = None,
) -> LRScheduler | None:
    if spec is None:
        return None
    scheduler_factory = resolve_target(spec.target)
    if not callable(scheduler_factory):
        raise TypeError(f"Resolved scheduler {spec.target!r} is not callable.")
    scheduler = scheduler_factory(optimizer, **callable_kwargs(spec, context))
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


def materialize_optimization_config(
    config: OptimizationConfig,
    *,
    context: TrainingRunContext | None = None,
) -> OptimizationConfig:
    """Resolve optional runtime-bound optimization recipe builders."""
    if config.builder is None:
        return config
    builder = resolve_target(config.builder.target)
    if not callable(builder):
        raise TypeError(
            f"Optimization builder {config.builder.target!r} is not callable."
        )
    built = builder(**callable_kwargs(config.builder, context))
    if isinstance(built, OptimizationConfig):
        builder_config = built
    elif isinstance(built, list):
        builder_config = OptimizationConfig(parameter_groups=built)
    else:
        raise TypeError(
            f"Optimization builder {config.builder.target!r} returned "
            f"{type(built).__name__}, expected OptimizationConfig or list."
        )
    return config.model_copy(
        update={
            "parameter_groups": [
                *builder_config.parameter_groups,
                *config.parameter_groups,
            ]
        }
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
    *,
    context: TrainingRunContext | None = None,
) -> list[OptimizerBinding]:
    """Build one optimizer per declared parameter group."""
    optimization = materialize_optimization_config(
        config.optimization,
        context=context,
    )
    if not optimization.parameter_groups:
        raise ValueError(
            "TrainingConfig.optimization.parameter_groups must not be empty."
        )
    optimizers: list[OptimizerBinding] = []
    seen_targets: dict[
        int, list[tuple[ParameterTargetSpec, _BindingView | None]]
    ] = {}
    for group in optimization.parameter_groups:
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
                scheduler=_build_scheduler(
                    group.scheduler,
                    optimizer,
                    context=context,
                ),
                base_parameter=resolved.base_parameter,
                field_name=resolved.field_name,
                view=view,
            )
        )
    return optimizers


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        try:
            return batch.to(device, non_blocking=device.type == "cuda")
        except TypeError:
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


def _call_before_step_hooks(
    hooks: Sequence[TrainingHook],
    state: TrainState,
) -> None:
    for hook in hooks:
        before_step = getattr(hook, "before_step", None)
        if before_step is not None:
            before_step(state)


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


def _profile_phase(
    profile: TrainingStepProfile | None,
    name: str,
) -> Any:
    if profile is None:
        return nullcontext()
    return profile.phase(name)


def _consume_step_diagnostics(state: TrainState) -> dict[str, float]:
    diagnostics = getattr(state, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        return {}
    raw_values = diagnostics.pop("metrics", {})
    return {
        str(name): float(value)
        for name, value in dict(raw_values).items()
        if isinstance(value, int | float)
    }


def train_step(
    state: TrainState,
    batch: Any,
    *,
    render_fn: RenderFn,
    render_fn_with_requirements: RenderFnWithRequirements | None = None,
    loss_fn: LossFn,
    optimizers: Sequence[OptimizerBinding],
    densification: DensificationMethod | None = None,
    probe_runtime: DensificationRuntime | None = None,
    hooks: Sequence[TrainingHook] = (),
    profile: TrainingStepProfile | None = None,
) -> dict[str, float]:
    """Run one optimization step."""
    with _profile_phase(profile, "before_hooks"):
        _call_before_step_hooks(hooks, state)
    with _profile_phase(profile, "transfer"):
        resolved_batch = _move_batch_to_device(batch, state.device)
    with _profile_phase(profile, "zero_grad"):
        for optimizer_binding in optimizers:
            optimizer_binding.zero_grad()
    with _profile_phase(profile, "render"):
        if render_fn_with_requirements is None:
            render_output = render_fn(state.model, resolved_batch.camera)
        else:
            render_output = render_fn_with_requirements(
                state.model,
                resolved_batch.camera,
                (
                    DensificationRenderRequirements()
                    if densification is None
                    else densification.get_render_requirements(state)
                ),
                resolved_batch,
            )
    with _profile_phase(profile, "loss"):
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
        with _profile_phase(profile, "densification_pre_backward"):
            densification.pre_backward(densification_context)
    with _profile_phase(profile, "pre_backward_hooks"):
        _call_pre_backward_hooks(
            hooks,
            state,
            resolved_batch,
            render_output,
            loss_result,
        )
    with _profile_phase(profile, "backward"):
        loss_result.loss.backward()
    if densification is not None and densification_context is not None:
        with _profile_phase(profile, "densification_post_backward"):
            densification.post_backward(densification_context)
    with _profile_phase(profile, "post_backward_hooks"):
        _call_post_backward_hooks(
            hooks,
            state,
            resolved_batch,
            render_output,
            loss_result,
        )
    if densification is not None and densification_context is not None:
        pre_optimizer_step = getattr(
            densification, "pre_optimizer_step", None
        )
        if pre_optimizer_step is not None:
            with _profile_phase(profile, "densification_pre_optimizer"):
                pre_optimizer_step(densification_context)
    with _profile_phase(profile, "optimizer"):
        for optimizer_binding in optimizers:
            optimizer_binding.step()
    if densification is not None and densification_context is not None:
        with _profile_phase(profile, "densification_post_optimizer"):
            densification.post_optimizer_step(densification_context)
    with _profile_phase(profile, "post_optimizer_hooks"):
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
        **_consume_step_diagnostics(state),
    }
    if densification is not None and densification_context is not None:
        with _profile_phase(profile, "densification_after_step"):
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
    with _profile_phase(profile, "after_step_hooks"):
        _call_after_step_hooks(hooks, state, metrics)
    return metrics


def set_torch_seed(seed: int) -> None:
    """Seed PyTorch RNGs, including CUDA when available."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_seed(seed: int) -> None:
    set_torch_seed(seed)


def cycle_dataloader(
    loader: DataLoader[PreparedFrameBatch],
) -> Iterator[PreparedFrameBatch]:
    """Repeat a dataloader forever."""
    while True:
        yield from loader


def _cycle(
    loader: DataLoader[PreparedFrameBatch],
) -> Iterator[PreparedFrameBatch]:
    return cycle_dataloader(loader)


def compute_frame_camera_extent(
    frame_dataset: PreparedFrameDataset,
    *,
    scale: float = 1.1,
) -> float:
    """Compute a scene extent estimate from prepared-frame camera centers."""
    if len(frame_dataset) == 0:
        return 0.0
    centers = [
        frame_dataset[index].camera.cam_to_world[..., :3, 3].reshape(-1, 3)
        for index in range(len(frame_dataset))
    ]
    camera_centers = torch.cat(centers, dim=0).to(torch.float32)
    mean_center = camera_centers.mean(dim=0, keepdim=True)
    return float(
        scale * (camera_centers - mean_center).norm(dim=-1).max().item()
    )


def build_training_run_context(
    frame_dataset: PreparedFrameDataset,
    config: TrainingConfig,
    *,
    device: torch.device | None = None,
) -> TrainingRunContext:
    """Build runtime-only values used to materialize training recipes."""
    return TrainingRunContext(
        frame_dataset=frame_dataset,
        camera_extent=compute_frame_camera_extent(frame_dataset),
        max_steps=config.runtime.max_steps,
        backend=config.render.backend,
        device=device or torch.device(config.runtime.device),
    )


def materialize_training_config(
    frame_dataset: PreparedFrameDataset,
    config: TrainingConfig | TrainingConfigSource,
) -> TrainingConfig:
    """Resolve a typed user config into the runtime ``TrainingConfig``."""
    if isinstance(config, TrainingConfig):
        return config
    materialize = getattr(config, "to_training_config", None)
    if not callable(materialize):
        raise TypeError(
            "run_training config must be TrainingConfig or provide "
            "to_training_config(frame_dataset)."
        )
    resolved = materialize(frame_dataset)
    if not isinstance(resolved, TrainingConfig):
        raise TypeError(
            "to_training_config(...) must return TrainingConfig, got "
            f"{type(resolved).__name__}."
        )
    return resolved


def run_training(
    frame_dataset: PreparedFrameDataset,
    config: TrainingConfig | TrainingConfigSource,
    *,
    runtime_hooks: Sequence[TrainingHook] = (),
) -> TrainingResult:
    """Run the declarative training loop and export a checkpoint directory."""
    from ember_core.training.checkpoints import (
        checkpoint_run_dir,
    )

    config = materialize_training_config(frame_dataset, config)
    concrete_checkpoint_dir = checkpoint_run_dir(
        config.checkpoint.output_dir,
        overwrite=config.checkpoint.overwrite,
    )
    config = config.model_copy(
        update={
            "checkpoint": config.checkpoint.model_copy(
                update={"output_dir": concrete_checkpoint_dir}
            )
        }
    )
    _set_seed(config.runtime.seed)
    device = torch.device(config.runtime.device)
    run_context = build_training_run_context(
        frame_dataset,
        config,
        device=device,
    )
    model = initialize_model(
        frame_dataset.scene_record,
        config,
        context=run_context,
    ).to(device)
    state = TrainState(
        model=model,
        step=0,
        seed=config.runtime.seed,
        device=device,
    )
    densification = build_densification_for_context(
        config,
        context=run_context,
    )
    dataloader = _build_dataloader_from_frame_dataset(frame_dataset, config)
    raw_render_fn = build_raw_render_fn(config)
    render_fn = build_training_render_fn(
        config,
        state,
        context=run_context,
    )
    render_fn_with_requirements = build_render_fn_with_requirements(
        config,
        state=state,
        context=run_context,
    )
    loss_fn = build_loss_fn(config)
    hooks = [*build_hooks(config), *runtime_hooks]
    optimizers = build_optimizer_set(state, config, context=run_context)
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
        densification.before_training(
            make_lifecycle_context(
                state=state,
                optimizers=optimizers,
                runtime=probe_runtime,
            )
        )
    history: list[dict[str, float]] = []
    iterator = _cycle(dataloader)
    logger = build_training_logger(
        config.logging,
        checkpoint_dir=config.checkpoint.output_dir,
    )
    profiler = build_training_profiler(config.profiler)
    training_started_at = time.perf_counter()
    for _ in range(config.runtime.max_steps):
        step_started_at = time.perf_counter()
        profile = None if profiler is None else profiler.start_step(state)
        with _profile_phase(profile, "dataloader"):
            batch = next(iterator)
        metrics = train_step(
            state,
            batch,
            render_fn=render_fn,
            render_fn_with_requirements=render_fn_with_requirements,
            loss_fn=loss_fn,
            optimizers=optimizers,
            densification=densification,
            probe_runtime=probe_runtime,
            hooks=hooks,
            profile=profile,
        )
        step_duration_seconds = max(
            time.perf_counter() - step_started_at,
            1e-12,
        )
        metrics["step_seconds"] = step_duration_seconds
        metrics["elapsed_seconds"] = time.perf_counter() - training_started_at
        metrics["iterations_per_second"] = 1.0 / step_duration_seconds
        if profiler is not None:
            profiler.finish_step(state, metrics, profile)
        if logger is not None:
            logger.write_step(state.step, metrics)
        history.append(metrics)
    if logger is not None:
        logger.close()
    from ember_core.training.checkpoints import save_checkpoint_dir

    if densification is not None:
        densification.after_training(
            make_lifecycle_context(
                state=state,
                optimizers=optimizers,
                runtime=probe_runtime,
            )
        )

    checkpoint_dir = save_checkpoint_dir(
        config.checkpoint.output_dir,
        state,
        config,
        frame_dataset=frame_dataset,
        run_context=run_context,
    )
    return TrainingResult(
        state=state,
        history=history,
        checkpoint_dir=str(checkpoint_dir),
    )


__all__ = [
    "build_dataloader",
    "build_densification_for_context",
    "build_densification_from_config",
    "build_hooks",
    "build_loss_fn",
    "build_modules",
    "build_optimizer_set",
    "build_parameters",
    "build_render_fn",
    "build_render_fn_with_requirements",
    "build_training_render_fn",
    "build_training_run_context",
    "compute_frame_camera_extent",
    "cycle_dataloader",
    "initialize_model",
    "instantiate_callable",
    "materialize_optimization_config",
    "materialize_training_config",
    "resolve_backend_options",
    "resolve_callable",
    "resolve_target",
    "run_training",
    "set_torch_seed",
    "train_step",
]
