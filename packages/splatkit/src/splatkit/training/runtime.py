"""Declarative training builders and loop."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import is_dataclass, replace
from functools import partial
from importlib import import_module
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from splatkit.core.registry import BACKEND_REGISTRY, render
from splatkit.data.adapters import FrameDataset, collate_frame_samples
from splatkit.data.contracts import (
    ImagePreparationSpec,
    PreparedFrameBatch,
    ResizeSpec,
    SceneDataset,
)
from splatkit.initialization import InitializedModel
from splatkit.training.config import (
    CallableSpec,
    ParameterGroupConfig,
    ParameterSpec,
    TrainingConfig,
)
from splatkit.training.protocols import (
    LossFn,
    LossResult,
    RenderFn,
    TrainingHook,
    TrainingResult,
    TrainState,
)


class OptimizerBinding:
    """Optimizer bound to one declared selector."""

    def __init__(
        self,
        selector: str,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.selector = selector
        self.optimizer = optimizer


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
    dataset: SceneDataset,
    config: TrainingConfig,
) -> InitializedModel:
    """Run the configured initializer."""
    initializer = resolve_callable(config.initialization.initializer)
    assert initializer is not None
    model = initializer(
        dataset,
        modules=build_modules(config),
        parameters=build_parameters(config),
    )
    if not isinstance(model, InitializedModel):
        raise TypeError(
            "Initializer must return InitializedModel, got "
            f"{type(model).__name__}."
        )
    return model


def _build_preparation(
    config: TrainingConfig,
) -> ImagePreparationSpec:
    resize = None
    if (
        config.batching.resize_width_scale is not None
        or config.batching.resize_width_target is not None
    ):
        resize = ResizeSpec(
            width_scale=config.batching.resize_width_scale,
            width_target=config.batching.resize_width_target,
            interpolation=config.batching.interpolation,
        )
    return ImagePreparationSpec(
        resize=resize,
        normalize=config.batching.normalize,
    )


def build_dataloader(
    dataset: SceneDataset,
    config: TrainingConfig,
) -> DataLoader[PreparedFrameBatch]:
    """Build the camera-batched dataloader."""
    frame_dataset = FrameDataset(
        dataset,
        preparation=_build_preparation(config),
        materialization_stage=config.batching.materialization_stage,
        materialization_mode=config.batching.materialization_mode,
        materialization_num_workers=(
            config.batching.materialization_num_workers
        ),
    )
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
    return replace(default_options, **config.render.backend_options)


def build_render_fn(config: TrainingConfig) -> RenderFn:
    """Build the stateless render pipeline."""
    feature_fn = resolve_callable(config.render.feature_fn)
    postprocess_fn = resolve_callable(config.render.postprocess_fn)
    options = _build_backend_options(config)

    def render_fn(model: InitializedModel, camera: Any) -> Any:
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
            return_normals=config.render.return_normals,
            return_2d_projections=config.render.return_2d_projections,
            return_projective_intersection_transforms=(
                config.render.return_projective_intersection_transforms
            ),
            options=options,
        )
        if postprocess_fn is None:
            return render_output
        return postprocess_fn(model, resolved_camera, render_output)

    return render_fn


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


def _resolve_selector(
    model: InitializedModel,
    selector: str,
) -> list[torch.Tensor]:
    scope, _, name = selector.partition(".")
    if scope == "scene":
        if not name:
            raise ValueError("Scene selector must include a field name.")
        value = getattr(model.scene, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Scene selector {selector!r} resolved to {type(value).__name__}."
            )
        return [value]
    if scope == "modules":
        if name not in model.modules:
            raise KeyError(f"Unknown module selector {selector!r}.")
        return list(model.modules[name].parameters())
    if scope == "parameters":
        if name not in model.parameters:
            raise KeyError(f"Unknown parameter selector {selector!r}.")
        return [model.parameters[name]]
    raise ValueError(
        f"Unsupported selector {selector!r}. Expected scene.*, modules.*, or parameters.*."
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
        )
    return torch.optim.SGD(
        list(parameters),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def build_optimizer_set(
    state: TrainState,
    config: TrainingConfig,
) -> list[OptimizerBinding]:
    """Build one optimizer per declared parameter selector."""
    if not config.optimization.parameter_groups:
        raise ValueError(
            "TrainingConfig.optimization.parameter_groups must not be empty."
        )
    optimizers: list[OptimizerBinding] = []
    for group in config.optimization.parameter_groups:
        parameters = _resolve_selector(state.model, group.selector)
        optimizers.append(
            OptimizerBinding(
                selector=group.selector,
                optimizer=_build_optimizer(group, parameters),
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


def train_step(
    state: TrainState,
    batch: Any,
    *,
    render_fn: RenderFn,
    loss_fn: LossFn,
    optimizers: Sequence[OptimizerBinding],
    hooks: Sequence[TrainingHook] = (),
) -> dict[str, float]:
    """Run one optimization step."""
    resolved_batch = _move_batch_to_device(batch, state.device)
    for optimizer_binding in optimizers:
        optimizer_binding.optimizer.zero_grad(set_to_none=True)
    render_output = render_fn(state.model, resolved_batch.camera)
    loss_result = loss_fn(state, resolved_batch, render_output)
    _call_pre_backward_hooks(
        hooks,
        state,
        resolved_batch,
        render_output,
        loss_result,
    )
    loss_result.loss.backward()
    _call_post_backward_hooks(
        hooks,
        state,
        resolved_batch,
        render_output,
        loss_result,
    )
    for optimizer_binding in optimizers:
        optimizer_binding.optimizer.step()
    state.step += 1
    metrics = {
        "loss": float(loss_result.loss.detach().item()),
        **loss_result.metrics,
    }
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
    dataset: SceneDataset,
    config: TrainingConfig,
) -> TrainingResult:
    """Run the declarative training loop and export a checkpoint directory."""
    _set_seed(config.runtime.seed)
    device = torch.device(config.runtime.device)
    model = initialize_model(dataset, config).to(device)
    state = TrainState(
        model=model,
        step=0,
        seed=config.runtime.seed,
        device=device,
    )
    dataloader = build_dataloader(dataset, config)
    render_fn = build_render_fn(config)
    loss_fn = build_loss_fn(config)
    hooks = build_hooks(config)
    optimizers = build_optimizer_set(state, config)
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
                hooks=hooks,
            )
        )
    from splatkit.training.checkpoints import save_checkpoint_dir

    checkpoint_dir = save_checkpoint_dir(
        config.checkpoint.output_dir,
        state,
        config,
        dataset=dataset,
    )
    return TrainingResult(
        state=state,
        history=history,
        checkpoint_dir=str(checkpoint_dir),
    )


__all__ = [
    "build_dataloader",
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
