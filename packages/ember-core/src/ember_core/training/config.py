"""Declarative training configuration models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from ember_core.core.backend_refs import BackendRef
from ember_core.core.enums import (
    BuiltinOptimizerKind,
    DeviceKind,
    ParameterScope,
)
from ember_core.core.keys import BackendId, OptimizerRef, serialized_id


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(
            right, torch.Tensor
        ):
            return False
        return bool(torch.equal(left, right))
    return left == right


def _dataclass_mapping(value: Any) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _options_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_mapping(value)
    raise TypeError(
        "RenderPipelineSpec.options must be a dataclass or Pydantic model."
    )


def _option_updates(value: Any, default: Any | None) -> dict[str, Any]:
    options = _options_mapping(value)
    if default is None:
        return options
    defaults = _options_mapping(default)
    return {
        name: option_value
        for name, option_value in options.items()
        if name not in defaults
        or not _values_equal(option_value, defaults[name])
    }


class TrainingConfigBase(BaseModel):
    """Base config with strict validation."""

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }


class CallableSpec(TrainingConfigBase):
    """Low-level importable callable target plus runtime-bound kwargs."""

    target: str
    object_ref: Callable[..., Any] | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    kwargs: dict[str, Any] = Field(default_factory=dict)
    context_kwargs: dict[str, str] = Field(default_factory=dict)


class ParameterSpec(TrainingConfigBase):
    """Declarative standalone parameter definition."""

    shape: tuple[int, ...]
    init: Literal["zeros", "ones", "normal", "constant"] = "zeros"
    requires_grad: bool = True
    mean: float = 0.0
    std: float = 1.0
    value: float = 0.0


class RuntimeConfig(TrainingConfigBase):
    """Runtime-level execution controls."""

    device: str = "cpu"
    seed: int = 0
    max_steps: int = Field(default=1, ge=1)

    @field_validator("device", mode="before")
    @classmethod
    def _serialize_device(cls, value: Any) -> str:
        if isinstance(value, DeviceKind):
            return value.value
        if isinstance(value, torch.device):
            return str(value)
        return value


class TrainingProfilerConfig(TrainingConfigBase):
    """Optional training profiler settings."""

    enabled: bool = False
    log_every: int = Field(default=10, ge=1)
    cuda_memory: bool = True
    sync_timing: bool = False
    output_path: Path | None = None


class TrainingLoggingConfig(TrainingConfigBase):
    """Checkpoint-local scalar logging settings."""

    enabled: bool = True
    log_every: int = Field(default=10, ge=1)
    metric_policy: Literal["active", "all"] = "active"


class BatchingConfig(TrainingConfigBase):
    """Dataloader batching settings."""

    batch_size: int = Field(default=1, ge=1)
    shuffle: bool = True
    num_workers: int = Field(default=8, ge=0)
    persistent_workers: bool = True
    pin_memory: bool = False


class InitializationSpec(TrainingConfigBase):
    """Initialization entrypoint for training payload creation."""

    initializer: CallableSpec = Field(
        default_factory=lambda: CallableSpec(
            target=(
                "ember_core.initialization.initialize_gaussian_model_from_scene_record"
            )
        )
    )


class ModelSpec(TrainingConfigBase):
    """Declarative auxiliary learnable components."""

    modules: dict[str, CallableSpec] = Field(default_factory=dict)
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)


class RenderPipelineSpec(TrainingConfigBase):
    """Declarative render pipeline definition."""

    backend: str
    backend_options: dict[str, Any] = Field(default_factory=dict)
    options: Any | None = Field(default=None, exclude=True, repr=False)
    feature_fn: CallableSpec | None = None
    postprocess_fn: CallableSpec | None = None
    training_backend_options_builder: CallableSpec | None = None
    return_alpha: bool = True
    return_depth: bool = False
    return_gaussian_impact_score: bool = False
    return_normals: bool = False
    return_2d_projections: bool = False
    return_projective_intersection_transforms: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_typed_authoring(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        backend = values.get("backend")
        backend_ref = backend if isinstance(backend, BackendRef) else None
        if isinstance(backend, BackendRef):
            values["backend"] = backend.serialized
        elif isinstance(backend, BackendId):
            values["backend"] = backend.serialized

        options = values.get("options")
        if options is not None:
            backend_options = values.get("backend_options")
            if backend_options:
                raise ValueError(
                    "Use either typed RenderPipelineSpec.options or "
                    "serialized backend_options, not both."
                )
            default_options = (
                backend_ref.options_type()
                if backend_ref is not None
                else None
            )
            values["backend_options"] = _option_updates(
                options,
                default_options,
            )
        return values

    @field_validator("backend", mode="before")
    @classmethod
    def _serialize_backend(cls, value: Any) -> str:
        if isinstance(value, BackendRef):
            return value.serialized
        if isinstance(value, BackendId):
            return value.serialized
        return value


class TensorSliceSpec(TrainingConfigBase):
    """Contiguous slice selection along one tensor axis."""

    axis: int = Field(ge=0)
    start: int | None = None
    stop: int | None = None


class TensorViewSpec(TrainingConfigBase):
    """Structured tensor view declaration."""

    slices: tuple[TensorSliceSpec, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_axes(self) -> TensorViewSpec:
        axes = [slice_spec.axis for slice_spec in self.slices]
        if len(set(axes)) != len(axes):
            raise ValueError("TensorViewSpec axes must be unique.")
        return self


class ParameterTargetSpec(TrainingConfigBase):
    """Declarative parameter target selection."""

    scope: Literal["scene", "modules", "parameters"]
    name: str
    view: TensorViewSpec | None = None

    @field_validator("scope", mode="before")
    @classmethod
    def _serialize_scope(cls, value: Any) -> str:
        if isinstance(value, ParameterScope):
            return value.value
        return value


class ParameterGroupConfig(TrainingConfigBase):
    """Optimization settings for one parameter target."""

    target: ParameterTargetSpec
    optimizer: str = "adam"
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = Field(default=0.0, ge=0.0)
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    scheduler: CallableSpec | None = None

    @field_validator("optimizer", mode="before")
    @classmethod
    def _serialize_optimizer(cls, value: Any) -> str:
        if isinstance(value, BuiltinOptimizerKind):
            return value.value
        if isinstance(value, OptimizerRef):
            return serialized_id(value)
        return value

    @model_validator(mode="after")
    def _validate_target(self) -> ParameterGroupConfig:
        if self.target.view is not None and self.target.scope == "modules":
            raise ValueError(
                "ParameterGroupConfig.target.view is only supported for "
                "scene and parameters targets."
            )
        return self


class OptimizationConfig(TrainingConfigBase):
    """Optimizer group declarations."""

    builder: CallableSpec | None = None
    parameter_groups: list[ParameterGroupConfig] = Field(default_factory=list)


class LossConfig(TrainingConfigBase):
    """Declarative loss entrypoint."""

    target: CallableSpec
    weights: dict[str, float] = Field(default_factory=dict)


class HookConfig(TrainingConfigBase):
    """Hook builder declarations."""

    builders: list[CallableSpec] = Field(default_factory=list)


class DensificationConfig(TrainingConfigBase):
    """Declarative densification builder configuration."""

    methods: list[Any] = Field(default_factory=list, exclude=True, repr=False)
    builders: list[CallableSpec] = Field(default_factory=list)


class CheckpointExportConfig(TrainingConfigBase):
    """Checkpoint directory export settings."""

    output_dir: Path = Path("checkpoints/latest")
    export_ply: bool = False
    overwrite: bool = False


class TrainingConfig(TrainingConfigBase):
    """Top-level declarative training configuration."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: TrainingLoggingConfig = Field(
        default_factory=TrainingLoggingConfig
    )
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    initialization: InitializationSpec = Field(
        default_factory=InitializationSpec
    )
    model: ModelSpec = Field(default_factory=ModelSpec)
    render: RenderPipelineSpec
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    loss: LossConfig
    densification: DensificationConfig | None = None
    hooks: HookConfig = Field(default_factory=HookConfig)
    checkpoint: CheckpointExportConfig = Field(
        default_factory=CheckpointExportConfig
    )


class CheckpointMetadata(TrainingConfigBase):
    """Persisted reproducibility metadata."""

    timestamp_utc: str
    seed: int
    git_commit: str | None = None
    git_dirty: bool | None = None
    scene_type: str
    backend_name: str
    export_ply: bool
    import_paths: list[str] = Field(default_factory=list)
    package_versions: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, dict[str, Any]] = Field(default_factory=dict)
    run_summary: dict[str, Any] = Field(default_factory=dict)
    dataset_summary: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "BatchingConfig",
    "CallableSpec",
    "CheckpointExportConfig",
    "CheckpointMetadata",
    "DensificationConfig",
    "HookConfig",
    "InitializationSpec",
    "LossConfig",
    "ModelSpec",
    "OptimizationConfig",
    "ParameterGroupConfig",
    "ParameterSpec",
    "ParameterTargetSpec",
    "RenderPipelineSpec",
    "RuntimeConfig",
    "TensorSliceSpec",
    "TensorViewSpec",
    "TrainingConfig",
    "TrainingLoggingConfig",
    "TrainingProfilerConfig",
]
