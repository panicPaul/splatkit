"""Declarative training configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

class TrainingConfigBase(BaseModel):
    """Base config with strict validation."""

    model_config = {
        "extra": "forbid",
    }


class CallableSpec(TrainingConfigBase):
    """Importable callable target plus bound kwargs."""

    target: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


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


class BatchingConfig(TrainingConfigBase):
    """Dataloader batching settings."""

    batch_size: int = Field(default=1, ge=1)
    shuffle: bool = True


class InitializationSpec(TrainingConfigBase):
    """Initialization entrypoint for training payload creation."""

    initializer: CallableSpec = Field(
        default_factory=lambda: CallableSpec(
            target=(
                "splatkit.initialization.initialize_gaussian_model_from_scene_record"
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
    feature_fn: CallableSpec | None = None
    postprocess_fn: CallableSpec | None = None
    return_alpha: bool = True
    return_depth: bool = False
    return_gaussian_impact_score: bool = False
    return_normals: bool = False
    return_2d_projections: bool = False
    return_projective_intersection_transforms: bool = False


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

    builder: CallableSpec | None = None


class CheckpointExportConfig(TrainingConfigBase):
    """Checkpoint directory export settings."""

    output_dir: Path = Path("checkpoints/latest")
    export_ply: bool = False


class TrainingConfig(TrainingConfigBase):
    """Top-level declarative training configuration."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
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
    "ParameterTargetSpec",
    "ParameterSpec",
    "RenderPipelineSpec",
    "RuntimeConfig",
    "TensorSliceSpec",
    "TensorViewSpec",
    "TrainingConfig",
]
