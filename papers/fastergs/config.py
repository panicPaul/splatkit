"""Canonical FasterGS paper config types, default loaders, and builders."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

import splatkit as sk
import torch
import torch.nn.functional as F
import tyro
from jaxtyping import Float
from pydantic import BaseModel, Field
from splatkit.training import LossResult, TrainState
from torch import Tensor

FasterGSBackendName = Literal["adapter.fastergs", "faster_gs.core"]
FasterGSDefaultName = Literal["garden_baseline", "garden_mcmc"]

_NOTEBOOK_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _NOTEBOOK_DIR.parents[1]
_DEFAULT_CHECKPOINT_ROOT = _REPO_ROOT / "checkpoints" / "papers" / "fastergs"
_DEFAULTS_DIR = _NOTEBOOK_DIR / "defaults"
_FASTERGS_BACKEND_MODULES = (
    "splatkit_adapter_backends.fastergs",
    "splatkit_native_faster_gs.faster_gs",
)


class FasterGSConfigBase(BaseModel):
    """Strict base class for FasterGS paper config sections."""

    model_config = {
        "extra": "forbid",
    }


class FasterGSSceneConfig(FasterGSConfigBase):
    """Scene-record loading options for the FasterGS paper notebook."""

    path: Path = Path("dataset/mipnerf360/garden")
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool = True


class FasterGSDataConfig(FasterGSConfigBase):
    """Prepared-frame dataset options for the FasterGS paper notebook."""

    camera_sensor_id: str | None = None
    image_scale_factor: float = Field(default=0.25, gt=0.0)
    split_target: Literal["train", "val", "all"] = "train"
    split_every_n: int | None = Field(default=8, ge=1)
    materialization_stage: Literal["none", "decoded", "prepared"] = "decoded"
    materialization_mode: Literal["lazy", "eager"] = "eager"
    materialization_num_workers: int | None = 0
    normalize_images: bool = True
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"


class FasterGSModelConfig(FasterGSConfigBase):
    """Gaussian initialization settings."""

    sh_degree: int = Field(default=3, ge=0)
    initial_scale: float = Field(default=0.01, gt=0.0)
    initial_opacity: float = Field(default=0.1, gt=0.0, lt=1.0)
    default_color: tuple[float, float, float] = (0.5, 0.5, 0.5)


class FasterGSRenderConfig(FasterGSConfigBase):
    """Explicit render options shared across FasterGS backends."""

    proper_antialiasing: bool = False
    near_plane: float = Field(default=0.2, gt=0.0)
    far_plane: float = Field(default=10_000.0, gt=0.0)
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)


class FasterGSOptimizationConfig(FasterGSConfigBase):
    """Optimizer settings following the FasterGS garden default config."""

    optimizer: str = "splatkit_gaussian_training.FusedAdam"
    means_lr_init: float = Field(default=1.6e-4, gt=0.0)
    means_lr_final: float = Field(default=1.6e-6, gt=0.0)
    means_lr_max_steps: int | None = Field(default=None, ge=1)
    sh_dc_lr: float = Field(default=2.5e-3, gt=0.0)
    sh_rest_lr: float = Field(default=1.25e-4, gt=0.0)
    opacity_lr: float = Field(default=2.5e-2, gt=0.0)
    scale_lr: float = Field(default=5e-3, gt=0.0)
    rotation_lr: float = Field(default=1e-3, gt=0.0)


class FasterGSLossConfig(FasterGSConfigBase):
    """Loss weights mirroring the FasterGS paper default config."""

    lambda_l1: float = Field(default=0.8, ge=0.0)
    lambda_dssim: float = Field(default=0.2, ge=0.0)
    lambda_opacity_regularization: float = Field(default=0.0, ge=0.0)
    lambda_scale_regularization: float = Field(default=0.0, ge=0.0)


class FasterGSDensificationConfig(FasterGSConfigBase):
    """Densification settings for the FasterGS paper notebook."""

    use_mcmc: bool = False
    refine_every: int = Field(default=100, ge=1)
    start_iter: int = Field(default=600, ge=0)
    stop_iter: int = Field(default=14_900, ge=0)
    grad_threshold: float = Field(default=2e-4, gt=0.0)
    dense_fraction: float = Field(default=0.01, gt=0.0)
    prune_opacity_threshold: float = Field(default=0.005, gt=0.0)
    opacity_reset_every: int = Field(default=3_000, ge=1)
    max_reset_opacity: float = Field(default=0.01, gt=0.0, lt=1.0)
    min_opacity: float = Field(default=0.005, gt=0.0, lt=1.0)
    max_primitives: int = Field(default=1_000_000, ge=1)
    noise_lr_scale: float = Field(default=5e5, gt=0.0)


class FasterGSCheckpointConfig(FasterGSConfigBase):
    """Checkpoint export settings."""

    output_dir: Path
    export_ply: bool = True


class FasterGSExecutionConfig(FasterGSConfigBase):
    """Training execution controls."""

    device: Literal["cpu", "cuda"] = "cuda"
    seed: int = 0
    max_steps: int = Field(default=30_000, ge=1)
    batch_size: int = Field(default=1, ge=1)
    shuffle: bool = True


class FasterGSExperimentConfig(FasterGSConfigBase):
    """Resolved experiment config for the FasterGS paper notebook."""

    preset: FasterGSDefaultName = "garden_baseline"
    backend: FasterGSBackendName = "adapter.fastergs"
    scene: FasterGSSceneConfig = Field(default_factory=FasterGSSceneConfig)
    data: FasterGSDataConfig = Field(default_factory=FasterGSDataConfig)
    model: FasterGSModelConfig = Field(default_factory=FasterGSModelConfig)
    render: FasterGSRenderConfig = Field(default_factory=FasterGSRenderConfig)
    optimization: FasterGSOptimizationConfig = Field(
        default_factory=FasterGSOptimizationConfig
    )
    loss: FasterGSLossConfig = Field(default_factory=FasterGSLossConfig)
    densification: FasterGSDensificationConfig = Field(
        default_factory=FasterGSDensificationConfig
    )
    checkpoint: FasterGSCheckpointConfig
    execution: FasterGSExecutionConfig = Field(
        default_factory=FasterGSExecutionConfig
    )


class FasterGSSceneOverrideConfig(FasterGSConfigBase):
    """Partial scene override config used by the script loader."""

    path: Path | None = None
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool | None = None


class FasterGSDataOverrideConfig(FasterGSConfigBase):
    """Partial prepared-dataset override config used by the script loader."""

    camera_sensor_id: str | None = None
    image_scale_factor: float | None = Field(default=None, gt=0.0)
    split_target: Literal["train", "val", "all"] | None = None
    split_every_n: int | None = Field(default=None, ge=1)
    materialization_stage: Literal["none", "decoded", "prepared"] | None = None
    materialization_mode: Literal["lazy", "eager"] | None = None
    materialization_num_workers: int | None = None
    normalize_images: bool | None = None
    interpolation: Literal["nearest", "bilinear", "bicubic"] | None = None


class FasterGSModelOverrideConfig(FasterGSConfigBase):
    """Partial model override config used by the script loader."""

    sh_degree: int | None = Field(default=None, ge=0)
    initial_scale: float | None = Field(default=None, gt=0.0)
    initial_opacity: float | None = Field(default=None, gt=0.0, lt=1.0)
    default_color: tuple[float, float, float] | None = None


class FasterGSRenderOverrideConfig(FasterGSConfigBase):
    """Partial render override config used by the script loader."""

    proper_antialiasing: bool | None = None
    near_plane: float | None = Field(default=None, gt=0.0)
    far_plane: float | None = Field(default=None, gt=0.0)
    background_color: tuple[float, float, float] | None = None


class FasterGSOptimizationOverrideConfig(FasterGSConfigBase):
    """Partial optimization override config used by the script loader."""

    optimizer: str | None = None
    means_lr_init: float | None = Field(default=None, gt=0.0)
    means_lr_final: float | None = Field(default=None, gt=0.0)
    means_lr_max_steps: int | None = Field(default=None, ge=1)
    sh_dc_lr: float | None = Field(default=None, gt=0.0)
    sh_rest_lr: float | None = Field(default=None, gt=0.0)
    opacity_lr: float | None = Field(default=None, gt=0.0)
    scale_lr: float | None = Field(default=None, gt=0.0)
    rotation_lr: float | None = Field(default=None, gt=0.0)


class FasterGSLossOverrideConfig(FasterGSConfigBase):
    """Partial loss override config used by the script loader."""

    lambda_l1: float | None = Field(default=None, ge=0.0)
    lambda_dssim: float | None = Field(default=None, ge=0.0)
    lambda_opacity_regularization: float | None = Field(default=None, ge=0.0)
    lambda_scale_regularization: float | None = Field(default=None, ge=0.0)


class FasterGSDensificationOverrideConfig(FasterGSConfigBase):
    """Partial densification override config used by the script loader."""

    use_mcmc: bool | None = None
    refine_every: int | None = Field(default=None, ge=1)
    start_iter: int | None = Field(default=None, ge=0)
    stop_iter: int | None = Field(default=None, ge=0)
    grad_threshold: float | None = Field(default=None, gt=0.0)
    dense_fraction: float | None = Field(default=None, gt=0.0)
    prune_opacity_threshold: float | None = Field(default=None, gt=0.0)
    opacity_reset_every: int | None = Field(default=None, ge=1)
    max_reset_opacity: float | None = Field(default=None, gt=0.0, lt=1.0)
    min_opacity: float | None = Field(default=None, gt=0.0, lt=1.0)
    max_primitives: int | None = Field(default=None, ge=1)
    noise_lr_scale: float | None = Field(default=None, gt=0.0)


class FasterGSCheckpointOverrideConfig(FasterGSConfigBase):
    """Partial checkpoint override config used by the script loader."""

    output_dir: Path | None = None
    export_ply: bool | None = None


class FasterGSExecutionOverrideConfig(FasterGSConfigBase):
    """Partial execution override config used by the script loader."""

    device: Literal["cpu", "cuda"] | None = None
    seed: int | None = None
    max_steps: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    shuffle: bool | None = None


class FasterGSExperimentOverrideConfig(FasterGSConfigBase):
    """Partial experiment override config used by the script loader."""

    preset: FasterGSDefaultName = "garden_baseline"
    backend: FasterGSBackendName | None = None
    scene: FasterGSSceneOverrideConfig = Field(
        default_factory=FasterGSSceneOverrideConfig
    )
    data: FasterGSDataOverrideConfig = Field(
        default_factory=FasterGSDataOverrideConfig
    )
    model: FasterGSModelOverrideConfig = Field(
        default_factory=FasterGSModelOverrideConfig
    )
    render: FasterGSRenderOverrideConfig = Field(
        default_factory=FasterGSRenderOverrideConfig
    )
    optimization: FasterGSOptimizationOverrideConfig = Field(
        default_factory=FasterGSOptimizationOverrideConfig
    )
    loss: FasterGSLossOverrideConfig = Field(
        default_factory=FasterGSLossOverrideConfig
    )
    densification: FasterGSDensificationOverrideConfig = Field(
        default_factory=FasterGSDensificationOverrideConfig
    )
    checkpoint: FasterGSCheckpointOverrideConfig = Field(
        default_factory=FasterGSCheckpointOverrideConfig
    )
    execution: FasterGSExecutionOverrideConfig = Field(
        default_factory=FasterGSExecutionOverrideConfig
    )


@dataclass(frozen=True)
class _JsonConfigSource:
    path: Annotated[Path, tyro.conf.Positional]


def _default_checkpoint_dir(
    preset: FasterGSDefaultName,
    backend: FasterGSBackendName,
) -> Path:
    return _DEFAULT_CHECKPOINT_ROOT / preset / backend


_DEFAULT_CONFIG_PATHS: dict[FasterGSDefaultName, Path] = {
    "garden_baseline": _DEFAULTS_DIR / "garden_baseline.json",
    "garden_mcmc": _DEFAULTS_DIR / "garden_mcmc.json",
}


def _resolve_relative_path(path: Path, *, base_dir: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path)


def _resolve_config_paths(
    config: FasterGSExperimentConfig,
    *,
    base_dir: Path,
    allow_default_scene_env_override: bool,
) -> FasterGSExperimentConfig:
    scene_path = config.scene.path
    env_scene_path = os.environ.get("SPLATKIT_FASTERGS_GARDEN_ROOT")
    if (
        allow_default_scene_env_override
        and env_scene_path is not None
        and scene_path == Path("dataset/mipnerf360/garden")
    ):
        scene_path = Path(env_scene_path)
    else:
        scene_path = _resolve_relative_path(scene_path, base_dir=base_dir)
    image_root = (
        _resolve_relative_path(config.scene.image_root, base_dir=base_dir)
        if config.scene.image_root is not None
        else None
    )
    undistort_output_dir = (
        _resolve_relative_path(
            config.scene.undistort_output_dir,
            base_dir=base_dir,
        )
        if config.scene.undistort_output_dir is not None
        else None
    )
    checkpoint_output_dir = _resolve_relative_path(
        config.checkpoint.output_dir,
        base_dir=base_dir,
    )
    return config.model_copy(
        update={
            "scene": config.scene.model_copy(
                update={
                    "path": scene_path,
                    "image_root": image_root,
                    "undistort_output_dir": undistort_output_dir,
                }
            ),
            "checkpoint": config.checkpoint.model_copy(
                update={
                    "output_dir": checkpoint_output_dir,
                }
            ),
        }
    )


def _load_config_json(path: Path) -> FasterGSExperimentConfig:
    resolved_path = path.expanduser().resolve()
    payload = json.loads(resolved_path.read_text())
    config = FasterGSExperimentConfig.model_validate(payload)
    base_dir = (
        _REPO_ROOT
        if resolved_path.is_relative_to(_DEFAULTS_DIR.resolve())
        else resolved_path.parent
    )
    return _resolve_config_paths(
        config,
        base_dir=base_dir,
        allow_default_scene_env_override=False,
    )


def _resolve_default_config_paths(
    config: FasterGSExperimentConfig,
) -> FasterGSExperimentConfig:
    return _resolve_config_paths(
        config,
        base_dir=_REPO_ROOT,
        allow_default_scene_env_override=True,
    )


def load_default_experiment_config(
    default: FasterGSDefaultName = "garden_baseline",
) -> FasterGSExperimentConfig:
    """Load a named default FasterGS experiment config from JSON."""
    return _resolve_default_config_paths(_load_config_json(_DEFAULT_CONFIG_PATHS[default]))


def _resolve_checkpoint_output_dir(
    config: FasterGSExperimentConfig,
) -> Path:
    default_parent = _DEFAULT_CHECKPOINT_ROOT / config.preset
    output_dir = config.checkpoint.output_dir.expanduser()
    if output_dir.parent == default_parent:
        return _default_checkpoint_dir(config.preset, config.backend)
    return output_dir


def _merge_model_overrides(
    base: BaseModel,
    override: BaseModel,
) -> BaseModel:
    updates: dict[str, Any] = {}
    for field_name in type(override).model_fields:
        override_value = getattr(override, field_name)
        base_value = getattr(base, field_name)
        if isinstance(override_value, BaseModel) and isinstance(
            base_value,
            BaseModel,
        ):
            updates[field_name] = _merge_model_overrides(
                base_value,
                override_value,
            )
            continue
        if override_value is not None:
            updates[field_name] = override_value
    return base.model_copy(update=updates)


def merge_experiment_config(
    base: FasterGSExperimentConfig,
    override: FasterGSExperimentOverrideConfig,
) -> FasterGSExperimentConfig:
    """Merge a partial override config into a resolved experiment config."""
    merged = _merge_model_overrides(base, override)
    assert isinstance(merged, FasterGSExperimentConfig)
    return merged.model_copy(update={"preset": override.preset})


def load_experiment_script_config(
    model_cls: type[BaseModel],
    value: BaseModel | dict[str, Any] | None = None,
    args: Sequence[str] | None = None,
) -> BaseModel:
    """Default-aware script loader for `marimo-config-gui`."""
    del model_cls
    default_config = (
        FasterGSExperimentConfig.model_validate(value)
        if value is not None
        else load_default_experiment_config()
    )
    script_input_type = (
        Annotated[
            FasterGSExperimentOverrideConfig,
            tyro.conf.subcommand(
                "cli",
                default=FasterGSExperimentOverrideConfig(
                    preset=default_config.preset
                ),
            ),
        ]
        | Annotated[_JsonConfigSource, tyro.conf.subcommand("json")]
    )
    parsed = tyro.cli(script_input_type, args=args)
    if isinstance(parsed, _JsonConfigSource):
        return _load_config_json(parsed.path)
    base = load_default_experiment_config(parsed.preset)
    return merge_experiment_config(base, parsed)


def register_fastergs_backends() -> tuple[str, ...]:
    """Best-effort registration for the FasterGS backends used by the notebook."""
    registered: list[str] = []
    for module_name in _FASTERGS_BACKEND_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        register = getattr(module, "register", None)
        if callable(register):
            register()
        registered.append(module_name)
    return tuple(registered)


def build_scene_load_config(
    config: FasterGSExperimentConfig,
) -> sk.ColmapSceneConfig:
    """Translate the paper config into a shared scene-load config."""
    source_pipes = (
        (sk.HorizonAlignPipeConfig(),) if config.scene.align_horizon else ()
    )
    return sk.ColmapSceneConfig(
        path=config.scene.path.expanduser(),
        image_root=(
            config.scene.image_root.expanduser()
            if config.scene.image_root is not None
            else None
        ),
        undistort_output_dir=(
            config.scene.undistort_output_dir.expanduser()
            if config.scene.undistort_output_dir is not None
            else None
        ),
        source_pipes=source_pipes,
    )


def build_prepared_frame_dataset_config(
    config: FasterGSExperimentConfig,
) -> sk.PreparedFrameDatasetConfig:
    """Translate the paper config into a prepared-frame dataset config."""
    split = None
    if config.data.split_target == "all":
        split = sk.SplitConfig(target="all", every_n=None, train_ratio=None)
    else:
        split = sk.SplitConfig(
            target=config.data.split_target,
            every_n=config.data.split_every_n,
            train_ratio=None,
        )
    return sk.PreparedFrameDatasetConfig(
        camera_sensor_id=config.data.camera_sensor_id,
        split=split,
        materialization=sk.MaterializationConfig(
            stage=config.data.materialization_stage,
            mode=config.data.materialization_mode,
            num_workers=config.data.materialization_num_workers,
        ),
        image_preparation=sk.ImagePreparationConfig(
            normalize=config.data.normalize_images,
            resize_width_scale=config.data.image_scale_factor,
            resize_width_target=None,
            interpolation=config.data.interpolation,
        ),
    )


def build_fastergs_mcmc_densification(
    *,
    refine_every: int,
    start_iter: int,
    stop_iter: int,
    min_opacity: float,
    max_primitives: int,
    noise_lr_scale: float,
) -> Any:
    """Construct the CUDA-backed Gaussian MCMC densification method."""
    from splatkit.densification import Schedule
    from splatkit_gaussian_training import GaussianMCMC

    return GaussianMCMC(
        schedule=Schedule(
            start_iteration=start_iter,
            end_iteration=stop_iter,
            frequency=refine_every,
        ),
        min_opacity=min_opacity,
        cap_max=max_primitives,
        noise_lr_scale=noise_lr_scale,
    )


def build_training_config(
    config: FasterGSExperimentConfig,
) -> sk.TrainingConfig:
    """Translate the paper config into a shared declarative training config."""
    means_lr_max_steps = (
        config.optimization.means_lr_max_steps
        if config.optimization.means_lr_max_steps is not None
        else config.execution.max_steps
    )
    parameter_groups = [
        sk.ParameterGroupConfig(
            target=sk.ParameterTargetSpec(
                scope="scene",
                name="center_position",
            ),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.means_lr_init,
            scheduler=sk.CallableSpec(
                target="splatkit.training.exponential_decay_to",
                kwargs={
                    "final_lr": config.optimization.means_lr_final,
                    "max_steps": means_lr_max_steps,
                },
            ),
        ),
        sk.ParameterGroupConfig(
            target=sk.ParameterTargetSpec(
                scope="scene",
                name="feature",
                view=sk.TensorViewSpec(
                    slices=(sk.TensorSliceSpec(axis=1, start=0, stop=1),)
                ),
            ),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.sh_dc_lr,
        ),
        sk.ParameterGroupConfig(
            target=sk.ParameterTargetSpec(
                scope="scene",
                name="feature",
                view=sk.TensorViewSpec(
                    slices=(sk.TensorSliceSpec(axis=1, start=1, stop=None),)
                ),
            ),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.sh_rest_lr,
        ),
        sk.ParameterGroupConfig(
            target=sk.ParameterTargetSpec(
                scope="scene",
                name="logit_opacity",
            ),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.opacity_lr,
        ),
        sk.ParameterGroupConfig(
            target=sk.ParameterTargetSpec(
                scope="scene",
                name="log_scales",
            ),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.scale_lr,
        ),
        sk.ParameterGroupConfig(
            target=sk.ParameterTargetSpec(
                scope="scene",
                name="quaternion_orientation",
            ),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.rotation_lr,
        ),
    ]
    if config.densification.use_mcmc:
        densification_builder = sk.CallableSpec(
            target="config.build_fastergs_mcmc_densification",
            kwargs={
                "refine_every": config.densification.refine_every,
                "start_iter": config.densification.start_iter,
                "stop_iter": config.densification.stop_iter,
                "min_opacity": config.densification.min_opacity,
                "max_primitives": config.densification.max_primitives,
                "noise_lr_scale": config.densification.noise_lr_scale,
            },
        )
    else:
        densification_builder = sk.CallableSpec(
            target="splatkit.Vanilla3DGS",
            kwargs={
                "refine_every": config.densification.refine_every,
                "start_iter": config.densification.start_iter,
                "stop_iter": config.densification.stop_iter,
                "grad_threshold": config.densification.grad_threshold,
                "relative_size_threshold": config.densification.dense_fraction,
                "prune_opacity_threshold": (
                    config.densification.prune_opacity_threshold
                ),
                "opacity_reset_every": config.densification.opacity_reset_every,
                "max_reset_opacity": config.densification.max_reset_opacity,
            },
        )
    return sk.TrainingConfig(
        runtime=sk.RuntimeConfig(
            device=config.execution.device,
            seed=config.execution.seed,
            max_steps=config.execution.max_steps,
        ),
        batching=sk.BatchingConfig(
            batch_size=config.execution.batch_size,
            shuffle=config.execution.shuffle,
        ),
        initialization=sk.InitializationSpec(
            initializer=sk.CallableSpec(
                target=(
                    "splatkit.initialization.initialize_gaussian_model_from_scene_record"
                ),
                kwargs={
                    "sh_degree": config.model.sh_degree,
                    "initial_scale": config.model.initial_scale,
                    "initial_opacity": config.model.initial_opacity,
                    "default_color": config.model.default_color,
                },
            )
        ),
        render=sk.RenderPipelineSpec(
            backend=config.backend,
            backend_options={
                "near_plane": config.render.near_plane,
                "far_plane": config.render.far_plane,
                "proper_antialiasing": config.render.proper_antialiasing,
                "background_color": list(config.render.background_color),
            },
        ),
        optimization=sk.OptimizationConfig(parameter_groups=parameter_groups),
        loss=sk.LossConfig(
            target=sk.CallableSpec(
                target="config.fastergs_training_loss",
                kwargs={
                    "lambda_l1": config.loss.lambda_l1,
                    "lambda_dssim": config.loss.lambda_dssim,
                    "lambda_opacity_regularization": (
                        config.loss.lambda_opacity_regularization
                    ),
                    "lambda_scale_regularization": (
                        config.loss.lambda_scale_regularization
                    ),
                },
            )
        ),
        densification=sk.DensificationConfig(builder=densification_builder),
        checkpoint=sk.CheckpointExportConfig(
            output_dir=_resolve_checkpoint_output_dir(config),
            export_ply=config.checkpoint.export_ply,
        ),
    )


def _gaussian_window(
    *,
    kernel_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, " channels 1 kernel_size kernel_size"]:
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = torch.outer(gauss_1d, gauss_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()


def _ssim_score(
    prediction: Float[Tensor, " batch height width 3"],
    target: Float[Tensor, " batch height width 3"],
    *,
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    prediction_nchw = prediction.permute(0, 3, 1, 2)
    target_nchw = target.permute(0, 3, 1, 2)
    channels = int(prediction_nchw.shape[1])
    window = _gaussian_window(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=channels,
        device=prediction_nchw.device,
        dtype=prediction_nchw.dtype,
    )
    padding = kernel_size // 2
    mu_prediction = F.conv2d(
        prediction_nchw,
        window,
        padding=padding,
        groups=channels,
    )
    mu_target = F.conv2d(
        target_nchw,
        window,
        padding=padding,
        groups=channels,
    )
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    sigma_prediction_sq = (
        F.conv2d(
            prediction_nchw.square(),
            window,
            padding=padding,
            groups=channels,
        )
        - mu_prediction_sq
    )
    sigma_target_sq = (
        F.conv2d(
            target_nchw.square(),
            window,
            padding=padding,
            groups=channels,
        )
        - mu_target_sq
    )
    sigma_product = (
        F.conv2d(
            prediction_nchw * target_nchw,
            window,
            padding=padding,
            groups=channels,
        )
        - mu_product
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_product + c1) * (2.0 * sigma_product + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        sigma_prediction_sq + sigma_target_sq + c2
    )
    return (numerator / denominator).mean()


def fastergs_training_loss(
    state: TrainState,
    batch: Any,
    render_output: Any,
    *,
    weights: dict[str, float],
    lambda_l1: float,
    lambda_dssim: float,
    lambda_opacity_regularization: float,
    lambda_scale_regularization: float,
) -> LossResult:
    """Paper-style FasterGS RGB loss over NHWC batched images."""
    del weights
    prediction = render_output.render
    target = batch.images
    if prediction.shape != target.shape:
        raise ValueError(
            "FasterGS training loss expects render output and batch images "
            f"to share the same NHWC shape, got {tuple(prediction.shape)!r} "
            f"vs {tuple(target.shape)!r}."
        )
    l1_loss = (prediction - target).abs().mean()
    dssim = (1.0 - _ssim_score(prediction, target)) / 2.0
    scene = state.model.scene
    assert isinstance(scene, sk.GaussianScene3D)
    opacity_regularization = torch.sigmoid(scene.logit_opacity).mean()
    scale_regularization = torch.exp(scene.log_scales).mean()
    loss = (
        lambda_l1 * l1_loss
        + lambda_dssim * dssim
        + lambda_opacity_regularization * opacity_regularization
        + lambda_scale_regularization * scale_regularization
    )
    return LossResult(
        loss=loss,
        metrics={
            "l1": float(l1_loss.detach().item()),
            "dssim": float(dssim.detach().item()),
            "opacity_regularization": float(
                opacity_regularization.detach().item()
            ),
            "scale_regularization": float(scale_regularization.detach().item()),
        },
    )


__all__ = [
    "FasterGSBackendName",
    "FasterGSCheckpointConfig",
    "FasterGSDataConfig",
    "FasterGSDensificationConfig",
    "FasterGSExecutionConfig",
    "FasterGSExperimentConfig",
    "FasterGSExperimentOverrideConfig",
    "FasterGSLossConfig",
    "FasterGSModelConfig",
    "FasterGSOptimizationConfig",
    "FasterGSDefaultName",
    "FasterGSRenderConfig",
    "FasterGSSceneConfig",
    "build_fastergs_mcmc_densification",
    "build_prepared_frame_dataset_config",
    "build_scene_load_config",
    "build_training_config",
    "fastergs_training_loss",
    "load_default_experiment_config",
    "load_experiment_script_config",
    "merge_experiment_config",
    "register_fastergs_backends",
]
