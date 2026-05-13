"""SVRaster paper training notebook for Ember."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import json
    import shutil
    import sys
    from pathlib import Path
    from typing import Any, Literal

    import ember_core as ember
    import ember_native_svraster
    import ember_splatting_training as ember_splatting
    import marimo as mo
    from ember_core.training import TrainingProfilerConfig, TrainingResult
    from marimo_config_gui import (
        ConfigPreset,
        ConfigPresetCatalog,
        create_config_gui,
    )
    from pydantic import BaseModel, Field

    NOTEBOOK_PATH = Path(__file__).resolve()
    NOTEBOOK_DIR = NOTEBOOK_PATH.parent
    REPO_ROOT = NOTEBOOK_DIR.parents[1]
    DEFAULTS_DIR = NOTEBOOK_DIR / "defaults"
    DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "papers" / "svraster"
    SVRasterBackendName = Literal["svraster.core"]
    SVRasterDefaultName = Literal[
        "garden_svraster",
        "garden_fast_train",
        "garden_fast_render",
        "garden_debug_val",
    ]
    sys.modules.setdefault("papers.svraster.notebook", sys.modules[__name__])
    ember_native_svraster.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # SVRaster training
    """)
    return


@app.cell(hide_code=True)
def _(training_controls):
    training_controls
    return


@app.cell
def _():
    svraster_presets = svraster_preset_catalog()
    config_gui = create_config_gui(
        SVRasterExperimentConfig,
        presets=svraster_presets,
        label="SVRaster config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    preset_selector = config_gui.preset_selector(
        label="SVRaster preset",
    )
    return (preset_selector,)


@app.cell
def _(config_gui):
    current_config = config_gui.validated_config()
    return (current_config,)


@app.cell(hide_code=True)
def _(preset_selector):
    preset_selector
    return


@app.cell(hide_code=True)
def _(config_gui):
    config_gui.stacked()
    return


@app.cell(hide_code=True)
def _(training_result_view):
    training_result_view
    return


@app.cell(hide_code=True)
def _(training_viewer):
    training_viewer
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Config definition
    """)
    return


@app.class_definition
class SVRasterConfigBase(BaseModel):
    """Strict base model for SVRaster paper configs."""

    model_config = {"extra": "forbid", "populate_by_name": True}


@app.class_definition
class SVRasterSceneConfig(SVRasterConfigBase):
    """Scene-record loading options."""

    path: Path = Path("dataset/mipnerf360/garden")
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool = True


@app.class_definition
class SVRasterDataConfig(SVRasterConfigBase):
    """Prepared-frame dataset options."""

    camera_sensor_id: str | None = None
    image_scale_factor: float = Field(default=0.25, gt=0.0)
    cache_resized_images: bool = True
    resized_image_cache_root: Path | None = None
    max_resized_image_caches: int = Field(default=4, ge=1)
    split_target: Literal["train", "val", "all"] = "train"
    split_every_n: int | None = Field(default=8, ge=1)
    normalize_images: bool = True
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"
    materialization_stage: Literal["none", "decoded", "prepared"] = "prepared"
    materialization_mode: Literal["lazy", "eager"] = "eager"
    materialization_num_workers: int | None = 8


@app.class_definition
class SVRasterModelConfig(SVRasterConfigBase):
    """SVRaster sparse-voxel model defaults."""

    backend_name: Literal["new_cuda"] = "new_cuda"
    max_num_levels: int = Field(default=21, ge=1)
    runtime_max_num_levels: int = Field(default=21, ge=1)
    sh_degree: int = Field(default=3, ge=0)
    initial_sh_degree: int = Field(default=3, ge=0)
    samples_per_voxel: int = Field(default=1, ge=1)
    render_supersampling: float = Field(default=1.5, ge=1.0)
    ss_aug_max: float = Field(default=1.5, ge=1.0)
    white_background: bool = False
    black_background: bool = False


@app.class_definition
class SVRasterInitializationConfig(SVRasterConfigBase):
    """Sparse-voxel initialization defaults."""

    initial_inside_level: int = Field(default=6, ge=1)
    outside_level: int = Field(default=5, ge=0)
    initial_outside_ratio: float = Field(default=2.0, ge=0.0)
    geometry_initial_value: float = -10.0
    sh0_initial_rgb: float = Field(default=0.5, ge=0.0, le=1.0)
    shs_initial_value: float = 0.0
    bound_mode: Literal["default", "forward", "camera_median", "camera_max"] = (
        "default"
    )
    bound_scale: float = Field(default=1.0, gt=0.0)
    forward_distance_scale: float = Field(default=1.0, gt=0.0)
    filter_zero_visibility: bool = True

    def build(
        self,
        context: ember.TrainingRunContext,
        *,
        model: SVRasterModelConfig,
    ) -> ember.InitializationSpec:
        """Build the runtime sparse-voxel initializer spec."""
        del context
        return ember.InitializationSpec(
            initializer=ember.bound_callable(
                target="ember_svraster_training.initialize_svraster_paper_scene",
                kwargs={
                    **self.model_dump(mode="python"),
                    "backend_name": model.backend_name,
                    "max_num_levels": model.max_num_levels,
                    "sh_degree": model.sh_degree,
                    "initial_sh_degree": model.initial_sh_degree,
                },
                bind={
                    "device": ember.ctx.run.device,
                    "frame_dataset": ember.ctx.run.frame_dataset,
                },
            )
        )


@app.class_definition
class SVRasterOptimizationRecipeConfig(SVRasterConfigBase):
    """Optimizer defaults from upstream SVRaster."""

    geo_lr: float = Field(default=0.025, gt=0.0)
    sh0_lr: float = Field(default=0.010, gt=0.0)
    shs_lr: float = Field(default=0.00025, gt=0.0)
    beta1: float = Field(default=0.1, ge=0.0, lt=1.0)
    beta2: float = Field(default=0.99, ge=0.0, lt=1.0)
    epsilon: float = Field(default=1e-15, gt=0.0)
    lr_decay_checkpoints: tuple[int, ...] = (19000,)
    lr_decay_multiplier: float = Field(default=0.1, gt=0.0)

    def build(self) -> ember.OptimizationConfig:
        """Build SVRaster optimizer groups through the training package."""
        return ember.OptimizationConfig(
            builder=ember.bound_callable(
                target="ember_svraster_training.svraster_optimization_config",
                kwargs={
                    "recipe": {
                        "geo_lr": self.geo_lr,
                        "sh0_lr": self.sh0_lr,
                        "shs_lr": self.shs_lr,
                        "betas": (self.beta1, self.beta2),
                        "eps": self.epsilon,
                        "lr_decay_checkpoints": self.lr_decay_checkpoints,
                        "lr_decay_multiplier": self.lr_decay_multiplier,
                    }
                },
            )
        )


@app.class_definition
class SVRasterLossConfig(SVRasterConfigBase):
    """SVRaster photometric loss weights."""

    lambda_photo: float = Field(default=1.0, ge=0.0)
    lambda_ssim: float = Field(default=0.02, ge=0.0)
    use_l1: bool = False
    use_huber: bool = False
    huber_threshold: float = Field(default=0.03, gt=0.0)
    lambda_t_concentration: float = Field(default=0.0, ge=0.0)
    lambda_t_inside: float = Field(default=0.01, ge=0.0)
    lambda_r_concentration: float = Field(default=0.01, ge=0.0)
    lambda_ascending: float = Field(default=0.0, ge=0.0)
    ascending_start_step: int = Field(default=0, ge=0)
    lambda_distortion: float = Field(default=0.1, ge=0.0)
    distortion_start_step: int = Field(default=10000, ge=0)
    ssim_backend: str = "cuda"

    def build(self) -> ember.LossConfig:
        """Build the runtime SVRaster loss spec."""
        return ember.loss_config(
            "ember_svraster_training.svraster_paper_rgb_loss",
            kwargs={
                "lambda_photo": self.lambda_photo,
                "lambda_ssim": self.lambda_ssim,
                "use_l1": self.use_l1,
                "use_huber": self.use_huber,
                "huber_threshold": self.huber_threshold,
                "lambda_t_concentration": self.lambda_t_concentration,
                "lambda_t_inside": self.lambda_t_inside,
                "ssim_backend": self.ssim_backend,
            },
        )


@app.class_definition
class SVRasterRegularizationConfig(SVRasterConfigBase):
    """SVRaster native training regularizers."""

    tv_density_weight: float = Field(default=1e-10, ge=0.0)
    tv_start_step: int = Field(default=0, ge=0)
    tv_end_step: int = Field(default=10000, ge=0)

    def build_hooks(self) -> list[ember.CallableSpec]:
        """Build hook specs for native SVRaster regularizers."""
        if self.tv_density_weight == 0.0:
            return []
        return [
            ember.bound_callable(
                target="ember_svraster_training.SVRasterTVDensityHook",
                kwargs={
                    "weight": self.tv_density_weight,
                    "start_step": self.tv_start_step,
                    "end_step": self.tv_end_step,
                },
            )
        ]


@app.class_definition
class SVRasterAdaptiveConfig(SVRasterConfigBase):
    """Notebook-local adaptive prune/subdivide schedule."""

    adapt_from: int = Field(default=1000, ge=0)
    adapt_every: int = Field(default=1000, ge=1)
    prune_until: int = Field(default=18000, ge=0)
    prune_threshold_initial: float = Field(default=0.0001, ge=0.0)
    prune_threshold_final: float = Field(default=0.05, ge=0.0)
    subdivide_until: int = Field(default=15000, ge=0)
    subdivide_all_until: int = Field(default=0, ge=0)
    subdivide_sample_threshold: float = Field(default=1.0, ge=0.0)
    subdivide_proportion: float = Field(default=0.05, ge=0.0, le=1.0)
    subdivide_max_voxels: int = Field(default=10_000_000, ge=1)

    def build(self, *, model: SVRasterModelConfig) -> ember.DensificationConfig:
        """Build the SVRaster adaptive pruning/subdivision config."""
        return ember.densification_config(
            ember.bound_callable(
                target=(
                    "ember_svraster_training.SVRasterAdaptivePruneSubdivide"
                ),
                kwargs={
                    **self.model_dump(mode="python"),
                    "max_num_levels": model.runtime_max_num_levels,
                },
            )
        )


@app.class_definition
class SVRasterTrainingConfig(SVRasterConfigBase):
    """Full training config for the SVRaster paper notebook."""

    runtime: ember.RuntimeConfig = Field(
        default_factory=lambda: ember.RuntimeConfig(
            device="cuda",
            seed=3721,
            max_steps=20000,
        )
    )
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    batching: ember.BatchingConfig = Field(
        default_factory=lambda: ember.BatchingConfig(
            batch_size=1,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )
    )
    checkpoint: ember.CheckpointExportConfig = Field(
        default_factory=lambda: ember.CheckpointExportConfig(
            output_dir=Path("checkpoints/latest"),
            export_ply=False,
            overwrite=True,
        )
    )
    model: SVRasterModelConfig = Field(default_factory=SVRasterModelConfig)
    initialization: SVRasterInitializationConfig = Field(
        default_factory=SVRasterInitializationConfig
    )
    optimization: SVRasterOptimizationRecipeConfig = Field(
        default_factory=SVRasterOptimizationRecipeConfig
    )
    loss: SVRasterLossConfig = Field(default_factory=SVRasterLossConfig)
    regularization: SVRasterRegularizationConfig = Field(
        default_factory=SVRasterRegularizationConfig
    )
    adaptive: SVRasterAdaptiveConfig = Field(
        default_factory=SVRasterAdaptiveConfig
    )
    viewer: ember_splatting.TrainingViewerConfig = Field(
        default_factory=ember_splatting.TrainingViewerConfig
    )


@app.class_definition
class SVRasterExperimentConfig(SVRasterConfigBase):
    """Top-level notebook config."""

    preset: SVRasterDefaultName | None = None
    scene: SVRasterSceneConfig = Field(default_factory=SVRasterSceneConfig)
    data: SVRasterDataConfig = Field(default_factory=SVRasterDataConfig)
    training: SVRasterTrainingConfig = Field(
        default_factory=SVRasterTrainingConfig
    )

    def to_training_config(
        self,
        frame_dataset: ember.PreparedFrameDataset | None = None,
    ) -> ember.TrainingConfig:
        """Build the runtime Ember training config."""
        del frame_dataset
        hooks = [*self.training.regularization.build_hooks()]
        return ember.TrainingConfig(
            runtime=self.training.runtime,
            profiler=self.training.profiler,
            batching=self.training.batching,
            initialization=self.training.initialization.build(
                None,
                model=self.training.model,
            ),
            render=ember.RenderPipelineSpec(
                backend="svraster.core",
                return_alpha=False,
                return_depth=(
                    self.training.loss.lambda_distortion > 0.0
                    and self.training.loss.distortion_start_step == 0
                ),
                backend_options={
                    "near_plane": 0.02,
                    "samples_per_voxel": (
                        self.training.model.samples_per_voxel
                    ),
                    "sort_rank_max_level": (
                        self.training.model.runtime_max_num_levels
                    ),
                    "supersampling": (self.training.model.render_supersampling),
                    "white_background": self.training.model.white_background,
                    "black_background": self.training.model.black_background,
                    "return_transmittance": (
                        self.training.loss.lambda_t_concentration > 0.0
                        or self.training.loss.lambda_t_inside > 0.0
                    ),
                    "track_max_weight": True,
                },
                training_backend_options_builder=ember.bound_callable(
                    target=(
                        "ember_svraster_training."
                        "svraster_paper_training_backend_options"
                    ),
                    kwargs={
                        "distortion_weight": (
                            self.training.loss.lambda_distortion
                        ),
                        "distortion_start_step": (
                            self.training.loss.distortion_start_step
                        ),
                        "ascending_weight": self.training.loss.lambda_ascending,
                        "ascending_start_step": (
                            self.training.loss.ascending_start_step
                        ),
                        "color_concentration_weight": (
                            self.training.loss.lambda_r_concentration
                        ),
                        "default_supersampling": (
                            self.training.model.render_supersampling
                        ),
                        "ss_aug_max": self.training.model.ss_aug_max,
                    },
                ),
            ),
            optimization=self.training.optimization.build(),
            loss=self.training.loss.build(),
            densification=self.training.adaptive.build(
                model=self.training.model
            ),
            hooks=ember.HookConfig(builders=hooks),
            checkpoint=self.training.checkpoint,
        )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Function definitions
    """)
    return


@app.function
def default_checkpoint_dir(
    preset: SVRasterDefaultName,
    backend: SVRasterBackendName,
) -> Path:
    """Return the default checkpoint directory for a preset/backend pair."""
    return DEFAULT_CHECKPOINT_ROOT / preset / backend


@app.function
def svraster_preset_catalog() -> ConfigPresetCatalog[SVRasterExperimentConfig]:
    """Return the notebook's named JSON preset catalog."""
    return ConfigPresetCatalog(
        model_cls=SVRasterExperimentConfig,
        presets={
            "garden_svraster": ConfigPreset(
                name="garden_svraster",
                path=DEFAULTS_DIR / "garden_svraster.json",
                label="Garden SVRaster",
                base_dir=REPO_ROOT,
            ),
            "garden_fast_train": ConfigPreset(
                name="garden_fast_train",
                path=DEFAULTS_DIR / "garden_fast_train.json",
                label="Garden fast train",
                base_dir=REPO_ROOT,
            ),
            "garden_fast_render": ConfigPreset(
                name="garden_fast_render",
                path=DEFAULTS_DIR / "garden_fast_render.json",
                label="Garden fast render",
                base_dir=REPO_ROOT,
            ),
            "garden_debug_val": ConfigPreset(
                name="garden_debug_val",
                path=DEFAULTS_DIR / "garden_debug_val.json",
                label="Garden debug validation",
                base_dir=REPO_ROOT,
            ),
        },
        default="garden_svraster",
    )


@app.function
def svraster_step_includes(
    step: int,
    start_step: int,
    end_step: int,
    frequency: int,
) -> bool:
    """Return whether an upstream-style adaptive schedule includes a step."""
    return step >= start_step and step <= end_step and step % frequency == 0


@app.function
def svraster_prune_threshold(
    step: int,
    *,
    adapt_from: int,
    prune_until: int,
    initial: float,
    final: float,
) -> float:
    """Linearly interpolate the upstream pruning threshold schedule."""
    if prune_until <= adapt_from:
        return final
    ratio = (step - adapt_from) / float(prune_until - adapt_from)
    clamped_ratio = min(max(ratio, 0.0), 1.0)
    return initial + clamped_ratio * (final - initial)


@app.function
def svraster_max_subdivide_count(
    current_voxels: int,
    max_voxels: int,
) -> int:
    """Return the upstream cap on voxels selected for subdivision."""
    return max(0, round((max_voxels - current_voxels) / 7))


@app.function
def resolve_checkpoint_output_dir(
    config: SVRasterExperimentConfig,
) -> Path:
    """Resolve the output checkpoint directory for the current config."""
    preset = config.preset or "garden_svraster"
    output_dir = config.training.checkpoint.output_dir
    default_dir = default_checkpoint_dir("garden_svraster", "svraster.core")
    is_latest = output_dir == Path("checkpoints/latest") or (
        output_dir.name == "latest" and output_dir.parent.name == "checkpoints"
    )
    if is_latest or output_dir == default_dir:
        return default_checkpoint_dir(preset, "svraster.core")
    return output_dir


@app.function
def resolve_training_config(
    experiment_config: SVRasterExperimentConfig,
    frame_dataset: ember.PreparedFrameDataset | None = None,
) -> ember.TrainingConfig:
    """Resolve notebook config to Ember's runtime TrainingConfig."""
    checkpoint = experiment_config.training.checkpoint.model_copy(
        update={
            "output_dir": resolve_checkpoint_output_dir(experiment_config),
        },
    )
    training = experiment_config.training.model_copy(
        update={"checkpoint": checkpoint},
        deep=True,
    )
    resolved_config = experiment_config.model_copy(
        update={"training": training},
        deep=True,
    )
    return resolved_config.to_training_config(frame_dataset)


@app.function
def format_duration(seconds: float) -> str:
    """Format a short ETA duration."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m"
    if minutes > 0:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


@app.function
def resolved_svraster_scene_path(config: SVRasterExperimentConfig) -> Path:
    """Resolve the configured scene path."""
    return config.scene.path.expanduser()


@app.function
def resolved_svraster_resized_cache_parent(
    config: SVRasterExperimentConfig,
) -> Path:
    """Return the resized-image cache parent for a config."""
    if config.data.resized_image_cache_root is not None:
        return config.data.resized_image_cache_root.expanduser()
    return (
        resolved_svraster_scene_path(config) / "ember_cache" / "resized_images"
    )


@app.function
def svraster_source_image_root(config: SVRasterExperimentConfig) -> Path:
    """Return the full-resolution source image root."""
    if config.scene.image_root is not None:
        return config.scene.image_root.expanduser()
    return resolved_svraster_scene_path(config) / "images"


@app.function
def svraster_resized_cache_enabled(
    config: SVRasterExperimentConfig,
) -> bool:
    """Return whether SVRaster should use a derived resized image cache."""
    return (
        config.data.cache_resized_images
        and config.data.image_scale_factor != 1.0
    )


@app.function
def svraster_resized_cache_root(config: SVRasterExperimentConfig) -> Path:
    """Return the derived resized image cache root for this config."""
    scale_name = f"{config.data.image_scale_factor:.6f}".rstrip("0").rstrip(".")
    scale_name = scale_name.replace(".", "p")
    return resolved_svraster_resized_cache_parent(config) / (
        f"scale_{scale_name}_{config.data.interpolation}"
    )


@app.function
def svraster_pillow_resampling(interpolation: str) -> Any:
    """Translate notebook interpolation names to Pillow resampling filters."""
    from PIL import Image

    if interpolation == "nearest":
        return Image.Resampling.NEAREST
    if interpolation == "bilinear":
        return Image.Resampling.BILINEAR
    if interpolation == "bicubic":
        return Image.Resampling.BICUBIC
    raise ValueError(f"Unsupported interpolation mode {interpolation!r}.")


@app.function
def enforce_svraster_resized_cache_limit(
    *,
    cache_root: Path,
    max_caches: int,
) -> None:
    """Keep only a bounded number of reusable resized image caches."""
    parent = cache_root.parent
    if not parent.exists():
        return
    cache_dirs = [
        path
        for path in parent.iterdir()
        if path.is_dir() and path.name.startswith("scale_")
    ]
    overflow = len(cache_dirs) - max_caches
    if overflow <= 0:
        return
    evictable = sorted(
        (path for path in cache_dirs if path != cache_root),
        key=lambda path: path.stat().st_mtime,
    )
    for stale_cache in evictable[:overflow]:
        shutil.rmtree(stale_cache)


@app.function
def materialize_svraster_resized_image_cache(
    *,
    source_root: Path,
    cache_root: Path,
    scale: float,
    interpolation: str,
    max_caches: int,
) -> Path:
    """Create/update a derived resized image cache from full-res images."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from PIL import Image
    from tqdm.auto import tqdm

    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    source_paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    if not source_paths:
        raise ValueError(f"No source images found under {source_root}.")
    resampling = svraster_pillow_resampling(interpolation)
    enforce_svraster_resized_cache_limit(
        cache_root=cache_root,
        max_caches=max_caches,
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    def resize_one(source_path: Path) -> None:
        relative_path = source_path.relative_to(source_root)
        target_path = cache_root / relative_path
        if (
            target_path.exists()
            and target_path.stat().st_mtime >= source_path.stat().st_mtime
        ):
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            resized_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            resized = rgb.resize(resized_size, resampling)
            save_kwargs = (
                {"quality": 95}
                if target_path.suffix.lower() in {".jpg", ".jpeg"}
                else {}
            )
            resized.save(target_path, **save_kwargs)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(resize_one, path) for path in source_paths]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preparing resized image cache",
        ):
            future.result()
    (cache_root / "cache_metadata.json").write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "scale": scale,
                "interpolation": interpolation,
                "num_images": len(source_paths),
            },
            indent=2,
            sort_keys=True,
        )
    )
    cache_root.touch()
    enforce_svraster_resized_cache_limit(
        cache_root=cache_root,
        max_caches=max_caches,
    )
    return cache_root


@app.function
def build_scene_load_config(
    config: SVRasterExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Build scene loading config from notebook options."""
    source_pipes = (
        (ember.HorizonAlignPipeConfig(),) if config.scene.align_horizon else ()
    )
    image_root = (
        materialize_svraster_resized_image_cache(
            source_root=svraster_source_image_root(config),
            cache_root=svraster_resized_cache_root(config),
            scale=config.data.image_scale_factor,
            interpolation=config.data.interpolation,
            max_caches=config.data.max_resized_image_caches,
        )
        if svraster_resized_cache_enabled(config)
        else (
            config.scene.image_root.expanduser()
            if config.scene.image_root is not None
            else None
        )
    )
    return ember.ColmapSceneConfig(
        path=resolved_svraster_scene_path(config),
        image_root=image_root,
        undistort_output_dir=(
            config.scene.undistort_output_dir.expanduser()
            if config.scene.undistort_output_dir is not None
            else None
        ),
        source_pipes=source_pipes,
    )


@app.function
def build_prepared_frame_dataset_config(
    config: SVRasterExperimentConfig,
) -> ember.PreparedFrameDatasetConfig:
    """Build prepared-frame dataset config from notebook options."""
    split = ember.SplitConfig(
        target=config.data.split_target,
        every_n=(
            None
            if config.data.split_target == "all"
            else config.data.split_every_n or 8
        ),
    )
    preparation = ember.ImagePreparationConfig(
        resize_width_scale=(
            None
            if svraster_resized_cache_enabled(config)
            else config.data.image_scale_factor
        ),
        resize_width_target=None,
        normalize=config.data.normalize_images,
        interpolation=config.data.interpolation,
    )
    materialization = ember.MaterializationConfig(
        stage=config.data.materialization_stage,
        mode=config.data.materialization_mode,
        num_workers=config.data.materialization_num_workers,
    )
    return ember.PreparedFrameDatasetConfig(
        camera_sensor_id=config.data.camera_sensor_id,
        split=split,
        image_preparation=preparation,
        materialization=materialization,
    )


@app.function
def run_svraster_training(
    frame_dataset: ember.PreparedFrameDataset,
    *,
    training_config: ember.TrainingConfig,
    training_viewer_handle: ember_splatting.TrainingViewerHandle | None = None,
) -> TrainingResult:
    """Run SVRaster training without Torch fallbacks for native failures."""
    try:
        return ember.run_training(
            frame_dataset,
            training_config,
            runtime_hooks=(
                ()
                if training_viewer_handle is None
                else training_viewer_handle.runtime_hooks()
            ),
        )
    except (ImportError, RuntimeError) as exc:
        message = str(exc).lower()
        extension_markers = ("cuda", "extension", "svraster", "ninja", "nvcc")
        if any(marker in message for marker in extension_markers):
            raise RuntimeError(
                "SVRaster native extension setup failed. Per notebook policy, "
                "no Torch fallback is attempted."
            ) from exc
        raise


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Training setup
    """)
    return


@app.cell
def _():
    prepare_button = mo.ui.run_button(
        label="Prepare training viewer",
        full_width=True,
    )
    train_button = mo.ui.run_button(
        label="Start training",
        full_width=True,
    )
    stop_button = mo.ui.run_button(
        label="Stop training",
        full_width=True,
    )
    training_status_refresh = mo.ui.refresh(
        options=["1s"],
        default_interval="1s",
        label="Training status",
    )
    training_controls = mo.vstack(
        [prepare_button, train_button, stop_button, training_status_refresh],
        gap=0.5,
    )
    return (
        prepare_button,
        stop_button,
        train_button,
        training_controls,
        training_status_refresh,
    )


@app.cell
def _():
    is_script_mode = mo.running_in_notebook() is False
    return (is_script_mode,)


@app.cell
def _(current_config, is_script_mode, prepare_button, train_button):
    should_prepare = (
        is_script_mode or bool(prepare_button.value) or bool(train_button.value)
    )
    scene_load_error = None
    try:
        scene_record = (
            ember.load_scene_record(build_scene_load_config(current_config))
            if should_prepare and current_config is not None
            else None
        )
    except Exception as error:
        scene_record = None
        scene_load_error = error
    return scene_load_error, scene_record


@app.cell
def _(current_config, scene_record):
    frame_dataset_error = None
    try:
        frame_dataset = (
            ember.prepare_frame_dataset(
                scene_record,
                build_prepared_frame_dataset_config(current_config),
            )
            if current_config is not None and scene_record is not None
            else None
        )
    except Exception as error:
        frame_dataset = None
        frame_dataset_error = error
    return frame_dataset, frame_dataset_error


@app.cell
def _(current_config, frame_dataset, is_script_mode):
    training_config = (
        resolve_training_config(current_config, frame_dataset)
        if current_config is not None and frame_dataset is not None
        else None
    )
    viewer_config = (
        current_config.training.viewer
        if current_config is not None
        else ember_splatting.TrainingViewerConfig()
    )
    resolved_training_config = training_config if is_script_mode else None
    return training_config, viewer_config


@app.cell
def _(
    current_config,
    frame_dataset,
    is_script_mode,
    training_config,
    viewer_config,
):
    training_viewer_error = None
    try:
        training_viewer_handle = (
            ember_splatting.create_training_viewer(
                frame_dataset,
                training_config,
                config=viewer_config,
                title="SVRaster training viewer",
            )
            if not is_script_mode
            and current_config is not None
            and frame_dataset is not None
            and training_config is not None
            else None
        )
    except Exception as error:
        training_viewer_handle = None
        training_viewer_error = error
    return training_viewer_error, training_viewer_handle


@app.cell
def _(training_viewer_handle):
    training_viewer = (
        None
        if training_viewer_handle is None
        else training_viewer_handle.viewer
    )
    return (training_viewer,)


@app.cell
def _(
    current_config,
    frame_dataset,
    is_script_mode,
    train_button,
    training_config,
    training_viewer_handle,
):
    should_train = bool(train_button.value)
    if (
        is_script_mode
        and current_config is not None
        and frame_dataset is not None
        and training_config is not None
    ):
        training_result = run_svraster_training(
            frame_dataset,
            training_config=training_config,
        )
    else:
        training_result = None
        if (
            should_train
            and frame_dataset is not None
            and training_config is not None
            and training_viewer_handle is not None
        ):
            training_viewer_handle.start_training(
                frame_dataset,
                training_config,
            )
    return (training_result,)


@app.cell
def _(stop_button, training_viewer_handle):
    should_stop = bool(stop_button.value)
    if should_stop and training_viewer_handle is not None:
        training_viewer_handle.request_stop()
    return


@app.cell
def _(
    frame_dataset_error,
    scene_load_error,
    training_result,
    training_status_refresh,
    training_viewer_error,
    training_viewer_handle,
):
    _ = training_status_refresh.value
    if scene_load_error is not None:
        training_result_view = mo.callout(
            f"Scene loading failed.\n\n```text\n{scene_load_error}\n```",
            kind="danger",
        )
    elif frame_dataset_error is not None:
        training_result_view = mo.callout(
            f"Frame dataset preparation failed.\n\n```text\n{frame_dataset_error}\n```",
            kind="danger",
        )
    elif training_viewer_error is not None:
        training_result_view = mo.callout(
            f"Training viewer preparation failed.\n\n```text\n{training_viewer_error}\n```",
            kind="danger",
        )
    elif training_result is not None:
        training_result_view = mo.md(
            f"Checkpoint: `{training_result.checkpoint_dir}`\n\n"
            f"Steps: `{len(training_result.history)}`"
        )
    elif training_viewer_handle is None:
        training_result_view = mo.md("Prepare the training viewer first.")
    else:
        snapshot = training_viewer_handle.snapshot()
        if snapshot.status == "idle":
            training_result_view = mo.md("Training has not started.")
        elif snapshot.status in {"running", "stopping"}:
            step_text = (
                f"{snapshot.step} / {snapshot.max_steps}"
                if snapshot.max_steps is not None
                else str(snapshot.step)
            )
            metric_parts = [
                f"{name}={value:.6g}"
                for name, value in sorted(snapshot.latest_metrics.items())
            ]
            if snapshot.primitive_count is not None:
                metric_parts.append(f"primitives={snapshot.primitive_count:,}")
            if snapshot.iterations_per_second is not None:
                metric_parts.append(
                    f"it/s={snapshot.iterations_per_second:.2f}"
                )
            metric_text = " | ".join(metric_parts)
            status_text = (
                "Stopping" if snapshot.status == "stopping" else "Training"
            )
            speed_text = (
                f"{snapshot.iterations_per_second:.2f} it/s"
                if snapshot.iterations_per_second is not None
                else "-- it/s"
            )
            elapsed_text = (
                f"elapsed {format_duration(snapshot.elapsed_seconds)}"
                if snapshot.elapsed_seconds is not None
                else "elapsed --"
            )
            eta_text = (
                f"ETA {format_duration(snapshot.eta_seconds)}"
                if snapshot.eta_seconds is not None
                else "ETA --"
            )
            training_result_view = mo.md(
                f"{status_text}: `{step_text}` {speed_text} "
                f"{elapsed_text} {eta_text}"
                + (f"\n\n{metric_text}" if metric_text else "")
            )
        elif snapshot.status == "cancelled":
            training_result_view = mo.md(
                f"Training cancelled at step `{snapshot.step}`."
            )
        elif snapshot.status == "failed":
            training_result_view = mo.callout(
                f"Training failed.\n\n```text\n{snapshot.error_text or ''}\n```",
                kind="danger",
            )
        else:
            assert snapshot.result is not None
            training_result_view = mo.md(
                f"Checkpoint: `{snapshot.result.checkpoint_dir}`\n\n"
                f"Steps: `{len(snapshot.result.history)}`"
            )
    return (training_result_view,)


if __name__ == "__main__":
    app.run()
