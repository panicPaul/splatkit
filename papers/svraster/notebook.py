"""SVRaster paper training notebook for Ember."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")

with app.setup:
    import json
    import sys
    from pathlib import Path
    from typing import Any, Literal

    import ember_core as ember
    import ember_native_svraster
    import marimo as mo
    import torch
    from ember_core.training import LossResult
    from jaxtyping import Float, Int
    from marimo_config_gui import (
        ConfigPreset,
        ConfigPresetCatalog,
        create_config_gui,
    )
    from pydantic import BaseModel, Field
    from torch import Tensor

    NOTEBOOK_PATH = Path(__file__).resolve()
    NOTEBOOK_DIR = NOTEBOOK_PATH.parent
    REPO_ROOT = NOTEBOOK_DIR.parents[1]
    DEFAULTS_DIR = NOTEBOOK_DIR / "defaults"
    DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "papers" / "svraster"
    SVRasterBackendName = Literal["svraster.core"]
    SVRasterDefaultName = Literal[
        "garden_svraster",
        "garden_fast_train",
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
def _(config_gui):
    config_gui.stacked()
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
    current_config = config_gui.validated_config()
    return (current_config,)


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
    align_horizon: bool = True


@app.class_definition
class SVRasterDataConfig(SVRasterConfigBase):
    """Prepared-frame dataset options."""

    camera_sensor_id: str | None = None
    image_scale_factor: float = Field(default=0.25, gt=0.0)
    split_target: Literal["train", "val", "all"] = "train"
    split_every_n: int | None = Field(default=8, ge=1)
    normalize_images: bool = True
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"
    materialization_stage: Literal["none", "decoded", "prepared"] = "none"
    materialization_mode: Literal["lazy", "eager"] = "lazy"
    materialization_num_workers: int | None = 0


@app.class_definition
class SVRasterModelConfig(SVRasterConfigBase):
    """SVRaster sparse-voxel model defaults."""

    backend_name: Literal["new_cuda"] = "new_cuda"
    max_num_levels: int = Field(default=16, ge=1)
    sh_degree: int = Field(default=3, ge=0)
    initial_sh_degree: int = Field(default=3, ge=0)
    samples_per_voxel: int = Field(default=1, ge=1)
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
    lambda_t_inside: float = Field(default=0.0, ge=0.0)
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

    def build(self) -> ember.DensificationConfig:
        """Build the SVRaster adaptive pruning/subdivision config."""
        return ember.densification_config(
            ember.bound_callable(
                target=(
                    "ember_svraster_training.SVRasterAdaptivePruneSubdivide"
                ),
                kwargs=self.model_dump(mode="python"),
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
    batching: ember.BatchingConfig = Field(
        default_factory=lambda: ember.BatchingConfig(
            batch_size=1, shuffle=False
        )
    )
    checkpoint: ember.CheckpointExportConfig = Field(
        default_factory=lambda: ember.CheckpointExportConfig(
            output_dir=(
                DEFAULT_CHECKPOINT_ROOT / "garden_svraster" / "svraster.core"
            ),
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
                    "supersampling": 1.0,
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
                        "ss_aug_max": self.training.model.ss_aug_max,
                    },
                ),
            ),
            optimization=self.training.optimization.build(),
            loss=self.training.loss.build(),
            densification=self.training.adaptive.build(),
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
def svraster_rgb_to_sh_zero(
    rgb: Float[Tensor, "... 3"],
) -> Float[Tensor, "... 3"]:
    """Convert RGB values to SVRaster SH-zero coefficients."""
    return ember.svraster_rgb_to_sh_zero(rgb)


@app.function
def encode_octpath_from_ijk(
    ijk: Int[Tensor, " num_voxels 3"],
    octlevel: Int[Tensor, " num_voxels 1"],
    *,
    max_num_levels: int,
) -> Int[Tensor, " num_voxels 1"]:
    """Encode voxel integer coordinates into SVRaster octree paths."""
    paths = torch.zeros(
        (ijk.shape[0],),
        dtype=torch.int64,
        device=ijk.device,
    )
    for level in range(1, max_num_levels + 1):
        active = octlevel.reshape(-1).to(torch.int64) >= level
        bit_shift = (octlevel.reshape(-1).to(torch.int64) - level).clamp_min(0)
        subtree = (
            (((ijk[:, 0] >> bit_shift) & 1) << 2)
            | (((ijk[:, 1] >> bit_shift) & 1) << 1)
            | ((ijk[:, 2] >> bit_shift) & 1)
        )
        paths |= torch.where(
            active,
            subtree << (3 * (max_num_levels - level)),
            torch.zeros_like(subtree),
        )
    return paths.reshape(-1, 1)


@app.function
def initialize_svraster_model_from_scene_record(
    scene_record: ember.SceneRecord,
    *,
    modules: dict[str, torch.nn.Module] | None = None,
    parameters: dict[str, torch.nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    initial_inside_level: int = 6,
    outside_level: int = 5,
    initial_outside_ratio: float = 2.0,
    geometry_initial_value: float = -10.0,
    sh0_initial_rgb: float = 0.5,
    shs_initial_value: float = 0.0,
    bound_scale: float = 1.0,
    device: torch.device | None = None,
) -> ember.InitializedModel:
    """Initialize a sparse-voxel scene from SceneRecord geometry."""
    del initial_outside_ratio
    target_device = device or torch.device("cpu")
    max_num_levels = 16
    sh_degree = 3
    point_cloud = scene_record.point_cloud
    camera = scene_record.resolve_camera_sensor().camera
    if point_cloud is not None and point_cloud.points.numel() > 0:
        points = point_cloud.points.to(
            device=target_device, dtype=torch.float32
        )
    else:
        camera_to_world = camera.cam_to_world
        points = camera_to_world[:, :3, 3].to(
            device=target_device,
            dtype=torch.float32,
        )
    scene_min = points.amin(dim=0)
    scene_max = points.amax(dim=0)
    scene_center = (scene_min + scene_max) * 0.5
    inside_extent = (scene_max - scene_min).amax().clamp_min(1e-6) * bound_scale
    scene_extent = inside_extent * float(2**outside_level)

    level = min(initial_inside_level + outside_level, max_num_levels)
    grid_resolution = 2**level
    normalized = (points - (scene_center - 0.5 * scene_extent)) / scene_extent
    ijk = (normalized.clamp(0.0, 1.0 - 1e-6) * grid_resolution).to(torch.int64)
    octlevel = torch.full(
        (ijk.shape[0], 1),
        level,
        dtype=torch.int64,
        device=target_device,
    )
    unique_ijkl = torch.cat([ijk, octlevel], dim=1).unique(dim=0)
    ijk = unique_ijkl[:, :3]
    octlevel = unique_ijkl[:, 3:].to(torch.int64)
    octpath = encode_octpath_from_ijk(
        ijk,
        octlevel,
        max_num_levels=max_num_levels,
    )
    _grid_points_key, voxel_keys = ember.svraster_build_grid_points_link(
        octpath,
        octlevel,
        backend_name=None,
        max_num_levels=max_num_levels,
    )
    num_voxels = int(octpath.shape[0])
    num_grid_points = (
        int(voxel_keys.max().item()) + 1 if voxel_keys.numel() else 0
    )
    sh_coefficients = (sh_degree + 1) ** 2 - 1
    scene = ember.SparseVoxelScene(
        backend_name="new_cuda",
        active_sh_degree=sh_degree,
        max_num_levels=max_num_levels,
        scene_center=scene_center,
        scene_extent=scene_extent.reshape(1),
        inside_extent=inside_extent.reshape(1),
        octpath=octpath,
        octlevel=octlevel,
        geo_grid_pts=torch.full(
            (num_grid_points, 1),
            geometry_initial_value,
            dtype=torch.float32,
            device=target_device,
        ).requires_grad_(True),
        sh0=svraster_rgb_to_sh_zero(
            torch.full(
                (num_voxels, 3),
                sh0_initial_rgb,
                dtype=torch.float32,
                device=target_device,
            )
        ).requires_grad_(True),
        shs=torch.full(
            (num_voxels, sh_coefficients, 3),
            shs_initial_value,
            dtype=torch.float32,
            device=target_device,
        ).requires_grad_(True),
        subdivision_priority=torch.ones(
            (num_voxels, 1),
            dtype=torch.float32,
            device=target_device,
        ).requires_grad_(True),
    )
    return ember.InitializedModel(
        scene=scene,
        modules=dict(modules or {}),
        parameters=dict(parameters or {}),
        buffers=dict(buffers or {}),
        metadata=dict(metadata or {}),
    )


@app.function
def svraster_rgb_loss(
    state: ember.TrainState,
    batch: Any,
    render_output: Any,
    *,
    lambda_photo: float = 1.0,
    lambda_ssim: float = 0.02,
    use_l1: bool = False,
    use_huber: bool = False,
    huber_threshold: float = 0.03,
    lambda_t_inside: float = 0.01,
    lambda_r_concentration: float = 0.01,
    lambda_distortion: float = 0.1,
    distortion_start_step: int = 10000,
    weights: dict[str, float] | None = None,
) -> LossResult:
    """Compute the SVRaster RGB loss terms available through current APIs."""
    del (
        lambda_ssim,
        lambda_r_concentration,
        lambda_distortion,
        distortion_start_step,
        weights,
    )
    prediction = render_output.render
    target = batch.images
    if use_l1:
        photo_loss = (prediction - target).abs().mean()
    elif use_huber:
        photo_loss = torch.nn.functional.huber_loss(
            prediction,
            target,
            delta=huber_threshold,
        )
    else:
        photo_loss = torch.nn.functional.mse_loss(prediction, target)
    loss = lambda_photo * photo_loss
    metrics = {"photo_loss": float(photo_loss.detach().item())}
    transmittance = getattr(render_output, "transmittance", None)
    if lambda_t_inside > 0.0 and transmittance is not None:
        transmittance_loss = transmittance.mean()
        loss = loss + lambda_t_inside * transmittance_loss
        metrics["transmittance_loss"] = float(
            transmittance_loss.detach().item()
        )
    return LossResult(
        loss=loss,
        metrics=metrics,
    )


@app.function
def svraster_training_backend_options(
    state: ember.TrainState,
    *,
    distortion_weight: float = 0.1,
    distortion_start_step: int = 10000,
) -> dict[str, float]:
    """Return per-step SVRaster native renderer regularizer weights."""
    return {
        "distortion_weight": (
            distortion_weight if state.step >= distortion_start_step else 0.0
        )
    }


@app.function
def resolve_training_config(
    experiment_config: SVRasterExperimentConfig,
    frame_dataset: ember.PreparedFrameDataset | None = None,
) -> ember.TrainingConfig:
    """Resolve notebook config to Ember's runtime TrainingConfig."""
    return experiment_config.to_training_config(frame_dataset)


@app.function
def build_scene_load_config(
    config: SVRasterExperimentConfig,
) -> ember.SceneLoadConfig:
    """Build scene loading config from notebook options."""
    postprocess = (
        ember.ScenePostprocessConfig(
            horizon_adjustment=ember.HorizonAdjustmentConfig(enabled=True)
        )
        if config.scene.align_horizon
        else None
    )
    return ember.SceneLoadConfig(
        source="colmap",
        path=config.scene.path,
        image_root=config.scene.image_root,
        postprocess=postprocess,
    )


@app.function
def build_prepared_frame_dataset_config(
    config: SVRasterExperimentConfig,
) -> ember.PreparedFrameDatasetConfig:
    """Build prepared-frame dataset config from notebook options."""
    split = ember.SplitConfig(
        target=config.data.split_target,
        mode="none" if config.data.split_target == "all" else "every_n",
        every_n=config.data.split_every_n or 8,
    )
    preparation = ember.ImagePreparationConfig(
        resize_width_scale=config.data.image_scale_factor,
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
) -> ember.TrainingResult:
    """Run SVRaster training without Torch fallbacks for native failures."""
    try:
        return ember.run_training(frame_dataset, training_config)
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
    ## Resolved config
    """)
    return


@app.cell
def _(current_config):
    training_config = (
        None
        if current_config is None
        else resolve_training_config(current_config)
    )
    return (training_config,)


@app.cell(hide_code=True)
def _(training_config):
    training_config
    return


if __name__ == "__main__":
    app.run()
