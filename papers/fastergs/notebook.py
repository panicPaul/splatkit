"""FasterGS paper training notebook for Ember."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")

with app.setup:
    import json
    import math
    import shutil
    import sys
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any, Literal, Protocol, runtime_checkable

    import ember_adapter_backends.fastergs as ember_fastergs_adapter
    import ember_core as ember
    import ember_native_faster_gs.faster_gs as ember_fastergs_native
    import ember_splatting_training as ember_splatting
    import marimo as mo
    import torch
    from ember_core.densification import (
        BaseDensificationMethod,
        DensificationContext,
        DensificationLifecycleContext,
        DensificationRenderRequirements,
        GaussianFamilyOps,
        Schedule,
    )
    from ember_core.training import (
        TrainingProfilerConfig,
        TrainingResult,
        TrainState,
    )
    from jaxtyping import Float
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
    DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "papers" / "fastergs"
    FasterGSBackendName = Literal["adapter.fastergs", "faster_gs.core"]
    FasterGSDefaultName = Literal[
        "garden_baseline",
        "garden_mcmc",
        "garden_debug_val",
        "garden_vanilla_gs_ablation",
    ]
    sys.modules.setdefault("papers.fastergs.notebook", sys.modules[__name__])
    ember_fastergs_adapter.register()
    ember_fastergs_native.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # FasterGS training
    """)
    return


@app.cell(hide_code=True)
def _(training_controls):
    training_controls
    return


@app.cell
def _():
    fastergs_presets = fastergs_preset_catalog()
    config_gui = create_config_gui(
        FasterGSExperimentConfig,
        presets=fastergs_presets,
        label="FasterGS config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    preset_selector = config_gui.preset_selector(
        label="FasterGS preset",
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
class FasterGSConfigBase(BaseModel):
    """Strict base model for FasterGS paper configs."""

    model_config = {"extra": "forbid", "populate_by_name": True}


@app.class_definition
class FasterGSSceneConfig(FasterGSConfigBase):
    """Scene-record loading options."""

    path: Path = Path("dataset/mipnerf360/garden")
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool = True


@app.class_definition
class FasterGSDataConfig(FasterGSConfigBase):
    """Prepared-frame dataset options."""

    camera_sensor_id: str | None = None
    image_scale_factor: float = Field(default=0.25, gt=0.0)
    cache_resized_images: bool = True
    resized_image_cache_root: Path | None = None
    max_resized_image_caches: int = Field(default=4, ge=1)
    split_target: Literal["train", "val", "all"] = "train"
    split_every_n: int | None = Field(default=8, ge=1)
    materialization_stage: Literal["none", "decoded", "prepared"] = "none"
    materialization_mode: Literal["lazy", "eager"] = "lazy"
    materialization_num_workers: int | None = 0
    normalize_images: bool = True
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"


@app.class_definition
class FasterGSScheduleConfig(FasterGSConfigBase):
    """Serializable step schedule config."""

    start_iteration: int = Field(default=0, ge=0)
    end_iteration: int = -1
    frequency: int = Field(default=1, ge=1)

    def to_schedule(self) -> Schedule:
        """Build the runtime densification schedule."""
        return Schedule(
            start_iteration=self.start_iteration,
            end_iteration=self.end_iteration,
            frequency=self.frequency,
        )


@app.class_definition
class FasterGSInitializationConfig(FasterGSConfigBase):
    """Typed FasterGS Gaussian initialization config."""

    sh_degree: int = Field(default=3, ge=0)
    use_mcmc: bool = False
    default_color: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def build(
        self,
        context: ember.TrainingRunContext,
    ) -> ember.InitializationSpec:
        """Build the runtime initializer spec."""
        del context
        return ember.InitializationSpec(
            initializer=ember.bound_callable(
                target=(
                    "papers.fastergs.notebook."
                    "initialize_fastergs_model_from_scene_record"
                ),
                kwargs={
                    "sh_degree": self.sh_degree,
                    "use_mcmc": self.use_mcmc,
                    "default_color": self.default_color,
                },
                bind={"device": ember.ctx.run.device},
            )
        )


@app.class_definition
class FasterGSTrainingBackendOptionsConfig(FasterGSConfigBase):
    """Typed per-step FasterGS training render options."""

    max_sh_degree: int = Field(default=3, ge=0)
    sh_start_step: int = Field(default=1000, ge=0)
    sh_step_interval: int = Field(default=1000, ge=1)
    clamp_output: bool = False

    def build(self) -> ember.CallableSpec:
        """Build the runtime training backend options builder."""
        return ember.bound_callable(
            target="ember_splatting_training.fastergs_training_backend_options",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class FasterGSRenderConfig(FasterGSConfigBase):
    """Typed FasterGS render pipeline config."""

    backend: FasterGSBackendName = "faster_gs.core"
    near_plane: float = Field(default=0.2, gt=0.0)
    far_plane: float = Field(default=10_000.0, gt=0.0)
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    return_alpha: bool = False
    training_backend_options: FasterGSTrainingBackendOptionsConfig = Field(
        default_factory=FasterGSTrainingBackendOptionsConfig
    )

    def build(
        self,
        context: ember.TrainingRunContext,
        *,
        mip_splatting_screen_filter: bool,
    ) -> ember.RenderPipelineSpec:
        """Build the runtime render pipeline spec."""
        del context
        return ember.RenderPipelineSpec(
            backend=self.backend,
            return_alpha=self.return_alpha,
            training_backend_options_builder=(
                self.training_backend_options.build()
            ),
            backend_options={
                "near_plane": self.near_plane,
                "far_plane": self.far_plane,
                "mip_splatting_screen_filter": mip_splatting_screen_filter,
                "background_color": list(self.background_color),
            },
        )


@app.class_definition
class FasterGSOptimizationConfig(FasterGSConfigBase):
    """Typed Gaussian 3DGS optimization config."""

    recipe: ember_splatting.Gaussian3DGSOptimizationRecipe = Field(
        default_factory=ember_splatting.Gaussian3DGSOptimizationRecipe
    )

    def build(
        self, context: ember.TrainingRunContext
    ) -> ember.OptimizationConfig:
        """Build runtime optimizer groups from the typed recipe."""
        return ember_splatting.gaussian_3dgs_optimization_config(
            self.recipe,
            position_lr_scale=context.camera_extent,
            max_steps=context.max_steps,
        )


@app.class_definition
class FasterGSLossConfig(FasterGSConfigBase):
    """Typed FasterGS training loss config."""

    lambda_l1: float = Field(default=0.8, ge=0.0)
    lambda_dssim: float = Field(default=0.2, ge=0.0)
    lambda_opacity_regularization: float = Field(default=0.0, ge=0.0)
    lambda_scale_regularization: float = Field(default=0.0, ge=0.0)

    def build(self, context: ember.TrainingRunContext) -> ember.LossConfig:
        """Build the runtime loss config."""
        del context
        return ember.loss_config(
            "ember_splatting_training.losses.rgb_l1_dssim_loss",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class FasterGSVanillaDensificationConfig(FasterGSConfigBase):
    """Typed notebook-local FasterGS adaptive density config."""

    refine_every: int = Field(default=100, ge=1)
    start_iter: int = Field(default=600, ge=0)
    stop_iter: int = Field(default=14_900, ge=0)
    grad_threshold: float = Field(default=2e-4, gt=0.0)
    dense_fraction: float = Field(default=0.01, gt=0.0)
    prune_opacity_threshold: float = Field(default=0.005, gt=0.0)
    opacity_reset_every: int = Field(default=3_000, ge=1)
    extra_opacity_reset_iter: int | None = Field(default=None, ge=0)
    max_reset_opacity: float = Field(default=0.01, gt=0.0, lt=1.0)

    def build(self, context: ember.TrainingRunContext) -> ember.CallableSpec:
        """Build the runtime vanilla FasterGS densification spec."""
        del context
        return ember.bound_callable(
            target="papers.fastergs.notebook.FasterGSVanillaDensification",
            kwargs=self.model_dump(mode="python"),
            bind={"camera_extent": ember.ctx.run.camera_extent},
        )


@app.class_definition
class FasterGSMCMCDensificationConfig(FasterGSConfigBase):
    """Typed FasterGS MCMC densification config."""

    refine_every: int = Field(default=100, ge=1)
    start_iter: int = Field(default=600, ge=0)
    stop_iter: int = Field(default=24_900, ge=0)
    min_opacity: float = Field(default=0.005, gt=0.0, lt=1.0)
    max_primitives: int = Field(default=1_000_000, ge=1)
    noise_lr_scale: float = Field(default=500_000.0, gt=0.0)

    def build(self, context: ember.TrainingRunContext) -> ember.CallableSpec:
        """Build the runtime MCMC densification spec."""
        del context
        return ember.bound_callable(
            target="papers.fastergs.notebook.build_fastergs_mcmc_densification",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class FasterGSMortonOrderingConfig(FasterGSConfigBase):
    """Typed scheduled Morton ordering config."""

    schedule: FasterGSScheduleConfig = Field(
        default_factory=lambda: FasterGSScheduleConfig(
            end_iteration=15_000,
            frequency=5_000,
        )
    )

    def build(self, context: ember.TrainingRunContext) -> ember.CallableSpec:
        """Build the runtime Morton ordering spec."""
        del context
        return ember.bound_callable(
            target="ember_splatting_training.GaussianMortonOrdering",
            kwargs={"schedule": self.schedule.model_dump(mode="python")},
        )


@app.class_definition
class FasterGSMipSplatting3DFilterConfig(FasterGSConfigBase):
    """Mip-Splatting 3D filter config.

    Computes a per-Gaussian world-space minimum filter radius across training
    cameras and applies it as a log-scale clamp.
    """

    recompute_schedule: FasterGSScheduleConfig = Field(
        default_factory=lambda: FasterGSScheduleConfig(
            start_iteration=15_000,
            end_iteration=29_899,
            frequency=100,
        )
    )
    near_plane: float | None = Field(default=0.2, gt=0.0)
    filter_variance: float = Field(default=0.2, gt=0.0)
    clipping_tolerance: float = Field(default=0.15, ge=0.0)

    def build(self, context: ember.TrainingRunContext) -> ember.CallableSpec:
        """Build the runtime Mip-Splatting 3D filter spec."""
        del context
        return ember.bound_callable(
            target="ember_splatting_training.GaussianMipSplatting3DFilter",
            kwargs={
                "recompute_schedule": self.recompute_schedule.model_dump(
                    mode="python"
                ),
                "near_plane": self.near_plane,
                "filter_variance": self.filter_variance,
                "clipping_tolerance": self.clipping_tolerance,
            },
        )


@app.class_definition
class FasterGSMipSplattingConfig(FasterGSConfigBase):
    """Full Mip-Splatting controls for FasterGS.

    The screen filter enables FasterGS rasterizer covariance/opacity
    compensation. The three-dimensional filter computes a per-Gaussian
    world-space radius over training cameras and clamps log-scales.
    """

    enabled: bool = True
    screen_filter_enabled: bool = True
    three_dimensional_filter: FasterGSMipSplatting3DFilterConfig = Field(
        default_factory=FasterGSMipSplatting3DFilterConfig
    )


@app.class_definition
class FasterGSFinalCleanupConfig(FasterGSConfigBase):
    """Typed FasterGS checkpoint cleanup config."""

    min_opacity: float = Field(default=1.0 / 255.0, gt=0.0, lt=1.0)

    def build(self, context: ember.TrainingRunContext) -> ember.CallableSpec:
        """Build the runtime final cleanup spec."""
        del context
        return ember.bound_callable(
            target="papers.fastergs.notebook.FasterGSFinalCleanup",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class FasterGSDensificationConfig(FasterGSConfigBase):
    """Typed FasterGS densification stack config."""

    mode: Literal["vanilla", "mcmc"] = "vanilla"
    vanilla: FasterGSVanillaDensificationConfig = Field(
        default_factory=FasterGSVanillaDensificationConfig
    )
    mcmc: FasterGSMCMCDensificationConfig = Field(
        default_factory=FasterGSMCMCDensificationConfig
    )
    morton: FasterGSMortonOrderingConfig = Field(
        default_factory=FasterGSMortonOrderingConfig
    )
    final_cleanup: FasterGSFinalCleanupConfig = Field(
        default_factory=FasterGSFinalCleanupConfig
    )

    def build(
        self,
        context: ember.TrainingRunContext,
        *,
        mip_splatting: FasterGSMipSplattingConfig,
    ) -> ember.DensificationConfig:
        """Build the runtime FasterGS densification stack."""
        primary = (
            self.mcmc.build(context)
            if self.mode == "mcmc"
            else self.vanilla.build(context)
        )
        builders = [
            primary,
            self.morton.build(context),
        ]
        if mip_splatting.enabled:
            builders.append(
                mip_splatting.three_dimensional_filter.build(context)
            )
        builders.append(self.final_cleanup.build(context))
        return ember.densification_config(*builders)


@app.class_definition
class FasterGSTrainingConfig(FasterGSConfigBase):
    """Typed user-facing FasterGS training config."""

    runtime: ember.RuntimeConfig = Field(default_factory=ember.RuntimeConfig)
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    batching: ember.BatchingConfig = Field(default_factory=ember.BatchingConfig)
    initialization: FasterGSInitializationConfig = Field(
        default_factory=FasterGSInitializationConfig
    )
    render: FasterGSRenderConfig = Field(default_factory=FasterGSRenderConfig)
    mip_splatting: FasterGSMipSplattingConfig = Field(
        default_factory=FasterGSMipSplattingConfig
    )
    optimization: FasterGSOptimizationConfig = Field(
        default_factory=FasterGSOptimizationConfig
    )
    loss: FasterGSLossConfig = Field(default_factory=FasterGSLossConfig)
    densification: FasterGSDensificationConfig = Field(
        default_factory=FasterGSDensificationConfig
    )
    checkpoint: ember.CheckpointExportConfig = Field(
        default_factory=ember.CheckpointExportConfig
    )
    viewer: ember_splatting.TrainingViewerConfig = Field(
        default_factory=ember_splatting.TrainingViewerConfig
    )

    def to_training_config(
        self,
        frame_dataset: ember.PreparedFrameDataset | None = None,
    ) -> ember.TrainingConfig:
        """Materialize this typed config into Ember's runtime config."""
        camera_extent = (
            ember.compute_frame_camera_extent(frame_dataset)
            if frame_dataset is not None
            else 1.0
        )
        context = ember.TrainingRunContext(
            frame_dataset=frame_dataset,
            camera_extent=camera_extent,
            max_steps=self.runtime.max_steps,
            backend=self.render.backend,
            device=torch.device(self.runtime.device),
        )
        return ember.TrainingConfig(
            runtime=self.runtime,
            profiler=self.profiler,
            batching=self.batching,
            initialization=self.initialization.build(context),
            render=self.render.build(
                context,
                mip_splatting_screen_filter=(
                    self.mip_splatting.enabled
                    and self.mip_splatting.screen_filter_enabled
                ),
            ),
            optimization=self.optimization.build(context),
            loss=self.loss.build(context),
            densification=self.densification.build(
                context,
                mip_splatting=self.mip_splatting,
            ),
            checkpoint=self.checkpoint,
        )


@app.class_definition
class FasterGSExperimentConfig(FasterGSConfigBase):
    """Resolved experiment config."""

    preset: FasterGSDefaultName = "garden_baseline"
    scene: FasterGSSceneConfig = Field(default_factory=FasterGSSceneConfig)
    data: FasterGSDataConfig = Field(default_factory=FasterGSDataConfig)
    training: FasterGSTrainingConfig


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Function definitions
    """)
    return


@app.function
def default_checkpoint_dir(
    preset: FasterGSDefaultName,
    backend: FasterGSBackendName,
) -> Path:
    """Return the default checkpoint directory for a preset/backend pair."""
    return DEFAULT_CHECKPOINT_ROOT / preset / backend


@app.function
def fastergs_preset_catalog() -> ConfigPresetCatalog[FasterGSExperimentConfig]:
    """Return the notebook's named JSON preset catalog."""
    return ConfigPresetCatalog(
        model_cls=FasterGSExperimentConfig,
        presets={
            "garden_baseline": ConfigPreset(
                name="garden_baseline",
                path=DEFAULTS_DIR / "garden_baseline.json",
                label="Garden baseline",
                base_dir=REPO_ROOT,
            ),
            "garden_mcmc": ConfigPreset(
                name="garden_mcmc",
                path=DEFAULTS_DIR / "garden_mcmc.json",
                label="Garden MCMC",
                base_dir=REPO_ROOT,
            ),
            "garden_debug_val": ConfigPreset(
                name="garden_debug_val",
                path=DEFAULTS_DIR / "garden_debug_val.json",
                label="Garden debug validation",
                base_dir=REPO_ROOT,
            ),
            "garden_vanilla_gs_ablation": ConfigPreset(
                name="garden_vanilla_gs_ablation",
                path=DEFAULTS_DIR / "garden_vanilla_gs_ablation.json",
                label="Garden vanilla GS ablation",
                base_dir=REPO_ROOT,
            ),
        },
        default="garden_baseline",
    )


@app.function
def resolve_fastergs_point_cloud(
    scene_record: ember.SceneRecord,
) -> ember.PointCloudState:
    """Return the SfM point cloud required by FasterGS initialization."""
    if scene_record.point_cloud is None:
        raise ValueError("FasterGS initialization requires an SfM point cloud.")
    return scene_record.point_cloud


@app.function
def fastergs_root_mean_squared_knn_distances(
    positions: Float[Tensor, " num_points 3"],
    *,
    torch_chunk_size: int = 512,
) -> Float[Tensor, " num_points"]:
    """Compute upstream FasterGS initial scale distances for the notebook."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "FasterGS KNN distances expect positions with shape "
            f"(num_points, 3), got {tuple(positions.shape)}."
        )
    if torch_chunk_size < 1:
        raise ValueError("torch_chunk_size must be at least 1.")
    num_points = int(positions.shape[0])
    if num_points == 0:
        return torch.empty(
            (0,),
            dtype=positions.dtype,
            device=positions.device,
        )
    if positions.device.type == "cuda":
        try:
            from simple_knn._C import distCUDA2

            mean_squared = distCUDA2(positions.contiguous())
            return mean_squared.clamp_min(1e-7).sqrt()
        except Exception:
            pass
    if num_points == 1:
        return torch.full(
            (1,),
            1e-3,
            dtype=positions.dtype,
            device=positions.device,
        )
    k = min(3, num_points - 1)
    nearest_distances = []
    for start in range(0, num_points, torch_chunk_size):
        stop = min(start + torch_chunk_size, num_points)
        distances = torch.cdist(positions[start:stop], positions)
        row_indices = torch.arange(
            stop - start,
            dtype=torch.long,
            device=positions.device,
        )
        col_indices = torch.arange(
            start,
            stop,
            dtype=torch.long,
            device=positions.device,
        )
        distances[row_indices, col_indices] = math.inf
        nearest_distances.append(distances.topk(k, largest=False).values)
    mean_squared = torch.cat(nearest_distances, dim=0).square().mean(dim=1)
    return mean_squared.clamp_min(1e-7).sqrt()


@app.function
def initialize_fastergs_model_from_scene_record(
    scene_record: ember.SceneRecord,
    *,
    modules: dict[str, torch.nn.Module] | None = None,
    parameters: dict[str, torch.nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    sh_degree: int = 3,
    use_mcmc: bool = False,
    default_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    device: torch.device | None = None,
) -> ember.InitializedModel:
    """Initialize Gaussians exactly like the FasterGS paper implementation."""
    point_cloud = resolve_fastergs_point_cloud(scene_record)
    target_device = device or torch.device("cpu")
    centers = point_cloud.points.to(device=target_device, dtype=torch.float32)
    num_points = int(centers.shape[0])

    colors = (
        torch.full(
            (num_points, 3),
            fill_value=0.5,
            dtype=torch.float32,
            device=target_device,
        )
        if point_cloud.colors is None
        else point_cloud.colors.to(device=target_device, dtype=torch.float32)
    )
    if point_cloud.colors is None and default_color != (0.5, 0.5, 0.5):
        colors = torch.tensor(
            default_color,
            dtype=torch.float32,
            device=target_device,
        ).expand(num_points, 3)

    sh_coeffs = (sh_degree + 1) ** 2
    feature = torch.zeros(
        (num_points, sh_coeffs, 3),
        dtype=torch.float32,
        device=target_device,
    )
    feature[:, 0, :] = (colors - 0.5) / 0.28209479177387814

    distances = fastergs_root_mean_squared_knn_distances(centers)
    if use_mcmc:
        distances = distances * 0.1
    log_scales = distances.log()[:, None].repeat(1, 3)

    quaternion_orientation = torch.zeros(
        (num_points, 4),
        dtype=torch.float32,
        device=target_device,
    )
    quaternion_orientation[:, 0] = 1.0

    initial_opacity = 0.5 if use_mcmc else 0.1
    opacity = torch.full(
        (num_points,),
        initial_opacity,
        dtype=torch.float32,
        device=target_device,
    )
    logit_opacity = opacity.clamp(1e-5, 1.0 - 1e-5).logit()
    scene = ember.GaussianScene3D(
        center_position=centers.requires_grad_(True),
        log_scales=log_scales.requires_grad_(True),
        quaternion_orientation=quaternion_orientation.requires_grad_(True),
        logit_opacity=logit_opacity.requires_grad_(True),
        feature=feature.requires_grad_(True),
        sh_degree=sh_degree,
    )
    return ember.InitializedModel(
        scene=scene,
        modules=dict(modules or {}),
        parameters=dict(parameters or {}),
        buffers=dict(buffers or {}),
        metadata=dict(metadata or {}),
    )


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
    is_script_mode = not mo.running_in_notebook()
    return (is_script_mode,)


@app.function
def fastergs_resized_cache_parent(config: FasterGSExperimentConfig) -> Path:
    """Return the reusable derived image cache parent for the scene."""
    if config.data.resized_image_cache_root is not None:
        return config.data.resized_image_cache_root.expanduser()
    return config.scene.path.expanduser() / "ember_cache" / "resized_images"


@app.function
def enforce_fastergs_resized_cache_limit(
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


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""
    # Training
    """)
    return


@app.function
def resolve_training_config(
    config: FasterGSExperimentConfig,
    frame_dataset: ember.PreparedFrameDataset | None = None,
) -> ember.TrainingConfig:
    """Apply paper notebook runtime defaults to native Ember training config."""
    checkpoint = config.training.checkpoint.model_copy(
        update={
            "output_dir": resolve_checkpoint_output_dir(config),
        },
    )
    training = config.training.model_copy(
        update={"checkpoint": checkpoint},
        deep=True,
    )
    return training.to_training_config(frame_dataset)


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


@app.cell
def _(current_config, is_script_mode, prepare_button):
    should_prepare = is_script_mode or bool(prepare_button.value)
    scene_record = (
        ember.load_scene_record(build_scene_load_config(current_config))
        if should_prepare and current_config is not None
        else None
    )
    return (scene_record,)


@app.cell
def _(current_config, scene_record):
    frame_dataset = (
        ember.prepare_frame_dataset(
            scene_record,
            build_prepared_frame_dataset_config(current_config),
        )
        if current_config is not None and scene_record is not None
        else None
    )
    return (frame_dataset,)


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
        training_result = run_fastergs_training(
            frame_dataset,
            current_config,
            training_config,
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
def _(current_config, frame_dataset):
    training_config = (
        resolve_training_config(current_config, frame_dataset)
        if current_config is not None and frame_dataset is not None
        else None
    )
    return (training_config,)


@app.cell
def _(current_config, frame_dataset, is_script_mode, training_config):
    training_viewer_handle = (
        ember_splatting.create_training_viewer(
            frame_dataset,
            training_config,
            config=current_config.training.viewer,
            title="FasterGS training viewer",
        )
        if not is_script_mode
        and current_config is not None
        and frame_dataset is not None
        and training_config is not None
        else None
    )
    return (training_viewer_handle,)


@app.cell
def _(training_viewer_handle):
    training_viewer = (
        None
        if training_viewer_handle is None
        else training_viewer_handle.viewer
    )
    return (training_viewer,)


@app.cell
def _(training_result, training_status_refresh, training_viewer_handle):
    _ = training_status_refresh.value
    if training_result is not None:
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


@app.cell(column=2, hide_code=True)
def _():
    mo.md("""
    # Densification
    """)
    return


@app.function
def build_fastergs_mcmc_densification(
    *,
    refine_every: int,
    start_iter: int,
    stop_iter: int,
    min_opacity: float,
    max_primitives: int,
    noise_lr_scale: float,
) -> Any:
    """Construct CUDA-backed Gaussian MCMC densification."""
    from ember_splatting_training import GaussianMCMC

    return GaussianMCMC(
        schedule=Schedule(
            start_iteration=max(0, start_iter - 1),
            end_iteration=max(0, stop_iter - 1),
            frequency=refine_every,
        ),
        min_opacity=min_opacity,
        cap_max=max_primitives,
        noise_lr_scale=noise_lr_scale,
    )


@app.class_definition
@runtime_checkable
class HasFasterGSDensificationInfo(Protocol):
    """Render-output trait for FasterGS densification accumulators."""

    densification_info: Float[Tensor, " 2 num_splats"]


@app.class_definition
class FasterGSVanillaDensification(BaseDensificationMethod):
    """Notebook-local FasterGS adaptive density control."""

    expected_scene_families = ("gaussian",)

    def __init__(
        self,
        *,
        refine_every: int = 100,
        start_iter: int = 600,
        stop_iter: int = 14_900,
        grad_threshold: float = 2e-4,
        dense_fraction: float = 0.01,
        prune_opacity_threshold: float = 0.005,
        opacity_reset_every: int = 3_000,
        extra_opacity_reset_iter: int | None = 500,
        max_reset_opacity: float = 0.01,
        camera_extent: float = 1.0,
    ) -> None:
        self.refine_schedule = Schedule(
            start_iteration=start_iter,
            end_iteration=stop_iter,
            frequency=refine_every,
        )
        self.stop_iter = stop_iter
        self.grad_threshold = grad_threshold
        self.dense_fraction = dense_fraction
        self.prune_opacity_threshold = prune_opacity_threshold
        self.opacity_reset_every = opacity_reset_every
        self.extra_opacity_reset_iter = extra_opacity_reset_iter
        self.max_reset_opacity = max_reset_opacity
        self.camera_extent = float(camera_extent)
        self.family_ops: GaussianFamilyOps | None = None
        self.grad_sum: Tensor | None = None
        self.visible_count: Tensor | None = None

    def get_render_requirements(
        self,
        state: TrainState,
    ) -> DensificationRenderRequirements:
        """Collect FasterGS visibility accumulators while densification runs."""
        return DensificationRenderRequirements(
            backend_options={
                "collect_densification_info": state.step + 1 < self.stop_iter
            }
        )

    def bind(
        self, state: Any, optimizers: Sequence[Any], family_ops: Any
    ) -> None:
        """Bind Gaussian topology operations."""
        del state, optimizers
        if not isinstance(family_ops, GaussianFamilyOps):
            raise TypeError(
                "FasterGSVanillaDensification requires GaussianFamilyOps."
            )
        self.family_ops = family_ops

    def post_backward(self, context: DensificationContext) -> None:
        """Accumulate FasterGS screen-space densification statistics."""
        if context.step + 1 >= self.stop_iter:
            return
        if not isinstance(context.render_output, HasFasterGSDensificationInfo):
            raise TypeError(
                "FasterGSVanillaDensification requires render outputs with "
                "densification_info."
            )
        densification_info = context.render_output.densification_info.detach()
        if densification_info.ndim != 2 or densification_info.shape[0] != 2:
            raise ValueError(
                "FasterGS densification_info must have shape "
                f"(2, num_splats), got {tuple(densification_info.shape)}."
            )
        if self.visible_count is None:
            self.visible_count = torch.zeros_like(densification_info[0])
            self.grad_sum = torch.zeros_like(densification_info[1])
        assert self.grad_sum is not None
        self.visible_count += densification_info[0]
        self.grad_sum += densification_info[1]

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Run scheduled clone/split/prune/reset actions."""
        if self.family_ops is None:
            return
        scene = context.state.model.scene
        if not isinstance(scene, ember.GaussianScene):
            return
        upstream_iteration = context.step + 1
        if self.refine_schedule.includes(upstream_iteration):
            self.adaptive_density_control(scene, upstream_iteration)
            self.reset_accumulators()
        if self.should_reset_opacity(upstream_iteration):
            self.family_ops.reset_opacity(self.max_reset_opacity)

    def adaptive_density_control(
        self,
        scene: ember.GaussianScene,
        step: int,
    ) -> None:
        if self.visible_count is None or self.grad_sum is None:
            return
        assert self.family_ops is not None
        avg_grad = self.grad_sum / self.visible_count.clamp_min(1.0)
        scales = torch.exp(scene.log_scales).max(dim=-1).values
        densify_mask = avg_grad >= self.grad_threshold
        small_mask = scales <= self.dense_fraction * self.camera_extent
        clone_mask = densify_mask & small_mask
        split_mask = densify_mask & ~small_mask
        self.family_ops.clone_and_split(
            clone_mask,
            split_mask,
            num_children=2,
            scale_shrink=0.625,
            prune_fn=lambda grown_scene: self.refinement_keep_mask(
                grown_scene,
                step,
            ),
            prune_field_names=(
                "center_position",
                "logit_opacity",
                "log_scales",
                "quaternion_orientation",
            ),
        )

    def refinement_keep_mask(
        self,
        scene: ember.GaussianScene,
        step: int,
    ) -> Tensor:
        keep_mask = torch.sigmoid(scene.logit_opacity) >= (
            self.prune_opacity_threshold
        )
        if (
            step > self.opacity_reset_every
            and scene.center_position.shape[0] > 0
        ):
            max_scale = torch.exp(scene.log_scales).max(dim=-1).values
            keep_mask &= max_scale <= 0.1 * self.camera_extent
        keep_mask &= scene.quaternion_orientation.square().sum(dim=1) >= 1e-8
        return keep_mask

    def should_reset_opacity(self, step: int) -> bool:
        scheduled = (
            step >= self.opacity_reset_every
            and step <= self.stop_iter
            and step % self.opacity_reset_every == 0
        )
        return scheduled or (
            self.extra_opacity_reset_iter is not None
            and step == self.extra_opacity_reset_iter
        )

    def reset_accumulators(self) -> None:
        self.visible_count = None
        self.grad_sum = None


@app.function
def run_fastergs_training(
    frame_dataset: ember.PreparedFrameDataset,
    experiment_config: FasterGSExperimentConfig,
    training_config: ember.TrainingConfig | None = None,
    training_viewer_handle: ember_splatting.TrainingViewerHandle | None = None,
) -> TrainingResult:
    """Run FasterGS training from a native Ember training config."""
    resolved_training_config = training_config or resolve_training_config(
        experiment_config,
        frame_dataset,
    )
    return ember.run_training(
        frame_dataset,
        resolved_training_config,
        runtime_hooks=(
            ()
            if training_viewer_handle is None
            else training_viewer_handle.runtime_hooks()
        ),
    )


@app.class_definition
class FasterGSFinalCleanup(BaseDensificationMethod):
    """Notebook-local FasterGS checkpoint cleanup before export."""

    expected_scene_families = ("gaussian",)

    def __init__(self, *, min_opacity: float = 1.0 / 255.0) -> None:
        self.min_opacity = float(min_opacity)
        self.family_ops: GaussianFamilyOps | None = None

    def bind(
        self,
        state: Any,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Bind Gaussian topology operations."""
        del state, optimizers
        if not isinstance(family_ops, GaussianFamilyOps):
            raise TypeError("FasterGSFinalCleanup requires GaussianFamilyOps.")
        self.family_ops = family_ops

    def after_training(self, context: DensificationLifecycleContext) -> None:
        """Prune invalid Gaussians and apply final Morton ordering."""
        del context
        if self.family_ops is None:
            return
        scene = self.family_ops.scene
        if not isinstance(scene, ember.GaussianScene):
            return
        keep_mask = torch.sigmoid(scene.logit_opacity) >= self.min_opacity
        keep_mask &= scene.quaternion_orientation.square().sum(dim=1) >= 1e-8
        if torch.any(~keep_mask):
            self.family_ops.prune(keep_mask)
            scene = self.family_ops.scene
        if scene.center_position.device.type != "cuda":
            return
        from ember_splatting_training import morton_order

        self.family_ops.reorder(morton_order(scene.center_position))


@app.cell(column=3, hide_code=True)
def _():
    mo.md("""
    # Support
    """)
    return


@app.function
def fastergs_resized_cache_enabled(config: FasterGSExperimentConfig) -> bool:
    """Return whether FasterGS should use a derived resized image cache."""
    return (
        config.data.cache_resized_images
        and config.data.image_scale_factor != 1.0
    )


@app.function
def fastergs_source_image_root(config: FasterGSExperimentConfig) -> Path:
    """Return the full-resolution source image root."""
    if config.scene.image_root is not None:
        return config.scene.image_root.expanduser()
    return config.scene.path.expanduser() / "images"


@app.function
def fastergs_resized_cache_root(config: FasterGSExperimentConfig) -> Path:
    """Return the derived resized image cache root for this config."""
    scale_name = f"{config.data.image_scale_factor:.6f}".rstrip("0").rstrip(".")
    scale_name = scale_name.replace(".", "p")
    return fastergs_resized_cache_parent(config) / (
        f"scale_{scale_name}_{config.data.interpolation}"
    )


@app.function
def fastergs_pillow_resampling(interpolation: str) -> Any:
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
def materialize_fastergs_resized_image_cache(
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
    resampling = fastergs_pillow_resampling(interpolation)
    enforce_fastergs_resized_cache_limit(
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
    enforce_fastergs_resized_cache_limit(
        cache_root=cache_root,
        max_caches=max_caches,
    )
    return cache_root


@app.function
def build_scene_load_config(
    config: FasterGSExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Translate paper config into an Ember scene loader config."""
    source_pipes = (
        (ember.HorizonAlignPipeConfig(),) if config.scene.align_horizon else ()
    )
    image_root = (
        materialize_fastergs_resized_image_cache(
            source_root=fastergs_source_image_root(config),
            cache_root=fastergs_resized_cache_root(config),
            scale=config.data.image_scale_factor,
            interpolation=config.data.interpolation,
            max_caches=config.data.max_resized_image_caches,
        )
        if fastergs_resized_cache_enabled(config)
        else (
            config.scene.image_root.expanduser()
            if config.scene.image_root is not None
            else None
        )
    )
    return ember.ColmapSceneConfig(
        path=config.scene.path.expanduser(),
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
    config: FasterGSExperimentConfig,
) -> ember.PreparedFrameDatasetConfig:
    """Translate paper config into an Ember frame dataset config."""
    split = (
        ember.SplitConfig(target="all", every_n=None, train_ratio=None)
        if config.data.split_target == "all"
        else ember.SplitConfig(
            target=config.data.split_target,
            every_n=config.data.split_every_n,
            train_ratio=None,
        )
    )
    return ember.PreparedFrameDatasetConfig(
        camera_sensor_id=config.data.camera_sensor_id,
        split=split,
        materialization=ember.MaterializationConfig(
            stage=config.data.materialization_stage,
            mode=config.data.materialization_mode,
            num_workers=config.data.materialization_num_workers,
        ),
        image_preparation=ember.ImagePreparationConfig(
            normalize=config.data.normalize_images,
            resize_width_scale=(
                None
                if fastergs_resized_cache_enabled(config)
                else config.data.image_scale_factor
            ),
            resize_width_target=None,
            interpolation=config.data.interpolation,
        ),
    )


@app.function
def resolve_checkpoint_output_dir(
    config: FasterGSExperimentConfig,
) -> Path:
    """Mirror checkpoint dirs by preset and backend unless user changed them."""
    default_parent = DEFAULT_CHECKPOINT_ROOT / config.preset
    output_dir = config.training.checkpoint.output_dir.expanduser()
    if output_dir.parent == default_parent:
        return default_checkpoint_dir(
            config.preset,
            config.training.render.backend,
        )
    return output_dir


if __name__ == "__main__":
    app.run()
