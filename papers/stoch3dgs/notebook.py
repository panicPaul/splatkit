"""Stoch3DGS paper training notebook for Ember."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import json
    import math
    import shutil
    import sys
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any, Literal, Protocol, runtime_checkable

    import ember_core as ember
    import ember_native_3dgrt as ember_3dgrt_native
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
        LossResult,
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
    DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "papers" / "stoch3dgs"
    Stoch3DGSBackendName = Literal["3dgrt.stoch3dgs"]
    Stoch3DGSDefaultName = Literal["garden_stoch", "garden_debug_val"]
    sys.modules.setdefault("papers.stoch3dgs.notebook", sys.modules[__name__])
    ember_3dgrt_native.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Stoch3DGS training
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Configuration
    """)
    return


@app.cell(hide_code=True)
def _(preset_selector):
    preset_selector
    return


@app.cell(hide_code=True)
def _(config_gui):
    config_gui.stacked()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Training controls
    """)
    return


@app.cell(hide_code=True)
def _(training_controls):
    training_controls
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Training output
    """)
    return


@app.cell(hide_code=True)
def _(training_result_view):
    training_result_view
    return


@app.cell(hide_code=True)
def _(training_viewer):
    training_viewer
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md("""
    # Configuration model
    """)
    return


@app.cell
def _():
    stoch3dgs_presets = stoch3dgs_preset_catalog()
    config_gui = create_config_gui(
        Stoch3DGSExperimentConfig,
        presets=stoch3dgs_presets,
        label="Stoch3DGS config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    preset_selector = config_gui.preset_selector(
        label="Stoch3DGS preset",
    )
    return (preset_selector,)


@app.cell
def _(config_gui):
    current_config = config_gui.validated_config()
    return (current_config,)


@app.class_definition
class Stoch3DGSConfigBase(BaseModel):
    """Strict base model for Stoch3DGS paper configs."""

    model_config = {"extra": "forbid", "populate_by_name": True}


@app.class_definition
class Stoch3DGSSceneConfig(Stoch3DGSConfigBase):
    """Scene-record loading options."""

    path: Path = Path("dataset/mipnerf360/garden")
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool = True


@app.class_definition
class Stoch3DGSDataConfig(Stoch3DGSConfigBase):
    """Prepared-frame dataset options."""

    camera_sensor_id: str | None = None
    image_scale_factor: float = Field(default=0.25, gt=0.0)
    cache_resized_images: bool = True
    resized_image_cache_root: Path | None = None
    max_resized_image_caches: int = Field(default=4, ge=1)
    split_target: Literal["train", "val", "all"] = "train"
    split_every_n: int | None = Field(default=8, ge=1)
    materialization_stage: Literal["none", "decoded", "prepared"] = "prepared"
    materialization_mode: Literal["lazy", "eager"] = "eager"
    materialization_num_workers: int | None = 8
    normalize_images: bool = True
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"


@app.class_definition
class Stoch3DGSScheduleConfig(Stoch3DGSConfigBase):
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
class Stoch3DGSInitializationConfig(Stoch3DGSConfigBase):
    """Typed Stoch3DGS Gaussian initialization config."""

    sh_degree: int = Field(default=3, ge=0)
    default_density: float = Field(default=0.1, gt=0.0, lt=1.0)
    default_scale_factor: float = Field(default=1.0, gt=0.0)
    observation_scale_factor: float = Field(default=0.01, gt=0.0)
    use_observation_points: bool = True
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
                    "papers.stoch3dgs.notebook."
                    "initialize_stoch3dgs_model_from_scene_record"
                ),
                kwargs=self.model_dump(mode="python"),
                bind={"device": ember.ctx.run.device},
            )
        )


@app.class_definition
class Stoch3DGSTrainingBackendOptionsConfig(Stoch3DGSConfigBase):
    """Typed per-step Stoch3DGS SH controls."""

    max_sh_degree: int = Field(default=3, ge=0)
    sh_start_step: int = Field(default=1000, ge=0)
    sh_step_interval: int = Field(default=1000, ge=1)

    def build_feature_fn(self) -> ember.CallableSpec:
        """Build the active-SH feature function spec."""
        return ember.bound_callable(
            target="papers.stoch3dgs.notebook.stoch3dgs_active_sh_scene",
        )


@app.class_definition
class Stoch3DGSRenderConfig(Stoch3DGSConfigBase):
    """Typed native Stoch3DGS render pipeline config."""

    backend: Stoch3DGSBackendName = "3dgrt.stoch3dgs"
    particle_kernel_degree: int = Field(default=4, ge=1)
    particle_kernel_density_clamping: bool = True
    particle_kernel_min_response: float = Field(default=0.0113, gt=0.0)
    particle_kernel_min_alpha: float = Field(default=1.0 / 255.0, gt=0.0)
    particle_kernel_max_alpha: float = Field(default=0.99, gt=0.0, le=1.0)
    primitive_type: Literal["instances"] = "instances"
    min_transmittance: float = Field(default=0.001, gt=0.0)
    enable_normals: bool = False
    enable_hitcounts: bool = True
    max_consecutive_bvh_update: int = Field(default=15, ge=1)
    ray_principal_point_mode: Literal["image_center", "intrinsics"] = (
        "image_center"
    )
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    return_alpha: bool = True
    return_depth: bool = True
    training_backend_options: Stoch3DGSTrainingBackendOptionsConfig = Field(
        default_factory=Stoch3DGSTrainingBackendOptionsConfig
    )

    def build(
        self, context: ember.TrainingRunContext
    ) -> ember.RenderPipelineSpec:
        """Build the runtime render pipeline spec."""
        del context
        return ember.RenderPipelineSpec(
            backend=self.backend,
            return_alpha=self.return_alpha,
            return_depth=self.return_depth,
            feature_fn=self.training_backend_options.build_feature_fn(),
            backend_options={
                "particle_kernel_degree": self.particle_kernel_degree,
                "particle_kernel_density_clamping": (
                    self.particle_kernel_density_clamping
                ),
                "particle_kernel_min_response": self.particle_kernel_min_response,
                "particle_kernel_min_alpha": self.particle_kernel_min_alpha,
                "particle_kernel_max_alpha": self.particle_kernel_max_alpha,
                "primitive_type": self.primitive_type,
                "min_transmittance": self.min_transmittance,
                "enable_normals": self.enable_normals,
                "enable_hitcounts": self.enable_hitcounts,
                "max_consecutive_bvh_update": self.max_consecutive_bvh_update,
                "ray_principal_point_mode": self.ray_principal_point_mode,
                "background_color": list(self.background_color),
            },
        )


@app.class_definition
class Stoch3DGSOptimizationConfig(Stoch3DGSConfigBase):
    """Typed Stoch3DGS optimization config."""

    center_position_lr_init: float = Field(default=1.6e-4, gt=0.0)
    center_position_lr_final: float = Field(default=1.6e-6, gt=0.0)
    center_position_lr_max_steps: int = Field(default=30_000, ge=1)
    center_position_lr_step_offset: int = Field(default=1, ge=0)
    features_albedo_lr: float = Field(default=2.5e-3, gt=0.0)
    features_specular_lr_divisor: float = Field(default=20.0, gt=0.0)
    density_lr: float = Field(default=0.04, gt=0.0)
    rotation_lr: float = Field(default=1e-3, gt=0.0)
    scale_lr: float = Field(default=5e-3, gt=0.0)
    adam_eps: float = Field(default=1e-15, gt=0.0)

    def build(
        self, context: ember.TrainingRunContext
    ) -> ember.OptimizationConfig:
        """Build Stoch3DGS optimizer groups with the upstream step schedule."""
        center_position_lr_max_steps = min(
            self.center_position_lr_max_steps,
            context.max_steps,
        )
        return ember.OptimizationConfig(
            parameter_groups=[
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene",
                        name="center_position",
                    ),
                    optimizer="adam",
                    lr=self.center_position_lr_init * context.camera_extent,
                    optimizer_kwargs={"eps": self.adam_eps},
                    scheduler=ember.bound_callable(
                        target="ember_core.training.exponential_decay_to",
                        kwargs={
                            "final_lr": (
                                self.center_position_lr_final
                                * context.camera_extent
                            ),
                            "max_steps": center_position_lr_max_steps,
                            "step_offset": self.center_position_lr_step_offset,
                        },
                    ),
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene",
                        name="feature",
                        view=ember.TensorViewSpec(
                            slices=(
                                ember.TensorSliceSpec(
                                    axis=1,
                                    start=0,
                                    stop=1,
                                ),
                            )
                        ),
                    ),
                    optimizer="adam",
                    lr=self.features_albedo_lr,
                    optimizer_kwargs={"eps": self.adam_eps},
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene",
                        name="feature",
                        view=ember.TensorViewSpec(
                            slices=(ember.TensorSliceSpec(axis=1, start=1),)
                        ),
                    ),
                    optimizer="adam",
                    lr=self.features_albedo_lr
                    / self.features_specular_lr_divisor,
                    optimizer_kwargs={"eps": self.adam_eps},
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene",
                        name="logit_opacity",
                    ),
                    optimizer="adam",
                    lr=self.density_lr,
                    optimizer_kwargs={"eps": self.adam_eps},
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene",
                        name="log_scales",
                    ),
                    optimizer="adam",
                    lr=self.scale_lr,
                    optimizer_kwargs={"eps": self.adam_eps},
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene",
                        name="quaternion_orientation",
                    ),
                    optimizer="adam",
                    lr=self.rotation_lr,
                    optimizer_kwargs={"eps": self.adam_eps},
                ),
            ]
        )


@app.class_definition
class Stoch3DGSLossConfig(Stoch3DGSConfigBase):
    """Typed Stoch3DGS training loss config."""

    lambda_l1: float = Field(default=0.8, ge=0.0)
    lambda_ssim: float = Field(default=0.2, ge=0.0)
    lambda_opacity_regularization: float = Field(default=0.0, ge=0.0)
    lambda_scale_regularization: float = Field(default=0.0, ge=0.0)
    ssim_backend: str = "cuda"

    def build(self, context: ember.TrainingRunContext) -> ember.LossConfig:
        """Build the runtime loss config."""
        del context
        return ember.loss_config(
            "papers.stoch3dgs.notebook.stoch3dgs_rgb_l1_ssim_loss",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class Stoch3DGSMipSplatting3DFilterConfig(Stoch3DGSConfigBase):
    """Mip-Splatting 3D filter config."""

    recompute_schedule: Stoch3DGSScheduleConfig = Field(
        default_factory=lambda: Stoch3DGSScheduleConfig(
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
class Stoch3DGSMipSplattingConfig(Stoch3DGSConfigBase):
    """Mip-Splatting 3D filter controls for Stoch3DGS."""

    enabled: bool = False
    three_dimensional_filter: Stoch3DGSMipSplatting3DFilterConfig = Field(
        default_factory=Stoch3DGSMipSplatting3DFilterConfig
    )


@app.class_definition
class Stoch3DGSDensificationConfig(Stoch3DGSConfigBase):
    """Typed notebook-local Stoch3DGS adaptive density config."""

    densify_frequency: int = Field(default=400, ge=1)
    densify_start_iteration: int = Field(default=500, ge=0)
    densify_end_iteration: int = Field(default=15_000, ge=0)
    clone_grad_threshold: float = Field(default=2e-4, gt=0.0)
    split_grad_threshold: float = Field(default=2e-4, gt=0.0)
    relative_size_threshold: float = Field(default=0.01, gt=0.0)
    split_n_gaussians: int = Field(default=2, ge=1)
    prune_frequency: int = Field(default=100, ge=1)
    prune_start_iteration: int = Field(default=500, ge=0)
    prune_end_iteration: int = Field(default=15_000, ge=0)
    prune_density_threshold: float = Field(default=0.005, gt=0.0)
    reset_density_frequency: int = Field(default=3_000, ge=1)
    reset_density_start_iteration: int = Field(default=0, ge=0)
    reset_density_end_iteration: int = Field(default=15_000, ge=0)
    reset_density_new_max_density: float = Field(default=0.01, gt=0.0, lt=1.0)
    max_primitives: int = Field(default=3_000_000, ge=1)
    max_primitives_keep: int = Field(default=2_700_000, ge=1)
    density_decay_gamma: float = Field(default=0.99, gt=0.0)
    density_decay_start_iteration: int = -1
    density_decay_end_iteration: int = -1
    density_decay_frequency: int = Field(default=50, ge=1)

    def build(
        self,
        context: ember.TrainingRunContext,
        *,
        mip_splatting: Stoch3DGSMipSplattingConfig,
    ) -> ember.DensificationConfig:
        """Build the runtime Stoch3DGS densification stack."""
        builders = [
            ember.bound_callable(
                target="papers.stoch3dgs.notebook.Stoch3DGSDensification",
                kwargs=self.model_dump(mode="python"),
            )
        ]
        if mip_splatting.enabled:
            builders.append(
                mip_splatting.three_dimensional_filter.build(context)
            )
        builders.append(
            ember.bound_callable(
                target="papers.stoch3dgs.notebook.Stoch3DGSFinalCleanup",
            )
        )
        return ember.densification_config(*builders)


@app.class_definition
class Stoch3DGSTrainingConfig(Stoch3DGSConfigBase):
    """Typed user-facing Stoch3DGS training config."""

    runtime: ember.RuntimeConfig = Field(default_factory=ember.RuntimeConfig)
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    batching: ember.BatchingConfig = Field(default_factory=ember.BatchingConfig)
    initialization: Stoch3DGSInitializationConfig = Field(
        default_factory=Stoch3DGSInitializationConfig
    )
    render: Stoch3DGSRenderConfig = Field(default_factory=Stoch3DGSRenderConfig)
    optimization: Stoch3DGSOptimizationConfig = Field(
        default_factory=Stoch3DGSOptimizationConfig
    )
    loss: Stoch3DGSLossConfig = Field(default_factory=Stoch3DGSLossConfig)
    mip_splatting: Stoch3DGSMipSplattingConfig = Field(
        default_factory=Stoch3DGSMipSplattingConfig
    )
    densification: Stoch3DGSDensificationConfig = Field(
        default_factory=Stoch3DGSDensificationConfig
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
            render=self.render.build(context),
            optimization=self.optimization.build(context),
            loss=self.loss.build(context),
            densification=self.densification.build(
                context,
                mip_splatting=self.mip_splatting,
            ),
            hooks=ember.HookConfig(
                builders=[
                    ember.bound_callable(
                        target=(
                            "papers.stoch3dgs.notebook.Stoch3DGSActiveSHHook"
                        ),
                        kwargs=(
                            self.render.training_backend_options.model_dump(
                                mode="python"
                            )
                        ),
                    )
                ]
            ),
            checkpoint=self.checkpoint,
        )


@app.class_definition
class Stoch3DGSExperimentConfig(Stoch3DGSConfigBase):
    """Resolved experiment config."""

    preset: Stoch3DGSDefaultName = "garden_stoch"
    scene: Stoch3DGSSceneConfig = Field(default_factory=Stoch3DGSSceneConfig)
    data: Stoch3DGSDataConfig = Field(default_factory=Stoch3DGSDataConfig)
    training: Stoch3DGSTrainingConfig


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Function definitions
    """)
    return


@app.function
def default_checkpoint_dir(
    preset: Stoch3DGSDefaultName,
    backend: Stoch3DGSBackendName,
) -> Path:
    """Return the default checkpoint directory for a preset/backend pair."""
    return DEFAULT_CHECKPOINT_ROOT / preset / backend


@app.function
def stoch3dgs_preset_catalog() -> ConfigPresetCatalog[
    Stoch3DGSExperimentConfig
]:
    """Return the notebook's named JSON preset catalog."""
    return ConfigPresetCatalog(
        model_cls=Stoch3DGSExperimentConfig,
        presets={
            "garden_stoch": ConfigPreset(
                name="garden_stoch",
                path=DEFAULTS_DIR / "garden_stoch.json",
                label="Garden stochastic",
                base_dir=REPO_ROOT,
            ),
            "garden_debug_val": ConfigPreset(
                name="garden_debug_val",
                path=DEFAULTS_DIR / "garden_debug_val.json",
                label="Garden debug validation",
                base_dir=REPO_ROOT,
            ),
        },
        default="garden_stoch",
    )


@app.function
def resolve_stoch3dgs_point_cloud(
    scene_record: ember.SceneRecord,
) -> ember.PointCloudState:
    """Return the SfM point cloud required by Stoch3DGS initialization."""
    if scene_record.point_cloud is None:
        raise ValueError(
            "Stoch3DGS initialization requires an SfM point cloud."
        )
    return scene_record.point_cloud


@app.function
def stoch3dgs_observer_points(
    scene_record: ember.SceneRecord,
) -> Float[Tensor, " num_cams 3"]:
    """Return camera centers in the same role as upstream observer points."""
    camera_to_world = scene_record.resolve_camera_sensor().camera.cam_to_world
    return camera_to_world[:, :3, 3]


@app.function
def stoch3dgs_scene_extent(
    camera_centers: Float[Tensor, " num_cams 3"],
) -> float:
    """Compute the upstream Colmap dataset camera extent."""
    center = camera_centers.mean(dim=0, keepdim=True)
    return float((camera_centers - center).norm(dim=1).max().item() * 1.1)


@app.function
def stoch3dgs_rgb_to_sh(
    rgb: Float[Tensor, " num_points 3"],
) -> Float[Tensor, " num_points 3"]:
    """Convert normalized RGB colors to SH DC coefficients."""
    return (rgb - 0.5) / 0.28209479177387814


@app.function
def stoch3dgs_nearest_neighbor_distances(
    positions: Float[Tensor, " num_points 3"],
    targets: Float[Tensor, " num_targets 3"] | None = None,
    *,
    torch_chunk_size: int = 8192,
) -> Float[Tensor, " num_points"]:
    """Compute differentiable nearest-neighbor distances without extension code."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "Stoch3DGS KNN distances expect positions with shape "
            f"(num_points, 3), got {tuple(positions.shape)}."
        )
    if targets is not None and (targets.ndim != 2 or targets.shape[1] != 3):
        raise ValueError(
            "Stoch3DGS observer distances expect targets with shape "
            f"(num_targets, 3), got {tuple(targets.shape)}."
        )
    if positions.shape[0] <= 1 and targets is None:
        return torch.ones(
            (positions.shape[0],),
            dtype=positions.dtype,
            device=positions.device,
        )
    resolved_targets = positions if targets is None else targets
    nearest_distances = []
    for start in range(0, int(positions.shape[0]), torch_chunk_size):
        stop = min(start + torch_chunk_size, int(positions.shape[0]))
        distances = torch.cdist(positions[start:stop], resolved_targets)
        if targets is None:
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
        nearest_distances.append(distances.min(dim=1).values)
    return torch.cat(nearest_distances, dim=0).clamp_min(1e-7)


@app.function
def initialize_stoch3dgs_model_from_scene_record(
    scene_record: ember.SceneRecord,
    *,
    modules: dict[str, torch.nn.Module] | None = None,
    parameters: dict[str, torch.nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    sh_degree: int = 3,
    default_density: float = 0.1,
    default_scale_factor: float = 1.0,
    observation_scale_factor: float = 0.01,
    use_observation_points: bool = True,
    default_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    device: torch.device | None = None,
) -> ember.InitializedModel:
    """Initialize Gaussians following the upstream Stoch3DGS Colmap path."""
    point_cloud = resolve_stoch3dgs_point_cloud(scene_record)
    target_device = device or torch.device("cpu")
    centers = point_cloud.points.to(device=target_device, dtype=torch.float32)
    num_points = int(centers.shape[0])

    colors = (
        torch.tensor(default_color, dtype=torch.float32, device=target_device)
        .view(1, 3)
        .expand(num_points, 3)
        if point_cloud.colors is None
        else point_cloud.colors.to(device=target_device, dtype=torch.float32)
    )
    sh_coeffs = (sh_degree + 1) ** 2
    feature = torch.zeros(
        (num_points, sh_coeffs, 3),
        dtype=torch.float32,
        device=target_device,
    )
    feature[:, 0, :] = stoch3dgs_rgb_to_sh(colors)

    if use_observation_points:
        observers = stoch3dgs_observer_points(scene_record).to(
            device=target_device,
            dtype=torch.float32,
        )
        distances = stoch3dgs_nearest_neighbor_distances(
            centers,
            observers,
        )
        scales = distances * observation_scale_factor
    else:
        scales = stoch3dgs_nearest_neighbor_distances(centers)
    log_scales = (scales * default_scale_factor).log()[:, None].repeat(1, 3)

    quaternion_orientation = torch.rand(
        (num_points, 4),
        dtype=torch.float32,
        device=target_device,
    )

    density = torch.full(
        (num_points,),
        default_density,
        dtype=torch.float32,
        device=target_device,
    )
    logit_opacity = density.clamp(1e-5, 1.0 - 1e-5).logit()
    scene = ember.GaussianScene3D(
        center_position=centers.requires_grad_(True),
        log_scales=log_scales.requires_grad_(True),
        quaternion_orientation=quaternion_orientation.requires_grad_(True),
        logit_opacity=logit_opacity.requires_grad_(True),
        feature=feature.requires_grad_(True),
        sh_degree=sh_degree,
    )
    camera_centers = stoch3dgs_observer_points(scene_record).to(torch.float32)
    model_metadata = dict(metadata or {})
    model_metadata.update(
        {
            "active_sh_degree": 0,
            "max_sh_degree": sh_degree,
            "stoch3dgs_scene_extent": stoch3dgs_scene_extent(camera_centers),
        }
    )
    model_buffers = dict(buffers or {})
    model_buffers["rolling_weight_contrib"] = torch.zeros(
        (num_points, 1),
        dtype=torch.float32,
        device=target_device,
    )
    return ember.InitializedModel(
        scene=scene,
        modules=dict(modules or {}),
        parameters=dict(parameters or {}),
        buffers=model_buffers,
        metadata=model_metadata,
    )


@app.function
def stoch3dgs_active_sh_scene(
    model: ember.InitializedModel,
    camera: ember.CameraState,
) -> ember.GaussianScene3D:
    """Return the Gaussian scene with inactive SH coefficients masked out."""
    del camera
    scene = model.scene
    if not isinstance(scene, ember.GaussianScene3D):
        raise TypeError("Stoch3DGS active SH expects a GaussianScene3D model.")
    active_sh_degree = int(
        model.metadata.get("active_sh_degree", scene.sh_degree)
    )
    max_coeffs = (scene.sh_degree + 1) ** 2
    active_coeffs = min((active_sh_degree + 1) ** 2, max_coeffs)
    if active_coeffs == max_coeffs:
        return scene
    feature = scene.feature.clone()
    feature[:, active_coeffs:, :] = 0.0
    return ember.GaussianScene3D(
        center_position=scene.center_position,
        log_scales=scene.log_scales,
        quaternion_orientation=scene.quaternion_orientation,
        logit_opacity=scene.logit_opacity,
        feature=feature,
        sh_degree=active_sh_degree,
    )


@app.function
def stoch3dgs_rgb_l1_ssim_loss(
    state: TrainState,
    batch: Any,
    render_output: Any,
    *,
    weights: dict[str, float],
    lambda_l1: float = 0.8,
    lambda_ssim: float = 0.2,
    lambda_opacity_regularization: float = 0.0,
    lambda_scale_regularization: float = 0.0,
    ssim_backend: str = "cuda",
) -> LossResult:
    """Upstream Stoch3DGS RGB reconstruction loss."""
    del weights
    prediction = render_output.render
    target = batch.images
    if prediction.shape != target.shape:
        raise ValueError(
            "Stoch3DGS RGB loss expects render and target images to share NHWC "
            f"shape, got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    l1_loss = (prediction - target).abs().mean()
    ssim_score = ember_splatting.ssim_score(
        prediction,
        target,
        backend=ssim_backend,
    )
    ssim_loss = 1.0 - ssim_score
    scene = state.model.scene
    if not isinstance(scene, ember.GaussianScene3D):
        raise TypeError("Stoch3DGS RGB loss expects a GaussianScene3D model.")
    opacity_regularization = torch.sigmoid(scene.logit_opacity).mean()
    scale_regularization = torch.exp(scene.log_scales).mean()
    loss = (
        lambda_l1 * l1_loss
        + lambda_ssim * ssim_loss
        + lambda_opacity_regularization * opacity_regularization
        + lambda_scale_regularization * scale_regularization
    )
    metrics = {
        "l1": float(l1_loss.detach().item()),
        "ssim_loss": float(ssim_loss.detach().item()),
    }
    if lambda_opacity_regularization > 0.0:
        metrics["opacity_regularization"] = float(
            opacity_regularization.detach().item()
        )
    if lambda_scale_regularization > 0.0:
        metrics["scale_regularization"] = float(
            scale_regularization.detach().item()
        )
    return LossResult(loss=loss, metrics=metrics)


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


@app.function
def resolve_checkpoint_output_dir(
    config: Stoch3DGSExperimentConfig,
) -> Path:
    """Resolve the output checkpoint directory for the current config."""
    output_dir = config.training.checkpoint.output_dir
    if output_dir == Path("checkpoints/latest"):
        return default_checkpoint_dir(
            config.preset,
            config.training.render.backend,
        )
    return output_dir


@app.function
def resolve_training_config(
    config: Stoch3DGSExperimentConfig,
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
            ember.PreparedFrameDataset(
                scene_record,
                config=build_prepared_frame_dataset_config(current_config),
            )
            if scene_record is not None and current_config is not None
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
    return resolved_training_config, training_config, viewer_config


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
                title="Stoch3DGS training viewer",
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
        training_result = run_stoch3dgs_training(
            current_config,
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


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Densification
    """)
    return


@app.class_definition
@runtime_checkable
class HasStoch3DGSTrainingInfo(Protocol):
    """Render output attributes used by Stoch3DGS training."""

    weights: Float[Tensor, " num_splats 1"]
    visibility: Float[Tensor, " num_splats 1"]


@app.class_definition
class Stoch3DGSActiveSHHook:
    """Update active SH degree according to upstream progressive training."""

    def __init__(
        self,
        *,
        max_sh_degree: int = 3,
        sh_start_step: int = 1000,
        sh_step_interval: int = 1000,
    ) -> None:
        self.max_sh_degree = max_sh_degree
        self.sh_start_step = sh_start_step
        self.sh_step_interval = sh_step_interval

    def before_step(self, state: TrainState) -> None:
        """Set active SH degree for the next render."""
        if state.step <= self.sh_start_step:
            active = 0
        else:
            active = (
                1
                + (state.step - self.sh_start_step - 1) // self.sh_step_interval
            )
        state.model.metadata["active_sh_degree"] = int(
            min(self.max_sh_degree, active)
        )


@app.function
def stoch3dgs_step_includes(
    step: int,
    start: int,
    end: int,
    frequency: int,
) -> bool:
    """Match upstream check_step_condition semantics."""
    return (
        start >= 0
        and step > start
        and (step < end or end == -1)
        and frequency > 0
        and step % frequency == 0
    )


@app.class_definition
class Stoch3DGSDensification(BaseDensificationMethod):
    """Notebook-local implementation of upstream Stoch3DGS GSStrategy."""

    expected_scene_families = ("gaussian",)

    def __init__(
        self,
        *,
        densify_frequency: int = 400,
        densify_start_iteration: int = 500,
        densify_end_iteration: int = 15_000,
        clone_grad_threshold: float = 2e-4,
        split_grad_threshold: float = 2e-4,
        relative_size_threshold: float = 0.01,
        split_n_gaussians: int = 2,
        prune_frequency: int = 100,
        prune_start_iteration: int = 500,
        prune_end_iteration: int = 15_000,
        prune_density_threshold: float = 0.005,
        reset_density_frequency: int = 3_000,
        reset_density_start_iteration: int = 0,
        reset_density_end_iteration: int = 15_000,
        reset_density_new_max_density: float = 0.01,
        max_primitives: int = 3_000_000,
        max_primitives_keep: int = 2_700_000,
        density_decay_gamma: float = 0.99,
        density_decay_start_iteration: int = -1,
        density_decay_end_iteration: int = -1,
        density_decay_frequency: int = 50,
    ) -> None:
        super().__init__()
        self.densify_frequency = densify_frequency
        self.densify_start_iteration = densify_start_iteration
        self.densify_end_iteration = densify_end_iteration
        self.clone_grad_threshold = clone_grad_threshold
        self.split_grad_threshold = split_grad_threshold
        self.relative_size_threshold = relative_size_threshold
        self.split_n_gaussians = split_n_gaussians
        self.prune_frequency = prune_frequency
        self.prune_start_iteration = prune_start_iteration
        self.prune_end_iteration = prune_end_iteration
        self.prune_density_threshold = prune_density_threshold
        self.reset_density_frequency = reset_density_frequency
        self.reset_density_start_iteration = reset_density_start_iteration
        self.reset_density_end_iteration = reset_density_end_iteration
        self.reset_density_new_max_density = reset_density_new_max_density
        self.max_primitives = max_primitives
        self.max_primitives_keep = max_primitives_keep
        self.density_decay_gamma = density_decay_gamma
        self.density_decay_start_iteration = density_decay_start_iteration
        self.density_decay_end_iteration = density_decay_end_iteration
        self.density_decay_frequency = density_decay_frequency
        self.densify_grad_norm_accum: Tensor | None = None
        self.densify_grad_norm_denom: Tensor | None = None
        self.family_ops: GaussianFamilyOps | None = None

    def bind(
        self,
        state: TrainState,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        """Bind topology operations and initialize buffers."""
        del optimizers
        if not isinstance(family_ops, GaussianFamilyOps):
            raise TypeError(
                "Stoch3DGSDensification requires GaussianFamilyOps."
            )
        self.family_ops = family_ops
        self._ensure_buffers(state.model.scene)
        self._ensure_weight_buffer(state)

    def before_training(
        self,
        context: DensificationLifecycleContext,
    ) -> None:
        """Ensure notebook-local buffers are present before training."""
        self._ensure_buffers(context.state.model.scene)
        self._ensure_weight_buffer(context.state)

    def get_render_requirements(
        self,
        state: object,
    ) -> DensificationRenderRequirements:
        """Native Stoch3DGS emits weights and visibility by default."""
        del state
        return DensificationRenderRequirements()

    def post_backward(self, context: DensificationContext) -> None:
        """Accumulate position-gradient norms and native weight contribution."""
        scene = context.state.model.scene
        if not isinstance(scene, ember.GaussianScene3D):
            raise TypeError("Stoch3DGSDensification expects GaussianScene3D.")
        self._ensure_buffers(scene)
        self._ensure_weight_buffer(context.state)
        if not isinstance(context.render_output, HasStoch3DGSTrainingInfo):
            raise TypeError(
                "Stoch3DGSDensification requires render outputs with native "
                "Stoch3DGS weights and visibility."
            )
        weights = context.render_output.weights.detach()
        if (
            weights.shape
            != context.state.model.buffers["rolling_weight_contrib"].shape
        ):
            raise ValueError(
                "Stoch3DGS weights shape must follow the current primitive "
                "count."
            )
        context.state.model.buffers["rolling_weight_contrib"] = (
            context.state.model.buffers["rolling_weight_contrib"] + weights
        )
        if stoch3dgs_step_includes(
            context.step,
            0,
            self.densify_end_iteration,
            1,
        ):
            self._update_gradient_buffer(context)

    def post_optimizer_step(self, context: DensificationContext) -> None:
        """Run upstream densification, pruning, density decay, and reset."""
        scene = context.state.model.scene
        if not isinstance(scene, ember.GaussianScene3D):
            raise TypeError("Stoch3DGSDensification expects GaussianScene3D.")
        self._ensure_buffers(scene)
        if stoch3dgs_step_includes(
            context.step,
            self.densify_start_iteration,
            self.densify_end_iteration,
            self.densify_frequency,
        ):
            self._densify_gaussians(context)
        if stoch3dgs_step_includes(
            context.step,
            self.prune_start_iteration,
            self.prune_end_iteration,
            self.prune_frequency,
        ):
            self._prune_gaussians_opacity(context)
        if stoch3dgs_step_includes(
            context.step,
            self.density_decay_start_iteration,
            self.density_decay_end_iteration,
            self.density_decay_frequency,
        ):
            self._decay_density(context)
        if stoch3dgs_step_includes(
            context.step,
            self.reset_density_start_iteration,
            self.reset_density_end_iteration,
            self.reset_density_frequency,
        ):
            self._reset_density(context)

    def _ensure_buffers(self, scene: ember.GaussianScene3D) -> None:
        num_splats = int(scene.center_position.shape[0])
        device = scene.center_position.device
        dtype = scene.center_position.dtype
        expected = (num_splats, 1)
        if (
            self.densify_grad_norm_accum is None
            or tuple(self.densify_grad_norm_accum.shape) != expected
        ):
            self.densify_grad_norm_accum = torch.zeros(
                expected,
                dtype=dtype,
                device=device,
            )
        if (
            self.densify_grad_norm_denom is None
            or tuple(self.densify_grad_norm_denom.shape) != expected
        ):
            self.densify_grad_norm_denom = torch.zeros(
                expected,
                dtype=torch.int32,
                device=device,
            )

    def _ensure_weight_buffer(self, state: TrainState) -> None:
        scene = state.model.scene
        if not isinstance(scene, ember.GaussianScene3D):
            return
        expected = (int(scene.center_position.shape[0]), 1)
        current = state.model.buffers.get("rolling_weight_contrib")
        if current is None or tuple(current.shape) != expected:
            state.model.buffers["rolling_weight_contrib"] = torch.zeros(
                expected,
                dtype=scene.center_position.dtype,
                device=scene.center_position.device,
            )

    def _update_gradient_buffer(self, context: DensificationContext) -> None:
        assert self.densify_grad_norm_accum is not None
        assert self.densify_grad_norm_denom is not None
        scene = context.state.model.scene
        gradients = scene.center_position.grad
        if gradients is None:
            return
        mask = (gradients != 0).max(dim=1).values
        if not bool(mask.any()):
            return
        camera_to_world = context.batch.camera.cam_to_world
        sensor_position = camera_to_world[0, :3, 3]
        distance_to_camera = (
            scene.center_position.detach()[mask] - sensor_position
        ).norm(dim=1, keepdim=True)
        self.densify_grad_norm_accum[mask] += (
            torch.norm(
                gradients[mask] * distance_to_camera,
                dim=-1,
                keepdim=True,
            )
            / 2.0
        )
        self.densify_grad_norm_denom[mask] += 1

    def _densify_gaussians(self, context: DensificationContext) -> None:
        assert self.family_ops is not None
        assert self.densify_grad_norm_accum is not None
        assert self.densify_grad_norm_denom is not None
        scene = context.state.model.scene
        denom = self.densify_grad_norm_denom.to(
            dtype=self.densify_grad_norm_accum.dtype
        ).clamp_min(1.0)
        densify_grad_norm = (self.densify_grad_norm_accum / denom).squeeze()
        densify_grad_norm = torch.nan_to_num(densify_grad_norm, nan=0.0)
        max_scale = torch.exp(scene.log_scales).max(dim=1).values
        relative_threshold = self.relative_size_threshold * self._scene_extent(
            context
        )
        clone_mask = torch.logical_and(
            densify_grad_norm >= self.clone_grad_threshold,
            max_scale <= relative_threshold,
        )
        split_mask = torch.logical_and(
            densify_grad_norm >= self.split_grad_threshold,
            max_scale > relative_threshold,
        )
        self.family_ops.clone_and_split(
            clone_mask,
            split_mask,
            num_children=self.split_n_gaussians,
            scale_shrink=1.0 / (0.8 * self.split_n_gaussians),
        )
        self._reset_all_dynamic_buffers(context.state)
        if scene.center_position.device.type == "cuda":
            torch.cuda.empty_cache()

    def _prune_gaussians_opacity(self, context: DensificationContext) -> None:
        assert self.family_ops is not None
        scene = context.state.model.scene
        density_keep = (
            torch.sigmoid(scene.logit_opacity) >= self.prune_density_threshold
        )
        keep_mask = density_keep
        if int(scene.center_position.shape[0]) > self.max_primitives:
            weights = context.state.model.buffers[
                "rolling_weight_contrib"
            ].squeeze()
            sorted_weights, _ = torch.sort(weights, descending=True)
            keep_index = min(
                self.max_primitives_keep, sorted_weights.shape[0] - 1
            )
            prune_weight_threshold = sorted_weights[keep_index]
            if float(sorted_weights.sum().item()) > 0.0:
                keep_mask = torch.logical_and(
                    keep_mask,
                    weights > prune_weight_threshold,
                )
        self.family_ops.prune(keep_mask)
        self._prune_buffers(keep_mask)
        self._reset_weight_buffer(context.state)

    def _decay_density(self, context: DensificationContext) -> None:
        assert self.family_ops is not None
        scene = context.state.model.scene
        opacity = torch.sigmoid(scene.logit_opacity) * self.density_decay_gamma
        self.family_ops._replace_scene(
            scene,
            {
                "logit_opacity": opacity.clamp(1e-5, 1.0 - 1e-5)
                .logit()
                .detach()
                .requires_grad_(scene.logit_opacity.requires_grad)
            },
            {"logit_opacity": lambda _key, old_value: old_value},
        )

    def _reset_density(self, context: DensificationContext) -> None:
        assert self.family_ops is not None
        self.family_ops.reset_opacity(self.reset_density_new_max_density)

    def _scene_extent(self, context: DensificationContext) -> float:
        metadata_extent = context.state.model.metadata.get(
            "stoch3dgs_scene_extent"
        )
        if isinstance(metadata_extent, int | float):
            return float(metadata_extent)
        assert self.family_ops is not None
        return self.family_ops.scene_extent()

    def _reset_all_dynamic_buffers(self, state: TrainState) -> None:
        scene = state.model.scene
        self.densify_grad_norm_accum = None
        self.densify_grad_norm_denom = None
        self._ensure_buffers(scene)
        self._reset_weight_buffer(state)

    def _reset_weight_buffer(self, state: TrainState) -> None:
        scene = state.model.scene
        state.model.buffers["rolling_weight_contrib"] = torch.zeros(
            (int(scene.center_position.shape[0]), 1),
            dtype=scene.center_position.dtype,
            device=scene.center_position.device,
        )

    def _prune_buffers(self, keep_mask: Tensor) -> None:
        if self.densify_grad_norm_accum is not None:
            self.densify_grad_norm_accum = self.densify_grad_norm_accum[
                keep_mask
            ]
        if self.densify_grad_norm_denom is not None:
            self.densify_grad_norm_denom = self.densify_grad_norm_denom[
                keep_mask
            ]


@app.function
def run_stoch3dgs_training(
    experiment_config: Stoch3DGSExperimentConfig,
    frame_dataset: ember.PreparedFrameDataset,
    *,
    training_config: ember.TrainingConfig | None = None,
) -> TrainingResult:
    """Run Stoch3DGS training from a prepared frame dataset."""
    resolved_training_config = training_config or resolve_training_config(
        experiment_config,
        frame_dataset,
    )
    try:
        return ember.run_training(frame_dataset, resolved_training_config)
    except (ImportError, RuntimeError) as exc:
        message = str(exc).lower()
        extension_markers = (
            "cuda",
            "optix",
            "slangc",
            "extension",
            "nvrtc",
            "threedgrt",
        )
        if any(marker in message for marker in extension_markers):
            raise RuntimeError(
                "Stoch3DGS native extension setup failed. Per notebook policy, "
                "no Torch fallback is attempted."
            ) from exc
        raise


@app.class_definition
class Stoch3DGSFinalCleanup(BaseDensificationMethod):
    """Notebook-local Stoch3DGS checkpoint cleanup before export."""

    expected_scene_families = ("gaussian",)

    def __init__(self, *, min_density: float = 1.0 / 255.0) -> None:
        super().__init__()
        self.min_density = min_density
        self.family_ops: GaussianFamilyOps | None = None

    def bind(
        self,
        state: TrainState,
        optimizers: Sequence[Any],
        family_ops: Any,
    ) -> None:
        del state, optimizers
        if not isinstance(family_ops, GaussianFamilyOps):
            raise TypeError("Stoch3DGSFinalCleanup requires GaussianFamilyOps.")
        self.family_ops = family_ops

    def after_training(self, context: DensificationLifecycleContext) -> None:
        """Remove nearly transparent primitives before checkpoint export."""
        del context
        assert self.family_ops is not None
        scene = self.family_ops.scene
        keep_mask = torch.sigmoid(scene.logit_opacity) >= self.min_density
        self.family_ops.prune(keep_mask)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Notebook support
    """)
    return


@app.function
def resolved_stoch3dgs_scene_path(config: Stoch3DGSExperimentConfig) -> Path:
    """Resolve the configured scene path without substituting sample scenes."""
    return config.scene.path.expanduser()


@app.function
def resolved_resized_image_cache_root(
    config: Stoch3DGSExperimentConfig,
) -> Path:
    """Return the resized-image cache root for a config."""
    if config.data.resized_image_cache_root is not None:
        return config.data.resized_image_cache_root.expanduser()
    return (
        resolved_stoch3dgs_scene_path(config) / "ember_cache" / "resized_images"
    )


@app.function
def stoch3dgs_source_image_root(config: Stoch3DGSExperimentConfig) -> Path:
    """Return the full-resolution source image root."""
    if config.scene.image_root is not None:
        return config.scene.image_root.expanduser()
    return resolved_stoch3dgs_scene_path(config) / "images"


@app.function
def stoch3dgs_resized_cache_enabled(
    config: Stoch3DGSExperimentConfig,
) -> bool:
    """Return whether Stoch3DGS should use a derived resized image cache."""
    return (
        config.data.cache_resized_images
        and config.data.image_scale_factor != 1.0
    )


@app.function
def stoch3dgs_resized_cache_root(
    config: Stoch3DGSExperimentConfig,
) -> Path:
    """Return the derived resized image cache root for this config."""
    scale_name = f"{config.data.image_scale_factor:.6f}".rstrip("0").rstrip(".")
    scale_name = scale_name.replace(".", "p")
    return resolved_resized_image_cache_root(config) / (
        f"scale_{scale_name}_{config.data.interpolation}"
    )


@app.function
def stoch3dgs_pillow_resampling(interpolation: str) -> Any:
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
def enforce_stoch3dgs_resized_cache_limit(
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
def materialize_stoch3dgs_resized_image_cache(
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
    resampling = stoch3dgs_pillow_resampling(interpolation)
    enforce_stoch3dgs_resized_cache_limit(
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
    enforce_stoch3dgs_resized_cache_limit(
        cache_root=cache_root,
        max_caches=max_caches,
    )
    return cache_root


@app.function
def build_scene_load_config(
    config: Stoch3DGSExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Build scene loading config from notebook options."""
    source_pipes = (
        (ember.HorizonAlignPipeConfig(),) if config.scene.align_horizon else ()
    )
    image_root = (
        materialize_stoch3dgs_resized_image_cache(
            source_root=stoch3dgs_source_image_root(config),
            cache_root=stoch3dgs_resized_cache_root(config),
            scale=config.data.image_scale_factor,
            interpolation=config.data.interpolation,
            max_caches=config.data.max_resized_image_caches,
        )
        if stoch3dgs_resized_cache_enabled(config)
        else (
            config.scene.image_root.expanduser()
            if config.scene.image_root is not None
            else None
        )
    )
    return ember.ColmapSceneConfig(
        path=resolved_stoch3dgs_scene_path(config),
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
    config: Stoch3DGSExperimentConfig,
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
            if stoch3dgs_resized_cache_enabled(config)
            else config.data.image_scale_factor
        ),
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


@app.cell
def _():
    is_script_mode = mo.running_in_notebook() is False
    return (is_script_mode,)


@app.cell
def _(resolved_training_config):
    if resolved_training_config is not None:
        print(
            json.dumps(
                resolved_training_config.model_dump(mode="json"),
                indent=2,
            )
        )
    return


if __name__ == "__main__":
    app.run()
