"""FastGS paper training notebook for Ember."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")

with app.setup:
    import json
    import math
    import shutil
    import sys
    import time
    from collections.abc import Sequence
    from dataclasses import replace
    from pathlib import Path
    from typing import Any, Literal, Protocol, runtime_checkable

    import ember_adapter_backends.fastgs as ember_fastgs_adapter
    import ember_core as ember
    import ember_native_faster_gs.fastgs as ember_fastgs_native
    import ember_splatting_training as ember_splatting
    import marimo as mo
    import torch
    from ember_core.densification import (
        BaseDensificationMethod,
        DensificationContext,
        DensificationLifecycleContext,
        DensificationRenderRequirements,
        GaussianFamilyOps,
        GaussianMetricAttribution,
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
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers.fastgs._densification import (
        FastGSDensification,
        compiled_fastgs_l1_metric_map,
    )

    DEFAULTS_DIR = NOTEBOOK_DIR / "defaults"
    DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "papers" / "fastgs"
    FastGSBackendName = Literal["adapter.fastgs", "faster_gs.fastgs"]
    FastGSDefaultName = Literal[
        "garden_base",
        "garden_big",
        "garden_debug_val",
        "mipsplatting-big",
    ]
    FastGSDensificationMode = Literal["fastgs", "mcmc"]
    FastGSMetricMapBackend = Literal["eager", "compile"]
    _COMPILED_FASTGS_L1_METRIC_MAP: Any | None = None
    sys.modules.setdefault("papers.fastgs.notebook", sys.modules[__name__])
    ember_fastgs_adapter.register()
    ember_fastgs_native.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # FastGS training
    """)
    return


@app.cell(hide_code=True)
def _(training_controls):
    training_controls
    return


@app.cell(hide_code=True)
def _(dataloader_benchmark_button):
    dataloader_benchmark_button
    return


@app.cell(hide_code=True)
def _(dataloader_benchmark_view):
    dataloader_benchmark_view
    return


@app.cell
def _():
    fastgs_presets = fastgs_preset_catalog()
    config_gui = create_config_gui(
        FastGSExperimentConfig,
        presets=fastgs_presets,
        label="FastGS config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    preset_selector = config_gui.preset_selector(
        label="FastGS preset",
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
class FastGSConfigBase(BaseModel):
    """Strict base model for FastGS paper configs."""

    model_config = {"extra": "forbid"}


@app.class_definition
class FastGSSceneConfig(FastGSConfigBase):
    """Scene-record loading options."""

    path: Path = Path("dataset/mipnerf360/garden")
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool = True


@app.class_definition
class FastGSDataConfig(FastGSConfigBase):
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
class FastGSScheduleConfig(FastGSConfigBase):
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
class FastGSInitializationConfig(FastGSConfigBase):
    """Typed FastGS Gaussian initialization config."""

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
                    "papers.fastgs.notebook."
                    "initialize_fastgs_model_from_scene_record"
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
class FastGSTrainingBackendOptionsConfig(FastGSConfigBase):
    """Typed per-step FastGS training render options."""

    max_sh_degree: int = Field(default=3, ge=0)
    sh_start_step: int = Field(default=1000, ge=0)
    sh_step_interval: int = Field(default=1000, ge=1)
    clamp_output: bool = False

    def build(self) -> ember.CallableSpec:
        """Build the runtime training backend options builder."""
        return ember.bound_callable(
            target="papers.fastgs.notebook.fastgs_training_backend_options",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class FastGSRenderConfig(FastGSConfigBase):
    """Typed FastGS render pipeline config."""

    backend: FastGSBackendName = "faster_gs.fastgs"
    near_plane: float = Field(default=0.2, gt=0.0)
    far_plane: float = Field(default=10_000.0, gt=0.0)
    compact_box_scale: float = Field(default=0.5, gt=0.0)
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    return_alpha: bool = False
    training_backend_options: FastGSTrainingBackendOptionsConfig = Field(
        default_factory=FastGSTrainingBackendOptionsConfig
    )

    def build(
        self,
        context: ember.TrainingRunContext,
        *,
        mip_splatting_screen_filter: bool,
    ) -> ember.RenderPipelineSpec:
        """Build the runtime render pipeline spec."""
        del context
        backend_options: dict[str, Any]
        if self.backend == "adapter.fastgs":
            backend_options = {
                "mult": self.compact_box_scale,
                "background_color": list(self.background_color),
            }
        else:
            backend_options = {
                "near_plane": self.near_plane,
                "far_plane": self.far_plane,
                "mip_splatting_screen_filter": mip_splatting_screen_filter,
                "compact_box_scale": self.compact_box_scale,
                "background_color": list(self.background_color),
            }
        return ember.RenderPipelineSpec(
            backend=self.backend,
            return_alpha=self.return_alpha,
            backend_options=backend_options,
        )


@app.class_definition
class FastGSOptimizationConfig(FastGSConfigBase):
    """Typed Gaussian 3DGS optimization config."""

    center_position_lr_init: float = Field(default=1.6e-4, gt=0.0)
    center_position_lr_final: float = Field(default=1.6e-6, gt=0.0)
    center_position_lr_max_steps: int = Field(default=30_000, ge=1)
    center_position_lr_step_offset: int = Field(default=1, ge=0)
    lowfeature_lr: float = Field(default=2.5e-3, gt=0.0)
    highfeature_lr: float = Field(default=5e-3, gt=0.0)
    opacity_lr: float = Field(default=2.5e-2, gt=0.0)
    scaling_lr: float = Field(default=5e-3, gt=0.0)
    rotation_lr: float = Field(default=1e-3, gt=0.0)
    adam_eps: float = Field(default=1e-15, gt=0.0)

    def build(
        self, context: ember.TrainingRunContext
    ) -> ember.OptimizationConfig:
        """Build FastGS optimizer groups with the paper step schedule."""
        optimizer = "papers.fastgs.notebook.FastGSScheduledAdam"
        optimizer_kwargs = {"eps": self.adam_eps, "schedule_kind": "main"}
        sh_optimizer_kwargs = {"eps": self.adam_eps, "schedule_kind": "sh"}
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
                    optimizer=optimizer,
                    lr=self.center_position_lr_init * context.camera_extent,
                    optimizer_kwargs=optimizer_kwargs,
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
                    optimizer=optimizer,
                    lr=self.lowfeature_lr,
                    optimizer_kwargs=optimizer_kwargs,
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene",
                        name="feature",
                        view=ember.TensorViewSpec(
                            slices=(ember.TensorSliceSpec(axis=1, start=1),)
                        ),
                    ),
                    optimizer=optimizer,
                    lr=self.highfeature_lr / 20.0,
                    optimizer_kwargs=sh_optimizer_kwargs,
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene",
                        name="logit_opacity",
                    ),
                    optimizer=optimizer,
                    lr=self.opacity_lr,
                    optimizer_kwargs=optimizer_kwargs,
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene",
                        name="log_scales",
                    ),
                    optimizer=optimizer,
                    lr=self.scaling_lr,
                    optimizer_kwargs=optimizer_kwargs,
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene",
                        name="quaternion_orientation",
                    ),
                    optimizer=optimizer,
                    lr=self.rotation_lr,
                    optimizer_kwargs=optimizer_kwargs,
                ),
            ]
        )


@app.class_definition
class FastGSLossConfig(FastGSConfigBase):
    """Typed FastGS training loss config."""

    lambda_l1: float = Field(default=0.8, ge=0.0)
    lambda_dssim: float = Field(default=0.2, ge=0.0)
    lambda_opacity_regularization: float = Field(default=0.0, ge=0.0)
    lambda_scale_regularization: float = Field(default=0.0, ge=0.0)

    def build(self, context: ember.TrainingRunContext) -> ember.LossConfig:
        """Build the runtime loss config."""
        del context
        return ember.loss_config(
            "papers.fastgs.notebook.rgb_l1_ssim_loss",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class FastGSDensificationMethodConfig(FastGSConfigBase):
    """Typed notebook-local FastGS adaptive density config."""

    refine_every: int = Field(default=500, ge=1)
    start_iter: int = Field(default=500, ge=0)
    stop_iter: int = Field(default=15_000, ge=0)
    loss_thresh: float = Field(default=0.1, gt=0.0)
    grad_threshold: float = Field(default=2e-4, gt=0.0)
    grad_abs_threshold: float = Field(default=1.2e-3, gt=0.0)
    dense_fraction: float = Field(default=0.001, gt=0.0)
    prune_opacity_threshold: float = Field(default=0.005, gt=0.0)
    opacity_reset_every: int = Field(default=3_000, ge=1)
    extra_opacity_reset_iter: int | None = Field(default=None, ge=0)
    max_reset_opacity: float = Field(default=0.8, gt=0.0, lt=1.0)
    scheduled_reset_opacity: float = Field(default=0.01, gt=0.0, lt=1.0)
    probe_view_count: int = Field(default=10, ge=1)
    importance_threshold: float = Field(default=5.0, ge=0.0)
    metric_map_backend: FastGSMetricMapBackend = "eager"
    final_prune_start_iter: int = Field(default=15_000, ge=0)
    final_prune_stop_iter: int = Field(default=30_000, ge=0)
    final_prune_every: int = Field(default=3_000, ge=1)
    final_prune_opacity_threshold: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
    )

    def build(self, context: ember.TrainingRunContext) -> ember.CallableSpec:
        """Build the runtime FastGS densification spec."""
        del context
        return ember.bound_callable(
            target="papers.fastgs.notebook.FastGSDensification",
            kwargs=self.model_dump(mode="python"),
            bind={
                "camera_extent": ember.ctx.run.camera_extent,
                "backend": ember.ctx.run.backend,
            },
        )


@app.class_definition
class FastGSMCMCDensificationConfig(FastGSConfigBase):
    """Typed FastGS MCMC densification config."""

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
            target="papers.fastgs.notebook.build_fastgs_mcmc_densification",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class FastGSMortonOrderingConfig(FastGSConfigBase):
    """Typed scheduled Morton ordering config."""

    schedule: FastGSScheduleConfig = Field(
        default_factory=lambda: FastGSScheduleConfig(
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
class FastGSMipSplatting3DFilterConfig(FastGSConfigBase):
    """Mip-Splatting 3D filter config."""

    recompute_schedule: FastGSScheduleConfig = Field(
        default_factory=lambda: FastGSScheduleConfig(
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
class FastGSMipSplattingConfig(FastGSConfigBase):
    """Full Mip-Splatting controls for FastGS."""

    enabled: bool = False
    screen_filter_enabled: bool = False
    three_dimensional_filter: FastGSMipSplatting3DFilterConfig = Field(
        default_factory=FastGSMipSplatting3DFilterConfig
    )


@app.class_definition
class FastGSFinalCleanupConfig(FastGSConfigBase):
    """Typed FastGS checkpoint cleanup config."""

    min_opacity: float = Field(default=1.0 / 255.0, gt=0.0, lt=1.0)

    def build(self, context: ember.TrainingRunContext) -> ember.CallableSpec:
        """Build the runtime final cleanup spec."""
        del context
        return ember.bound_callable(
            target="papers.fastgs.notebook.FastGSFinalCleanup",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class FastGSDensificationConfig(FastGSConfigBase):
    """Typed FastGS densification stack config."""

    mode: FastGSDensificationMode = "fastgs"
    fastgs: FastGSDensificationMethodConfig = Field(
        default_factory=FastGSDensificationMethodConfig
    )
    mcmc: FastGSMCMCDensificationConfig = Field(
        default_factory=FastGSMCMCDensificationConfig
    )
    morton: FastGSMortonOrderingConfig = Field(
        default_factory=FastGSMortonOrderingConfig
    )
    final_cleanup: FastGSFinalCleanupConfig = Field(
        default_factory=FastGSFinalCleanupConfig
    )

    def build(
        self,
        context: ember.TrainingRunContext,
        *,
        mip_splatting: FastGSMipSplattingConfig,
    ) -> ember.DensificationConfig:
        """Build the runtime FastGS densification stack."""
        if self.mode == "mcmc":
            primary = self.mcmc.build(context)
        else:
            primary = self.fastgs.build(context)
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
class FastGSTrainingConfig(FastGSConfigBase):
    """Typed user-facing FastGS training config."""

    runtime: ember.RuntimeConfig = Field(default_factory=ember.RuntimeConfig)
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    batching: ember.BatchingConfig = Field(default_factory=ember.BatchingConfig)
    initialization: FastGSInitializationConfig = Field(
        default_factory=FastGSInitializationConfig
    )
    render: FastGSRenderConfig = Field(default_factory=FastGSRenderConfig)
    mip_splatting: FastGSMipSplattingConfig = Field(
        default_factory=FastGSMipSplattingConfig
    )
    optimization: FastGSOptimizationConfig = Field(
        default_factory=FastGSOptimizationConfig
    )
    loss: FastGSLossConfig = Field(default_factory=FastGSLossConfig)
    densification: FastGSDensificationConfig = Field(
        default_factory=FastGSDensificationConfig
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
            hooks=ember.HookConfig(
                builders=[
                    ember.bound_callable(
                        target="papers.fastgs.notebook.FastGSSHTrainingHook",
                        kwargs=self.render.training_backend_options.model_dump(
                            mode="python"
                        ),
                    )
                ]
            ),
            checkpoint=self.checkpoint,
        )


@app.class_definition
class FastGSExperimentConfig(FastGSConfigBase):
    """Resolved experiment config."""

    preset: FastGSDefaultName = "garden_base"
    scene: FastGSSceneConfig = Field(default_factory=FastGSSceneConfig)
    data: FastGSDataConfig = Field(default_factory=FastGSDataConfig)
    training: FastGSTrainingConfig


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Function definitions
    """)
    return


@app.function
def default_checkpoint_dir(
    preset: FastGSDefaultName,
    backend: FastGSBackendName,
) -> Path:
    """Return the default checkpoint directory for a preset/backend pair."""
    return DEFAULT_CHECKPOINT_ROOT / preset / backend


@app.function
def fastgs_preset_catalog() -> ConfigPresetCatalog[FastGSExperimentConfig]:
    """Return the notebook's named JSON preset catalog."""
    return ConfigPresetCatalog(
        model_cls=FastGSExperimentConfig,
        presets={
            "garden_base": ConfigPreset(
                name="garden_base",
                path=DEFAULTS_DIR / "garden_base.json",
                label="Garden base",
                base_dir=REPO_ROOT,
            ),
            "garden_big": ConfigPreset(
                name="garden_big",
                path=DEFAULTS_DIR / "garden_big.json",
                label="Garden big",
                base_dir=REPO_ROOT,
            ),
            "garden_debug_val": ConfigPreset(
                name="garden_debug_val",
                path=DEFAULTS_DIR / "garden_debug_val.json",
                label="Garden debug validation",
                base_dir=REPO_ROOT,
            ),
            "mipsplatting-big": ConfigPreset(
                name="mipsplatting-big",
                path=DEFAULTS_DIR / "mipsplatting-big.json",
                label="Mip-Splatting AA big",
                base_dir=REPO_ROOT,
            ),
        },
        default="garden_base",
    )


@app.function
def resolve_fastgs_point_cloud(
    scene_record: ember.SceneRecord,
) -> ember.PointCloudState:
    """Return the SfM point cloud required by FastGS initialization."""
    if scene_record.point_cloud is None:
        raise ValueError("FastGS initialization requires an SfM point cloud.")
    return scene_record.point_cloud


@app.function
def fastgs_root_mean_squared_knn_distances(
    positions: Float[Tensor, " num_points 3"],
    *,
    torch_chunk_size: int = 512,
) -> Float[Tensor, " num_points"]:
    """Compute upstream FastGS initial scale distances for the notebook."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "FastGS KNN distances expect positions with shape "
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
def initialize_fastgs_model_from_scene_record(
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
    """Initialize Gaussians exactly like the FastGS paper implementation."""
    point_cloud = resolve_fastgs_point_cloud(scene_record)
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

    distances = fastgs_root_mean_squared_knn_distances(centers)
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


@app.function
def fastgs_training_backend_options(
    state: TrainState,
    *,
    max_sh_degree: int = 3,
    sh_start_step: int = 1000,
    sh_step_interval: int = 1000,
    clamp_output: bool = False,
) -> dict[str, bool]:
    """Notebook-local placeholder for paper training render options."""
    del state, max_sh_degree, sh_start_step, sh_step_interval
    return {"clamp_output": clamp_output} if clamp_output else {}


@app.class_definition
class FastGSScheduledAdam(torch.optim.Adam):
    """Adam with the optimizer-step cadence from the FastGS code release."""

    def __init__(
        self,
        params: Any,
        *,
        lr: float,
        schedule_kind: Literal["main", "sh"] = "main",
        **kwargs: Any,
    ) -> None:
        super().__init__(params, lr=lr, **kwargs)
        if schedule_kind not in {"main", "sh"}:
            raise ValueError(
                "FastGSScheduledAdam schedule_kind must be 'main' or 'sh'."
            )
        self.schedule_kind = schedule_kind
        self.fastgs_iteration = 0
        self._zero_grad_on_next_call = True

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Only clear gradients after a real FastGS optimizer step."""
        if not self._zero_grad_on_next_call:
            return
        super().zero_grad(set_to_none=set_to_none)
        self._zero_grad_on_next_call = False

    def step(self, closure: Any = None) -> Any:
        self.fastgs_iteration += 1
        if not self.should_step(self.fastgs_iteration):
            self._zero_grad_on_next_call = False
            return None
        result = super().step(closure=closure)
        self._zero_grad_on_next_call = True
        return result

    def should_step(self, iteration: int) -> bool:
        """Return whether this optimizer group steps at an iteration."""
        if self.schedule_kind == "sh" and iteration <= 15_000:
            return iteration % 16 == 0
        if iteration <= 15_000:
            return True
        if iteration <= 20_000:
            return iteration % 32 == 0
        return iteration % 64 == 0


@app.class_definition
class FastGSSHTrainingHook:
    """Increase active SH degree every 1000 steps like the FastGS script."""

    def __init__(
        self,
        *,
        max_sh_degree: int = 3,
        sh_start_step: int = 1000,
        sh_step_interval: int = 1000,
        clamp_output: bool = False,
    ) -> None:
        del clamp_output
        self.max_sh_degree = int(max_sh_degree)
        self.sh_start_step = int(sh_start_step)
        self.sh_step_interval = int(sh_step_interval)

    def after_step(self, state: TrainState, metrics: dict[str, float]) -> None:
        """Update the scene SH degree after each completed training step."""
        del metrics
        scene = state.model.scene
        if not isinstance(scene, ember.GaussianScene3D):
            return
        active_degree = active_fastgs_sh_degree(
            state.step,
            max_degree=self.max_sh_degree,
            start_step=self.sh_start_step,
            step_interval=self.sh_step_interval,
        )
        if scene.sh_degree == active_degree:
            return
        state.model = replace(
            state.model,
            scene=replace(scene, sh_degree=active_degree),
        )


@app.function
def active_fastgs_sh_degree(
    step: int,
    *,
    max_degree: int = 3,
    start_step: int = 1000,
    step_interval: int = 1000,
) -> int:
    """Return the active SH degree used by the FastGS release."""
    if step < start_step:
        return 0
    return min(max_degree, 1 + (step - start_step) // step_interval)


@app.function
def fastgs_normalize_score(score: Tensor) -> Tensor:
    """Normalize a score tensor without synchronizing CUDA to Python."""
    min_value = torch.amin(score)
    max_value = torch.amax(score)
    denom = max_value - min_value
    normalized = (score - min_value) / denom.clamp_min(1e-12)
    return torch.where(
        denom.abs() <= 1e-12, torch.zeros_like(score), normalized
    )


@app.function
def fastgs_l1_metric_map(
    predicted: Tensor,
    target: Tensor,
    loss_thresh: float,
) -> Tensor:
    """Build FastGS' normalized L1 metric map from one rendered probe image."""
    l1_map = (predicted - target).abs().mean(dim=-1)
    normalized_l1 = fastgs_normalize_score(l1_map)
    return (normalized_l1 > loss_thresh).to(torch.int32)


@app.function
def rgb_l1_ssim_loss(
    state: TrainState,
    batch: Any,
    render_output: Any,
    *,
    weights: dict[str, float],
    lambda_l1: float = 0.8,
    lambda_dssim: float = 0.2,
    lambda_opacity_regularization: float = 0.0,
    lambda_scale_regularization: float = 0.0,
) -> LossResult:
    """FastGS paper RGB loss: 0.8 L1 + 0.2 (1 - SSIM)."""
    del weights
    prediction = render_output.render
    target = batch.images
    if prediction.shape != target.shape:
        raise ValueError(
            "FastGS RGB loss expects render and target images to share NHWC "
            f"shape, got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    l1_loss = (prediction - target).abs().mean()
    from ember_splatting_training.losses import ssim_score

    one_minus_ssim = 1.0 - ssim_score(prediction, target)
    scene = state.model.scene
    if not isinstance(scene, ember.GaussianScene3D):
        raise TypeError("rgb_l1_ssim_loss expects a GaussianScene3D model.")
    opacity_regularization = torch.sigmoid(scene.logit_opacity).mean()
    scale_regularization = torch.exp(scene.log_scales).mean()
    loss = (
        lambda_l1 * l1_loss
        + lambda_dssim * one_minus_ssim
        + lambda_opacity_regularization * opacity_regularization
        + lambda_scale_regularization * scale_regularization
    )
    return LossResult(
        loss=loss,
        metrics={
            "l1": float(l1_loss.detach().item()),
            "one_minus_ssim": float(one_minus_ssim.detach().item()),
            "opacity_regularization": float(
                opacity_regularization.detach().item()
            ),
            "scale_regularization": float(scale_regularization.detach().item()),
        },
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


@app.cell
def _():
    dataloader_benchmark_button = mo.ui.run_button(label="Benchmark dataloader")
    return (dataloader_benchmark_button,)


@app.function
def benchmark_fastgs_dataloader(
    experiment_config: FastGSExperimentConfig,
    *,
    warmup_steps: int = 50,
    measured_steps: int = 300,
) -> Any:
    """Benchmark the exact FastGS training dataloader for a config."""
    from ember_core.benchmarks import benchmark_dataloader
    from ember_core.training.runtime import build_dataloader

    scene_record = ember.load_scene_record(
        build_scene_load_config(experiment_config)
    )
    frame_dataset = ember.prepare_frame_dataset(
        scene_record,
        build_prepared_frame_dataset_config(experiment_config),
    )
    training_config = resolve_training_config(
        experiment_config,
        frame_dataset,
    )
    dataloader = build_dataloader(frame_dataset, training_config)
    device = torch.device(training_config.runtime.device)

    def move_batch_to_device(batch: Any) -> Any:
        moved = batch.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return moved

    return benchmark_dataloader(
        dataloader,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        prepare_batch=move_batch_to_device,
        show_progress=True,
    )


@app.cell
def _(current_config, dataloader_benchmark_button):
    should_benchmark_dataloader = bool(dataloader_benchmark_button.value)
    dataloader_benchmark_result = (
        benchmark_fastgs_dataloader(current_config)
        if should_benchmark_dataloader and current_config is not None
        else None
    )
    return (dataloader_benchmark_result,)


@app.cell
def _(dataloader_benchmark_result):
    dataloader_benchmark_view = (
        mo.md("Dataloader benchmark has not run.")
        if dataloader_benchmark_result is None
        else mo.md(
            "Dataloader: "
            f"`{dataloader_benchmark_result.iters_per_sec:.1f}` it/s "
            f"(`{dataloader_benchmark_result.mean_ms_per_batch:.3f}` ms/batch, "
            f"p90 `{dataloader_benchmark_result.p90_ms_per_batch:.3f}` ms). "
            + (
                "Meets the 200 it/s target."
                if dataloader_benchmark_result.iters_per_sec >= 200.0
                else "Below the 200 it/s target."
            )
        )
    )
    return (dataloader_benchmark_view,)


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""
    # Training
    """)
    return


@app.function
def resolve_training_config(
    config: FastGSExperimentConfig,
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
def _(current_config, is_script_mode, prepare_button, train_button):
    should_prepare = (
        is_script_mode or bool(prepare_button.value) or bool(train_button.value)
    )
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
            title="FastGS training viewer",
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
        training_result = run_fastgs_training(
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
def build_fastgs_mcmc_densification(
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
class HasFastGSDensificationInfo(Protocol):
    """Render-output trait for FastGS densification accumulators."""

    densification_info: Float[Tensor, " 4 num_splats"]


@app.cell
def _():
    class _FastGSDensificationCellLocal(BaseDensificationMethod):
        """Notebook-local FastGS adaptive density control."""

        expected_scene_families = ("gaussian",)

        def __init__(
            self,
            *,
            refine_every: int = 100,
            start_iter: int = 600,
            stop_iter: int = 14_900,
            backend: FastGSBackendName = "adapter.fastgs",
            loss_thresh: float = 0.1,
            grad_threshold: float = 2e-4,
            grad_abs_threshold: float = 1.2e-3,
            dense_fraction: float = 0.01,
            prune_opacity_threshold: float = 0.005,
            opacity_reset_every: int = 3_000,
            extra_opacity_reset_iter: int | None = 500,
            max_reset_opacity: float = 0.01,
            scheduled_reset_opacity: float = 0.01,
            probe_view_count: int = 10,
            importance_threshold: float = 5.0,
            metric_map_backend: FastGSMetricMapBackend = "eager",
            final_prune_start_iter: int = 15_000,
            final_prune_stop_iter: int = 30_000,
            final_prune_every: int = 3_000,
            final_prune_opacity_threshold: float = 0.1,
            camera_extent: float = 1.0,
        ) -> None:
            self.refine_schedule = Schedule(
                start_iteration=start_iter,
                end_iteration=stop_iter,
                frequency=refine_every,
            )
            self.backend = backend
            self.stop_iter = stop_iter
            self.loss_thresh = loss_thresh
            self.grad_threshold = grad_threshold
            self.grad_abs_threshold = grad_abs_threshold
            self.dense_fraction = dense_fraction
            self.prune_opacity_threshold = prune_opacity_threshold
            self.opacity_reset_every = opacity_reset_every
            self.extra_opacity_reset_iter = extra_opacity_reset_iter
            self.max_reset_opacity = max_reset_opacity
            self.scheduled_reset_opacity = scheduled_reset_opacity
            self.probe_view_count = probe_view_count
            self.importance_threshold = importance_threshold
            self.metric_map_backend = metric_map_backend
            self.final_prune_start_iter = final_prune_start_iter
            self.final_prune_stop_iter = final_prune_stop_iter
            self.final_prune_every = final_prune_every
            self.final_prune_opacity_threshold = final_prune_opacity_threshold
            self.camera_extent = float(camera_extent)
            self.family_ops: GaussianFamilyOps | None = None
            self.clone_grad_sum: Tensor | None = None
            self.split_grad_sum: Tensor | None = None
            self.visible_count: Tensor | None = None
            self.max_screen_radii: Tensor | None = None

        def get_render_requirements(
            self,
            state: TrainState,
        ) -> DensificationRenderRequirements:
            """Collect FastGS visibility accumulators while densification runs."""
            if self.backend == "adapter.fastgs":
                return DensificationRenderRequirements()
            return DensificationRenderRequirements(
                backend_options={
                    "collect_densification_info": state.step + 1
                    < self.stop_iter
                }
            )

        def bind(
            self, state: Any, optimizers: Sequence[Any], family_ops: Any
        ) -> None:
            """Bind Gaussian topology operations."""
            del state, optimizers
            if not isinstance(family_ops, GaussianFamilyOps):
                raise TypeError(
                    "FastGSDensification requires GaussianFamilyOps."
                )
            self.family_ops = family_ops

        def post_backward(self, context: DensificationContext) -> None:
            """Accumulate FastGS screen-space densification statistics."""
            if context.step + 1 >= self.stop_iter:
                return
            if self.backend == "adapter.fastgs":
                self._accumulate_adapter_gradients(context)
                return
            self._accumulate_native_densification_info(context)

        def pre_optimizer_step(self, context: DensificationContext) -> None:
            """Run scheduled clone/split/prune/reset actions before optimizer."""
            if self.family_ops is None:
                return
            scene = context.state.model.scene
            if not isinstance(scene, ember.GaussianScene):
                return
            upstream_iteration = context.step + 1
            if self.refine_schedule.includes(upstream_iteration):
                self.adaptive_density_control(
                    context, scene, upstream_iteration
                )
                self.reset_accumulators()
            if self.should_reset_opacity(upstream_iteration):
                self.family_ops.reset_opacity(self.scheduled_reset_opacity)
            if self.should_final_prune(upstream_iteration):
                pruning_score = self.compute_pruning_score(context)
                self.final_prune(pruning_score)

        def adaptive_density_control(
            self,
            context: DensificationContext,
            scene: ember.GaussianScene,
            step: int,
        ) -> None:
            if (
                self.visible_count is None
                or self.clone_grad_sum is None
                or self.split_grad_sum is None
                or self.max_screen_radii is None
            ):
                return
            assert self.family_ops is not None
            score_started_at = time.perf_counter()
            importance_score, pruning_score = self.compute_fastgs_scores(
                context,
                densify=True,
            )
            self._record_metric(
                context,
                "refinement_fastgs_score_ms",
                (time.perf_counter() - score_started_at) * 1000.0,
            )
            avg_clone_grad = self.clone_grad_sum / self.visible_count.clamp_min(
                1.0
            )
            avg_split_grad = self.split_grad_sum / self.visible_count.clamp_min(
                1.0
            )
            scales = torch.exp(scene.log_scales).max(dim=-1).values
            metric_mask = importance_score > self.importance_threshold
            clone_mask = (
                (avg_clone_grad >= self.grad_threshold)
                & (scales <= self.dense_fraction * self.camera_extent)
                & metric_mask
            )
            split_mask = (
                (avg_split_grad >= self.grad_abs_threshold)
                & (scales > self.dense_fraction * self.camera_extent)
                & metric_mask
            )
            grown_max_screen_radii = self._grown_zero_accumulator_values(
                self.max_screen_radii,
                clone_mask,
                split_mask,
                num_children=2,
            )

            prune_sampling_ms = 0.0

            def refinement_keep_mask(
                grown_scene: ember.GaussianScene,
            ) -> Tensor:
                nonlocal prune_sampling_ms
                keep_mask = torch.sigmoid(grown_scene.logit_opacity) >= (
                    self.prune_opacity_threshold
                )
                if (
                    step > self.opacity_reset_every
                    and grown_scene.center_position.shape[0] > 0
                ):
                    max_scale = (
                        torch.exp(grown_scene.log_scales).max(dim=-1).values
                    )
                    keep_mask &= max_scale <= 0.1 * self.camera_extent
                    keep_mask &= grown_max_screen_radii <= 20.0
                prune_started_at = time.perf_counter()
                sampled_prune_mask = self._sample_refinement_prune_mask(
                    ~keep_mask,
                    pruning_score,
                )
                prune_sampling_ms += (
                    time.perf_counter() - prune_started_at
                ) * 1000.0
                keep_mask = ~sampled_prune_mask
                keep_mask &= (
                    grown_scene.quaternion_orientation.square().sum(dim=1)
                    >= 1e-8
                )
                return keep_mask

            topology_started_at = time.perf_counter()
            self.family_ops.clone_and_split(
                clone_mask,
                split_mask,
                num_children=2,
                scale_shrink=0.625,
                prune_fn=refinement_keep_mask,
                prune_field_names=(
                    "center_position",
                    "logit_opacity",
                    "log_scales",
                    "quaternion_orientation",
                ),
            )
            self._record_metric(
                context,
                "refinement_fastgs_topology_ms",
                (time.perf_counter() - topology_started_at) * 1000.0,
            )
            self._record_metric(
                context,
                "refinement_fastgs_prune_sampling_ms",
                prune_sampling_ms,
            )
            reset_started_at = time.perf_counter()
            self.family_ops.reset_opacity(self.max_reset_opacity)
            self._record_metric(
                context,
                "refinement_fastgs_opacity_reset_ms",
                (time.perf_counter() - reset_started_at) * 1000.0,
            )

        def _sample_refinement_prune_mask(
            self,
            prune_mask: Tensor,
            pruning_score: Tensor,
        ) -> Tensor:
            """Apply upstream FastGS' budgeted refinement prune sampling."""
            remove_budget = int(0.5 * int(prune_mask.sum().item()))
            if remove_budget <= 0 or pruning_score.numel() == 0:
                return torch.zeros_like(prune_mask)
            scores = 1.0 - pruning_score.reshape(-1)
            weighted_count = min(int(scores.numel()), int(prune_mask.numel()))
            padded_importance = torch.zeros(
                (int(prune_mask.numel()),),
                dtype=torch.float32,
                device=prune_mask.device,
            )
            padded_importance[:weighted_count] = 1.0 / (
                1e-6 + scores[:weighted_count].clamp_min(0.0)
            ).to(device=prune_mask.device, dtype=torch.float32)
            sampled_indices = torch.multinomial(
                padded_importance,
                remove_budget,
                replacement=False,
            )
            sampled_mask = torch.zeros_like(prune_mask)
            sampled_mask[sampled_indices] = True
            return prune_mask & sampled_mask

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

        def should_final_prune(self, step: int) -> bool:
            return (
                step > self.final_prune_start_iter
                and step < self.final_prune_stop_iter
                and step % self.final_prune_every == 0
            )

        def reset_accumulators(self) -> None:
            self.clone_grad_sum = None
            self.split_grad_sum = None
            self.visible_count = None
            self.max_screen_radii = None

        def _append_zero_accumulator_values(self, count: int) -> None:
            if count <= 0:
                return
            for name in (
                "clone_grad_sum",
                "split_grad_sum",
                "visible_count",
                "max_screen_radii",
            ):
                value = getattr(self, name)
                if value is None:
                    continue
                setattr(
                    self,
                    name,
                    torch.cat(
                        [
                            value,
                            torch.zeros(
                                (count,),
                                dtype=value.dtype,
                                device=value.device,
                            ),
                        ]
                    ),
                )

        def _split_accumulator_values(
            self,
            split_mask: torch.Tensor,
            *,
            num_children: int,
        ) -> None:
            for name in (
                "clone_grad_sum",
                "split_grad_sum",
                "visible_count",
                "max_screen_radii",
            ):
                value = getattr(self, name)
                if value is None:
                    continue
                child_count = int(split_mask.sum().item()) * num_children
                setattr(
                    self,
                    name,
                    torch.cat(
                        [
                            value[~split_mask],
                            torch.zeros(
                                (child_count,),
                                dtype=value.dtype,
                                device=value.device,
                            ),
                        ]
                    ),
                )

        def _grown_zero_accumulator_values(
            self,
            value: Tensor,
            clone_mask: Tensor,
            split_mask: Tensor,
            *,
            num_children: int,
        ) -> Tensor:
            """Return accumulator values in fused clone/split output order."""
            clone_count = int(clone_mask.sum().item())
            split_child_count = int(split_mask.sum().item()) * num_children
            return torch.cat(
                [
                    value[~split_mask],
                    torch.zeros(
                        (clone_count,),
                        dtype=value.dtype,
                        device=value.device,
                    ),
                    torch.zeros(
                        (split_child_count,),
                        dtype=value.dtype,
                        device=value.device,
                    ),
                ]
            )

        def _ensure_buffers(
            self,
            *,
            num_splats: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> None:
            shape = (num_splats,)
            if (
                self.clone_grad_sum is None
                or self.clone_grad_sum.shape != shape
            ):
                self.clone_grad_sum = torch.zeros(
                    shape, dtype=dtype, device=device
                )
                self.split_grad_sum = torch.zeros(
                    shape, dtype=dtype, device=device
                )
                self.visible_count = torch.zeros(
                    shape, dtype=dtype, device=device
                )
                self.max_screen_radii = torch.zeros(
                    shape,
                    dtype=dtype,
                    device=device,
                )

        def _accumulate_native_densification_info(
            self,
            context: DensificationContext,
        ) -> None:
            if not isinstance(
                context.render_output, HasFastGSDensificationInfo
            ):
                raise TypeError(
                    "FastGSDensification requires render outputs with "
                    "densification_info."
                )
            scene = context.state.model.scene
            densification_info = (
                context.render_output.densification_info.detach()
            )
            if densification_info.ndim != 2 or densification_info.shape[0] != 4:
                raise ValueError(
                    "FastGS densification_info must have shape "
                    f"(4, num_splats), got {tuple(densification_info.shape)}."
                )
            self._ensure_buffers(
                num_splats=int(scene.center_position.shape[0]),
                dtype=scene.center_position.dtype,
                device=scene.center_position.device,
            )
            assert self.clone_grad_sum is not None
            assert self.split_grad_sum is not None
            assert self.visible_count is not None
            assert self.max_screen_radii is not None
            self.visible_count += densification_info[0]
            self.clone_grad_sum += densification_info[1]
            self.split_grad_sum += densification_info[2]
            self.max_screen_radii = torch.maximum(
                self.max_screen_radii,
                densification_info[3].to(dtype=self.max_screen_radii.dtype),
            )

        def _accumulate_adapter_gradients(
            self,
            context: DensificationContext,
        ) -> None:
            output = context.render_output
            if not all(
                hasattr(output, name)
                for name in ("viewspace_points", "visibility_filter", "radii")
            ):
                raise TypeError(
                    "adapter.fastgs densification requires viewspace_points, "
                    "visibility_filter, and radii render outputs."
                )
            gradients = output.viewspace_points.grad
            if gradients is None:
                return
            scene = context.state.model.scene
            self._ensure_buffers(
                num_splats=int(scene.center_position.shape[0]),
                dtype=scene.center_position.dtype,
                device=scene.center_position.device,
            )
            assert self.clone_grad_sum is not None
            assert self.split_grad_sum is not None
            assert self.visible_count is not None
            assert self.max_screen_radii is not None
            visibility = output.visibility_filter.to(gradients.dtype)
            self.clone_grad_sum += (
                gradients[..., :2].norm(dim=-1) * visibility
            ).sum(dim=0)
            self.split_grad_sum += (
                gradients[..., 2:].norm(dim=-1) * visibility
            ).sum(dim=0)
            self.visible_count += visibility.sum(dim=0)
            visible_radii = torch.where(
                output.visibility_filter,
                output.radii.to(scene.center_position.dtype),
                torch.zeros_like(
                    output.radii, dtype=scene.center_position.dtype
                ),
            )
            self.max_screen_radii = torch.maximum(
                self.max_screen_radii,
                visible_radii.max(dim=0).values,
            )

        def compute_fastgs_scores(
            self,
            context: DensificationContext,
            *,
            densify: bool,
        ) -> tuple[Tensor, Tensor]:
            """Compute FastGS multi-view consistency scores."""
            attribution = self.require_runtime_trait(
                context,
                GaussianMetricAttribution,
            )
            if context.runtime is None:
                raise RuntimeError("FastGS scoring requires a runtime.")
            probe_views = context.runtime.sample_views(self.probe_view_count)
            scene = context.state.model.scene
            importance_sum = torch.zeros(
                int(scene.center_position.shape[0]),
                dtype=scene.center_position.dtype,
                device=scene.center_position.device,
            )
            pruning_sum = torch.zeros_like(importance_sum)
            if not probe_views:
                return importance_sum, pruning_sum
            for sample in probe_views:
                probe_output = context.runtime.render_raw(
                    context.state.model,
                    sample.camera,
                )
                predicted = probe_output.render[0]
                target = sample.image
                metric_map = self._metric_map(predicted, target)
                photometric_loss = self._photometric_loss(predicted, target)
                attributed = attribution.attribute_metric_map(
                    scene,
                    sample.camera,
                    metric_map,
                    options=context.runtime.render_options,
                )
                if densify:
                    importance_sum += attributed
                pruning_sum += photometric_loss * attributed
            importance_score = torch.div(
                importance_sum,
                float(len(probe_views)),
                rounding_mode="floor",
            )
            return importance_score, self._normalize_score(pruning_sum)

        def compute_pruning_score(
            self, context: DensificationContext
        ) -> Tensor:
            """Compute only the FastGS VCP score."""
            _importance_score, pruning_score = self.compute_fastgs_scores(
                context,
                densify=False,
            )
            return pruning_score

        def final_prune(self, pruning_score: Tensor) -> None:
            """Apply FastGS final-stage VCP pruning."""
            if self.family_ops is None:
                return
            scene = self.family_ops.scene
            prune_mask = (
                torch.sigmoid(scene.logit_opacity)
                < self.final_prune_opacity_threshold
            )
            if pruning_score.numel() == prune_mask.numel():
                prune_mask |= pruning_score > 0.9
            if torch.any(prune_mask):
                self.family_ops.prune(~prune_mask)

        def _normalize_score(self, score: Tensor) -> Tensor:
            return fastgs_normalize_score(score)

        def _metric_map(
            self,
            predicted: Tensor,
            target: Tensor,
        ) -> Tensor:
            if self.metric_map_backend == "compile":
                try:
                    return compiled_fastgs_l1_metric_map(
                        predicted,
                        target,
                        self.loss_thresh,
                    )
                except Exception:
                    return fastgs_l1_metric_map(
                        predicted,
                        target,
                        self.loss_thresh,
                    )
            return fastgs_l1_metric_map(
                predicted,
                target,
                self.loss_thresh,
            )

        def _photometric_loss(
            self,
            predicted: Tensor,
            target: Tensor,
        ) -> Tensor:
            l1_loss = (predicted - target).abs().mean()
            from ember_splatting_training.losses import ssim_score

            one_minus_ssim = 1.0 - ssim_score(
                predicted[None, ...],
                target[None, ...],
            )
            return 0.8 * l1_loss + 0.2 * one_minus_ssim

        def _record_metric(
            self,
            context: DensificationContext,
            name: str,
            value: float,
        ) -> None:
            diagnostics = getattr(context.state, "diagnostics", None)
            if not isinstance(diagnostics, dict):
                return
            diagnostics.setdefault("metrics", {})[name] = float(value)

    return


@app.function
def run_fastgs_training(
    frame_dataset: ember.PreparedFrameDataset,
    experiment_config: FastGSExperimentConfig,
    training_config: ember.TrainingConfig | None = None,
    training_viewer_handle: ember_splatting.TrainingViewerHandle | None = None,
) -> TrainingResult:
    """Run FastGS training from a native Ember training config."""
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
class FastGSFinalCleanup(BaseDensificationMethod):
    """Notebook-local FastGS checkpoint cleanup before export."""

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
            raise TypeError("FastGSFinalCleanup requires GaussianFamilyOps.")
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
def fastgs_resized_cache_enabled(config: FastGSExperimentConfig) -> bool:
    """Return whether FastGS should use a derived resized image cache."""
    return (
        config.data.cache_resized_images
        and config.data.image_scale_factor != 1.0
    )


@app.function
def fastgs_resized_cache_parent(config: FastGSExperimentConfig) -> Path:
    """Return the reusable derived image cache parent for the scene."""
    if config.data.resized_image_cache_root is not None:
        return config.data.resized_image_cache_root.expanduser()
    return config.scene.path.expanduser() / "ember_cache" / "resized_images"


@app.function
def fastgs_source_image_root(config: FastGSExperimentConfig) -> Path:
    """Return the full-resolution source image root."""
    if config.scene.image_root is not None:
        return config.scene.image_root.expanduser()
    return config.scene.path.expanduser() / "images"


@app.function
def fastgs_resized_cache_root(config: FastGSExperimentConfig) -> Path:
    """Return the derived resized image cache root for this config."""
    scale_name = f"{config.data.image_scale_factor:.6f}".rstrip("0").rstrip(".")
    scale_name = scale_name.replace(".", "p")
    return fastgs_resized_cache_parent(config) / (
        f"scale_{scale_name}_{config.data.interpolation}"
    )


@app.function
def fastgs_pillow_resampling(interpolation: str) -> Any:
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
def enforce_fastgs_resized_cache_limit(
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
def materialize_fastgs_resized_image_cache(
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
    resampling = fastgs_pillow_resampling(interpolation)
    enforce_fastgs_resized_cache_limit(
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
    enforce_fastgs_resized_cache_limit(
        cache_root=cache_root,
        max_caches=max_caches,
    )
    return cache_root


@app.function
def build_scene_load_config(
    config: FastGSExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Translate paper config into an Ember scene loader config."""
    source_pipes = (
        (ember.HorizonAlignPipeConfig(),) if config.scene.align_horizon else ()
    )
    image_root = (
        materialize_fastgs_resized_image_cache(
            source_root=fastgs_source_image_root(config),
            cache_root=fastgs_resized_cache_root(config),
            scale=config.data.image_scale_factor,
            interpolation=config.data.interpolation,
            max_caches=config.data.max_resized_image_caches,
        )
        if fastgs_resized_cache_enabled(config)
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
    config: FastGSExperimentConfig,
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
                if fastgs_resized_cache_enabled(config)
                else config.data.image_scale_factor
            ),
            resize_width_target=None,
            interpolation=config.data.interpolation,
        ),
    )


@app.function
def resolve_checkpoint_output_dir(
    config: FastGSExperimentConfig,
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
