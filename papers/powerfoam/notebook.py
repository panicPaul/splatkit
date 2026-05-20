"""PowerFoam paper training notebook for Ember."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import sys
    from pathlib import Path
    from typing import Literal

    import ember_core as ember
    import ember_native_powerfoam as ember_powerfoam
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
    DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "papers" / "powerfoam"
    PowerFoamDefaultName = Literal["garden_debug", "garden_base"]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from papers._foam_notebook_utils import (
        build_foam_prepared_frame_dataset_config,
        build_foam_scene_load_config,
    )

    sys.modules.setdefault("papers.powerfoam.notebook", sys.modules[__name__])
    ember_powerfoam.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # PowerFoam training
    """)
    return


@app.cell(hide_code=True)
def _(training_controls):
    training_controls
    return


@app.cell(hide_code=True)
def _(training_preparation_status):
    training_preparation_status
    return


@app.cell
def _():
    presets = powerfoam_preset_catalog()
    config_gui = create_config_gui(
        PowerFoamExperimentConfig,
        presets=presets,
        path_defaults_source=DEFAULTS_DIR,
        label="PowerFoam config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    current_config = config_gui.validated_config()
    return (current_config,)


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
    ## Training setup
    """)
    return


@app.cell
def _():
    prepare_button = mo.ui.run_button(
        label="Prepare training inspector",
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
    training_inspector_refresh = mo.ui.refresh(
        options=["5s", "10s", "30s", "1m"],
        default_interval="10s",
        label="Image refresh",
    )
    training_controls = mo.vstack(
        [
            prepare_button,
            train_button,
            stop_button,
            training_status_refresh,
            training_inspector_refresh,
        ],
        gap=0.5,
    )
    return (
        prepare_button,
        stop_button,
        train_button,
        training_controls,
        training_inspector_refresh,
        training_status_refresh,
    )


@app.cell
def _():
    is_script_mode = not mo.running_in_notebook()
    return (is_script_mode,)


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""
    # Training
    """)
    return


@app.function(column=1)
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


@app.cell(column=1)
def _(current_config):
    training_preparation_handle = None
    training_preparation_snapshot = None
    if current_config is not None:
        training_preparation_handle, training_preparation_snapshot = (
            ember_splatting.create_training_preparation(
                load_scene=lambda: ember.load_scene_record(
                    build_scene_load_config(current_config)
                ),
                prepare_frame_view_catalog=lambda scene_record: (
                    ember.build_prepared_frame_view_catalog(
                        scene_record,
                        build_prepared_frame_dataset_config(current_config),
                    )
                ),
            )
        )
    return training_preparation_handle, training_preparation_snapshot


@app.cell(column=1)
def _(
    current_config,
    is_script_mode,
    prepare_button,
    train_button,
    training_preparation_handle,
):
    should_prepare = (
        is_script_mode or bool(prepare_button.value) or bool(train_button.value)
    )
    if (
        should_prepare
        and current_config is not None
        and training_preparation_handle is not None
    ):
        training_preparation_handle.start(wait=is_script_mode)
    return


@app.cell(column=1)
def _(ember_splatting, training_preparation_snapshot):
    _snapshot = (
        training_preparation_snapshot()
        if training_preparation_snapshot is not None
        else None
    )
    training_preparation_status = (
        ember_splatting.render_training_preparation_status(_snapshot)
    )
    return (training_preparation_status,)


@app.cell(column=1)
def _(ember_splatting, training_preparation_snapshot):
    _snapshot = (
        training_preparation_snapshot()
        if training_preparation_snapshot is not None
        else None
    )
    (
        scene_load_error,
        scene_record,
        frame_dataset,
        frame_dataset_error,
        frame_view_catalog,
    ) = ember_splatting.training_preparation_outputs(_snapshot)
    return (
        scene_load_error,
        scene_record,
        frame_dataset,
        frame_dataset_error,
        frame_view_catalog,
    )


@app.cell(column=1)
def _(current_config, frame_dataset):
    training_config = (
        resolve_training_config(current_config)
        if current_config is not None and frame_dataset is not None
        else None
    )
    viewer_config = (
        current_config.training.viewer
        if current_config is not None
        else ember_splatting.TrainingViewerConfig()
    )
    return training_config, viewer_config


@app.cell(column=1)
def _(frame_dataset, is_script_mode, training_config, viewer_config):
    training_viewer_error = None
    try:
        training_viewer_handle = (
            ember_splatting.create_training_run(
                frame_dataset,
                training_config,
                config=viewer_config,
                title="PowerFoam training inspector",
            )
            if not is_script_mode
            and frame_dataset is not None
            and training_config is not None
            else None
        )
    except Exception as error:
        training_viewer_handle = None
        training_viewer_error = error
    return training_viewer_error, training_viewer_handle


@app.cell(column=1)
def _(frame_view_catalog, is_script_mode):
    training_inspector = (
        None
        if is_script_mode or frame_view_catalog is None
        else ember_splatting.create_training_view_inspector(
            frame_view_catalog,
        )
    )
    return (training_inspector,)


@app.cell(column=1)
def _(
    frame_view_catalog,
    training_inspector,
    training_inspector_refresh,
    training_viewer_handle,
):
    training_viewer = (
        None
        if training_inspector is None
        else training_inspector.panel(
            training_viewer_handle,
            frame_view_catalog,
            refresh=training_inspector_refresh,
        )
    )
    return (training_viewer,)


@app.cell(column=1)
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
        training_result = run_powerfoam_training(
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


@app.cell(column=1)
def _(stop_button, training_viewer_handle):
    should_stop = bool(stop_button.value)
    if should_stop and training_viewer_handle is not None:
        training_viewer_handle.request_stop()
    return


@app.cell(column=1)
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
            f"Training inspector preparation failed.\n\n```text\n{training_viewer_error}\n```",
            kind="danger",
        )
    elif training_result is not None:
        training_result_view = mo.md(
            f"Checkpoint: `{training_result.checkpoint_dir}`\n\n"
            f"Steps: `{len(training_result.history)}`"
        )
    elif training_viewer_handle is None:
        training_result_view = mo.md("Prepare the training inspector first.")
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
    # Support Code
    """)
    return


@app.class_definition(column=2)
class PowerFoamConfigBase(BaseModel):
    """Strict base model for PowerFoam paper configs."""

    model_config = {"extra": "forbid", "populate_by_name": True}


@app.class_definition(column=2)
class PowerFoamSceneConfig(PowerFoamConfigBase):
    """Scene-record loading options."""

    path: Path = Path("dataset/mipnerf360/garden")
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool = True


@app.class_definition(column=2)
class PowerFoamDataConfig(PowerFoamConfigBase):
    """Prepared-frame dataset options."""

    camera_sensor_id: str | None = None
    image_scale_factor: float = Field(default=0.125, gt=0.0)
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


@app.class_definition(column=2)
class PowerFoamInitializationConfig(PowerFoamConfigBase):
    """Typed PowerFoam initialization config."""

    init_type: Literal["sfm", "random_bounded", "random_unbounded"] = "sfm"
    init_points: int = Field(default=100_000, ge=1)
    render_objective: Literal["volume", "surface"] = "volume"
    sv_dof: int = Field(default=8, ge=1)
    num_texel_sites: int = Field(default=8, ge=1)
    attr_dtype: Literal["float", "half"] = "float"
    seed: int | None = 0

    def build(
        self,
        context: ember.TrainingRunContext,
    ) -> ember.InitializationSpec:
        """Build the runtime initializer spec."""
        del context
        return ember.InitializationSpec(
            initializer=ember.bound_callable(
                target=(
                    "ember_native_powerfoam."
                    "initialize_powerfoam_model_from_scene_record"
                ),
                kwargs=self.model_dump(mode="python"),
                bind={"device": ember.ctx.run.device},
            )
        )


@app.class_definition(column=2)
class PowerFoamRenderConfig(PowerFoamConfigBase):
    """Typed PowerFoam render pipeline config."""

    backend: Literal["powerfoam.rasterize"] = "powerfoam.rasterize"
    render_objective: Literal["volume", "surface"] | None = None
    density_beta: float = Field(default=100.0, gt=0.0)
    radii_beta: float = Field(default=100.0, gt=0.0)
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    return_alpha: bool = True
    return_depth: bool = False
    return_normals: bool = False
    clamp_output: bool = False
    disable_coop_prim_load: bool = False
    disable_coop_adj_load: bool = False
    is_pinhole: bool = True

    def build(self) -> ember.RenderPipelineSpec:
        """Build the runtime render pipeline spec."""
        return ember.RenderPipelineSpec(
            backend=self.backend,
            return_alpha=self.return_alpha,
            return_depth=self.return_depth,
            return_normals=self.return_normals,
            training_backend_options_builder=ember.CallableSpec(
                target=(
                    "ember_native_powerfoam.powerfoam_training_backend_options"
                )
            ),
            backend_options={
                "render_objective": self.render_objective,
                "density_beta": self.density_beta,
                "radii_beta": self.radii_beta,
                "background_color": list(self.background_color),
                "clamp_output": self.clamp_output,
                "disable_coop_prim_load": self.disable_coop_prim_load,
                "disable_coop_adj_load": self.disable_coop_adj_load,
                "is_pinhole": self.is_pinhole,
            },
        )


@app.class_definition(column=2)
class PowerFoamLossConfig(PowerFoamConfigBase):
    """Typed PowerFoam loss config."""

    rgb: float = Field(default=1.0, ge=0.0)
    ssim: float = Field(default=0.2, ge=0.0)
    normal: float = Field(default=0.1, ge=0.0)
    contribution: float = Field(default=0.1, ge=0.0)
    interpenetration: float = Field(default=1e-4, ge=0.0)

    def build(self, *, max_steps: int) -> ember.LossConfig:
        """Build the runtime loss spec."""
        return ember.LossConfig(
            target=ember.CallableSpec(
                target="ember_native_powerfoam.powerfoam_training_loss",
            ),
            weights={
                "rgb": self.rgb,
                "ssim": self.ssim,
                "normal": self.normal,
                "contribution": self.contribution,
                "interpenetration": self.interpenetration,
                "max_steps": float(max_steps),
            },
        )


@app.class_definition(column=2)
class PowerFoamDensificationConfig(PowerFoamConfigBase):
    """Typed PowerFoam resampling config."""

    enabled: bool = True
    resample_every: int = Field(default=100, ge=1)
    resample_offset: int = Field(default=99, ge=0)
    densify_from: int = Field(default=1_000, ge=0)
    densify_until: int = Field(default=24_000, ge=2)
    final_points: int = Field(default=1_200_000, ge=1)
    stop_fraction: float = Field(default=0.95, ge=0.0, le=1.0)
    adjacency_max_interval: int = Field(default=20, ge=1)
    stats_epsilon: float = Field(default=1e-5, gt=0.0)
    sort_after_resample: bool = True

    def build(self, *, max_steps: int) -> ember.DensificationConfig | None:
        """Build the runtime densification spec."""
        if not self.enabled:
            return None
        return ember.DensificationConfig(
            builders=[
                ember.CallableSpec(
                    target="ember_native_powerfoam.PowerFoamResampling",
                    kwargs={
                        "max_steps": max_steps,
                        "resample_every": self.resample_every,
                        "resample_offset": self.resample_offset,
                        "densify_from": self.densify_from,
                        "densify_until": self.densify_until,
                        "final_points": self.final_points,
                        "stop_fraction": self.stop_fraction,
                        "adjacency_max_interval": self.adjacency_max_interval,
                        "stats_epsilon": self.stats_epsilon,
                        "sort_after_resample": self.sort_after_resample,
                    },
                )
            ]
        )


@app.class_definition(column=2)
class PowerFoamTrainingConfig(PowerFoamConfigBase):
    """Top-level runtime PowerFoam training knobs."""

    runtime: ember.RuntimeConfig = Field(
        default_factory=lambda: ember.RuntimeConfig(
            device="cuda",
            seed=0,
            max_steps=30_000,
        )
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
    initialization: PowerFoamInitializationConfig = Field(
        default_factory=PowerFoamInitializationConfig
    )
    render: PowerFoamRenderConfig = Field(default_factory=PowerFoamRenderConfig)
    optimization: ember_powerfoam.PowerFoamOptimizationRecipe = Field(
        default_factory=ember_powerfoam.PowerFoamOptimizationRecipe
    )
    loss: PowerFoamLossConfig = Field(default_factory=PowerFoamLossConfig)
    densification: PowerFoamDensificationConfig = Field(
        default_factory=PowerFoamDensificationConfig
    )
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    checkpoint: ember.CheckpointExportConfig = Field(
        default_factory=lambda: ember.CheckpointExportConfig(
            output_dir=DEFAULT_CHECKPOINT_ROOT / "latest",
            export_ply=False,
            overwrite=False,
        )
    )
    viewer: ember_splatting.TrainingViewerConfig = Field(
        default_factory=ember_splatting.TrainingViewerConfig
    )


@app.class_definition(column=2)
class PowerFoamExperimentConfig(PowerFoamConfigBase):
    """Serializable PowerFoam experiment config."""

    preset: PowerFoamDefaultName = "garden_base"
    scene: PowerFoamSceneConfig = Field(default_factory=PowerFoamSceneConfig)
    data: PowerFoamDataConfig = Field(default_factory=PowerFoamDataConfig)
    training: PowerFoamTrainingConfig = Field(
        default_factory=PowerFoamTrainingConfig
    )


@app.function(column=2)
def powerfoam_preset_catalog() -> ConfigPresetCatalog:
    """Load PowerFoam defaults from JSON files."""
    return ConfigPresetCatalog(
        model_cls=PowerFoamExperimentConfig,
        presets={
            "garden_debug": ConfigPreset(
                name="garden_debug",
                path=DEFAULTS_DIR / "garden_debug.json",
                label="Garden debug",
                base_dir=REPO_ROOT,
            ),
            "garden_base": ConfigPreset(
                name="garden_base",
                path=DEFAULTS_DIR / "garden_base.json",
                label="Garden base",
                base_dir=REPO_ROOT,
            ),
        },
        default="garden_base",
    )


@app.function(column=2)
def resolve_checkpoint_output_dir(config: PowerFoamExperimentConfig) -> Path:
    """Resolve the checkpoint directory for the current PowerFoam config."""
    output_dir = config.training.checkpoint.output_dir.expanduser()
    latest_dir = DEFAULT_CHECKPOINT_ROOT / "latest"
    if output_dir == latest_dir or output_dir == Path("checkpoints/latest"):
        return DEFAULT_CHECKPOINT_ROOT / config.preset
    return output_dir


@app.function(column=2)
def resolve_training_config(
    experiment_config: PowerFoamExperimentConfig,
) -> ember.TrainingConfig:
    """Resolve the user-facing PowerFoam config into an Ember TrainingConfig."""
    checkpoint = experiment_config.training.checkpoint.model_copy(
        update={
            "output_dir": resolve_checkpoint_output_dir(experiment_config),
        },
    )
    training = experiment_config.training.model_copy(
        update={"checkpoint": checkpoint},
        deep=True,
    )
    return ember.TrainingConfig(
        runtime=training.runtime,
        profiler=training.profiler,
        batching=training.batching,
        initialization=training.initialization.build(None),
        render=training.render.build(),
        optimization=ember_powerfoam.powerfoam_optimization_config(
            training.optimization,
            max_steps=training.runtime.max_steps,
        ),
        loss=training.loss.build(max_steps=training.runtime.max_steps),
        densification=training.densification.build(
            max_steps=training.runtime.max_steps
        ),
        checkpoint=training.checkpoint,
    )


@app.function(column=2)
def build_scene_load_config(
    config: PowerFoamExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Translate PowerFoam config into an Ember scene loader config."""
    return build_foam_scene_load_config(config)


@app.function(column=2)
def build_prepared_frame_dataset_config(
    config: PowerFoamExperimentConfig,
) -> ember.PreparedFrameDatasetConfig:
    """Translate PowerFoam config into an Ember frame dataset config."""
    return build_foam_prepared_frame_dataset_config(config)


@app.function(column=2)
def run_powerfoam_training(
    frame_dataset: ember.PreparedFrameDataset,
    *,
    training_config: ember.TrainingConfig,
    training_viewer_handle: ember_splatting.TrainingViewerHandle | None = None,
) -> TrainingResult:
    """Run PowerFoam training from a native Ember training config."""
    return ember.run_training(
        frame_dataset,
        training_config,
        runtime_hooks=(
            ()
            if training_viewer_handle is None
            else training_viewer_handle.runtime_hooks()
        ),
    )


if __name__ == "__main__":
    app.run()
