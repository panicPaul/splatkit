"""NHT-Fast-GS paper training notebook for Ember."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import importlib.util
    import sys
    import types
    from pathlib import Path
    from typing import Any, Literal

    import ember_core as ember
    import ember_native_nht
    import ember_splatting_training as ember_splatting
    import marimo as mo
    import torch
    from ember_core.training import TrainingProfilerConfig
    from marimo_config_gui import (
        ConfigPreset,
        ConfigPresetCatalog,
        create_config_gui,
    )
    from pydantic import Field
    from torch import Tensor

    NOTEBOOK_PATH = Path(__file__).resolve()
    NOTEBOOK_DIR = NOTEBOOK_PATH.parent
    REPO_ROOT = NOTEBOOK_DIR.parents[1]

    def _ensure_namespace_package(name: str, path: Path) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        module.__package__ = name
        paths = list(getattr(module, "__path__", []))
        path_str = str(path)
        if path_str not in paths:
            paths.append(path_str)
        module.__path__ = paths
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            setattr(sys.modules[parent_name], attr, module)
        return module

    _ensure_namespace_package("papers", REPO_ROOT / "papers")
    _ensure_namespace_package("papers.nht", REPO_ROOT / "papers" / "nht")
    _ensure_namespace_package(
        "papers.nht_fast_gs",
        REPO_ROOT / "papers" / "nht_fast_gs",
    )
    if "papers.nht.notebook" in sys.modules:
        nht = sys.modules["papers.nht.notebook"]
    else:
        nht_path = REPO_ROOT / "papers" / "nht" / "notebook.py"
        nht_spec = importlib.util.spec_from_file_location(
            "papers.nht.notebook",
            nht_path,
        )
        if nht_spec is None or nht_spec.loader is None:
            raise ImportError(f"Could not load {nht_path}.")
        nht = importlib.util.module_from_spec(nht_spec)
        sys.modules[nht_spec.name] = nht
        nht_spec.loader.exec_module(nht)
    initialize_nht_model_from_scene_record = (
        nht.initialize_nht_model_from_scene_record
    )
    nht_feature_scene = nht.nht_feature_scene
    nht_decode_render = nht.nht_decode_render
    nht_rgb_l1_dssim_loss = nht.nht_rgb_l1_dssim_loss
    NHTDeferredShader = nht.NHTDeferredShader
    NHTColorRefineAndEMAHook = nht.NHTColorRefineAndEMAHook
    DEFAULTS_DIR = NOTEBOOK_DIR / "defaults"
    DEFAULT_CHECKPOINT_ROOT = (
        REPO_ROOT / "checkpoints" / "papers" / "nht_fast_gs"
    )
    NHTFastGSBackendName = Literal["nht.3dgut_fast_gs"]
    NHTFastGSDefaultName = Literal[
        "garden_nht_fast_gs",
        "garden_big",
        "garden_debug_val",
    ]
    sys.modules.setdefault("papers.nht_fast_gs.notebook", sys.modules[__name__])
    ember_native_nht.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # NHT-Fast-GS training
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
    ## Training setup
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
    nht_fast_gs_presets = nht_fast_gs_preset_catalog()
    config_gui = create_config_gui(
        NHTFastGSExperimentConfig,
        presets=nht_fast_gs_presets,
        label="NHT-Fast-GS config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    preset_selector = config_gui.preset_selector(
        label="NHT-Fast-GS preset",
    )
    return (preset_selector,)


@app.cell
def _(config_gui):
    current_config = config_gui.validated_config()
    return (current_config,)


@app.class_definition
class NHTFastGSSceneConfig(nht.NHTSceneConfig):
    """Scene-record loading options."""


@app.class_definition
class NHTFastGSDataConfig(nht.NHTDataConfig):
    """Prepared-frame dataset options."""


@app.class_definition
class NHTFastGSInitializationConfig(nht.NHTInitializationConfig):
    """NHT initialization options."""


@app.class_definition
class NHTFastGSShaderConfig(nht.NHTShaderConfig):
    """Deferred shader module config."""


@app.class_definition
class NHTFastGSModelConfig(nht.NHTModelConfig):
    """Auxiliary learnable module config."""

    shader: NHTFastGSShaderConfig = Field(default_factory=NHTFastGSShaderConfig)


@app.class_definition
class NHTFastGSRenderConfig(nht.NHTConfigBase):
    """Typed native NHT render pipeline config."""

    backend: NHTFastGSBackendName = "nht.3dgut_fast_gs"
    ray_dir_scale: float | None = Field(default=None, gt=0.0)
    center_ray_mode: bool = False

    def build(
        self,
        context: ember.TrainingRunContext,
        *,
        shader: NHTFastGSShaderConfig,
        mip_splatting_screen_filter: bool,
    ) -> ember.RenderPipelineSpec:
        """Build the runtime render pipeline spec."""
        del context
        ray_dir_scale = (
            shader.ray_dir_scale()
            if self.ray_dir_scale is None
            else self.ray_dir_scale
        )
        return ember.RenderPipelineSpec(
            backend=self.backend,
            return_alpha=True,
            return_depth=True,
            feature_fn=ember.bound_callable(
                target="papers.nht.notebook.nht_feature_scene",
            ),
            postprocess_fn=ember.bound_callable(
                target="papers.nht.notebook.nht_decode_render",
            ),
            backend_options={
                "ray_dir_scale": ray_dir_scale,
                "center_ray_mode": self.center_ray_mode,
                "mip_splatting_screen_filter": mip_splatting_screen_filter,
            },
        )


@app.class_definition
class NHTFastGSMipSplatting3DFilterConfig(nht.NHTConfigBase):
    """Mip-Splatting 3D filter config."""

    recompute_schedule: nht.NHTScheduleConfig = Field(
        default_factory=lambda: nht.NHTScheduleConfig(
            start_iteration=15_000,
            end_iteration=29_899,
            frequency=100,
        )
    )
    near_plane: float | None = Field(default=0.2, gt=0.0)
    filter_variance: float = Field(default=0.2, gt=0.0)
    clipping_tolerance: float = Field(default=0.15, ge=0.0)

    def build(
        self,
        context: ember.TrainingRunContext,
    ) -> ember.CallableSpec:
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
class NHTFastGSMipSplattingConfig(nht.NHTConfigBase):
    """Full Mip-Splatting controls for NHT-Fast-GS."""

    enabled: bool = False
    screen_filter_enabled: bool = True
    three_dimensional_filter: NHTFastGSMipSplatting3DFilterConfig = Field(
        default_factory=NHTFastGSMipSplatting3DFilterConfig
    )


@app.class_definition
class NHTFastGSOptimizationConfig(nht.NHTOptimizationConfig):
    """Optimization config shared with NHT."""


@app.class_definition
class NHTFastGSLossConfig(nht.NHTLossConfig):
    """Training loss config shared with NHT."""


@app.class_definition
class NHTFastGSDensificationConfig(nht.NHTConfigBase):
    """FastGS adaptive density config for the NHT renderer."""

    refine_every: int = Field(default=500, ge=1)
    start_iter: int = Field(default=500, ge=0)
    stop_iter: int = Field(default=15_000, ge=0)
    loss_thresh: float = Field(default=0.06, ge=0.0)
    grad_threshold: float = Field(default=2e-4, gt=0.0)
    grad_abs_threshold: float = Field(default=8e-4, gt=0.0)
    dense_fraction: float = Field(default=0.001, gt=0.0)
    prune_opacity_threshold: float = Field(default=0.005, gt=0.0)
    opacity_reset_every: int = Field(default=3_000, ge=1)
    extra_opacity_reset_iter: int | None = Field(default=None, ge=0)
    max_reset_opacity: float = Field(default=0.8, gt=0.0, lt=1.0)
    scheduled_reset_opacity: float = Field(default=0.01, gt=0.0, lt=1.0)
    probe_view_count: int = Field(default=10, ge=1)
    importance_threshold: float = Field(default=5.0, gt=0.0)
    final_prune_start_iter: int = Field(default=15_000, ge=0)
    final_prune_stop_iter: int = Field(default=30_000, ge=0)
    final_prune_every: int = Field(default=3_000, ge=1)
    final_prune_opacity_threshold: float = Field(default=0.1, gt=0.0, lt=1.0)
    final_prune_mode: Literal["fastgs", "disabled"] = "disabled"

    def build(
        self,
        context: ember.TrainingRunContext,
    ) -> ember.DensificationConfig:
        """Build FastGS densification for NHT training."""
        return ember.densification_config(
            ember.bound_callable(
                target="papers.nht_fast_gs.notebook.NHTFastGSDensification",
                kwargs={
                    **self.model_dump(mode="python"),
                    "camera_extent": context.camera_extent,
                },
            )
        )


@app.class_definition
class NHTFastGSTrainingConfig(nht.NHTConfigBase):
    """Typed user-facing NHT-Fast-GS training config."""

    runtime: ember.RuntimeConfig = Field(default_factory=ember.RuntimeConfig)
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    batching: ember.BatchingConfig = Field(default_factory=ember.BatchingConfig)
    initialization: NHTFastGSInitializationConfig = Field(
        default_factory=NHTFastGSInitializationConfig
    )
    model: NHTFastGSModelConfig = Field(default_factory=NHTFastGSModelConfig)
    render: NHTFastGSRenderConfig = Field(default_factory=NHTFastGSRenderConfig)
    mip_splatting: NHTFastGSMipSplattingConfig = Field(
        default_factory=NHTFastGSMipSplattingConfig
    )
    optimization: NHTFastGSOptimizationConfig = Field(
        default_factory=NHTFastGSOptimizationConfig
    )
    densification: NHTFastGSDensificationConfig = Field(
        default_factory=NHTFastGSDensificationConfig
    )
    loss: NHTFastGSLossConfig = Field(default_factory=NHTFastGSLossConfig)
    color_refine_steps: int = Field(default=3000, ge=0)
    ema_enabled: bool = True
    ema_decay: float = Field(default=0.95, ge=0.0, lt=1.0)
    ema_start_step: int = Field(default=0, ge=0)
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
        color_refine_start = max(
            self.runtime.max_steps - self.color_refine_steps,
            0,
        )
        fastgs_densification = self.densification.build(context)
        densification_entries = [
            *fastgs_densification.methods,
            *fastgs_densification.builders,
        ]
        if self.mip_splatting.enabled:
            densification_entries.append(
                self.mip_splatting.three_dimensional_filter.build(context)
            )
        return ember.TrainingConfig(
            runtime=self.runtime,
            profiler=self.profiler,
            batching=self.batching,
            initialization=self.initialization.build(context),
            model=self.model.build(),
            render=self.render.build(
                context,
                shader=self.model.shader,
                mip_splatting_screen_filter=(
                    self.mip_splatting.enabled
                    and self.mip_splatting.screen_filter_enabled
                ),
            ),
            optimization=self.optimization.build(
                context,
                batch_size=self.batching.batch_size,
            ),
            densification=ember.densification_config(*densification_entries),
            loss=self.loss.build(
                context,
                color_refine_start=color_refine_start,
            ),
            hooks=ember.hooks_config(
                ember.bound_callable(
                    target="papers.nht.notebook.NHTColorRefineAndEMAHook",
                    kwargs={
                        "color_refine_start": color_refine_start,
                        "ema_enabled": self.ema_enabled,
                        "ema_decay": self.ema_decay,
                        "ema_start_step": self.ema_start_step,
                    },
                )
            ),
            checkpoint=self.checkpoint,
        )


@app.class_definition
class NHTFastGSExperimentConfig(nht.NHTConfigBase):
    """Resolved NHT-Fast-GS experiment config."""

    preset: NHTFastGSDefaultName = "garden_nht_fast_gs"
    scene: NHTFastGSSceneConfig = Field(default_factory=NHTFastGSSceneConfig)
    data: NHTFastGSDataConfig = Field(default_factory=NHTFastGSDataConfig)
    training: NHTFastGSTrainingConfig = Field(
        default_factory=NHTFastGSTrainingConfig
    )


@app.class_definition
class NHTFastGSDensification(ember_splatting.GaussianFastGS):
    """FastGS densification that scores decoded NHT RGB probes."""

    def probe_prediction(
        self,
        context: Any,
        sample: Any,
        probe_output: Any,
    ) -> Tensor:
        """Decode raw NHT features before FastGS metric-map construction."""
        decoded = nht_decode_render(
            context.state.model,
            sample.camera,
            probe_output,
        )
        return decoded.render[0]


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Function definitions
    """)
    return


@app.function
def default_checkpoint_dir(
    preset: NHTFastGSDefaultName,
    backend: NHTFastGSBackendName,
) -> Path:
    """Return the default checkpoint directory for a preset/backend pair."""
    return DEFAULT_CHECKPOINT_ROOT / preset / backend


@app.function
def resolved_nht_fast_gs_scene_path(
    config: NHTFastGSExperimentConfig,
) -> Path:
    """Resolve the configured scene path without substituting sample scenes."""
    return config.scene.path.expanduser()


@app.function
def nht_fast_gs_resized_cache_enabled(
    config: NHTFastGSExperimentConfig,
) -> bool:
    """Return whether NHT-Fast-GS should use a derived image cache."""
    return (
        config.data.cache_resized_images
        and config.data.image_scale_factor != 1.0
    )


@app.function
def nht_fast_gs_source_image_root(
    config: NHTFastGSExperimentConfig,
) -> Path:
    """Return the full-resolution source image root."""
    if config.scene.image_root is not None:
        return config.scene.image_root.expanduser()
    return resolved_nht_fast_gs_scene_path(config) / "images"


@app.function
def nht_fast_gs_resized_cache_parent(
    config: NHTFastGSExperimentConfig,
) -> Path:
    """Return the reusable derived image cache parent for the scene."""
    if config.data.resized_image_cache_root is not None:
        return config.data.resized_image_cache_root.expanduser()
    return (
        resolved_nht_fast_gs_scene_path(config)
        / "ember_cache"
        / "resized_images"
    )


@app.function
def nht_fast_gs_resized_cache_root(
    config: NHTFastGSExperimentConfig,
) -> Path:
    """Return the derived resized image cache root for this config."""
    scale_name = f"{config.data.image_scale_factor:.6f}".rstrip("0").rstrip(".")
    scale_name = scale_name.replace(".", "p")
    return nht_fast_gs_resized_cache_parent(config) / (
        f"scale_{scale_name}_{config.data.interpolation}"
    )


@app.function
def nht_scene_load_config(
    config: NHTFastGSExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Build the configured scene-record loader."""
    source_pipes = (
        (ember.HorizonAlignPipeConfig(),) if config.scene.align_horizon else ()
    )
    scene_path = resolved_nht_fast_gs_scene_path(config)
    image_root = (
        nht.materialize_nht_resized_image_cache(
            source_root=nht_fast_gs_source_image_root(config),
            cache_root=nht_fast_gs_resized_cache_root(config),
            scale=config.data.image_scale_factor,
            interpolation=config.data.interpolation,
            max_caches=config.data.max_resized_image_caches,
        )
        if nht_fast_gs_resized_cache_enabled(config)
        else (
            config.scene.image_root.expanduser()
            if config.scene.image_root is not None
            else None
        )
    )
    return ember.ColmapSceneConfig(
        path=scene_path,
        image_root=image_root,
        undistort_output_dir=config.scene.undistort_output_dir,
        source_pipes=source_pipes,
    )


@app.function
def nht_prepared_frame_dataset_config(
    config: NHTFastGSExperimentConfig,
) -> ember.PreparedFrameDatasetConfig:
    """Build the configured prepared-frame dataset options."""
    return ember.PreparedFrameDatasetConfig(
        camera_sensor_id=config.data.camera_sensor_id,
        split=ember.SplitConfig(
            target=config.data.split_target,
            every_n=(
                None
                if config.data.split_target == "all"
                else config.data.split_every_n
            ),
            train_ratio=None,
        ),
        materialization=ember.MaterializationConfig(
            stage=config.data.materialization_stage,
            mode=config.data.materialization_mode,
            num_workers=config.data.materialization_num_workers,
        ),
        image_preparation=ember.ImagePreparationConfig(
            resize_width_scale=(
                None
                if nht_fast_gs_resized_cache_enabled(config)
                else config.data.image_scale_factor
            ),
            normalize=config.data.normalize_images,
            interpolation=config.data.interpolation,
        ),
    )


@app.function
def nht_fast_gs_preset_catalog() -> ConfigPresetCatalog[
    NHTFastGSExperimentConfig
]:
    """Return the notebook's named JSON preset catalog."""
    return ConfigPresetCatalog(
        model_cls=NHTFastGSExperimentConfig,
        presets={
            "garden_nht_fast_gs": ConfigPreset(
                name="garden_nht_fast_gs",
                path=DEFAULTS_DIR / "garden_nht_fast_gs.json",
                label="Garden NHT-Fast-GS",
                base_dir=REPO_ROOT,
            ),
            "garden_big": ConfigPreset(
                name="garden_big",
                path=DEFAULTS_DIR / "garden_big.json",
                label="Garden NHT-Fast-GS Big",
                base_dir=REPO_ROOT,
            ),
            "garden_debug_val": ConfigPreset(
                name="garden_debug_val",
                path=DEFAULTS_DIR / "garden_debug_val.json",
                label="Garden debug validation",
                base_dir=REPO_ROOT,
            ),
        },
        default="garden_nht_fast_gs",
    )


@app.function
def resolve_training_config(
    config: NHTFastGSExperimentConfig,
    frame_dataset: ember.PreparedFrameDataset | None = None,
) -> ember.TrainingConfig:
    """Apply paper notebook runtime defaults to native Ember training config."""
    checkpoint = config.training.checkpoint.model_copy(
        update={
            "output_dir": default_checkpoint_dir(
                config.preset,
                config.training.render.backend,
            )
        }
    )
    training = config.training.model_copy(update={"checkpoint": checkpoint})
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


@app.function
def nht_fast_gs_should_show_jit_compile_notice(
    config: NHTFastGSExperimentConfig,
    snapshot: Any,
    *,
    is_script_mode: bool,
) -> bool:
    """Return whether NHT-Fast-GS is likely waiting on first JIT compile."""
    return (
        not is_script_mode
        and config.training.model.shader.jit_fusion
        and snapshot.status == "running"
        and snapshot.step == 0
        and not snapshot.latest_metrics
    )


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
def _(current_config, is_script_mode, prepare_button):
    should_prepare = is_script_mode or bool(prepare_button.value)
    scene_load_error = None
    try:
        scene_record = (
            ember.load_scene_record(nht_scene_load_config(current_config))
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
                config=nht_prepared_frame_dataset_config(current_config),
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
                title="NHT-Fast-GS training viewer",
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
        training_result = run_nht_fast_gs_training(
            current_config,
            frame_dataset,
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
def _(
    current_config,
    frame_dataset_error,
    is_script_mode,
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
        elif nht_fast_gs_should_show_jit_compile_notice(
            current_config,
            snapshot,
            is_script_mode=is_script_mode,
        ):
            training_result_view = mo.callout(
                "NHT shader JIT compilation is likely running. The first "
                "training step can take a while; progress and metrics will "
                "appear after compilation finishes.",
                kind="warn",
            )
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


@app.cell
def _():
    is_script_mode = mo.running_in_notebook() is False
    return (is_script_mode,)


@app.cell(hide_code=True)
def _(resolved_training_config):
    if resolved_training_config is not None:
        resolved_training_config_view = mo.ui.code_editor(
            value=resolved_training_config.model_dump_json(indent=2),
            language="json",
            disabled=True,
        )
    else:
        resolved_training_config_view = mo.md(
            "Prepare the training viewer to inspect the resolved config."
        )
    resolved_training_config_view
    return


@app.function
def run_nht_fast_gs_training(
    config: NHTFastGSExperimentConfig,
    frame_dataset: ember.PreparedFrameDataset,
    training_config: ember.TrainingConfig | None = None,
) -> ember.TrainingResult:
    """Run NHT-Fast-GS training from a prepared frame dataset."""
    resolved_training_config = training_config or resolve_training_config(
        config,
        frame_dataset,
    )
    return ember.run_training(frame_dataset, resolved_training_config)


if __name__ == "__main__":
    app.run()
