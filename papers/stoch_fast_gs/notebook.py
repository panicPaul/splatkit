"""Stoch-Fast-GS paper training notebook for Ember."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import importlib.util
    import sys
    import types
    from pathlib import Path
    from typing import Literal

    import ember_core as ember
    import ember_native_3dgrt as ember_3dgrt_native
    import ember_splatting_training as ember_splatting
    import marimo as mo
    import torch
    from ember_core.training import TrainingProfilerConfig, TrainingResult
    from marimo_config_gui import (
        ConfigPreset,
        ConfigPresetCatalog,
        create_config_gui,
    )
    from pydantic import Field

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
    _ensure_namespace_package(
        "papers.stoch3dgs", REPO_ROOT / "papers" / "stoch3dgs"
    )
    _ensure_namespace_package(
        "papers.stoch_fast_gs",
        REPO_ROOT / "papers" / "stoch_fast_gs",
    )
    if "papers.stoch3dgs.notebook" in sys.modules:
        stoch3dgs = sys.modules["papers.stoch3dgs.notebook"]
    else:
        stoch3dgs_path = REPO_ROOT / "papers" / "stoch3dgs" / "notebook.py"
        stoch3dgs_spec = importlib.util.spec_from_file_location(
            "papers.stoch3dgs.notebook",
            stoch3dgs_path,
        )
        if stoch3dgs_spec is None or stoch3dgs_spec.loader is None:
            raise ImportError(f"Could not load {stoch3dgs_path}.")
        stoch3dgs = importlib.util.module_from_spec(stoch3dgs_spec)
        sys.modules[stoch3dgs_spec.name] = stoch3dgs
        stoch3dgs_spec.loader.exec_module(stoch3dgs)
    build_scene_load_config = stoch3dgs.build_scene_load_config
    build_prepared_frame_dataset_config = (
        stoch3dgs.build_prepared_frame_dataset_config
    )
    initialize_stoch3dgs_model_from_scene_record = (
        stoch3dgs.initialize_stoch3dgs_model_from_scene_record
    )
    stoch3dgs_active_sh_scene = stoch3dgs.stoch3dgs_active_sh_scene
    stoch3dgs_rgb_l1_ssim_loss = stoch3dgs.stoch3dgs_rgb_l1_ssim_loss
    DEFAULTS_DIR = NOTEBOOK_DIR / "defaults"
    DEFAULT_CHECKPOINT_ROOT = (
        REPO_ROOT / "checkpoints" / "papers" / "stoch_fast_gs"
    )
    StochFastGSBackendName = Literal["3dgrt.stoch_fast_gs"]
    StochFastGSDefaultName = Literal[
        "garden_stoch_fast_gs",
        "garden_big",
        "garden_debug_val",
    ]
    sys.modules.setdefault(
        "papers.stoch_fast_gs.notebook", sys.modules[__name__]
    )
    ember_3dgrt_native.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Stoch-Fast-GS training
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
    stoch_fast_gs_presets = stoch_fast_gs_preset_catalog()
    config_gui = create_config_gui(
        StochFastGSExperimentConfig,
        presets=stoch_fast_gs_presets,
        label="Stoch-Fast-GS config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    preset_selector = config_gui.preset_selector(
        label="Stoch-Fast-GS preset",
    )
    return (preset_selector,)


@app.cell
def _(config_gui):
    current_config = config_gui.validated_config()
    return (current_config,)


@app.class_definition
class StochFastGSSceneConfig(stoch3dgs.Stoch3DGSSceneConfig):
    """Scene-record loading options."""


@app.class_definition
class StochFastGSDataConfig(stoch3dgs.Stoch3DGSDataConfig):
    """Prepared-frame dataset options."""


@app.class_definition
class StochFastGSInitializationConfig(stoch3dgs.Stoch3DGSInitializationConfig):
    """Stoch3DGS initialization options reused by Stoch-Fast-GS."""


@app.class_definition
class StochFastGSTrainingBackendOptionsConfig(
    stoch3dgs.Stoch3DGSTrainingBackendOptionsConfig
):
    """Typed per-step active-SH controls."""


@app.class_definition
class StochFastGSRenderConfig(stoch3dgs.Stoch3DGSRenderConfig):
    """Typed native Stoch3DGS render config for FastGS densification."""

    backend: StochFastGSBackendName = "3dgrt.stoch_fast_gs"
    training_backend_options: StochFastGSTrainingBackendOptionsConfig = Field(
        default_factory=StochFastGSTrainingBackendOptionsConfig
    )


@app.class_definition
class StochFastGSOptimizationConfig(stoch3dgs.Stoch3DGSOptimizationConfig):
    """Optimization config shared with Stoch3DGS."""


@app.class_definition
class StochFastGSLossConfig(stoch3dgs.Stoch3DGSLossConfig):
    """Training loss config shared with Stoch3DGS."""


@app.class_definition
class StochFastGSDensificationConfig(stoch3dgs.Stoch3DGSConfigBase):
    """FastGS adaptive density config for the Stoch3DGS renderer."""

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
        """Build FastGS densification with Stoch3DGS final cleanup."""
        return ember.densification_config(
            ember.bound_callable(
                target="ember_splatting_training.GaussianFastGS",
                kwargs={
                    **self.model_dump(mode="python"),
                    "camera_extent": context.camera_extent,
                },
            ),
            ember.bound_callable(
                target="papers.stoch3dgs.notebook.Stoch3DGSFinalCleanup",
            ),
        )


@app.class_definition
class StochFastGSTrainingConfig(stoch3dgs.Stoch3DGSConfigBase):
    """Typed user-facing Stoch-Fast-GS training config."""

    runtime: ember.RuntimeConfig = Field(default_factory=ember.RuntimeConfig)
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    batching: ember.BatchingConfig = Field(default_factory=ember.BatchingConfig)
    initialization: StochFastGSInitializationConfig = Field(
        default_factory=StochFastGSInitializationConfig
    )
    render: StochFastGSRenderConfig = Field(
        default_factory=StochFastGSRenderConfig
    )
    optimization: StochFastGSOptimizationConfig = Field(
        default_factory=StochFastGSOptimizationConfig
    )
    loss: StochFastGSLossConfig = Field(default_factory=StochFastGSLossConfig)
    densification: StochFastGSDensificationConfig = Field(
        default_factory=StochFastGSDensificationConfig
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
            densification=self.densification.build(context),
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
class StochFastGSExperimentConfig(stoch3dgs.Stoch3DGSConfigBase):
    """Resolved Stoch-Fast-GS experiment config."""

    preset: StochFastGSDefaultName = "garden_stoch_fast_gs"
    scene: StochFastGSSceneConfig = Field(
        default_factory=StochFastGSSceneConfig
    )
    data: StochFastGSDataConfig = Field(default_factory=StochFastGSDataConfig)
    training: StochFastGSTrainingConfig = Field(
        default_factory=StochFastGSTrainingConfig
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Function definitions
    """)
    return


@app.function
def default_checkpoint_dir(
    preset: StochFastGSDefaultName,
    backend: StochFastGSBackendName,
) -> Path:
    """Return the default checkpoint directory for a preset/backend pair."""
    return DEFAULT_CHECKPOINT_ROOT / preset / backend


@app.function
def stoch_fast_gs_preset_catalog() -> ConfigPresetCatalog[
    StochFastGSExperimentConfig
]:
    """Return the notebook's named JSON preset catalog."""
    return ConfigPresetCatalog(
        model_cls=StochFastGSExperimentConfig,
        presets={
            "garden_stoch_fast_gs": ConfigPreset(
                name="garden_stoch_fast_gs",
                path=DEFAULTS_DIR / "garden_stoch_fast_gs.json",
                label="Garden Stoch-Fast-GS",
                base_dir=REPO_ROOT,
            ),
            "garden_big": ConfigPreset(
                name="garden_big",
                path=DEFAULTS_DIR / "garden_big.json",
                label="Garden Stoch-Fast-GS Big",
                base_dir=REPO_ROOT,
            ),
            "garden_debug_val": ConfigPreset(
                name="garden_debug_val",
                path=DEFAULTS_DIR / "garden_debug_val.json",
                label="Garden debug validation",
                base_dir=REPO_ROOT,
            ),
        },
        default="garden_stoch_fast_gs",
    )


@app.function
def resolve_checkpoint_output_dir(
    config: StochFastGSExperimentConfig,
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
    config: StochFastGSExperimentConfig,
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
                title="Stoch-Fast-GS training viewer",
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
        training_result = run_stoch_fast_gs_training(
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
def run_stoch_fast_gs_training(
    experiment_config: StochFastGSExperimentConfig,
    frame_dataset: ember.PreparedFrameDataset,
    *,
    training_config: ember.TrainingConfig | None = None,
) -> TrainingResult:
    """Run Stoch-Fast-GS training from a prepared frame dataset."""
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
                "Stoch-Fast-GS native extension setup failed. Per notebook "
                "policy, no Torch fallback is attempted."
            ) from exc
        raise


if __name__ == "__main__":
    app.run()
