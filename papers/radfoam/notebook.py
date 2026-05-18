"""RADFOAM paper training notebook for Ember."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import sys
    from pathlib import Path
    from typing import Literal

    import ember_core as ember
    import ember_native_radfoam as ember_radfoam
    import marimo as mo
    from ember_core.training import TrainingProfilerConfig
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
    DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "papers" / "radfoam"
    RadFoamDefaultName = Literal["garden_debug", "garden_base"]
    sys.modules.setdefault("papers.radfoam.notebook", sys.modules[__name__])
    ember_radfoam.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # RADFOAM training
    """)
    return


@app.cell
def _():
    presets = radfoam_preset_catalog()
    config_gui = create_config_gui(
        RadFoamExperimentConfig,
        presets=presets,
        label="RADFOAM config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell(hide_code=True)
def _(config_gui):
    config_gui.stacked()
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md("""
    # Support Code
    """)
    return


@app.class_definition(column=2)
class RadFoamConfigBase(BaseModel):
    """Strict base model for RADFOAM paper configs."""

    model_config = {"extra": "forbid", "populate_by_name": True}


@app.class_definition(column=2)
class RadFoamSceneConfig(RadFoamConfigBase):
    """Scene-record loading options."""

    path: Path = Path("dataset/mipnerf360/garden")
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool = True


@app.class_definition(column=2)
class RadFoamDataConfig(RadFoamConfigBase):
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


@app.class_definition(column=2)
class RadFoamInitializationConfig(RadFoamConfigBase):
    """Typed RADFOAM point-cloud initialization config."""

    sh_degree: int = Field(default=3, ge=0)
    init_points: int = Field(default=131_072, ge=ember_radfoam.MIN_RADFOAM_POINTS)
    random_points: int = Field(default=5_000, ge=0)
    point_cloud_sample_ratio: float = Field(default=0.9, gt=0.0, le=1.0)
    activation_scale: float = Field(default=1.0, gt=0.0)
    jitter_std: float = Field(default=1e-2, ge=0.0)
    random_point_scale: float = Field(default=10.0, gt=0.0)
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
                    "ember_native_radfoam."
                    "initialize_radfoam_model_from_scene_record"
                ),
                kwargs=self.model_dump(mode="python"),
                bind={"device": ember.ctx.run.device},
            )
        )


@app.class_definition(column=2)
class RadFoamRenderConfig(RadFoamConfigBase):
    """Typed RADFOAM render pipeline config."""

    backend: Literal["radfoam.core"] = "radfoam.core"
    weight_threshold: float = Field(default=0.001, gt=0.0)
    max_intersections: int = Field(default=1024, ge=1)
    density_beta: float = Field(default=10.0, gt=0.0)
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    return_alpha: bool = False
    return_depth: bool = False
    clamp_output: bool = False

    def build(self) -> ember.RenderPipelineSpec:
        """Build the runtime render pipeline spec."""
        return ember.RenderPipelineSpec(
            backend=self.backend,
            return_alpha=self.return_alpha,
            return_depth=self.return_depth,
            backend_options={
                "weight_threshold": self.weight_threshold,
                "max_intersections": self.max_intersections,
                "density_beta": self.density_beta,
                "background_color": list(self.background_color),
                "clamp_output": self.clamp_output,
            },
        )


@app.class_definition(column=2)
class RadFoamLossConfig(RadFoamConfigBase):
    """Typed RADFOAM loss config."""

    loss_type: Literal["l1", "mse"] = "l1"

    def build(self) -> ember.LossConfig:
        """Build the runtime loss spec."""
        return ember.LossConfig(
            target=ember.CallableSpec(
                target="ember_native_radfoam.radfoam_rgb_loss",
                kwargs={"loss_type": self.loss_type},
            )
        )


@app.class_definition(column=2)
class RadFoamDensificationConfig(RadFoamConfigBase):
    """Typed RADFOAM topology-refresh config."""

    enabled: bool = True
    refine_every: int = Field(default=100, ge=1)
    start_iter: int = Field(default=0, ge=0)
    stop_iter: int = -1

    def build(self) -> ember.DensificationConfig | None:
        """Build the runtime densification spec."""
        if not self.enabled:
            return None
        return ember.DensificationConfig(
            builders=[
                ember.CallableSpec(
                    target="ember_native_radfoam.RadFoamTopologyRefresh",
                    kwargs={
                        "refine_every": self.refine_every,
                        "start_iter": self.start_iter,
                        "stop_iter": self.stop_iter,
                    },
                )
            ]
        )


@app.class_definition(column=2)
class RadFoamTrainingConfig(RadFoamConfigBase):
    """Top-level runtime RADFOAM training knobs."""

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
    initialization: RadFoamInitializationConfig = Field(
        default_factory=RadFoamInitializationConfig
    )
    render: RadFoamRenderConfig = Field(default_factory=RadFoamRenderConfig)
    optimization: ember_radfoam.RadFoamOptimizationRecipe = Field(
        default_factory=ember_radfoam.RadFoamOptimizationRecipe
    )
    loss: RadFoamLossConfig = Field(default_factory=RadFoamLossConfig)
    densification: RadFoamDensificationConfig = Field(
        default_factory=RadFoamDensificationConfig
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


@app.class_definition(column=2)
class RadFoamExperimentConfig(RadFoamConfigBase):
    """Serializable RADFOAM experiment config."""

    preset: RadFoamDefaultName = "garden_debug"
    scene: RadFoamSceneConfig = Field(default_factory=RadFoamSceneConfig)
    data: RadFoamDataConfig = Field(default_factory=RadFoamDataConfig)
    training: RadFoamTrainingConfig = Field(
        default_factory=RadFoamTrainingConfig
    )


@app.function(column=2)
def radfoam_preset_catalog() -> ConfigPresetCatalog:
    """Load RADFOAM defaults from JSON files."""
    return ConfigPresetCatalog(
        model_cls=RadFoamExperimentConfig,
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
        default="garden_debug",
    )


@app.function(column=2)
def resolve_training_config(
    experiment_config: RadFoamExperimentConfig,
) -> ember.TrainingConfig:
    """Resolve the user-facing RADFOAM config into an Ember TrainingConfig."""
    training = experiment_config.training
    return ember.TrainingConfig(
        runtime=training.runtime,
        profiler=training.profiler,
        batching=training.batching,
        initialization=training.initialization.build(None),
        render=training.render.build(),
        optimization=ember_radfoam.radfoam_optimization_config(
            training.optimization,
            max_steps=training.runtime.max_steps,
        ),
        loss=training.loss.build(),
        densification=training.densification.build(),
        checkpoint=training.checkpoint,
    )


if __name__ == "__main__":
    app.run()
