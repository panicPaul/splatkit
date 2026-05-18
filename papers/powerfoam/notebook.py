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
    import marimo as mo
    import torch
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
    DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "papers" / "powerfoam"
    PowerFoamDefaultName = Literal["garden_debug", "garden_base"]
    sys.modules.setdefault("papers.powerfoam.notebook", sys.modules[__name__])
    ember_powerfoam.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # PowerFoam training
    """)
    return


@app.cell
def _():
    presets = powerfoam_preset_catalog()
    config_gui = create_config_gui(
        PowerFoamExperimentConfig,
        presets=presets,
        label="PowerFoam config",
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
        kwargs = self.model_dump(mode="python")
        attr_dtype = kwargs.pop("attr_dtype")
        kwargs["attr_dtype"] = (
            torch.float16 if attr_dtype == "half" else torch.float32
        )
        return ember.InitializationSpec(
            initializer=ember.bound_callable(
                target=(
                    "ember_native_powerfoam."
                    "initialize_powerfoam_model_from_scene_record"
                ),
                kwargs=kwargs,
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
                    "ember_native_powerfoam."
                    "powerfoam_training_backend_options"
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

    def build(self) -> ember.LossConfig:
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


@app.class_definition(column=2)
class PowerFoamExperimentConfig(PowerFoamConfigBase):
    """Serializable PowerFoam experiment config."""

    preset: PowerFoamDefaultName = "garden_debug"
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
        default="garden_debug",
    )


@app.function(column=2)
def resolve_training_config(
    experiment_config: PowerFoamExperimentConfig,
) -> ember.TrainingConfig:
    """Resolve the user-facing PowerFoam config into an Ember TrainingConfig."""
    training = experiment_config.training
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
        loss=training.loss.build(),
        densification=training.densification.build(
            max_steps=training.runtime.max_steps
        ),
        checkpoint=training.checkpoint,
    )


if __name__ == "__main__":
    app.run()
