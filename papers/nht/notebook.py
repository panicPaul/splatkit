"""Neural Harmonic Textures paper training notebook for Ember."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import json
    import math
    import shutil
    import sys
    from pathlib import Path
    from typing import Any, Literal

    import ember_core as ember
    import ember_native_nht
    import ember_splatting_training as ember_splatting
    import marimo as mo
    import torch
    from ember_core.training import LossResult, TrainingProfilerConfig
    from jaxtyping import Float
    from marimo_config_gui import (
        ConfigPreset,
        ConfigPresetCatalog,
        create_config_gui,
    )
    from pydantic import BaseModel, Field
    from torch import Tensor, nn

    NOTEBOOK_PATH = Path(__file__).resolve()
    NOTEBOOK_DIR = NOTEBOOK_PATH.parent
    REPO_ROOT = NOTEBOOK_DIR.parents[1]
    DEFAULTS_DIR = NOTEBOOK_DIR / "defaults"
    DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "papers" / "nht"
    NHTBackendName = Literal["nht.3dgut"]
    NHTDefaultName = Literal["garden_nht", "garden_debug_val"]
    sys.modules.setdefault("papers.nht.notebook", sys.modules[__name__])
    ember_native_nht.register()


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Neural Harmonic Textures
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
def _(NHTExperimentConfig, nht_preset_catalog):
    nht_presets = nht_preset_catalog()
    config_gui = create_config_gui(
        NHTExperimentConfig,
        presets=nht_presets,
        label="NHT config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    preset_selector = config_gui.preset_selector(label="NHT preset")
    return (preset_selector,)


@app.cell
def _(config_gui):
    current_config = config_gui.validated_config()
    return (current_config,)


@app.class_definition
class NHTConfigBase(BaseModel):
    """Strict base model for NHT paper configs."""

    model_config = {"extra": "forbid", "populate_by_name": True}


@app.class_definition
class NHTSceneConfig(NHTConfigBase):
    """Scene-record loading options."""

    path: Path = Path("dataset/mipnerf360/garden")
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool = True


@app.class_definition
class NHTDataConfig(NHTConfigBase):
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
class NHTInitializationConfig(NHTConfigBase):
    """Typed NHT feature initialization config."""

    feature_dim: int = Field(default=48, ge=4, multiple_of=4)
    features_init_min: float = -math.pi / 2.0
    features_init_max: float = math.pi / 2.0
    initial_scale: float = Field(default=0.1, gt=0.0)
    initial_opacity: float = Field(default=0.5, gt=0.0, lt=1.0)

    def build(
        self, context: ember.TrainingRunContext
    ) -> ember.InitializationSpec:
        """Build the runtime initializer spec."""
        del context
        return ember.InitializationSpec(
            initializer=ember.bound_callable(
                target="papers.nht.notebook.initialize_nht_model_from_scene_record",
                kwargs=self.model_dump(mode="python"),
            )
        )


@app.class_definition
class NHTShaderConfig(NHTConfigBase):
    """Deferred shader module config."""

    feature_dim: int = Field(default=48, ge=4, multiple_of=4)
    hidden_dim: int = Field(default=128, ge=1)
    num_hidden_layers: int = Field(default=3, ge=1)
    enable_view_encoding: bool = True
    view_encoding: Literal["sh", "fourier", "frequency"] = "sh"
    view_encoding_degree: int = Field(default=3, ge=1)
    view_encoding_scale: float = Field(default=3.0, gt=0.0)
    view_encoding_frequencies: int = Field(default=4, ge=0)
    jit_fusion: bool = True

    def ray_dir_scale(self) -> float:
        """Return the upstream ray scale implied by this shader config."""
        if not self.enable_view_encoding:
            return 1.0
        return self.view_encoding_scale if self.view_encoding == "sh" else 1.0

    def build(self) -> ember.CallableSpec:
        """Build the shader module spec."""
        return ember.bound_callable(
            target="papers.nht.notebook.NHTDeferredShader",
            kwargs=self.model_dump(mode="python"),
        )


@app.class_definition
class NHTModelConfig(NHTConfigBase):
    """Auxiliary learnable module config."""

    shader: NHTShaderConfig = Field(default_factory=NHTShaderConfig)

    def build(self) -> ember.ModelSpec:
        """Build the runtime model spec."""
        return ember.ModelSpec(modules={"deferred_shader": self.shader.build()})


@app.class_definition
class NHTOptimizationConfig(NHTConfigBase):
    """Typed NHT optimization config."""

    center_position_lr_init: float = Field(default=1.6e-4, gt=0.0)
    center_position_lr_final: float = Field(default=1.6e-6, gt=0.0)
    center_position_lr_max_steps: int | None = Field(default=None, ge=1)
    feature_lr: float = Field(default=15e-3, gt=0.0)
    shader_lr: float = Field(default=68e-5, gt=0.0)
    feature_lr_decay: bool = True
    feature_lr_decay_scheduler: Literal["cosine", "exponential"] = "cosine"
    feature_lr_decay_final: float = Field(default=0.1, gt=0.0)
    shader_lr_decay: bool = True
    shader_lr_decay_scheduler: Literal["cosine", "exponential"] = "cosine"
    shader_lr_decay_final: float = Field(default=0.1, gt=0.0)
    logit_opacity_lr: float = Field(default=5e-2, gt=0.0)
    log_scales_lr: float = Field(default=5e-3, gt=0.0)
    quaternion_orientation_lr: float = Field(default=1e-3, gt=0.0)
    shader_weight_decay: float = Field(default=0.0, ge=0.0)
    adam_eps: float = Field(default=1e-15, gt=0.0)
    adam_fused: bool = True

    def build(
        self,
        context: ember.TrainingRunContext,
        *,
        batch_size: int = 1,
    ) -> ember.OptimizationConfig:
        """Build optimizer groups for NHT scene state and deferred shader."""
        max_steps = self.center_position_lr_max_steps or context.max_steps
        batch_scale = math.sqrt(float(batch_size))
        splat_optimizer_kwargs = {
            "eps": self.adam_eps / batch_scale,
            "fused": self.adam_fused,
        }
        betas = (
            1.0 - batch_size * (1.0 - 0.9),
            1.0 - batch_size * (1.0 - 0.999),
        )
        feature_scheduler = self._lr_decay_scheduler(
            initial_lr=self.feature_lr * batch_scale,
            final_ratio=self.feature_lr_decay_final,
            max_steps=context.max_steps,
            kind=self.feature_lr_decay_scheduler,
        )
        shader_scheduler = self._lr_decay_scheduler(
            initial_lr=self.shader_lr * batch_scale,
            final_ratio=self.shader_lr_decay_final,
            max_steps=context.max_steps,
            kind=self.shader_lr_decay_scheduler,
        )
        return ember.OptimizationConfig(
            parameter_groups=[
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene", name="center_position"
                    ),
                    optimizer="adam",
                    lr=(
                        self.center_position_lr_init
                        * context.camera_extent
                        * batch_scale
                    ),
                    betas=betas,
                    optimizer_kwargs=splat_optimizer_kwargs,
                    scheduler=ember.bound_callable(
                        target="ember_core.training.exponential_decay_to",
                        kwargs={
                            "final_lr": (
                                self.center_position_lr_final
                                * context.camera_extent
                                * batch_scale
                            ),
                            "max_steps": max_steps,
                        },
                    ),
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene", name="feature"
                    ),
                    optimizer="adam",
                    lr=self.feature_lr * batch_scale,
                    betas=betas,
                    optimizer_kwargs=splat_optimizer_kwargs,
                    scheduler=feature_scheduler
                    if self.feature_lr_decay
                    else None,
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="modules", name="deferred_shader"
                    ),
                    optimizer="adam",
                    lr=self.shader_lr * batch_scale,
                    weight_decay=self.shader_weight_decay,
                    scheduler=shader_scheduler
                    if self.shader_lr_decay
                    else None,
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene", name="logit_opacity"
                    ),
                    optimizer="adam",
                    lr=self.logit_opacity_lr * batch_scale,
                    betas=betas,
                    optimizer_kwargs=splat_optimizer_kwargs,
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene", name="log_scales"
                    ),
                    optimizer="adam",
                    lr=self.log_scales_lr * batch_scale,
                    betas=betas,
                    optimizer_kwargs=splat_optimizer_kwargs,
                ),
                ember.ParameterGroupConfig(
                    target=ember.ParameterTargetSpec(
                        scope="scene", name="quaternion_orientation"
                    ),
                    optimizer="adam",
                    lr=self.quaternion_orientation_lr * batch_scale,
                    betas=betas,
                    optimizer_kwargs=splat_optimizer_kwargs,
                ),
            ]
        )

    def _lr_decay_scheduler(
        self,
        *,
        initial_lr: float,
        final_ratio: float,
        max_steps: int,
        kind: Literal["cosine", "exponential"],
    ) -> ember.CallableSpec:
        """Build the feature/MLP scheduler used by upstream NHT."""
        final_lr = initial_lr * final_ratio
        if kind == "cosine":
            return ember.bound_callable(
                target="torch.optim.lr_scheduler.CosineAnnealingLR",
                kwargs={"T_max": max_steps, "eta_min": final_lr},
            )
        return ember.bound_callable(
            target="torch.optim.lr_scheduler.ExponentialLR",
            kwargs={"gamma": final_ratio ** (1.0 / max_steps)},
        )


@app.class_definition
class NHTLossConfig(NHTConfigBase):
    """Typed NHT training loss config."""

    lambda_l1: float = Field(default=0.8, ge=0.0)
    lambda_ssim: float = Field(default=0.2, ge=0.0)
    lambda_opacity_regularization: float = Field(default=0.02, ge=0.0)
    lambda_scale_regularization: float = Field(default=0.005, ge=0.0)
    ssim_backend: str = "cuda"

    def build(
        self,
        context: ember.TrainingRunContext,
        *,
        color_refine_start: int,
    ) -> ember.LossConfig:
        """Build the runtime loss config."""
        del context
        kwargs = self.model_dump(mode="python")
        kwargs["color_refine_start"] = color_refine_start
        return ember.loss_config(
            "papers.nht.notebook.nht_rgb_l1_dssim_loss",
            kwargs=kwargs,
        )


@app.class_definition
class NHTScheduleConfig(NHTConfigBase):
    """Serializable step schedule config."""

    start_iteration: int = Field(default=0, ge=0)
    end_iteration: int = -1
    frequency: int = Field(default=1, ge=1)


@app.class_definition
class NHTMCMCConfig(NHTConfigBase):
    """Typed NHT MCMC densification config."""

    enabled: bool = True
    start_iteration: int = Field(default=500, ge=0)
    end_iteration: int = Field(default=25_000, ge=-1)
    frequency: int = Field(default=100, ge=1)
    min_opacity: float = Field(default=0.005, ge=0.0)
    cap_growth_factor: float = Field(default=1.05, gt=1.0)
    cap_max: int = Field(default=1_000_000, ge=1)
    inject_position_noise: bool = True
    noise_lr_scale: float = Field(default=5e5, gt=0.0)

    def build(
        self,
        context: ember.TrainingRunContext,
    ) -> ember.DensificationConfig | None:
        """Build the runtime MCMC densification config."""
        del context
        if not self.enabled:
            return None
        return ember.densification_config(
            ember.bound_callable(
                target="papers.nht.notebook.build_nht_mcmc",
                kwargs={
                    "start_iteration": self.start_iteration,
                    "end_iteration": self.end_iteration,
                    "frequency": self.frequency,
                    "min_opacity": self.min_opacity,
                    "cap_growth_factor": self.cap_growth_factor,
                    "cap_max": self.cap_max,
                    "inject_position_noise": self.inject_position_noise,
                    "noise_lr_scale": self.noise_lr_scale,
                },
            )
        )


@app.cell
def _():
    class NHTRenderConfig(NHTConfigBase):
        """Typed native NHT render pipeline config."""

        backend: NHTBackendName = "nht.3dgut"
        ray_dir_scale: float | None = Field(default=None, gt=0.0)
        center_ray_mode: bool = False

        def build(
            self,
            context: ember.TrainingRunContext,
            *,
            shader: NHTShaderConfig,
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

    return (NHTRenderConfig,)


@app.cell
def _():
    class NHTMipSplatting3DFilterConfig(NHTConfigBase):
        """Mip-Splatting 3D filter config."""

        recompute_schedule: NHTScheduleConfig = Field(
            default_factory=lambda: NHTScheduleConfig(
                start_iteration=15_000,
                end_iteration=29_899,
                frequency=100,
            )
        )
        near_plane: float | None = Field(default=0.2, gt=0.0)
        filter_variance: float = Field(default=0.2, gt=0.0)
        clipping_tolerance: float = Field(default=0.15, ge=0.0)

        def build(
            self, context: ember.TrainingRunContext
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

    return (NHTMipSplatting3DFilterConfig,)


@app.cell
def _(NHTMipSplatting3DFilterConfig):
    class NHTMipSplattingConfig(NHTConfigBase):
        """Full Mip-Splatting controls for NHT."""

        enabled: bool = False
        screen_filter_enabled: bool = True
        three_dimensional_filter: NHTMipSplatting3DFilterConfig = Field(
            default_factory=NHTMipSplatting3DFilterConfig
        )

    return (NHTMipSplattingConfig,)


@app.cell
def _(NHTMipSplattingConfig, NHTRenderConfig):
    class NHTTrainingConfig(NHTConfigBase):
        """Typed user-facing NHT training config."""

        runtime: ember.RuntimeConfig = Field(
            default_factory=ember.RuntimeConfig
        )
        profiler: TrainingProfilerConfig = Field(
            default_factory=TrainingProfilerConfig
        )
        batching: ember.BatchingConfig = Field(
            default_factory=ember.BatchingConfig
        )
        initialization: NHTInitializationConfig = Field(
            default_factory=NHTInitializationConfig
        )
        model: NHTModelConfig = Field(default_factory=NHTModelConfig)
        render: NHTRenderConfig = Field(default_factory=NHTRenderConfig)
        mip_splatting: NHTMipSplattingConfig = Field(
            default_factory=NHTMipSplattingConfig
        )
        optimization: NHTOptimizationConfig = Field(
            default_factory=NHTOptimizationConfig
        )
        mcmc: NHTMCMCConfig = Field(default_factory=NHTMCMCConfig)
        loss: NHTLossConfig = Field(default_factory=NHTLossConfig)
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
            densification_builders = []
            mcmc_config = self.mcmc.build(context)
            if mcmc_config is not None:
                densification_builders.extend(mcmc_config.builders)
            if self.mip_splatting.enabled:
                densification_builders.append(
                    self.mip_splatting.three_dimensional_filter.build(context)
                )
            densification = (
                ember.densification_config(*densification_builders)
                if densification_builders
                else None
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
                densification=densification,
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

    return (NHTTrainingConfig,)


@app.cell
def _(NHTTrainingConfig):
    class NHTExperimentConfig(NHTConfigBase):
        """Resolved experiment config."""

        preset: NHTDefaultName = "garden_nht"
        scene: NHTSceneConfig = Field(default_factory=NHTSceneConfig)
        data: NHTDataConfig = Field(default_factory=NHTDataConfig)
        training: NHTTrainingConfig

    return (NHTExperimentConfig,)


@app.class_definition
class NHTDecodedRenderOutput:
    """Decoded NHT render output preserving feature render attributes."""

    def __init__(
        self,
        raw_output: object,
        render: Tensor,
        extras: Tensor | None = None,
    ) -> None:
        self.raw_output = raw_output
        self.render = render
        self.extras = extras
        self.features = raw_output.features
        self.alphas = raw_output.alphas
        self.depth = raw_output.depth
        self.visibility = raw_output.visibility
        self.weights = raw_output.weights


@app.class_definition
class NHTDeferredShader(nn.Module):
    """tiny-cuda-nn deferred RGB shader for accumulated NHT features."""

    def __init__(
        self,
        *,
        feature_dim: int = 48,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        enable_view_encoding: bool = True,
        view_encoding: Literal["sh", "fourier", "frequency"] = "sh",
        view_encoding_degree: int = 3,
        view_encoding_scale: float = 3.0,
        view_encoding_frequencies: int = 4,
        jit_fusion: bool = False,
    ) -> None:
        super().__init__()
        if feature_dim % 4 != 0:
            raise ValueError("NHT feature_dim must be divisible by four.")
        import tinycudann as tcnn

        self.feature_dim = feature_dim
        self.encoded_feature_dim = feature_dim // 4 * 2
        self.enable_view_encoding = enable_view_encoding
        self.input_dim = (
            self.encoded_feature_dim + 3
            if enable_view_encoding
            else self.encoded_feature_dim
        )
        self.view_encoding = view_encoding
        self.view_encoding_scale = view_encoding_scale
        if view_encoding == "sh":
            view_encoding_config = {
                "n_dims_to_encode": 3,
                "otype": "SphericalHarmonics",
                "degree": view_encoding_degree,
            }
        else:
            view_encoding_config = {
                "n_dims_to_encode": 3,
                "otype": "Frequency",
                "n_frequencies": view_encoding_frequencies,
            }
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Sigmoid",
            "n_neurons": hidden_dim,
            "n_hidden_layers": max(num_hidden_layers, 1),
        }
        if enable_view_encoding:
            self.network = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.input_dim,
                n_output_dims=3,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": self.encoded_feature_dim,
                            "otype": "Identity",
                        },
                        view_encoding_config,
                    ],
                },
                network_config=network_config,
            )
        else:
            self.network = tcnn.Network(
                n_input_dims=self.input_dim,
                n_output_dims=3,
                network_config=network_config,
            )
        if hasattr(tcnn, "supports_jit_fusion"):
            self.network.jit_fusion = bool(
                jit_fusion and tcnn.supports_jit_fusion()
            )
        self.register_buffer(
            "_nht_ema_ready",
            torch.tensor(False, dtype=torch.bool),
        )
        for name, parameter in self.named_parameters():
            self.register_buffer(
                self._ema_buffer_name(name),
                parameter.detach().clone(),
            )

    @property
    def ray_dir_scale(self) -> float:
        """Return the ray direction scale expected by the renderer."""
        if not self.enable_view_encoding:
            return 1.0
        return self.view_encoding_scale if self.view_encoding == "sh" else 1.0

    def forward(
        self,
        features: Float[Tensor, " batch height width input_dim"],
        view_dirs: Float[Tensor, " batch height width 3"] | None = None,
    ) -> Float[Tensor, " batch height width 3"]:
        """Decode accumulated features into RGB."""
        colors, _extras = self.decode(features, view_dirs)
        return colors

    def decode(
        self,
        features: Float[Tensor, " batch height width input_dim"],
        view_dirs: Float[Tensor, " batch height width 3"] | None = None,
    ) -> tuple[Float[Tensor, " batch height width 3"], Tensor | None]:
        """Decode accumulated features and preserve optional extra channels."""
        del view_dirs
        expected_backend_dim = self.encoded_feature_dim + 3
        if features.shape[-1] < expected_backend_dim:
            raise ValueError(
                "NHT shader expects encoded feature channels plus ray dirs: "
                f"at least {expected_backend_dim}, got {features.shape[-1]}."
            )
        extras = (
            features[..., expected_backend_dim:]
            if features.shape[-1] > expected_backend_dim
            else None
        )
        features = features[..., :expected_backend_dim]
        original_shape = features.shape[:-1]
        if self.enable_view_encoding:
            shader_features = features
        else:
            shader_features = features[..., : self.encoded_feature_dim]
        flattened = shader_features.contiguous().view(-1, self.input_dim)
        colors = self.network(flattened).view(*original_shape, 3)
        return colors, extras

    def apply_ema(self) -> dict[str, Tensor] | None:
        """Swap current parameters to EMA values and return originals."""
        if not bool(self._nht_ema_ready.item()):
            return None
        saved = {}
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                saved[name] = parameter.detach().clone()
                parameter.copy_(getattr(self, self._ema_buffer_name(name)))
        return saved

    def restore_from_ema(self, saved: dict[str, Tensor] | None) -> None:
        """Restore parameters returned by apply_ema."""
        if saved is None:
            return
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(saved[name])

    @staticmethod
    def _ema_buffer_name(parameter_name: str) -> str:
        return f"_nht_ema_{parameter_name.replace('.', '_')}"


@app.class_definition
class NHTColorRefineAndEMAHook:
    """Freeze geometry for color refinement and maintain shader EMA buffers."""

    def __init__(
        self,
        *,
        color_refine_start: int,
        ema_enabled: bool = True,
        ema_decay: float = 0.95,
        ema_start_step: int = 0,
        module_name: str = "deferred_shader",
    ) -> None:
        self.color_refine_start = color_refine_start
        self.ema_enabled = ema_enabled
        self.ema_decay = ema_decay
        self.ema_start_step = ema_start_step
        self.module_name = module_name
        self._ema_buffers: dict[str, str] = {}

    def before_step(self, state: ember.TrainState) -> None:
        """Disable geometry gradients once the upstream color-refine phase starts."""
        is_refine = state.step >= self.color_refine_start
        state.diagnostics["nht/color_refine_active"] = float(is_refine)
        if not is_refine:
            return
        scene = state.model.scene
        for name in (
            "center_position",
            "log_scales",
            "quaternion_orientation",
            "logit_opacity",
        ):
            value = getattr(scene, name, None)
            if isinstance(value, Tensor):
                value.requires_grad_(False)

    def post_optimizer_step(
        self,
        state: ember.TrainState,
        batch: object,
        render_output: object,
        loss_result: LossResult,
    ) -> None:
        """Update EMA buffers after optimizer steps."""
        del batch, render_output, loss_result
        if not self.ema_enabled or state.step < self.ema_start_step:
            return
        shader = state.model.modules.get(self.module_name)
        if not isinstance(shader, nn.Module):
            return
        self._ensure_ema_buffers(shader)
        one_minus_decay = 1.0 - self.ema_decay
        with torch.no_grad():
            for name, parameter in shader.named_parameters():
                buffer = getattr(shader, self._ema_buffers[name])
                buffer.lerp_(parameter.detach(), one_minus_decay)
            ready = getattr(shader, "_nht_ema_ready", None)
            if isinstance(ready, Tensor):
                ready.fill_(True)

    def _ensure_ema_buffers(self, shader: nn.Module) -> None:
        if self._ema_buffers:
            return
        for name, parameter in shader.named_parameters():
            if hasattr(shader, "_ema_buffer_name"):
                buffer_name = shader._ema_buffer_name(name)
            else:
                buffer_name = f"_nht_ema_{name.replace('.', '_')}"
            if not hasattr(shader, buffer_name):
                shader.register_buffer(buffer_name, parameter.detach().clone())
            self._ema_buffers[name] = buffer_name


@app.class_definition
class NHTGaussianMCMC(ember_splatting.GaussianMCMC):
    """NHT MCMC variant that sanitizes opacity logits before sampling."""

    def _sanitize_opacity_logits(self, scene: ember.GaussianScene3D) -> None:
        with torch.no_grad():
            needs_fix = torch.isnan(scene.logit_opacity)
            if bool(needs_fix.any().item()):
                scene.logit_opacity.data[needs_fix] = 0.0

    def _relocate_dead(self, scene: ember.GaussianScene3D) -> None:
        self._sanitize_opacity_logits(scene)
        super()._relocate_dead(scene)

    def _append_new(self, scene: ember.GaussianScene3D) -> None:
        self._sanitize_opacity_logits(scene)
        super()._append_new(scene)


@app.cell
def _(NHTExperimentConfig):
    def nht_preset_catalog() -> ConfigPresetCatalog[NHTExperimentConfig]:
        """Return the notebook's named JSON preset catalog."""
        return ConfigPresetCatalog(
            model_cls=NHTExperimentConfig,
            presets={
                "garden_nht": ConfigPreset(
                    name="garden_nht",
                    path=DEFAULTS_DIR / "garden_nht.json",
                    label="Garden NHT",
                    base_dir=REPO_ROOT,
                ),
                "garden_debug_val": ConfigPreset(
                    name="garden_debug_val",
                    path=DEFAULTS_DIR / "garden_debug_val.json",
                    label="Garden debug validation",
                    base_dir=REPO_ROOT,
                ),
            },
            default="garden_nht",
        )

    return (nht_preset_catalog,)


@app.function
def initialize_nht_model_from_scene_record(
    scene_record: ember.SceneRecord,
    *,
    modules: dict[str, nn.Module] | None = None,
    parameters: dict[str, nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, object] | None = None,
    feature_dim: int = 32,
    features_init_min: float = -math.pi / 2.0,
    features_init_max: float = math.pi / 2.0,
    initial_scale: float = 0.01,
    initial_opacity: float = 0.1,
) -> ember.InitializedModel:
    """Initialize Gaussian geometry with flat learnable NHT features."""
    point_cloud = scene_record.point_cloud
    if point_cloud is None:
        raise ValueError("NHT initialization requires an SfM point cloud.")
    centers = point_cloud.points.to(torch.float32)
    num_splats = int(centers.shape[0])
    distances = nht_root_mean_squared_knn_distances(centers)
    log_scales = torch.log(distances * initial_scale).unsqueeze(-1).repeat(1, 3)
    quaternion_orientation = torch.rand((num_splats, 4), dtype=torch.float32)
    opacity = torch.full((num_splats,), initial_opacity, dtype=torch.float32)
    opacity = opacity.clamp(1e-5, 1.0 - 1e-5)
    feature = torch.empty((num_splats, feature_dim), dtype=torch.float32)
    feature.uniform_(features_init_min, features_init_max)
    return ember.InitializedModel(
        scene=ember.GaussianScene3D(
            center_position=centers.detach().clone().requires_grad_(True),
            log_scales=log_scales.detach().clone().requires_grad_(True),
            quaternion_orientation=quaternion_orientation.detach()
            .clone()
            .requires_grad_(True),
            logit_opacity=torch.logit(opacity)
            .detach()
            .clone()
            .requires_grad_(True),
            feature=feature.detach().clone().requires_grad_(True),
            sh_degree=0,
        ),
        modules=dict(modules or {}),
        parameters=dict(parameters or {}),
        buffers=dict(buffers or {}),
        metadata=dict(metadata or {}),
    )


@app.function
def nht_root_mean_squared_knn_distances(
    positions: Float[Tensor, " num_points 3"],
    *,
    torch_chunk_size: int = 512,
) -> Float[Tensor, " num_points"]:
    """Compute the upstream NHT KNN scale distances."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "NHT KNN distances expect positions with shape "
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
def nht_feature_scene(
    model: ember.InitializedModel,
    camera: ember.CameraState,
) -> ember.GaussianScene3D:
    """Return the flat-feature scene consumed by the native NHT backend."""
    del camera
    scene = model.scene
    if not isinstance(scene, ember.GaussianScene3D):
        raise TypeError("nht_feature_scene expects a GaussianScene3D model.")
    if scene.feature.ndim != 2:
        raise ValueError("NHT expects flat feature tensors.")
    return scene


@app.function
def nht_decode_render(
    model: ember.InitializedModel,
    camera: ember.CameraState,
    render_output: object,
) -> NHTDecodedRenderOutput:
    """Decode native accumulated NHT features into RGB."""
    shader = model.modules["deferred_shader"]
    if not isinstance(shader, NHTDeferredShader):
        raise TypeError("NHT postprocess expects an NHTDeferredShader module.")
    del camera
    saved = shader.apply_ema() if "checkpoint_step" in model.metadata else None
    try:
        render, extras = shader.decode(render_output.features)
    finally:
        shader.restore_from_ema(saved)
    return NHTDecodedRenderOutput(render_output, render, extras)


@app.function
def nht_rgb_l1_dssim_loss(
    state: ember.TrainState,
    batch: object,
    render_output: NHTDecodedRenderOutput,
    *,
    lambda_l1: float = 0.8,
    lambda_ssim: float = 0.2,
    lambda_opacity_regularization: float = 0.0,
    lambda_scale_regularization: float = 0.0,
    ssim_backend: str = "cuda",
    color_refine_start: int = 0,
    weights: dict[str, float] | None = None,
) -> LossResult:
    """NHT RGB reconstruction loss with optional Gaussian regularizers."""
    del weights
    prediction = render_output.render.to(torch.float32)
    target = batch.images.to(torch.float32)
    if prediction.shape != target.shape:
        raise ValueError(
            "NHT reconstruction loss expects render and target images to "
            f"share NHWC shape, got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}."
        )
    l1_loss = (prediction - target).abs().mean()
    ssim_loss = 1.0 - ember_splatting.ssim_score(
        prediction,
        target,
        padding="valid",
        backend=ssim_backend,
    )
    scene = state.model.scene
    if not isinstance(scene, ember.GaussianScene3D):
        raise TypeError(
            "nht_rgb_l1_dssim_loss expects a GaussianScene3D model."
        )
    if state.step < color_refine_start:
        opacity_regularization = torch.sigmoid(scene.logit_opacity).mean()
        scale_regularization = torch.exp(scene.log_scales).mean()
    else:
        opacity_regularization = torch.zeros((), device=prediction.device)
        scale_regularization = torch.zeros((), device=prediction.device)
    loss = (
        lambda_l1 * l1_loss
        + lambda_ssim * ssim_loss
        + lambda_opacity_regularization * opacity_regularization
        + lambda_scale_regularization * scale_regularization
    )
    return LossResult(
        loss=loss,
        metrics={
            "l1": float(l1_loss.detach().item()),
            "ssim_loss": float(ssim_loss.detach().item()),
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


@app.function
def build_nht_mcmc(
    *,
    start_iteration: int = 500,
    end_iteration: int = -1,
    frequency: int = 100,
    min_opacity: float = 0.005,
    cap_growth_factor: float = 1.05,
    cap_max: int = 1_000_000,
    inject_position_noise: bool = True,
    noise_lr_scale: float = 5e5,
) -> NHTGaussianMCMC:
    """Build the NHT MCMC densifier matching the upstream training schedule."""
    from ember_core.densification import Schedule

    return NHTGaussianMCMC(
        schedule=Schedule(
            start_iteration=start_iteration,
            end_iteration=end_iteration,
            frequency=frequency,
        ),
        min_opacity=min_opacity,
        cap_growth_factor=cap_growth_factor,
        cap_max=cap_max,
        inject_position_noise=inject_position_noise,
        noise_lr_scale=noise_lr_scale,
    )


@app.function
def default_checkpoint_dir(
    preset: NHTDefaultName,
    backend: NHTBackendName,
) -> Path:
    """Return the default checkpoint directory for a preset/backend pair."""
    return DEFAULT_CHECKPOINT_ROOT / preset / backend


@app.cell
def _(NHTExperimentConfig):
    def resolve_training_config(
        config: NHTExperimentConfig,
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

    return (resolve_training_config,)


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
def _(NHTExperimentConfig):
    def nht_should_show_jit_compile_notice(
        config: NHTExperimentConfig,
        snapshot: Any,
        *,
        is_script_mode: bool,
    ) -> bool:
        """Return whether the NHT notebook is likely waiting on first JIT compile."""
        return (
            not is_script_mode
            and config.training.model.shader.jit_fusion
            and snapshot.status == "running"
            and snapshot.step == 0
            and not snapshot.latest_metrics
        )

    return (nht_should_show_jit_compile_notice,)


@app.cell
def _(NHTExperimentConfig):
    def resolved_nht_scene_path(config: NHTExperimentConfig) -> Path:
        """Resolve the configured scene path without substituting sample scenes."""
        return config.scene.path.expanduser()

    return (resolved_nht_scene_path,)


@app.cell
def _(NHTExperimentConfig):
    def nht_resized_cache_enabled(config: NHTExperimentConfig) -> bool:
        """Return whether NHT should use a derived resized image cache."""
        return (
            config.data.cache_resized_images
            and config.data.image_scale_factor != 1.0
        )

    return (nht_resized_cache_enabled,)


@app.cell
def _(NHTExperimentConfig, resolved_nht_scene_path):
    def nht_source_image_root(config: NHTExperimentConfig) -> Path:
        """Return the full-resolution source image root."""
        if config.scene.image_root is not None:
            return config.scene.image_root.expanduser()
        return resolved_nht_scene_path(config) / "images"

    return (nht_source_image_root,)


@app.cell
def _(NHTExperimentConfig, resolved_nht_scene_path):
    def nht_resized_cache_parent(config: NHTExperimentConfig) -> Path:
        """Return the reusable derived image cache parent for the scene."""
        if config.data.resized_image_cache_root is not None:
            return config.data.resized_image_cache_root.expanduser()
        return (
            resolved_nht_scene_path(config) / "ember_cache" / "resized_images"
        )

    return (nht_resized_cache_parent,)


@app.cell
def _(NHTExperimentConfig, nht_resized_cache_parent):
    def nht_resized_cache_root(config: NHTExperimentConfig) -> Path:
        """Return the derived resized image cache root for this config."""
        scale_name = f"{config.data.image_scale_factor:.6f}".rstrip("0").rstrip(
            "."
        )
        scale_name = scale_name.replace(".", "p")
        return nht_resized_cache_parent(config) / (
            f"scale_{scale_name}_{config.data.interpolation}"
        )

    return (nht_resized_cache_root,)


@app.function
def nht_pillow_resampling(interpolation: str) -> Any:
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
def enforce_nht_resized_cache_limit(
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
def materialize_nht_resized_image_cache(
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
    resampling = nht_pillow_resampling(interpolation)
    enforce_nht_resized_cache_limit(
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
    enforce_nht_resized_cache_limit(
        cache_root=cache_root,
        max_caches=max_caches,
    )
    return cache_root


@app.cell
def _(
    NHTExperimentConfig,
    nht_resized_cache_enabled,
    nht_resized_cache_root,
    nht_source_image_root,
    resolved_nht_scene_path,
):
    def nht_scene_load_config(
        config: NHTExperimentConfig,
    ) -> ember.ColmapSceneConfig:
        """Build the configured scene-record loader."""
        source_pipes = (
            (ember.HorizonAlignPipeConfig(),)
            if config.scene.align_horizon
            else ()
        )
        scene_path = resolved_nht_scene_path(config)
        image_root = (
            materialize_nht_resized_image_cache(
                source_root=nht_source_image_root(config),
                cache_root=nht_resized_cache_root(config),
                scale=config.data.image_scale_factor,
                interpolation=config.data.interpolation,
                max_caches=config.data.max_resized_image_caches,
            )
            if nht_resized_cache_enabled(config)
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

    return (nht_scene_load_config,)


@app.cell
def _(NHTExperimentConfig, nht_scene_load_config):
    def nht_load_scene_record(config: NHTExperimentConfig) -> ember.SceneRecord:
        """Load the configured scene record."""
        return ember.load_scene_record(
            nht_scene_load_config(config),
        )

    return (nht_load_scene_record,)


@app.cell
def _(NHTExperimentConfig, nht_resized_cache_enabled):
    def nht_prepared_frame_dataset_config(
        config: NHTExperimentConfig,
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
                    if nht_resized_cache_enabled(config)
                    else config.data.image_scale_factor
                ),
                normalize=config.data.normalize_images,
                interpolation=config.data.interpolation,
            ),
        )

    return (nht_prepared_frame_dataset_config,)


@app.cell
def _(NHTExperimentConfig, nht_prepared_frame_dataset_config):
    def nht_prepare_frame_dataset(
        config: NHTExperimentConfig,
        scene_record: ember.SceneRecord,
    ) -> ember.PreparedFrameDataset:
        """Prepare the configured frame dataset."""
        return ember.prepare_frame_dataset(
            scene_record,
            config=nht_prepared_frame_dataset_config(config),
        )

    return (nht_prepare_frame_dataset,)


@app.cell
def _(NHTExperimentConfig, resolve_training_config):
    def run_nht_training(
        config: NHTExperimentConfig,
        frame_dataset: ember.PreparedFrameDataset,
        training_config: ember.TrainingConfig | None = None,
    ) -> ember.TrainingResult:
        """Run NHT training from a prepared frame dataset."""
        resolved_training_config = training_config or resolve_training_config(
            config,
            frame_dataset,
        )
        return ember.run_training(frame_dataset, resolved_training_config)

    return (run_nht_training,)


@app.cell
def _():
    is_script_mode = mo.running_in_notebook() is False
    return (is_script_mode,)


@app.cell
def _(current_config, is_script_mode, nht_load_scene_record, prepare_button):
    should_prepare = is_script_mode or bool(prepare_button.value)
    scene_record = (
        nht_load_scene_record(current_config)
        if should_prepare and current_config is not None
        else None
    )
    return (scene_record,)


@app.cell
def _(current_config, nht_prepare_frame_dataset, scene_record):
    frame_dataset = (
        nht_prepare_frame_dataset(current_config, scene_record)
        if scene_record is not None and current_config is not None
        else None
    )
    return (frame_dataset,)


@app.cell
def _(current_config, frame_dataset, is_script_mode, resolve_training_config):
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
    training_viewer_handle = (
        ember_splatting.create_training_viewer(
            frame_dataset,
            training_config,
            config=viewer_config,
            title="NHT training viewer",
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
    run_nht_training,
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
        training_result = run_nht_training(
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
    is_script_mode,
    nht_should_show_jit_compile_notice,
    training_result,
    training_status_refresh,
    training_viewer_handle,
):
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
        elif nht_should_show_jit_compile_notice(
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


@app.cell(hide_code=True)
def _(resolved_training_config):
    mo.ui.code_editor(
        value=json.dumps(
            resolved_training_config.model_dump(mode="json"), indent=2
        ),
        language="json",
        disabled=True,
    )
    return


if __name__ == "__main__":
    app.run()
