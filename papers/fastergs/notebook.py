"""FasterGS paper training notebook for Ember."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")

with app.setup:
    import json
    import sys
    from collections.abc import Sequence
    from dataclasses import replace
    from pathlib import Path
    from typing import Any, Literal, Protocol, runtime_checkable

    import ember_core as ember
    import marimo as mo
    import torch
    import torch.nn as nn
    from ember_core.core.registry import BACKEND_REGISTRY
    from ember_core.densification import (
        BaseDensificationMethod,
        DensificationContext,
        DensificationRenderRequirements,
        GaussianFamilyOps,
        Schedule,
    )
    from ember_core.densification.runtime import (
        bind_densification,
        build_densification,
    )
    from ember_core.training import (
        LossResult,
        TrainingResult,
        TrainState,
        build_dataloader,
        build_loss_fn,
        build_optimizer_set,
        initialize_model,
        save_checkpoint_dir,
        train_step,
    )
    from jaxtyping import Float
    from marimo_config_gui import (
        ConfigPreset,
        ConfigPresetCatalog,
        PydanticGui,
        config_error,
        config_form,
        config_json,
        config_preset_selector,
        config_value,
        create_config_state,
    )
    from pydantic import BaseModel, Field
    from torch import Tensor

    NOTEBOOK_PATH = Path(__file__).resolve()
    NOTEBOOK_DIR = NOTEBOOK_PATH.parent
    REPO_ROOT = NOTEBOOK_DIR.parents[1]
    DEFAULTS_DIR = NOTEBOOK_DIR / "defaults"
    DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "papers" / "fastergs"
    FasterGSBackendName = Literal["adapter.fastergs", "faster_gs.core"]
    FasterGSDefaultName = Literal["garden_baseline", "garden_mcmc"]
    sys.modules.setdefault("papers.fastergs.notebook", sys.modules[__name__])


@app.cell(hide_code=True)
def _():
    mo.md("""
    # FasterGS training
    """)
    return


@app.cell(hide_code=True)
def _(train_button):
    train_button
    return


@app.cell(hide_code=True)
def _(preset_selector):
    preset_selector
    return


@app.cell(hide_code=True)
def _(densification_form):
    densification_form
    return


@app.cell
def _():
    fastergs_presets = fastergs_preset_catalog()
    form_gui_state, json_gui_state, config_bindings = create_config_state(
        FasterGSExperimentConfig,
        presets=fastergs_presets,
    )
    return config_bindings, fastergs_presets, form_gui_state, json_gui_state


@app.cell
def _(config_bindings, fastergs_presets, form_gui_state, json_gui_state):
    preset_selector = config_preset_selector(
        config_bindings,
        presets=fastergs_presets,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
        label="FasterGS preset",
    )
    return (preset_selector,)


@app.cell
def _(config_bindings, form_gui_state, json_gui_state):
    current_config = config_value(
        config_bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    return (current_config,)


@app.cell(hide_code=True)
def _(config_bindings, form_gui_state):
    config_form(
        config_bindings,
        form_gui_state=form_gui_state,
        label="FasterGS config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
        exclude_fields=frozenset({"densification"}),
    )
    return


@app.cell(hide_code=True)
def _(config_bindings, form_gui_state, json_gui_state):
    config_json(
        config_bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    return


@app.cell(hide_code=True)
def _(config_bindings, form_gui_state, json_gui_state):
    config_error(
        config_bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    return


@app.cell(hide_code=True)
def _(training_result_view):
    training_result_view
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    Viewer integration is intentionally left as a placeholder while the viewer
    refactor is in flight. Training and checkpoint export do not depend on it.
    """)
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

    model_config = {"extra": "forbid"}


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
    split_target: Literal["train", "val", "all"] = "train"
    split_every_n: int | None = Field(default=8, ge=1)
    materialization_stage: Literal["none", "decoded", "prepared"] = "decoded"
    materialization_mode: Literal["lazy", "eager"] = "eager"
    materialization_num_workers: int | None = 0
    normalize_images: bool = True
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"


@app.class_definition
class FasterGSModelConfig(FasterGSConfigBase):
    """Gaussian initialization settings."""

    sh_degree: int = Field(default=3, ge=0)
    initial_scale: float = Field(default=0.01, gt=0.0)
    initial_opacity: float = Field(default=0.1, gt=0.0, lt=1.0)
    default_color: tuple[float, float, float] = (0.5, 0.5, 0.5)


@app.class_definition
class FasterGSRenderConfig(FasterGSConfigBase):
    """FasterGS render options."""

    proper_antialiasing: bool = False
    near_plane: float = Field(default=0.2, gt=0.0)
    far_plane: float = Field(default=10_000.0, gt=0.0)
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)


@app.class_definition
class FasterGSOptimizationConfig(FasterGSConfigBase):
    """Optimizer settings following FasterGS defaults."""

    optimizer: str = "ember_splatting_training.FusedAdam"
    means_lr_init: float = Field(default=1.6e-4, gt=0.0)
    means_lr_final: float = Field(default=1.6e-6, gt=0.0)
    means_lr_max_steps: int | None = Field(default=None, ge=1)
    sh_dc_lr: float = Field(default=2.5e-3, gt=0.0)
    sh_rest_lr: float = Field(default=1.25e-4, gt=0.0)
    opacity_lr: float = Field(default=2.5e-2, gt=0.0)
    scale_lr: float = Field(default=5e-3, gt=0.0)
    rotation_lr: float = Field(default=1e-3, gt=0.0)


@app.class_definition
class FasterGSLossConfig(FasterGSConfigBase):
    """Loss weights used for training."""

    lambda_l1: float = Field(default=0.8, ge=0.0)
    lambda_dssim: float = Field(default=0.2, ge=0.0)
    lambda_opacity_regularization: float = Field(default=0.0, ge=0.0)
    lambda_scale_regularization: float = Field(default=0.0, ge=0.0)


@app.class_definition
class FasterGSDensificationConfig(FasterGSConfigBase):
    """FasterGS densification schedule and thresholds."""

    use_mcmc: bool = False
    refine_every: int = Field(default=100, ge=1)
    start_iter: int = Field(default=600, ge=0)
    stop_iter: int = Field(default=14_900, ge=0)
    grad_threshold: float = Field(default=2e-4, gt=0.0)
    dense_fraction: float = Field(default=0.01, gt=0.0)
    prune_opacity_threshold: float = Field(default=0.005, gt=0.0)
    opacity_reset_every: int = Field(default=3_000, ge=1)
    extra_opacity_reset_iter: int | None = Field(default=500, ge=0)
    max_reset_opacity: float = Field(default=0.01, gt=0.0, lt=1.0)
    min_opacity: float = Field(default=0.005, gt=0.0, lt=1.0)
    max_primitives: int = Field(default=1_000_000, ge=1)
    noise_lr_scale: float = Field(default=5e5, gt=0.0)


@app.class_definition
class FasterGSCheckpointConfig(FasterGSConfigBase):
    """Checkpoint export settings."""

    output_dir: Path
    export_ply: bool = True
    overwrite: bool = False


@app.class_definition
class FasterGSExecutionConfig(FasterGSConfigBase):
    """Runtime controls for the notebook training loop."""

    device: Literal["cpu", "cuda"] = "cuda"
    seed: int = 0
    max_steps: int = Field(default=30_000, ge=1)
    batch_size: int = Field(default=1, ge=1)
    shuffle: bool = True
    preview_every: int = Field(default=250, ge=1)


@app.class_definition
class FasterGSExperimentConfig(FasterGSConfigBase):
    """Resolved experiment config."""

    preset: FasterGSDefaultName = "garden_baseline"
    backend: FasterGSBackendName = "adapter.fastergs"
    scene: FasterGSSceneConfig = Field(default_factory=FasterGSSceneConfig)
    data: FasterGSDataConfig = Field(default_factory=FasterGSDataConfig)
    model: FasterGSModelConfig = Field(default_factory=FasterGSModelConfig)
    render: FasterGSRenderConfig = Field(default_factory=FasterGSRenderConfig)
    optimization: FasterGSOptimizationConfig = Field(
        default_factory=FasterGSOptimizationConfig
    )
    loss: FasterGSLossConfig = Field(default_factory=FasterGSLossConfig)
    densification: FasterGSDensificationConfig = Field(
        default_factory=FasterGSDensificationConfig
    )
    checkpoint: FasterGSCheckpointConfig
    execution: FasterGSExecutionConfig = Field(
        default_factory=FasterGSExecutionConfig
    )


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
        },
        default="garden_baseline",
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Training setup
    """)
    return


@app.cell
def _():
    train_button = mo.ui.run_button(label="Start training")
    return (train_button,)


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""
    # Training
    """)
    return


@app.function(column=1)
def compute_training_camera_extent(
    frame_dataset: ember.PreparedFrameDataset,
) -> float:
    """Compute FasterGS camera extent from prepared training cameras."""
    centers = [
        frame_dataset[index].camera.cam_to_world[..., :3, 3].reshape(-1, 3)
        for index in range(len(frame_dataset))
    ]
    camera_centers = torch.cat(centers, dim=0).to(torch.float32)
    mean_center = camera_centers.mean(dim=0, keepdim=True)
    return float(1.1 * (camera_centers - mean_center).norm(dim=-1).max().item())


@app.function(column=1)
def build_training_config(
    config: FasterGSExperimentConfig,
    *,
    camera_extent: float = 1.0,
) -> ember.TrainingConfig:
    """Translate paper config into Ember's declarative training config."""
    means_lr_max_steps = (
        config.optimization.means_lr_max_steps
        if config.optimization.means_lr_max_steps is not None
        else config.execution.max_steps
    )
    optimizer_kwargs = {"eps": 1e-15}
    parameter_groups = [
        ember.ParameterGroupConfig(
            target=ember.ParameterTargetSpec(
                scope="scene",
                name="center_position",
            ),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.means_lr_init * camera_extent,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=ember.CallableSpec(
                target="ember_core.training.exponential_decay_to",
                kwargs={
                    "final_lr": config.optimization.means_lr_final
                    * camera_extent,
                    "max_steps": means_lr_max_steps,
                },
            ),
        ),
        ember.ParameterGroupConfig(
            target=ember.ParameterTargetSpec(
                scope="scene",
                name="feature",
                view=ember.TensorViewSpec(
                    slices=(ember.TensorSliceSpec(axis=1, start=0, stop=1),)
                ),
            ),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.sh_dc_lr,
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
            optimizer=config.optimization.optimizer,
            lr=config.optimization.sh_rest_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
        ember.ParameterGroupConfig(
            target=ember.ParameterTargetSpec(
                scope="scene",
                name="logit_opacity",
            ),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.opacity_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
        ember.ParameterGroupConfig(
            target=ember.ParameterTargetSpec(scope="scene", name="log_scales"),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.scale_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
        ember.ParameterGroupConfig(
            target=ember.ParameterTargetSpec(
                scope="scene",
                name="quaternion_orientation",
            ),
            optimizer=config.optimization.optimizer,
            lr=config.optimization.rotation_lr,
            optimizer_kwargs=optimizer_kwargs,
        ),
    ]
    if config.densification.use_mcmc:
        densification_builder = ember.CallableSpec(
            target="papers.fastergs.notebook.build_fastergs_mcmc_densification",
            kwargs={
                "refine_every": config.densification.refine_every,
                "start_iter": config.densification.start_iter,
                "stop_iter": config.densification.stop_iter,
                "min_opacity": config.densification.min_opacity,
                "max_primitives": config.densification.max_primitives,
                "noise_lr_scale": config.densification.noise_lr_scale,
            },
        )
    else:
        densification_builder = ember.CallableSpec(
            target="papers.fastergs.notebook.FasterGSVanillaDensification",
            kwargs={
                "refine_every": config.densification.refine_every,
                "start_iter": config.densification.start_iter,
                "stop_iter": config.densification.stop_iter,
                "grad_threshold": config.densification.grad_threshold,
                "dense_fraction": config.densification.dense_fraction,
                "prune_opacity_threshold": (
                    config.densification.prune_opacity_threshold
                ),
                "opacity_reset_every": config.densification.opacity_reset_every,
                "extra_opacity_reset_iter": (
                    config.densification.extra_opacity_reset_iter
                ),
                "max_reset_opacity": config.densification.max_reset_opacity,
                "camera_extent": camera_extent,
            },
        )
    return ember.TrainingConfig(
        runtime=ember.RuntimeConfig(
            device=config.execution.device,
            seed=config.execution.seed,
            max_steps=config.execution.max_steps,
        ),
        batching=ember.BatchingConfig(
            batch_size=config.execution.batch_size,
            shuffle=config.execution.shuffle,
        ),
        initialization=ember.InitializationSpec(
            initializer=ember.CallableSpec(
                target=(
                    "ember_core.initialization."
                    "initialize_gaussian_model_from_scene_record"
                ),
                kwargs={
                    "sh_degree": config.model.sh_degree,
                    "initial_scale": config.model.initial_scale,
                    "initial_opacity": config.model.initial_opacity,
                    "default_color": config.model.default_color,
                },
            )
        ),
        render=ember.RenderPipelineSpec(
            backend=config.backend,
            return_alpha=False,
            backend_options={
                "near_plane": config.render.near_plane,
                "far_plane": config.render.far_plane,
                "proper_antialiasing": config.render.proper_antialiasing,
                "background_color": list(config.render.background_color),
            },
        ),
        optimization=ember.OptimizationConfig(
            parameter_groups=parameter_groups
        ),
        loss=ember.LossConfig(
            target=ember.CallableSpec(
                target="papers.fastergs.notebook.fastergs_training_loss",
                kwargs={
                    "lambda_l1": config.loss.lambda_l1,
                    "lambda_dssim": config.loss.lambda_dssim,
                    "lambda_opacity_regularization": (
                        config.loss.lambda_opacity_regularization
                    ),
                    "lambda_scale_regularization": (
                        config.loss.lambda_scale_regularization
                    ),
                },
            )
        ),
        densification=ember.DensificationConfig(builder=densification_builder),
        checkpoint=ember.CheckpointExportConfig(
            output_dir=resolve_checkpoint_output_dir(config),
            export_ply=config.checkpoint.export_ply,
        ),
    )


@app.function(column=1)
def gaussian_window(
    *,
    kernel_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, " channels 1 kernel_size kernel_size"]:
    """Build a depthwise Gaussian SSIM window."""
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = torch.outer(gauss_1d, gauss_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()


@app.function(column=1)
def ssim_score(
    prediction: Float[Tensor, " batch height width 3"],
    target: Float[Tensor, " batch height width 3"],
    *,
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Compute mean SSIM over NHWC RGB tensors."""
    prediction_nchw = prediction.permute(0, 3, 1, 2)
    target_nchw = target.permute(0, 3, 1, 2)
    channels = int(prediction_nchw.shape[1])
    window = gaussian_window(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=channels,
        device=prediction_nchw.device,
        dtype=prediction_nchw.dtype,
    )
    padding = kernel_size // 2
    mu_prediction = nn.functional.conv2d(
        prediction_nchw,
        window,
        padding=padding,
        groups=channels,
    )
    mu_target = nn.functional.conv2d(
        target_nchw,
        window,
        padding=padding,
        groups=channels,
    )
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    sigma_prediction_sq = (
        nn.functional.conv2d(
            prediction_nchw.square(),
            window,
            padding=padding,
            groups=channels,
        )
        - mu_prediction_sq
    )
    sigma_target_sq = (
        nn.functional.conv2d(
            target_nchw.square(),
            window,
            padding=padding,
            groups=channels,
        )
        - mu_target_sq
    )
    sigma_product = (
        nn.functional.conv2d(
            prediction_nchw * target_nchw,
            window,
            padding=padding,
            groups=channels,
        )
        - mu_product
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_product + c1) * (2.0 * sigma_product + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        sigma_prediction_sq + sigma_target_sq + c2
    )
    return (numerator / denominator).mean()


@app.function(column=1)
def fastergs_training_loss(
    state: TrainState,
    batch: Any,
    render_output: Any,
    *,
    weights: dict[str, float],
    lambda_l1: float,
    lambda_dssim: float,
    lambda_opacity_regularization: float,
    lambda_scale_regularization: float,
) -> LossResult:
    """Paper-style FasterGS RGB reconstruction loss."""
    del weights
    prediction = render_output.render
    target = batch.images
    if prediction.shape != target.shape:
        raise ValueError(
            "FasterGS loss expects render and target images to share NHWC "
            f"shape, got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    l1_loss = (prediction - target).abs().mean()
    dssim = (1.0 - ssim_score(prediction, target)) / 2.0
    scene = state.model.scene
    assert isinstance(scene, ember.GaussianScene3D)
    opacity_regularization = torch.sigmoid(scene.logit_opacity).mean()
    scale_regularization = torch.exp(scene.log_scales).mean()
    loss = (
        lambda_l1 * l1_loss
        + lambda_dssim * dssim
        + lambda_opacity_regularization * opacity_regularization
        + lambda_scale_regularization * scale_regularization
    )
    return LossResult(
        loss=loss,
        metrics={
            "l1": float(l1_loss.detach().item()),
            "dssim": float(dssim.detach().item()),
            "opacity_regularization": float(
                opacity_regularization.detach().item()
            ),
            "scale_regularization": float(scale_regularization.detach().item()),
        },
    )


@app.cell(column=1)
def _(register_fastergs_backends):
    def run_fastergs_training(
        frame_dataset: ember.PreparedFrameDataset,
        experiment_config: FasterGSExperimentConfig,
    ) -> TrainingResult:
        """Run FasterGS training through notebook-local paper logic."""
        register_fastergs_backends()
        set_seed(experiment_config.execution.seed)
        device = torch.device(experiment_config.execution.device)
        camera_extent = compute_training_camera_extent(frame_dataset)
        training_config = build_training_config(
            experiment_config,
            camera_extent=camera_extent,
        )
        ensure_checkpoint_output_writable(
            training_config.checkpoint.output_dir,
            overwrite=experiment_config.checkpoint.overwrite,
        )
        model = initialize_model(
            frame_dataset.scene_record, training_config
        ).to(device)
        state = TrainState(
            model=model,
            step=0,
            seed=experiment_config.execution.seed,
            device=device,
        )
        dataloader = build_dataloader(frame_dataset, training_config)
        loss_fn = build_loss_fn(training_config)
        optimizers = build_optimizer_set(state, training_config)
        densification = build_densification(training_config.densification)
        densification = bind_densification(densification, state, optimizers)
        render_fn = make_dynamic_fastergs_render_fn(
            training_config,
            state,
            collect_until_step=experiment_config.densification.stop_iter,
            use_mcmc=experiment_config.densification.use_mcmc,
        )
        history: list[dict[str, float]] = []
        iterator = cycle(dataloader)
        for _ in range(training_config.runtime.max_steps):
            history.append(
                train_step(
                    state,
                    next(iterator),
                    render_fn=render_fn,
                    loss_fn=loss_fn,
                    optimizers=optimizers,
                    densification=densification,
                    hooks=(),
                )
            )
        checkpoint_dir = save_checkpoint_dir(
            training_config.checkpoint.output_dir,
            state,
            training_config,
            frame_dataset=frame_dataset,
        )
        return TrainingResult(
            state=state,
            history=history,
            checkpoint_dir=str(checkpoint_dir),
        )

    return (run_fastergs_training,)


@app.cell(column=1)
def _(current_config, train_button):
    should_prepare = bool(train_button.value)
    scene_record = (
        ember.load_scene_record(build_scene_load_config(current_config))
        if should_prepare and current_config is not None
        else None
    )
    return (scene_record,)


@app.cell(column=1)
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


@app.cell(column=1)
def _(current_config, frame_dataset, run_fastergs_training, train_button):
    should_train = bool(train_button.value)
    training_result = (
        run_fastergs_training(frame_dataset, current_config)
        if should_train
        and current_config is not None
        and frame_dataset is not None
        else None
    )
    return (training_result,)


@app.cell(column=1)
def _(training_result):
    training_result_view = (
        mo.md("Training has not started.")
        if training_result is None
        else mo.md(
            f"Checkpoint: `{training_result.checkpoint_dir}`\n\n"
            f"Steps: `{len(training_result.history)}`"
        )
    )
    return (training_result_view,)


@app.cell(column=2, hide_code=True)
def _():
    mo.md("""
    # Densification
    """)
    return


@app.function(column=2)
def update_densification_payload(
    config_payload: dict[str, Any],
    densification: FasterGSDensificationConfig,
) -> dict[str, Any]:
    """Return a full experiment payload with updated densification settings."""
    current_config = FasterGSExperimentConfig.model_validate(config_payload)
    next_config = current_config.model_copy(
        update={"densification": densification}
    )
    return next_config.model_dump(mode="json")


@app.function(column=2)
def config_payload_json(config_payload: dict[str, Any]) -> str:
    """Serialize a config payload for the JSON editor state."""
    config = FasterGSExperimentConfig.model_validate(config_payload)
    return json.dumps(config.model_dump(mode="json"), indent=2)


@app.function(column=2)
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
            start_iteration=start_iter,
            end_iteration=stop_iter,
            frequency=refine_every,
        ),
        min_opacity=min_opacity,
        cap_max=max_primitives,
        noise_lr_scale=noise_lr_scale,
    )


@app.class_definition(column=2)
@runtime_checkable
class HasFasterGSDensificationInfo(Protocol):
    """Render-output trait for FasterGS densification accumulators."""

    densification_info: Float[Tensor, " 2 num_splats"]


@app.function(column=2)
def render_options(
    training_config: ember.TrainingConfig,
    *,
    collect_densification_info: bool,
) -> Any:
    """Build backend options with dynamic FasterGS densification collection."""
    backend = BACKEND_REGISTRY[training_config.render.backend]
    option_updates = dict(training_config.render.backend_options)
    option_updates["collect_densification_info"] = collect_densification_info
    default_options = backend.default_options
    resolved: dict[str, Any] = {}
    for field_name, value in option_updates.items():
        current = getattr(default_options, field_name)
        if isinstance(current, torch.Tensor):
            value = torch.as_tensor(
                value,
                dtype=current.dtype,
                device=current.device,
            )
        resolved[field_name] = value
    return replace(default_options, **resolved)


@app.function(column=2)
def make_dynamic_fastergs_render_fn(
    training_config: ember.TrainingConfig,
    state: TrainState,
    *,
    collect_until_step: int,
    use_mcmc: bool,
) -> Any:
    """Create a render function that toggles FasterGS accumulation by step."""

    def render_fn(model: Any, camera: Any) -> Any:
        collect = (not use_mcmc) and state.step < collect_until_step
        return ember.render(
            model.scene,
            camera,
            backend=training_config.render.backend,
            return_alpha=False,
            return_depth=False,
            return_gaussian_impact_score=False,
            return_normals=False,
            return_2d_projections=False,
            return_projective_intersection_transforms=False,
            options=render_options(
                training_config,
                collect_densification_info=collect,
            ),
        )

    return render_fn


@app.cell(column=2)
def _(config_bindings, form_gui_state):
    densification_form_ref: dict[
        str, PydanticGui[FasterGSDensificationConfig]
    ] = {}

    def on_densification_change(_: Any) -> None:
        densification = densification_form_ref["form"].value
        if densification is None:
            return
        next_payload = update_densification_payload(
            form_gui_state(),
            densification,
        )
        config_bindings.set_form_gui_state(next_payload)
        config_bindings.set_json_gui_state(config_payload_json(next_payload))

    densification_form = PydanticGui(
        FasterGSDensificationConfig,
        value=form_gui_state()["densification"],
        label="Densification",
        include_json_editor=False,
        bordered=False,
        nested_models_multiple_open=False,
        nested_models_flat_after_level=1,
        on_change=on_densification_change,
    )
    densification_form_ref["form"] = densification_form
    return (densification_form,)


@app.class_definition(column=2)
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

    def get_render_requirements(self) -> DensificationRenderRequirements:
        """Document the backend option this strategy consumes."""
        return DensificationRenderRequirements(
            backend_options={"collect_densification_info": True}
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
        if context.step >= self.stop_iter:
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
        if self.refine_schedule.includes(context.step):
            self.adaptive_density_control(scene)
            self.reset_accumulators()
        if self.should_reset_opacity(context.step):
            self.family_ops.reset_opacity(self.max_reset_opacity)

    def adaptive_density_control(self, scene: ember.GaussianScene) -> None:
        if self.visible_count is None or self.grad_sum is None:
            return
        assert self.family_ops is not None
        avg_grad = self.grad_sum / self.visible_count.clamp_min(1.0)
        scales = torch.exp(scene.log_scales).max(dim=-1).values
        densify_mask = avg_grad >= self.grad_threshold
        small_mask = scales <= self.dense_fraction * self.camera_extent
        clone_mask = densify_mask & small_mask
        split_mask = densify_mask & ~small_mask
        n_cloned = int(clone_mask.sum().item())
        if torch.any(clone_mask):
            self.family_ops.clone(clone_mask)
        if torch.any(split_mask):
            split_mask = torch.cat(
                [
                    split_mask,
                    torch.zeros(
                        n_cloned,
                        dtype=torch.bool,
                        device=split_mask.device,
                    ),
                ]
            )
            self.family_ops.split(
                split_mask,
                num_children=2,
                scale_shrink=0.8,
            )
        scene = self.family_ops.scene
        keep_mask = torch.sigmoid(scene.logit_opacity) >= (
            self.prune_opacity_threshold
        )
        keep_mask &= scene.quaternion_orientation.square().sum(dim=1) >= 1e-8
        if torch.any(~keep_mask):
            self.family_ops.prune(keep_mask)

    def should_reset_opacity(self, step: int) -> bool:
        scheduled = (
            step >= self.opacity_reset_every
            and step <= self.stop_iter
            and step % self.opacity_reset_every == 0
        )
        return scheduled or step == self.extra_opacity_reset_iter

    def reset_accumulators(self) -> None:
        self.visible_count = None
        self.grad_sum = None


@app.cell(column=3, hide_code=True)
def _():
    mo.md("""
    # Support
    """)
    return


@app.cell(column=3)
def _(importlib):
    def register_fastergs_backends() -> tuple[str, ...]:
        """Register FasterGS backends used by the paper notebook."""
        registered: list[str] = []
        for module_name in (
            "ember_adapter_backends.fastergs",
            "ember_native_faster_gs.faster_gs",
        ):
            module = importlib.import_module(module_name)
            register = module.register
            register()
            registered.append(module_name)
        return tuple(registered)

    return (register_fastergs_backends,)


@app.function(column=3)
def build_scene_load_config(
    config: FasterGSExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Translate paper config into an Ember scene loader config."""
    source_pipes = (
        (ember.HorizonAlignPipeConfig(),) if config.scene.align_horizon else ()
    )
    return ember.ColmapSceneConfig(
        path=config.scene.path.expanduser(),
        image_root=(
            config.scene.image_root.expanduser()
            if config.scene.image_root is not None
            else None
        ),
        undistort_output_dir=(
            config.scene.undistort_output_dir.expanduser()
            if config.scene.undistort_output_dir is not None
            else None
        ),
        source_pipes=source_pipes,
    )


@app.function(column=3)
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
            resize_width_scale=config.data.image_scale_factor,
            resize_width_target=None,
            interpolation=config.data.interpolation,
        ),
    )


@app.function(column=3)
def resolve_checkpoint_output_dir(
    config: FasterGSExperimentConfig,
) -> Path:
    """Mirror checkpoint dirs by preset and backend unless user changed them."""
    default_parent = DEFAULT_CHECKPOINT_ROOT / config.preset
    output_dir = config.checkpoint.output_dir.expanduser()
    if output_dir.parent == default_parent:
        return default_checkpoint_dir(config.preset, config.backend)
    return output_dir


@app.function(column=3)
def ensure_checkpoint_output_writable(
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """Fail before training overwrites an existing checkpoint artifact."""
    artifacts = ("config.json", "metadata.json", "model.ckpt", "scene.ply")
    if overwrite:
        return
    existing = [output_dir / artifact for artifact in artifacts]
    if any(path.exists() for path in existing):
        raise FileExistsError(
            "Checkpoint output directory already contains training artifacts: "
            f"{output_dir}. Set checkpoint.overwrite=true to replace them."
        )


@app.function(column=3)
def set_seed(seed: int) -> None:
    """Seed PyTorch RNGs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.function(column=3)
def cycle(loader: Any) -> Any:
    """Repeat a dataloader forever."""
    while True:
        yield from loader


if __name__ == "__main__":
    app.run()
