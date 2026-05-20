"""Scaffold-GS paper training notebook for Ember."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")

with app.setup:
    import json
    import math
    import shutil
    import sys
    from pathlib import Path
    from typing import Any, Literal

    import ember_core as ember
    import ember_native_faster_gs.faster_gs as ember_fastergs_native
    import ember_splatting_training as ember_splatting
    import marimo as mo
    import torch
    from ember_core.training import (
        LossResult,
        TrainingProfilerConfig,
        TrainingResult,
    )
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
    DEFAULT_CHECKPOINT_ROOT = (
        REPO_ROOT / "checkpoints" / "papers" / "scaffold_gs"
    )
    ScaffoldGSBackendName = Literal["faster_gs.core"]
    ScaffoldGSDefaultName = Literal["garden_scaffold_gs", "garden_debug_val"]
    sys.modules.setdefault("papers.scaffold_gs.notebook", sys.modules[__name__])
    ember_fastergs_native.register()


class ScaffoldGSConfigBase(BaseModel):
    """Strict base model for Scaffold-GS paper configs."""

    model_config = {"extra": "forbid", "populate_by_name": True}


class ScaffoldGSSceneConfig(ScaffoldGSConfigBase):
    """Scene-record loading options."""

    path: Path = Path("dataset/mipnerf360/garden")
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    align_horizon: bool = True
    white_background: bool = False


class ScaffoldGSDataConfig(ScaffoldGSConfigBase):
    """Prepared-frame dataset options."""

    camera_sensor_id: str | None = None
    image_scale_factor: float = Field(default=1.0, gt=0.0)
    split_target: Literal["train", "val", "all"] = "train"
    split_every_n: int | None = Field(default=8, ge=1)
    materialization_stage: Literal["none", "decoded", "prepared"] = "none"
    materialization_mode: Literal["lazy", "eager"] = "lazy"
    materialization_num_workers: int | None = 0
    normalize_images: bool = True
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"


class ScaffoldGSModelConfig(ScaffoldGSConfigBase):
    """Scaffold-GS anchor and neural-Gaussian model options."""

    spherical_harmonics_degree: int = Field(default=3, ge=0)
    anchor_feature_dimension: int = Field(default=32, ge=1)
    neural_offsets_per_anchor: int = Field(default=10, ge=1)
    initial_voxel_size: float = Field(default=0.001, gt=0.0)
    anchor_update_depth: int = Field(default=3, ge=1)
    anchor_update_initial_factor: int = Field(default=16, ge=1)
    anchor_update_hierarchy_factor: int = Field(default=4, ge=1)
    use_feature_bank: bool = False
    appearance_embedding_dimension: int = Field(default=32, ge=0)
    point_cloud_sampling_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
    level_of_detail: int = Field(default=0, ge=0)
    downsample_factor: int = Field(default=1, ge=1)
    use_lowpoly_mode: bool = False
    use_undistorted_inputs: bool = False
    include_distance_in_opacity_mlp: bool = False
    include_distance_in_covariance_mlp: bool = False
    include_distance_in_color_mlp: bool = False


class ScaffoldGSRenderConfig(ScaffoldGSConfigBase):
    """Typed FasterGS render options for Scaffold-GS."""

    backend: ScaffoldGSBackendName = "faster_gs.core"
    near_plane: float = Field(default=0.01, gt=0.0)
    far_plane: float = Field(default=1000.0, gt=0.0)
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mip_splatting_screen_filter: bool = True
    clamp_output: bool = True
    opacity_selection_threshold: float = Field(default=0.0, ge=0.0, lt=1.0)

    def build(self) -> ember.RenderPipelineSpec:
        """Build the direct-RGB FasterGS render pipeline."""
        return ember.RenderPipelineSpec(
            backend=self.backend,
            return_alpha=False,
            feature_fn=ember.bound_callable(
                target="papers.scaffold_gs.notebook.scaffold_gs_render_scene"
            ),
            backend_options={
                "near_plane": self.near_plane,
                "far_plane": self.far_plane,
                "color_source": "direct_rgb",
                "mip_splatting_screen_filter": self.mip_splatting_screen_filter,
                "clamp_output": self.clamp_output,
                "background_color": list(self.background_color),
            },
        )


class ScaffoldGSOptimizationConfig(ScaffoldGSConfigBase):
    """Optimizer defaults transcribed from upstream Scaffold-GS."""

    iterations: int = Field(default=30_000, ge=1)
    position_learning_rate_initial: float = Field(default=0.0, ge=0.0)
    position_learning_rate_final: float = Field(default=0.0, ge=0.0)
    position_learning_rate_delay_multiplier: float = Field(default=0.01, ge=0.0)
    position_learning_rate_max_steps: int = Field(default=30_000, ge=1)
    offset_learning_rate_initial: float = Field(default=0.01, gt=0.0)
    offset_learning_rate_final: float = Field(default=0.0001, gt=0.0)
    offset_learning_rate_delay_multiplier: float = Field(default=0.01, ge=0.0)
    offset_learning_rate_max_steps: int = Field(default=30_000, ge=1)
    feature_learning_rate: float = Field(default=0.0075, gt=0.0)
    opacity_learning_rate: float = Field(default=0.02, gt=0.0)
    scaling_learning_rate: float = Field(default=0.007, gt=0.0)
    rotation_learning_rate: float = Field(default=0.002, gt=0.0)
    opacity_mlp_learning_rate_initial: float = Field(default=0.002, gt=0.0)
    opacity_mlp_learning_rate_final: float = Field(default=0.00002, gt=0.0)
    covariance_mlp_learning_rate_initial: float = Field(default=0.004, gt=0.0)
    covariance_mlp_learning_rate_final: float = Field(default=0.004, gt=0.0)
    color_mlp_learning_rate_initial: float = Field(default=0.008, gt=0.0)
    color_mlp_learning_rate_final: float = Field(default=0.00005, gt=0.0)
    feature_bank_mlp_learning_rate_initial: float = Field(default=0.01, gt=0.0)
    feature_bank_mlp_learning_rate_final: float = Field(default=0.00001, gt=0.0)
    appearance_learning_rate_initial: float = Field(default=0.05, gt=0.0)
    appearance_learning_rate_final: float = Field(default=0.0005, gt=0.0)
    percent_dense: float = Field(default=0.01, ge=0.0)

    def build(self) -> ember.OptimizationConfig:
        """Build optimizer groups for anchor fields and Scaffold-GS MLPs."""
        optimizer = "ember_splatting_training.FusedAdam"
        optimizer_kwargs = {"eps": 1e-15}
        groups = [
            ember.parameter_group(
                "scene",
                "center_position",
                lr=max(self.position_learning_rate_initial, 1e-12),
                optimizer=optimizer,
                **optimizer_kwargs,
            ),
            ember.parameter_group(
                "scene",
                "feature",
                lr=self.feature_learning_rate,
                optimizer=optimizer,
                **optimizer_kwargs,
            ),
            ember.parameter_group(
                "scene",
                "logit_opacity",
                lr=self.opacity_learning_rate,
                optimizer=optimizer,
                **optimizer_kwargs,
            ),
            ember.parameter_group(
                "scene",
                "log_scales",
                lr=self.scaling_learning_rate,
                optimizer=optimizer,
                **optimizer_kwargs,
            ),
            ember.parameter_group(
                "scene",
                "quaternion_orientation",
                lr=self.rotation_learning_rate,
                optimizer=optimizer,
                **optimizer_kwargs,
            ),
            ember.parameter_group(
                "parameters",
                "anchor_offsets",
                lr=self.offset_learning_rate_initial,
                optimizer=optimizer,
                **optimizer_kwargs,
            ),
            ember.parameter_group(
                "modules",
                "opacity_mlp",
                lr=self.opacity_mlp_learning_rate_initial,
                optimizer=optimizer,
                **optimizer_kwargs,
            ),
            ember.parameter_group(
                "modules",
                "covariance_mlp",
                lr=self.covariance_mlp_learning_rate_initial,
                optimizer=optimizer,
                **optimizer_kwargs,
            ),
            ember.parameter_group(
                "modules",
                "color_mlp",
                lr=self.color_mlp_learning_rate_initial,
                optimizer=optimizer,
                **optimizer_kwargs,
            ),
        ]
        groups.extend(
            [
                ember.parameter_group(
                    "modules",
                    "feature_bank_mlp",
                    lr=self.feature_bank_mlp_learning_rate_initial,
                    optimizer=optimizer,
                    **optimizer_kwargs,
                ),
                ember.parameter_group(
                    "modules",
                    "appearance_embedding",
                    lr=self.appearance_learning_rate_initial,
                    optimizer=optimizer,
                    **optimizer_kwargs,
                ),
            ]
        )
        return ember.OptimizationConfig(parameter_groups=groups)


class ScaffoldGSLossConfig(ScaffoldGSConfigBase):
    """Photometric and scale regularization weights."""

    structural_similarity_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    scaling_regularization_weight: float = Field(default=0.01, ge=0.0)

    def build(self) -> ember.LossConfig:
        """Build the Scaffold-GS loss callable."""
        return ember.loss_config(
            "papers.scaffold_gs.notebook.scaffold_gs_rgb_loss",
            kwargs=self.model_dump(mode="python"),
        )


class ScaffoldGSDensificationConfig(ScaffoldGSConfigBase):
    """Anchor growth and pruning schedule from upstream Scaffold-GS."""

    statistics_start_iteration: int = Field(default=500, ge=0)
    anchor_update_start_iteration: int = Field(default=1500, ge=0)
    anchor_update_interval: int = Field(default=100, ge=1)
    anchor_update_until_iteration: int = Field(default=15_000, ge=0)
    minimum_opacity: float = Field(default=0.005, ge=0.0, lt=1.0)
    success_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    densification_gradient_threshold: float = Field(default=0.0002, ge=0.0)


class ScaffoldGSTrainingConfig(ScaffoldGSConfigBase):
    """Typed user-facing Scaffold-GS training config."""

    runtime: ember.RuntimeConfig = Field(default_factory=ember.RuntimeConfig)
    profiler: TrainingProfilerConfig = Field(
        default_factory=TrainingProfilerConfig
    )
    batching: ember.BatchingConfig = Field(default_factory=ember.BatchingConfig)
    model: ScaffoldGSModelConfig = Field(default_factory=ScaffoldGSModelConfig)
    render: ScaffoldGSRenderConfig = Field(
        default_factory=ScaffoldGSRenderConfig
    )
    optimization: ScaffoldGSOptimizationConfig = Field(
        default_factory=ScaffoldGSOptimizationConfig
    )
    loss: ScaffoldGSLossConfig = Field(default_factory=ScaffoldGSLossConfig)
    densification: ScaffoldGSDensificationConfig = Field(
        default_factory=ScaffoldGSDensificationConfig
    )
    checkpoint: ember.CheckpointExportConfig = Field(
        default_factory=ember.CheckpointExportConfig
    )
    viewer: ember_splatting.TrainingViewerConfig = Field(
        default_factory=ember_splatting.TrainingViewerConfig
    )

    def to_training_config(self) -> ember.TrainingConfig:
        """Materialize this typed config into Ember's runtime config."""
        return ember.TrainingConfig(
            runtime=self.runtime.model_copy(
                update={"max_steps": self.optimization.iterations}
            ),
            profiler=self.profiler,
            batching=self.batching,
            initialization=ember.InitializationSpec(
                initializer=ember.bound_callable(
                    target=(
                        "papers.scaffold_gs.notebook."
                        "initialize_scaffold_gs_model_from_scene_record"
                    ),
                    kwargs=self.model.model_dump(mode="python"),
                    bind={"device": ember.ctx.run.device},
                )
            ),
            render=self.render.build(),
            optimization=self.optimization.build(),
            loss=self.loss.build(),
            checkpoint=self.checkpoint,
        )


class ScaffoldGSExperimentConfig(ScaffoldGSConfigBase):
    """Resolved Scaffold-GS experiment config."""

    preset: ScaffoldGSDefaultName = "garden_scaffold_gs"
    scene: ScaffoldGSSceneConfig = Field(default_factory=ScaffoldGSSceneConfig)
    data: ScaffoldGSDataConfig = Field(default_factory=ScaffoldGSDataConfig)
    training: ScaffoldGSTrainingConfig


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Scaffold-GS training
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
    scaffold_gs_presets = scaffold_gs_preset_catalog()
    config_gui = create_config_gui(
        ScaffoldGSExperimentConfig,
        presets=scaffold_gs_presets,
        label="Scaffold-GS config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    preset_selector = config_gui.preset_selector(label="Scaffold-GS preset")
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


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""
    # Training
    """)
    return


@app.cell
def _():
    prepare_button = mo.ui.run_button(
        label="Prepare training inspector",
        full_width=True,
    )
    train_button = mo.ui.run_button(label="Start training", full_width=True)
    stop_button = mo.ui.run_button(label="Stop training", full_width=True)
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
def _(current_config, frame_dataset):
    training_config = (
        resolve_training_config(current_config)
        if current_config is not None and frame_dataset is not None
        else None
    )
    return (training_config,)


@app.cell
def _(current_config, frame_dataset, is_script_mode, training_config):
    training_viewer_handle = (
        ember_splatting.create_training_run(
            frame_dataset,
            training_config,
            config=current_config.training.viewer,
            title="Scaffold-GS training inspector",
        )
        if not is_script_mode
        and current_config is not None
        and frame_dataset is not None
        and training_config is not None
        else None
    )
    return (training_viewer_handle,)


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
        training_result = run_scaffold_gs_training(
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
                frame_dataset, training_config
            )
    return (training_result,)


@app.cell
def _(stop_button, training_viewer_handle):
    should_stop = bool(stop_button.value)
    if should_stop and training_viewer_handle is not None:
        training_viewer_handle.request_stop()
    return


@app.cell
def _(frame_view_catalog, is_script_mode):
    training_inspector = (
        None
        if is_script_mode or frame_view_catalog is None
        else ember_splatting.create_training_view_inspector(
            frame_view_catalog,
        )
    )
    return (training_inspector,)


@app.cell
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


@app.cell
def _(training_result, training_status_refresh, training_viewer_handle):
    _ = training_status_refresh.value
    if training_result is not None:
        training_result_view = mo.md(
            f"Checkpoint: `{training_result.checkpoint_dir}`\n\n"
            f"Steps: `{len(training_result.history)}`"
        )
    elif training_viewer_handle is None:
        training_result_view = mo.md("Prepare the training inspector first.")
    else:
        snapshot = training_viewer_handle.snapshot()
        training_result_view = mo.md(
            f"Status: `{snapshot.status}` step `{snapshot.step}`"
        )
    return (training_result_view,)


@app.cell(column=2, hide_code=True)
def _():
    mo.md("""
    # Scaffold-GS helpers
    """)
    return


class ScaffoldGSMLP(nn.Module):
    """Small ReLU MLP used by Scaffold-GS anchor heads."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Evaluate the MLP."""
        return self.net(inputs)


def scaffold_gs_preset_catalog() -> ConfigPresetCatalog[
    ScaffoldGSExperimentConfig
]:
    """Return the notebook's named JSON preset catalog."""
    return ConfigPresetCatalog(
        model_cls=ScaffoldGSExperimentConfig,
        presets={
            "garden_scaffold_gs": ConfigPreset(
                name="garden_scaffold_gs",
                path=DEFAULTS_DIR / "garden_scaffold_gs.json",
                label="Garden Scaffold-GS",
                base_dir=REPO_ROOT,
            ),
            "garden_debug_val": ConfigPreset(
                name="garden_debug_val",
                path=DEFAULTS_DIR / "garden_debug_val.json",
                label="Garden debug validation",
                base_dir=REPO_ROOT,
            ),
        },
        default="garden_scaffold_gs",
    )


def _require_point_cloud(
    scene_record: ember.SceneRecord,
) -> ember.PointCloudState:
    if scene_record.point_cloud is None:
        raise ValueError(
            "Scaffold-GS initialization requires an SfM point cloud."
        )
    return scene_record.point_cloud


def _sample_point_cloud(
    point_cloud: ember.PointCloudState,
    ratio: float,
) -> ember.PointCloudState:
    if ratio >= 1.0:
        return point_cloud
    num_points = int(point_cloud.points.shape[0])
    keep_count = max(1, math.ceil(num_points * ratio))
    indices = torch.linspace(0, num_points - 1, keep_count).long()
    return ember.PointCloudState(
        points=point_cloud.points[indices],
        colors=None
        if point_cloud.colors is None
        else point_cloud.colors[indices],
        normals=None
        if point_cloud.normals is None
        else point_cloud.normals[indices],
    )


def initialize_scaffold_gs_model_from_scene_record(
    scene_record: ember.SceneRecord,
    *,
    modules: dict[str, nn.Module] | None = None,
    parameters: dict[str, nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    spherical_harmonics_degree: int = 3,
    anchor_feature_dimension: int = 32,
    neural_offsets_per_anchor: int = 10,
    initial_voxel_size: float = 0.001,
    anchor_update_depth: int = 3,
    anchor_update_initial_factor: int = 16,
    anchor_update_hierarchy_factor: int = 4,
    use_feature_bank: bool = False,
    appearance_embedding_dimension: int = 32,
    point_cloud_sampling_ratio: float = 1.0,
    level_of_detail: int = 0,
    downsample_factor: int = 1,
    use_lowpoly_mode: bool = False,
    use_undistorted_inputs: bool = False,
    include_distance_in_opacity_mlp: bool = False,
    include_distance_in_covariance_mlp: bool = False,
    include_distance_in_color_mlp: bool = False,
    device: torch.device | None = None,
) -> ember.InitializedModel:
    """Initialize Scaffold-GS anchors from a COLMAP point cloud."""
    del (
        spherical_harmonics_degree,
        anchor_update_depth,
        anchor_update_initial_factor,
        anchor_update_hierarchy_factor,
        level_of_detail,
        downsample_factor,
        use_lowpoly_mode,
        use_undistorted_inputs,
    )
    target_device = device or torch.device("cpu")
    point_cloud = _sample_point_cloud(
        _require_point_cloud(scene_record),
        point_cloud_sampling_ratio,
    )
    centers = point_cloud.points.to(device=target_device, dtype=torch.float32)
    num_anchors = int(centers.shape[0])
    colors = (
        torch.full(
            (num_anchors, 3),
            0.5,
            dtype=torch.float32,
            device=target_device,
        )
        if point_cloud.colors is None
        else point_cloud.colors.to(device=target_device, dtype=torch.float32)
    )
    anchor_features = torch.zeros(
        (num_anchors, anchor_feature_dimension),
        dtype=torch.float32,
        device=target_device,
    )
    anchor_features[:, : min(3, anchor_feature_dimension)] = colors[
        :, : min(3, anchor_feature_dimension)
    ]
    log_scales = torch.full(
        (num_anchors, 3),
        math.log(initial_voxel_size),
        dtype=torch.float32,
        device=target_device,
    )
    quaternion_orientation = torch.zeros(
        (num_anchors, 4),
        dtype=torch.float32,
        device=target_device,
    )
    quaternion_orientation[:, 0] = 1.0
    logit_opacity = torch.full(
        (num_anchors,),
        torch.tensor(0.1).logit().item(),
        dtype=torch.float32,
        device=target_device,
    )
    scene = ember.GaussianScene3D(
        center_position=centers.requires_grad_(True),
        log_scales=log_scales.requires_grad_(True),
        quaternion_orientation=quaternion_orientation.requires_grad_(True),
        logit_opacity=logit_opacity.requires_grad_(True),
        feature=anchor_features.requires_grad_(True),
        sh_degree=0,
    )
    head_input_dim = anchor_feature_dimension + 3
    opacity_input_dim = head_input_dim + int(include_distance_in_opacity_mlp)
    covariance_input_dim = head_input_dim + int(
        include_distance_in_covariance_mlp
    )
    color_input_dim = (
        head_input_dim
        + int(include_distance_in_color_mlp)
        + appearance_embedding_dimension
    )
    built_modules = dict(modules or {})
    try:
        appearance_count = max(1, scene_record.num_frames)
    except ValueError:
        appearance_count = 1
    built_modules.update(
        {
            "opacity_mlp": ScaffoldGSMLP(
                opacity_input_dim,
                neural_offsets_per_anchor,
            ),
            "covariance_mlp": ScaffoldGSMLP(
                covariance_input_dim,
                neural_offsets_per_anchor * 7,
            ),
            "color_mlp": ScaffoldGSMLP(
                color_input_dim,
                neural_offsets_per_anchor * 3,
            ),
            "feature_bank_mlp": ScaffoldGSMLP(anchor_feature_dimension + 4, 3),
            "appearance_embedding": nn.Embedding(
                appearance_count,
                appearance_embedding_dimension,
            ),
        }
    )
    built_parameters = dict(parameters or {})
    built_parameters["anchor_offsets"] = nn.Parameter(
        torch.zeros(
            (num_anchors, neural_offsets_per_anchor, 3),
            dtype=torch.float32,
            device=target_device,
        )
    )
    built_metadata = {
        **dict(metadata or {}),
        "neural_offsets_per_anchor": neural_offsets_per_anchor,
        "use_feature_bank": use_feature_bank,
        "appearance_embedding_dimension": appearance_embedding_dimension,
        "include_distance_in_opacity_mlp": include_distance_in_opacity_mlp,
        "include_distance_in_covariance_mlp": include_distance_in_covariance_mlp,
        "include_distance_in_color_mlp": include_distance_in_color_mlp,
    }
    return ember.InitializedModel(
        scene=scene,
        modules=built_modules,
        parameters=built_parameters,
        buffers=dict(buffers or {}),
        metadata=built_metadata,
    )


def _camera_conditioning(
    scene: ember.GaussianScene3D,
    camera: ember.CameraState,
) -> tuple[Tensor, Tensor]:
    camera_center = camera.cam_to_world[0, :3, 3]
    anchor_to_camera = camera_center[None, :] - scene.center_position
    distance = anchor_to_camera.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction = anchor_to_camera / distance
    return direction, distance


def _append_optional_distance(
    inputs: Tensor,
    distance: Tensor,
    *,
    enabled: bool,
) -> Tensor:
    return torch.cat([inputs, distance], dim=-1) if enabled else inputs


def scaffold_gs_render_scene(
    model: ember.InitializedModel,
    camera: ember.CameraState,
) -> ember.GaussianScene3D:
    """Generate view-adaptive direct-RGB Gaussians from Scaffold-GS anchors."""
    scene = model.scene
    if not isinstance(scene, ember.GaussianScene3D):
        raise TypeError("Scaffold-GS expects a GaussianScene3D anchor scene.")
    offsets = model.parameters["anchor_offsets"]
    num_offsets = int(offsets.shape[1])
    direction, distance = _camera_conditioning(scene, camera)
    base_inputs = torch.cat([scene.feature, direction], dim=-1)
    opacity_inputs = _append_optional_distance(
        base_inputs,
        distance,
        enabled=bool(model.metadata["include_distance_in_opacity_mlp"]),
    )
    covariance_inputs = _append_optional_distance(
        base_inputs,
        distance,
        enabled=bool(model.metadata["include_distance_in_covariance_mlp"]),
    )
    appearance_module = model.modules["appearance_embedding"]
    if isinstance(appearance_module, nn.Embedding):
        appearance = appearance_module.weight[:1].expand(
            scene.feature.shape[0], -1
        )
    else:
        appearance = scene.feature.new_empty((scene.feature.shape[0], 0))
    color_inputs = torch.cat(
        [
            _append_optional_distance(
                base_inputs,
                distance,
                enabled=bool(model.metadata["include_distance_in_color_mlp"]),
            ),
            appearance,
        ],
        dim=-1,
    )
    opacity = torch.sigmoid(
        model.modules["opacity_mlp"](opacity_inputs)
    ).reshape(-1)
    covariance = model.modules["covariance_mlp"](covariance_inputs).reshape(
        -1,
        num_offsets,
        7,
    )
    colors = torch.sigmoid(model.modules["color_mlp"](color_inputs)).reshape(
        -1,
        num_offsets,
        3,
    )
    expanded_centers = (
        scene.center_position[:, None, :]
        + offsets * torch.exp(scene.log_scales)[:, None, :]
    ).reshape(-1, 3)
    expanded_scales = (
        scene.log_scales[:, None, :] + 0.1 * torch.tanh(covariance[..., :3])
    ).reshape(-1, 3)
    expanded_rotations = (
        scene.quaternion_orientation[:, None, :] + 0.01 * covariance[..., 3:7]
    ).reshape(-1, 4)
    expanded_rotations = expanded_rotations / expanded_rotations.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-6)
    keep_mask = opacity > 0.0
    return ember.GaussianScene3D(
        center_position=expanded_centers[keep_mask],
        log_scales=expanded_scales[keep_mask],
        quaternion_orientation=expanded_rotations[keep_mask],
        logit_opacity=opacity[keep_mask].clamp(1e-5, 1.0 - 1e-5).logit(),
        feature=colors.reshape(-1, 3)[keep_mask],
        sh_degree=0,
    )


def scaffold_gs_rgb_loss(
    state: ember.TrainState,
    batch: Any,
    render_output: Any,
    *,
    weights: dict[str, float] | None = None,
    structural_similarity_weight: float = 0.2,
    scaling_regularization_weight: float = 0.01,
) -> LossResult:
    """Scaffold-GS loss: L1/DSSIM plus scale-volume regularization."""
    del weights
    prediction = render_output.render
    target = batch.images
    l1 = (prediction - target).abs().mean()
    dssim = ember_splatting.dssim_loss(prediction, target)
    scene = state.model.scene
    scaling_regularization = (
        torch.exp(scene.log_scales).prod(dim=1).mean()
        if isinstance(scene, ember.GaussianScene3D)
        else prediction.new_tensor(0.0)
    )
    loss = (
        (1.0 - structural_similarity_weight) * l1
        + structural_similarity_weight * dssim
        + scaling_regularization_weight * scaling_regularization
    )
    return LossResult(
        loss=loss,
        metrics={
            "l1": float(l1.detach().item()),
            "dssim": float(dssim.detach().item()),
            "scaling_regularization": float(
                scaling_regularization.detach().item()
            ),
        },
    )


@app.cell(column=3, hide_code=True)
def _():
    mo.md("""
    # Support
    """)
    return


def resolve_training_config(
    config: ScaffoldGSExperimentConfig,
) -> ember.TrainingConfig:
    """Apply paper notebook runtime defaults to native Ember training config."""
    checkpoint = config.training.checkpoint.model_copy(
        update={"output_dir": resolve_checkpoint_output_dir(config)}
    )
    training = config.training.model_copy(
        update={"checkpoint": checkpoint},
        deep=True,
    )
    return training.to_training_config()


def run_scaffold_gs_training(
    frame_dataset: ember.PreparedFrameDataset,
    experiment_config: ScaffoldGSExperimentConfig,
    training_config: ember.TrainingConfig | None = None,
) -> TrainingResult:
    """Run Scaffold-GS training from a native Ember training config."""
    resolved_training_config = training_config or resolve_training_config(
        experiment_config
    )
    return ember.run_training(frame_dataset, resolved_training_config)


def build_scene_load_config(
    config: ScaffoldGSExperimentConfig,
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


def build_prepared_frame_dataset_config(
    config: ScaffoldGSExperimentConfig,
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


def resolve_checkpoint_output_dir(config: ScaffoldGSExperimentConfig) -> Path:
    """Mirror checkpoint dirs by preset and backend unless user changed them."""
    default_parent = DEFAULT_CHECKPOINT_ROOT / config.preset
    output_dir = config.training.checkpoint.output_dir.expanduser()
    if output_dir.parent == default_parent:
        return (
            DEFAULT_CHECKPOINT_ROOT
            / config.preset
            / config.training.render.backend
        )
    return output_dir


if __name__ == "__main__":
    app.run()
