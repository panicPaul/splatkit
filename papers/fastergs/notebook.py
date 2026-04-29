"""FasterGS paper training notebook for Ember."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")

with app.setup:
    import math
    import sys
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any, Literal, Protocol, runtime_checkable

    import ember_core as ember
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
        TrainingResult,
        TrainState,
    )
    from jaxtyping import Float
    from marimo_config_gui import (
        ConfigPreset,
        ConfigPresetCatalog,
        config_gui_panel,
        config_json_editor,
        config_preset_selector,
        config_status_panel,
        create_config_state,
        validated_config,
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
    current_config = validated_config(
        config_bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    return (current_config,)


@app.cell(hide_code=True)
def _(preset_selector):
    preset_selector
    return


@app.cell(hide_code=True)
def _(config_bindings, form_gui_state):
    config_gui_panel(
        config_bindings,
        form_gui_state=form_gui_state,
        label="FasterGS config",
        nested_models_multiple_open=False,
        nested_models_flat_after_level=2,
    )
    return


@app.cell(hide_code=True)
def _(config_bindings, form_gui_state, json_gui_state):
    config_json_editor(
        config_bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    return


@app.cell(hide_code=True)
def _(config_bindings, form_gui_state, json_gui_state):
    config_status_panel(
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
class FasterGSExperimentConfig(FasterGSConfigBase):
    """Resolved experiment config."""

    preset: FasterGSDefaultName = "garden_baseline"
    scene: FasterGSSceneConfig = Field(default_factory=FasterGSSceneConfig)
    data: FasterGSDataConfig = Field(default_factory=FasterGSDataConfig)
    training: ember.TrainingConfig


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


@app.function
def _resolve_fastergs_point_cloud(
    scene_record: ember.SceneRecord,
) -> ember.PointCloudState:
    """Return the SfM point cloud required by FasterGS initialization."""
    if scene_record.point_cloud is None:
        raise ValueError("FasterGS initialization requires an SfM point cloud.")
    return scene_record.point_cloud


@app.function
def fastergs_root_mean_squared_knn_distances(
    positions: Float[Tensor, " num_points 3"],
    *,
    fallback_max_points: int = 20_000,
) -> Float[Tensor, " num_points"]:
    """Compute upstream FasterGS initial scale distances for the notebook."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "FasterGS KNN distances expect positions with shape "
            f"(num_points, 3), got {tuple(positions.shape)}."
        )
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
    if num_points > fallback_max_points:
        raise RuntimeError(
            "FasterGS KNN initialization needs simple_knn on CUDA for large "
            f"point clouds; got {num_points} points without that backend."
        )
    if num_points == 1:
        return torch.full(
            (1,),
            1e-3,
            dtype=positions.dtype,
            device=positions.device,
        )
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(math.inf)
    k = min(3, num_points - 1)
    mean_squared = distances.topk(k, largest=False).values.square().mean(dim=1)
    return mean_squared.clamp_min(1e-7).sqrt()


@app.function
def initialize_fastergs_model_from_scene_record(
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
    """Initialize Gaussians exactly like the FasterGS paper implementation."""
    point_cloud = _resolve_fastergs_point_cloud(scene_record)
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

    distances = fastergs_root_mean_squared_knn_distances(centers)
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


@app.function
def resolve_training_config(
    config: FasterGSExperimentConfig,
) -> ember.TrainingConfig:
    """Apply paper notebook runtime defaults to native Ember training config."""
    checkpoint = config.training.checkpoint.model_copy(
        update={
            "output_dir": resolve_checkpoint_output_dir(config),
        },
    )
    return config.training.model_copy(
        update={"checkpoint": checkpoint},
        deep=True,
    )


@app.cell
def _(register_fastergs_backends):
    def run_fastergs_training(
        frame_dataset: ember.PreparedFrameDataset,
        experiment_config: FasterGSExperimentConfig,
    ) -> TrainingResult:
        """Run FasterGS training from a native Ember training config."""
        register_fastergs_backends()
        return ember.run_training(
            frame_dataset,
            resolve_training_config(experiment_config),
        )

    return (run_fastergs_training,)


@app.cell
def _(current_config, train_button):
    should_prepare = bool(train_button.value)
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


@app.cell
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


@app.function
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
class HasFasterGSDensificationInfo(Protocol):
    """Render-output trait for FasterGS densification accumulators."""

    densification_info: Float[Tensor, " 2 num_splats"]


@app.class_definition
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

    def get_render_requirements(
        self,
        state: TrainState,
    ) -> DensificationRenderRequirements:
        """Collect FasterGS visibility accumulators while densification runs."""
        return DensificationRenderRequirements(
            backend_options={
                "collect_densification_info": state.step + 1 < self.stop_iter
            }
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
        if context.step + 1 >= self.stop_iter:
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
        upstream_iteration = context.step + 1
        if self.refine_schedule.includes(upstream_iteration):
            self.adaptive_density_control(scene, upstream_iteration)
            self.reset_accumulators()
        if self.should_reset_opacity(upstream_iteration):
            self.family_ops.reset_opacity(self.max_reset_opacity)

    def adaptive_density_control(
        self,
        scene: ember.GaussianScene,
        step: int,
    ) -> None:
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
                scale_shrink=0.625,
            )
        scene = self.family_ops.scene
        keep_mask = torch.sigmoid(scene.logit_opacity) >= (
            self.prune_opacity_threshold
        )
        if step > self.opacity_reset_every and scene.center_position.shape[0] > 0:
            max_scale = torch.exp(scene.log_scales).max(dim=-1).values
            keep_mask &= max_scale <= 0.1 * self.camera_extent
        keep_mask &= scene.quaternion_orientation.square().sum(dim=1) >= 1e-8
        if torch.any(~keep_mask):
            self.family_ops.prune(keep_mask)

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

    def reset_accumulators(self) -> None:
        self.visible_count = None
        self.grad_sum = None


@app.class_definition
class FasterGSFinalCleanup(BaseDensificationMethod):
    """Notebook-local FasterGS checkpoint cleanup before export."""

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
            raise TypeError("FasterGSFinalCleanup requires GaussianFamilyOps.")
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


@app.cell
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


@app.function
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


@app.function
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


@app.function
def resolve_checkpoint_output_dir(
    config: FasterGSExperimentConfig,
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
