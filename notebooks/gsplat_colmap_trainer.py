"""Simple COLMAP 3DGS training notebook using splatkit + gsplat backend."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    from dataclasses import replace
    from pathlib import Path
    from typing import Literal

    import marimo as mo
    import numpy as np
    import splatkit as sk
    import splatkit_backends.gsplat as sk_gsplat
    import torch
    import torch.nn.functional as F
    from marimo_3dv import json_gui
    from PIL import Image
    from pydantic import BaseModel, Field
    from splatkit.data import (
        ColmapDatasetConfig,
        DatasetRuntimeConfig,
        HorizonAlignPipeConfig,
        MaterializationConfig,
        NormalizePipeConfig,
        ResizePipeConfig,
        SplitConfig,
        collate_frame_samples,
        load_dataset,
    )
    from splatkit.io.scene import save_scene
    from torch.utils.data import DataLoader
    from tqdm import tqdm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # gsplat COLMAP Trainer

    Minimal 3DGS training notebook for COLMAP reconstructions using
    `splatkit` data loading + initialization and
    `splatkit-backends.gsplat` rendering.

    This omits densification by design and keeps the loop explicit to stress
    backend behavior.

    ## Findings

    - GT images from our data pipeline are produced as `N,H,W,3` (NHWC) from
      preprocessing and batching, so render output must be permuted before
      MSE.
    - `GsplatRenderOutput.render` is HWC-style: `num_cams,height,width,3`.
    - `model.to(device)` from dataclass `scene.to(...)` returns non-leaf tensors,
      so training needs a leaf reconstruction step before optimizer creation.
    - A long bundle build is expected in a fresh run: `build_trainer_bundle` does
      COLMAP scene hydration + optional eager frame materialization, and
      dataloader startup still needs worker bootstrap on first iterator use.
      This is true even if source images already exist on disk, because the
      prepared tensors/caches are not persisted unless you explicitly keep that
      in-memory dataset state warm for the notebook session.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Inputs
    """)
    return


@app.cell
def _(Config):
    train_form = json_gui(
        Config,
        value=Config(),
        label="COLMAP Trainer",
        live_update=False,
    )
    return (train_form,)


@app.cell
def _(train_form):
    train_form
    return


@app.cell
def _():
    run_button = mo.ui.button(
        value=0,
        label="Run 3DGS training",
        on_click=lambda value: (0 if value is None else int(value)) + 1,
    )
    mo.vstack([run_button], gap=1.0)
    return (run_button,)


@app.cell
def _(config, run_button, run_training, trainer_bundle):
    training_artifact: tuple | None
    if (run_button.value or 0) <= 0:
        training_artifact = None
        if config is None:
            _ = mo.callout(
                "Choose a COLMAP root and submit the form to run training.",
                kind="info",
            )
        else:
            _ = mo.callout(
                "Press 'Run 3DGS training' to start.",
                kind="info",
            )
    elif config is None:
        _ = mo.callout(
            "Choose a COLMAP root and submit the form to run training.",
            kind="info",
        )
        training_artifact = None
    else:
        training_artifact = run_training(config, trainer_bundle)
        if training_artifact is None and config is not None:
            _ = mo.callout(
                "Training did not run. Check form configuration and try again.",
                kind="info",
            )
    return (training_artifact,)


@app.cell
def _():
    benchmark_button = mo.ui.button(
        value=0,
        label="Run dataloader benchmark",
        on_click=lambda value: (0 if value is None else int(value)) + 1,
    )
    mo.vstack([benchmark_button], gap=1.0)
    return (benchmark_button,)


@app.cell
def _(benchmark_button, trainer_bundle):
    if (benchmark_button.value or 0) <= 0:
        _ = mo.callout(
            "Press 'Run dataloader benchmark' to measure loader speed.",
            kind="info",
        )
        dataloader_benchmark = None
    elif trainer_bundle is None:
        _ = mo.callout(
            "Load a valid dataset and build dataloader before benchmarking.",
            kind="warn",
        )
        dataloader_benchmark = None
    else:
        dataloader = trainer_bundle[1]
        dataloader_benchmark = benchmark_dataloader(
            dataloader, measured_steps=1_000
        )
        _ = mo.callout(
            (
                f"Benchmark done. Initialization: {dataloader_benchmark['initialization_ms']:.2f} ms. "
                f"Warmup: {dataloader_benchmark['warmup_ms_per_batch']:.2f} ms/batch "
                f"(steps={dataloader_benchmark['warmup_steps']}). "
                f"Measured: {dataloader_benchmark['ms_per_batch']:.2f} ms/batch, "
                f"{dataloader_benchmark['iters_per_sec']:.2f} it/s "
                f"(steps={dataloader_benchmark['measured_steps']})."
            ),
            kind="success",
        )
    return (dataloader_benchmark,)


@app.cell
def _(dataloader_benchmark):
    dataloader_benchmark
    return


@app.cell
def _(config, load_dataset_from_config):
    dataset = load_dataset_from_config(config)
    return (dataset,)


@app.cell
def _(dataloader_benchmark):
    dataloader_benchmark
    return


@app.cell
def _(build_trainer_bundle, config, dataset):
    trainer_bundle = build_trainer_bundle(config, dataset)
    return (trainer_bundle,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Setup
    """)
    return


@app.cell
def _():
    sk_gsplat.register()
    return


@app.class_definition
class ExecutionConfig(BaseModel):
    """Runtime configuration for notebook-level execution settings."""

    device: Literal["cpu", "cuda"] = "cuda"
    output_dir: Path = Path("debug/gsplat_notebook")
    num_data_workers: int = Field(default=8, ge=0, le=32)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Types
    """)
    return


@app.cell
def _():
    class DataConfig(BaseModel):
        colmap_root: Path = Path(
            "/home/schlack/Documents/3DGS_scenes/360/garden"
        )
        enable_undistort: bool = False
        undistort_output_dir: Path = Path(
            "/tmp/splatkit_gsplat_colmap_undistorted"
        )
        apply_horizon_adjustment: bool = True
        split_target: Literal["all", "train", "val"] = "all"
        materialization_stage: Literal["decoded", "prepared"] = "prepared"
        materialization_mode: Literal["lazy", "eager"] = "eager"
        materialization_num_workers: int = Field(default=0, ge=0, le=32)
        resize_width_target: int = Field(default=1297, ge=1)
        resize_interpolation: Literal[
            "nearest",
            "bilinear",
            "bicubic",
        ] = "bicubic"
        normalize_images: bool = True

    class ModelConfig(BaseModel):
        sh_degree: int = 3
        initial_scale: float = 0.01
        initial_opacity: float = 0.1

    class OptimizationConfig(BaseModel):
        means_lr: float = 1.6e-4
        scales_lr: float = 5e-3
        opacities_lr: float = 5e-2
        quats_lr: float = 1e-3
        feature_lr: float = 2.5e-3

    class TrainingConfig(BaseModel):
        batch_size: int = Field(default=1, ge=1, le=8)
        max_steps: int = Field(default=30_000, ge=1, le=120_000)
        log_every: int = Field(default=250, ge=1, le=120_000)

    class Config(BaseModel):
        execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
        data: DataConfig = Field(default_factory=DataConfig)
        model: ModelConfig = Field(default_factory=ModelConfig)
        optimization: OptimizationConfig = Field(
            default_factory=OptimizationConfig
        )
        training: TrainingConfig = Field(default_factory=TrainingConfig)
        run_immediately: bool = True

    return (Config,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Utilities
    """)
    return


@app.function
def check_uniform_batch(batch) -> bool:
    """Validate that all samples in a batch share image resolution."""
    width = batch.camera.width
    height = batch.camera.height
    return torch.equal(width, width[:1].expand_as(width)) and torch.equal(
        height, height[:1].expand_as(height)
    )


@app.function
def move_batch_to_device(batch, device: torch.device):
    """Move batched images and cameras to the target device."""
    camera = batch.camera
    if camera.intrinsics is None:
        camera_device = replace(
            camera,
            width=camera.width.to(device),
            height=camera.height.to(device),
            fov_degrees=camera.fov_degrees.to(device),
            cam_to_world=camera.cam_to_world.to(device),
        )
    else:
        camera_device = replace(
            camera,
            width=camera.width.to(device),
            height=camera.height.to(device),
            fov_degrees=camera.fov_degrees.to(device),
            cam_to_world=camera.cam_to_world.to(device),
            intrinsics=camera.intrinsics.to(device),
        )
    return replace(
        batch,
        images=batch.images.to(device),
        camera=camera_device,
    )


@app.function
def benchmark_dataloader(
    dataloader, warmup_steps: int = 10, measured_steps: int = 100
):
    """Benchmark dataloader iteration speed with warmup + timed steps."""
    if measured_steps <= 0:
        return {
            "initialization_ms": 0.0,
            "warmup_steps": int(warmup_steps),
            "measured_steps": 0,
            "warmup_ms_per_batch": 0.0,
            "ms_per_batch": 0.0,
            "iters_per_sec": 0.0,
        }

    import time

    init_start = time.perf_counter()
    iterator = iter(dataloader)
    try:
        _ = next(iterator)
    except StopIteration:
        initialization_ms = (time.perf_counter() - init_start) * 1000.0
        return {
            "initialization_ms": float(initialization_ms),
            "warmup_steps": int(warmup_steps),
            "measured_steps": 0,
            "warmup_ms_per_batch": 0.0,
            "ms_per_batch": 0.0,
            "iters_per_sec": 0.0,
        }
    initialization_ms = (time.perf_counter() - init_start) * 1000.0

    warmup_iter = tqdm(
        range(max(0, warmup_steps)),
        desc="dataloader warmup",
        leave=False,
    )
    warmup_start = time.perf_counter()
    for _ in warmup_iter:
        try:
            _ = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            _ = next(iterator)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000.0

    start = time.perf_counter()
    measured_iter = tqdm(
        range(measured_steps),
        desc="dataloader timed",
        leave=False,
    )
    for _ in measured_iter:
        try:
            _ = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            _ = next(iterator)
    elapsed = time.perf_counter() - start

    warmup_time = warmup_ms / max(1, warmup_steps)
    mean_ms = (elapsed / measured_steps) * 1000.0
    return {
        "initialization_ms": float(initialization_ms),
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "warmup_ms_per_batch": float(warmup_time),
        "ms_per_batch": float(mean_ms),
        "iters_per_sec": float(
            measured_steps / elapsed if elapsed > 0 else 0.0
        ),
    }


@app.function
def move_model_to_device_as_leaf(model, device: torch.device):
    """Move the initialized scene tensors to device while keeping leaf params."""
    scene = model.scene
    return replace(
        model,
        scene=replace(
            scene,
            center_position=scene.center_position.to(device)
            .detach()
            .requires_grad_(True),
            log_scales=scene.log_scales.to(device)
            .detach()
            .requires_grad_(True),
            quaternion_orientation=scene.quaternion_orientation.to(device)
            .detach()
            .requires_grad_(True),
            logit_opacity=scene.logit_opacity.to(device)
            .detach()
            .requires_grad_(True),
            feature=scene.feature.to(device).detach().requires_grad_(True),
        ),
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _(Config):
    def _build_split_config(
        target: Literal["all", "train", "val"],
    ) -> SplitConfig:
        if target == "all":
            return SplitConfig(
                target="all",
                every_n=None,
                train_ratio=None,
            )
        if target == "val":
            return SplitConfig(target="val")
        return SplitConfig(target="train")

    def build_colmap_dataset_config(config: Config) -> ColmapDatasetConfig:
        data = config.data
        runtime = DatasetRuntimeConfig(
            split=_build_split_config(data.split_target),
            materialization=MaterializationConfig(
                stage=data.materialization_stage,
                mode=data.materialization_mode,
                num_workers=data.materialization_num_workers,
            ),
        )
        source_pipes = (
            HorizonAlignPipeConfig(
                enabled=data.apply_horizon_adjustment,
            ),
        )
        cache_pipes = (
            ResizePipeConfig(
                width_target=data.resize_width_target,
                interpolation=data.resize_interpolation,
            ),
        )
        prepare_pipes = (NormalizePipeConfig(enabled=data.normalize_images),)
        undistort_output_dir = (
            data.undistort_output_dir.expanduser()
            if data.enable_undistort
            else None
        )
        return ColmapDatasetConfig(
            path=data.colmap_root,
            runtime=runtime,
            source_pipes=source_pipes,
            cache_pipes=cache_pipes,
            prepare_pipes=prepare_pipes,
            undistort_output_dir=undistort_output_dir,
        )

    def build_config_from_form(form) -> Config | None:
        if not str(form.value.data.colmap_root).strip():
            return None

        root = form.value.data.colmap_root.expanduser()
        if not root.exists():
            raise ValueError(f"COLMAP root `{root}` does not exist.")
        if (
            form.value.execution.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA selected but not available in this environment."
            )

        output_dir = form.value.execution.output_dir.expanduser()
        if not output_dir:
            raise ValueError("output_dir must be set.")

        data = form.value.data.model_copy(
            update={
                "colmap_root": root,
                "undistort_output_dir": form.value.data.undistort_output_dir.expanduser(),
            }
        )
        return Config(
            execution=form.value.execution.model_copy(
                update={"output_dir": output_dir}
            ),
            data=data,
            model=form.value.model,
            optimization=form.value.optimization,
            training=form.value.training,
            run_immediately=True,
        )

    return build_colmap_dataset_config, build_config_from_form


@app.cell
def _(build_config_from_form, train_form):
    config = build_config_from_form(train_form)
    return (config,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Pipeline
    """)
    return


@app.function
def build_frame_dataset_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
):
    """Build a dataloader from a prepared frame dataset."""
    if num_workers <= 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_frame_samples,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_frame_samples,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


@app.cell
def _(Config, build_colmap_dataset_config):
    def load_dataset_from_config(config: Config | None):
        if config is None:
            return None
        dataset_config = build_colmap_dataset_config(config)
        dataset = load_dataset(dataset_config)
        scene_dataset = (
            dataset.dataset if hasattr(dataset, "dataset") else dataset
        )
        if scene_dataset.point_cloud is None:
            raise RuntimeError(
                "Loaded COLMAP dataset has no point cloud; this backend needs points."
            )
        return dataset

    return (load_dataset_from_config,)


@app.cell
def _(Config):
    def _get_scene_dataset(dataset):
        return dataset.dataset if hasattr(dataset, "dataset") else dataset

    def build_trainer_bundle(config: Config | None, dataset):
        if config is None or dataset is None:
            return None

        scene_dataset = _get_scene_dataset(dataset)
        if scene_dataset.camera.camera_convention != "opencv":
            raise RuntimeError(
                "gsplat backend in this version expects opencv camera convention."
            )

        if scene_dataset.point_cloud is None:
            raise RuntimeError(
                "Initialized models require point cloud data in the dataset."
            )

        if config.training.batch_size > 1:
            widths = set(
                int(value) for value in scene_dataset.camera.width.tolist()
            )
            heights = set(
                int(value) for value in scene_dataset.camera.height.tolist()
            )
            if len(widths) != 1 or len(heights) != 1:
                raise RuntimeError(
                    "Mixed-resolution mini-batches are not supported with "
                    "batch_size > 1."
                )

        dataloader = build_frame_dataset_dataloader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            num_workers=config.execution.num_data_workers,
        )
        model = sk.initialize_gaussian_model_from_dataset(
            scene_dataset,
            sh_degree=config.model.sh_degree,
            initial_scale=config.model.initial_scale,
            initial_opacity=config.model.initial_opacity,
        )
        return model, dataloader

    return (build_trainer_bundle,)


@app.cell
def _(Config):
    def run_training(config: Config | None, trainer_bundle):
        if (
            config is None
            or trainer_bundle is None
            or not config.run_immediately
        ):
            return None

        device = torch.device(config.execution.device)
        model, dataloader = trainer_bundle
        model = model.to(device)
        model = move_model_to_device_as_leaf(model, device)

        render_options = sk_gsplat.GsplatRenderOptions(
            packed=False,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="classic",
            depth_render_mode="ED",
            eps_2d=0.3,
        )

        optimizers = (
            torch.optim.Adam(
                [model.scene.center_position], lr=config.optimization.means_lr
            ),
            torch.optim.Adam(
                [model.scene.log_scales], lr=config.optimization.scales_lr
            ),
            torch.optim.Adam(
                [model.scene.logit_opacity], lr=config.optimization.opacities_lr
            ),
            torch.optim.Adam(
                [model.scene.quaternion_orientation],
                lr=config.optimization.quats_lr,
            ),
            torch.optim.Adam(
                [model.scene.feature], lr=config.optimization.feature_lr
            ),
        )

        iterator = iter(dataloader)
        loss_history: list[dict[str, float]] = []
        output_dir = config.execution.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        for step in tqdm(range(1, config.training.max_steps + 1)):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            batch = move_batch_to_device(batch, device)
            if config.training.batch_size > 1 and not check_uniform_batch(
                batch
            ):
                raise RuntimeError(
                    "Batch images must all match width/height for gsplat backend "
                    "when batch_size > 1."
                )

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

            render_output = sk.render(
                model.scene,
                batch.camera,
                backend="gsplat",
                return_alpha=False,
                return_depth=False,
                return_normals=False,
                return_2d_projections=False,
                return_projective_intersection_transforms=False,
                options=render_options,
            )
            loss = F.mse_loss(
                render_output.render.permute(0, 3, 1, 2),
                batch.images,
            )
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at step {step}.")

            if (
                step % config.training.log_every == 0
                or step == 1
                or step == config.training.max_steps
            ):
                loss_history.append(
                    {"step": step, "loss": float(loss.detach().cpu().item())}
                )

        final_model_path = output_dir / "model_final.ply"
        save_scene(model.scene, final_model_path)
        return (model, loss_history, final_model_path)

    return (run_training,)


@app.cell(hide_code=True)
def _(training_artifact: tuple | None):
    if training_artifact is not None:
        _ = mo.callout(
            f"Saved final model to {training_artifact[2]}.",
            kind="success",
        )
    else:
        _ = mo.callout(
            "No training artifact yet. Fill the form and run once.",
            kind="info",
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Notes

    - No densification branch is used.
    - A single top-level `Config` now flows through setup → dataloader → training,
      with `data`, `model`, `optimization`, and `training` subconfigs.
    - Data backend behavior is driven by a single `ColmapDatasetConfig` built
      from the UI `DataConfig` in `build_colmap_dataset_config`.
    - This notebook keeps batch checks strict for backend compatibility.
    - Defaults intentionally mirror the simple gsplat training baseline.
    """)
    return


if __name__ == "__main__":
    app.run()
