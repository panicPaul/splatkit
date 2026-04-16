"""Simple COLMAP 3DGS training notebook using splatkit + gsplat backend."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    from dataclasses import dataclass, replace
    from pathlib import Path
    from typing import Literal

    import marimo as mo
    import splatkit as sk
    import splatkit_backends.gsplat as sk_gsplat
    import torch
    import torch.nn.functional as F
    from marimo_3dv import form_gui
    from pydantic import BaseModel, Field
    from splatkit.io.scene import save_scene
    from splatkit.training.config import BatchingConfig
    from splatkit.training.runtime import build_dataloader


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # gsplat COLMAP Trainer

    Minimal 3DGS training notebook for COLMAP reconstructions using
    `splatkit` data loading + initialization and
    `splatkit-backends.gsplat` rendering.

    This omits densification by design and keeps the loop explicit to stress
    backend behavior.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Setup
    """)
    return


@app.cell
def _(sk_gsplat):
    sk_gsplat.register()
    return


@app.cell
def _():
    DEFAULTS = {
        "sh_degree": 3,
        "initial_scale": 0.01,
        "initial_opacity": 0.1,
        "means_lr": 1.6e-4,
        "scales_lr": 5e-3,
        "opacities_lr": 5e-2,
        "quats_lr": 1e-3,
        "feature_lr": 2.5e-3,
        "log_every": 250,
        "max_steps": 30_000,
        "batch_size": 1,
    }
    return (DEFAULTS,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Types
    """)
    return


@app.cell
def _():
    @dataclass(frozen=True)
    class _LoaderConfig:
        batching: BatchingConfig

    return (_LoaderConfig,)


@app.cell
def _():
    @dataclass(frozen=True)
    class _TrainerConfig:
        colmap_root: Path
        output_dir: Path
        device: str
        batch_size: int
        max_steps: int
        sh_degree: int
        initial_scale: float
        initial_opacity: float
        means_lr: float
        scales_lr: float
        opacities_lr: float
        quats_lr: float
        feature_lr: float
        log_every: int
        resize_max_long_edge: int | None
        undistort: bool
        undistort_output_dir: Path | None
        apply_horizon_adjustment: bool
        run_immediately: bool

    return (_TrainerConfig,)


@app.cell(hide_code=True)
def _(DEFAULTS):
    class TrainerConfigModel(BaseModel):
        colmap_root: Path = Path()
        output_dir: Path = Path("/tmp/splatkit_gsplat_colmap_run")
        device: Literal["cpu", "cuda"] = "cuda"
        batch_size: int = Field(default=DEFAULTS["batch_size"], ge=1, le=8)
        max_steps: int = Field(default=DEFAULTS["max_steps"], ge=1, le=120_000)
        log_every: int = Field(default=DEFAULTS["log_every"], ge=1, le=120_000)
        resize_max_long_edge: int = Field(default=0, ge=0, le=4_096)
        undistort: bool = False
        undistort_output_dir: Path = Path(
            "/tmp/splatkit_gsplat_colmap_undistorted"
        )
        apply_horizon_adjustment: bool = False

    return (TrainerConfigModel,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Utilities
    """)
    return


@app.cell
def _():
    def _check_uniform_batch(batch) -> bool:
        width = batch.camera.width
        height = batch.camera.height
        return torch.equal(width, width[:1].expand_as(width)) and torch.equal(
            height, height[:1].expand_as(height)
        )

    return (_check_uniform_batch,)


@app.cell
def _():
    def _move_batch_to_device(batch, device: torch.device):
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

    return (_move_batch_to_device,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Inputs
    """)
    return


@app.cell
def _(TrainerConfigModel):
    train_form = form_gui(
        TrainerConfigModel,
        value=TrainerConfigModel(),
        label="COLMAP Trainer",
        live_update=False,
    )
    return (train_form,)


@app.cell
def _(train_form):
    mo.vstack([train_form], gap=1.0)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _(DEFAULTS):
    def build_config_from_form(form) -> _TrainerConfig | None:
        if not str(form.value.colmap_root).strip():
            return None

        root = form.value.colmap_root.expanduser()
        if not root.exists():
            raise ValueError(f"COLMAP root `{root}` does not exist.")
        if form.value.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA selected but not available in this environment."
            )

        resize = (
            None
            if int(form.value.resize_max_long_edge) == 0
            else int(form.value.resize_max_long_edge)
        )
        if form.value.batch_size > 1 and resize is None:
            _ = mo.callout(
                "Batch size > 1 is supported only when image resolutions are uniform.",
                kind="warn",
            )

        return _TrainerConfig(
            colmap_root=root,
            output_dir=form.value.output_dir.expanduser(),
            device=form.value.device,
            batch_size=int(form.value.batch_size),
            max_steps=int(form.value.max_steps),
            sh_degree=DEFAULTS["sh_degree"],
            initial_scale=DEFAULTS["initial_scale"],
            initial_opacity=DEFAULTS["initial_opacity"],
            means_lr=DEFAULTS["means_lr"],
            scales_lr=DEFAULTS["scales_lr"],
            opacities_lr=DEFAULTS["opacities_lr"],
            quats_lr=DEFAULTS["quats_lr"],
            feature_lr=DEFAULTS["feature_lr"],
            log_every=int(form.value.log_every),
            resize_max_long_edge=resize,
            undistort=bool(form.value.undistort),
            undistort_output_dir=(
                form.value.undistort_output_dir.expanduser()
                if form.value.undistort
                else None
            ),
            apply_horizon_adjustment=bool(form.value.apply_horizon_adjustment),
            run_immediately=True,
        )

    return (build_config_from_form,)


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


@app.cell
def _(sk):
    def load_dataset_from_config(config: _TrainerConfig | None):
        if config is None:
            return None

        horizon_adjustment = (
            sk.HorizonAdjustmentSpec(enabled=config.apply_horizon_adjustment)
            if config.apply_horizon_adjustment
            else None
        )
        dataset = sk.load_colmap_dataset(
            config.colmap_root,
            undistort_output_dir=(
                config.undistort_output_dir if config.undistort else None
            ),
            horizon_adjustment=horizon_adjustment,
        )
        if dataset.point_cloud is None:
            raise RuntimeError(
                "Loaded COLMAP dataset has no point cloud; this backend needs points."
            )
        return dataset

    return (load_dataset_from_config,)


@app.cell
def _(config, load_dataset_from_config):
    dataset = load_dataset_from_config(config)
    return (dataset,)


@app.cell
def _(_LoaderConfig, BatchingConfig, build_dataloader, sk):
    def build_trainer_bundle(config, dataset):
        if config is None or dataset is None:
            return None

        if dataset.camera.camera_convention != "opencv":
            raise RuntimeError(
                "gsplat backend in this version expects opencv camera convention."
            )

        if config.batch_size > 1:
            widths = set(int(value) for value in dataset.camera.width.tolist())
            heights = set(
                int(value) for value in dataset.camera.height.tolist()
            )
            if len(widths) != 1 or len(heights) != 1:
                raise RuntimeError(
                    "Mixed-resolution mini-batches are not supported with "
                    "batch_size > 1."
                )

        batching_config = BatchingConfig(
            batch_size=config.batch_size,
            shuffle=True,
            normalize=True,
            resize_max_long_edge=config.resize_max_long_edge,
            interpolation="lanczos",
        )
        dataloader = build_dataloader(
            dataset,
            _LoaderConfig(batching=batching_config),
        )
        model = sk.initialize_gaussian_model_from_dataset(
            dataset,
            sh_degree=config.sh_degree,
            initial_scale=config.initial_scale,
            initial_opacity=config.initial_opacity,
        )
        return model, dataloader

    return (build_trainer_bundle,)


@app.cell
def _(config, dataset, build_trainer_bundle):
    trainer_bundle = build_trainer_bundle(config, dataset)
    return (trainer_bundle,)


@app.cell
def _(
    _check_uniform_batch,
    _move_batch_to_device,
    F,
    save_scene,
    sk,
    sk_gsplat,
    torch,
):
    def run_training(config, trainer_bundle):
        if (
            config is None
            or trainer_bundle is None
            or not config.run_immediately
        ):
            return None

        model, dataloader = trainer_bundle
        model = model.to(torch.device(config.device))

        render_options = sk_gsplat.GsplatRenderOptions(
            packed=False,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="classic",
            depth_render_mode="ED",
            eps_2d=0.3,
        )

        optimizers = (
            torch.optim.Adam([model.scene.center_position], lr=config.means_lr),
            torch.optim.Adam([model.scene.log_scales], lr=config.scales_lr),
            torch.optim.Adam(
                [model.scene.logit_opacity], lr=config.opacities_lr
            ),
            torch.optim.Adam(
                [model.scene.quaternion_orientation],
                lr=config.quats_lr,
            ),
            torch.optim.Adam([model.scene.feature], lr=config.feature_lr),
        )

        iterator = iter(dataloader)
        loss_history: list[dict[str, float]] = []
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        for step in range(1, config.max_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            batch = _move_batch_to_device(batch, torch.device(config.device))
            if config.batch_size > 1 and not _check_uniform_batch(batch):
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
                step % config.log_every == 0
                or step == 1
                or step == config.max_steps
            ):
                loss_history.append(
                    {"step": step, "loss": float(loss.detach().cpu().item())}
                )

        final_model_path = output_dir / "model_final.ply"
        save_scene(model.scene, final_model_path)
        return (model, loss_history, final_model_path)

    return (run_training,)


@app.cell
def _(config, trainer_bundle, run_training):
    training_artifact = run_training(config, trainer_bundle)
    if training_artifact is None and config is None:
        _ = mo.callout(
            "Choose a COLMAP root and submit the form to run training.",
            kind="info",
        )
    return (training_artifact,)


@app.cell(hide_code=True)
def _(training_artifact):
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
    - This notebook keeps batch checks strict for backend compatibility.
    - Defaults intentionally mirror the simple gsplat training baseline.
    """)
    return


if __name__ == "__main__":
    app.run()
