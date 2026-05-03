"""TensorBoard checkpoint analysis notebook."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")

with app.setup:
    from pathlib import Path

    import ember_native_3dgrt
    import ember_native_faster_gs
    import ember_native_faster_gs_mojo
    import ember_native_svraster
    import marimo as mo
    import numpy as np
    import torch
    from ember_core.core import CameraState
    from marimo_3dv import Viewer, ViewerState
    from marimo_config_gui import create_config_gui
    from pydantic import BaseModel, Field

    ember_native_faster_gs.register()
    ember_native_faster_gs_mojo.register()
    ember_native_svraster.register()
    ember_native_3dgrt.register()


    class TensorBoardAnalysisConfig(BaseModel):
        """Notebook inputs for checkpoint-local TensorBoard logs."""

        checkpoint_dir: Path = Field(
            default=Path(
                "checkpoints/papers/fastergs/garden_baseline/"
                "faster_gs.core_run_1"
            ),
            description="Checkpoint run directory or logs directory.",
        )
        chart_title: str = "Selected scalar metrics"
        load_checkpoint: bool = Field(
            default=False,
            description="Load checkpoint metadata in addition to TensorBoard scalars.",
        )


@app.cell(hide_code=True)
def _():
    mo.md("""
    # TensorBoard checkpoint analysis
    """)
    return


@app.cell
def _():
    config_gui = create_config_gui(
        TensorBoardAnalysisConfig,
        label="TensorBoard analysis config",
        nested_models_multiple_open=False,
    )
    return (config_gui,)


@app.cell
def _(config_gui):
    current_config = config_gui.validated_config()
    return (current_config,)


@app.cell(hide_code=True)
def _(config_gui):
    config_gui.stacked()
    return


@app.cell(hide_code=True)
def _(tags):
    default_tags = ["train/loss"] if "train/loss" in tags else []
    metric_selector = mo.ui.multiselect(
        options=list(tags),
        value=default_tags,
        label="Metrics",
        full_width=True,
    )
    metric_selector
    return (metric_selector,)


@app.cell(hide_code=True)
def _(current_config, metric_selector, scalar_frame):
    from ember_splatting_training.tensorboard_analysis import scalar_line_chart

    scalar_line_chart(
        scalar_frame,
        tags=metric_selector.value,
        title=current_config.chart_title,
    )
    return


@app.cell
def _(scalar_frame):
    import polars as pl

    metric_summary = (
        scalar_frame.group_by("tag")
        .agg(
            pl.len().alias("points"),
            pl.min("step").alias("first_step"),
            pl.max("step").alias("last_step"),
            pl.min("value").alias("min_value"),
            pl.max("value").alias("max_value"),
        )
        .sort("tag")
        if not scalar_frame.is_empty()
        else scalar_frame.select("tag")
    )
    return (metric_summary,)


@app.cell(hide_code=True)
def _(metric_summary):
    metric_summary
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""
    ## Scalar Data
    """)
    return


@app.cell
def _(current_config):
    from ember_splatting_training.tensorboard_analysis import (
        empty_scalar_frame,
        find_event_files,
        read_scalars,
    )

    checkpoint_dir = current_config.checkpoint_dir.expanduser()
    event_files = find_event_files(checkpoint_dir)
    scalar_frame = read_scalars(checkpoint_dir) if event_files else empty_scalar_frame()
    return checkpoint_dir, event_files, scalar_frame


@app.cell
def _(scalar_frame):
    from ember_splatting_training.tensorboard_analysis import scalar_tags

    tags = scalar_tags(scalar_frame)
    return (tags,)


@app.cell
def _(checkpoint_dir, event_files, metric_summary, scalar_frame):
    summary = mo.md(
        "\n".join(
            [
                f"Checkpoint/log path: `{checkpoint_dir}`",
                f"Event files: `{len(event_files)}`",
                f"Scalar rows: `{scalar_frame.height}`",
                f"Metrics: `{metric_summary.height}`",
            ]
        )
    )
    summary
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md("""
    ## Checkpoint
    """)
    return


@app.cell
def _(checkpoint_dir, current_config):
    from ember_splatting_training.tensorboard_analysis import load_checkpoint

    checkpoint_files = (
        checkpoint_dir / "config.json",
        checkpoint_dir / "metadata.json",
        checkpoint_dir / "model.ckpt",
    )
    checkpoint_error = ""
    if current_config.load_checkpoint and all(
        checkpoint_file.exists() for checkpoint_file in checkpoint_files
    ):
        try:
            checkpoint = load_checkpoint(checkpoint_dir)
        except Exception as error:
            checkpoint = None
            checkpoint_error = str(error)
    else:
        checkpoint = None
    return checkpoint, checkpoint_error


@app.cell
def _():
    viewer_state = ViewerState(camera_convention="opencv")
    return (viewer_state,)


@app.cell
def _(checkpoint):
    if checkpoint is None:
        render_device = torch.device("cpu")
        render_model = None
    else:
        requested_device = torch.device(checkpoint.config.runtime.device)
        render_device = (
            requested_device
            if requested_device.type != "cuda" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        render_model = checkpoint.model.to(render_device)
    return render_device, render_model


@app.cell
def _(checkpoint, checkpoint_error):
    if checkpoint is None:
        checkpoint_view = mo.md(
            f"Checkpoint not loaded: `{checkpoint_error}`"
            if checkpoint_error
            else "No checkpoint loaded."
        )
    else:
        checkpoint_view = mo.md(
            "\n".join(
                [
                    f"Backend: `{checkpoint.config.render.backend}`",
                    f"Scene: `{type(checkpoint.model.scene).__name__}`",
                    f"Step: `{checkpoint.model.metadata.get('checkpoint_step', 0)}`",
                ]
            )
        )
    checkpoint_view
    return


@app.cell(hide_code=True)
def _(checkpoint, render_device, render_model, viewer_state):
    def render_checkpoint(camera):
        if checkpoint is None or render_model is None:
            return np.full(
                (camera.height, camera.width, 3),
                245,
                dtype=np.uint8,
            )
        core_camera = CameraState(
            width=torch.tensor([camera.width], dtype=torch.int64, device=render_device),
            height=torch.tensor(
                [camera.height],
                dtype=torch.int64,
                device=render_device,
            ),
            fov_degrees=torch.tensor(
                [camera.fov_degrees],
                dtype=torch.float32,
                device=render_device,
            ),
            cam_to_world=torch.as_tensor(
                camera.cam_to_world,
                dtype=torch.float32,
                device=render_device,
            )[None],
            camera_convention=camera.camera_convention,
        )
        with torch.no_grad():
            render_output = checkpoint.render_fn(render_model, core_camera)
        image = render_output.render[0].detach().clamp(0.0, 1.0).cpu().numpy()
        return image

    viewer = Viewer(render_checkpoint, state=viewer_state)
    viewer
    return


if __name__ == "__main__":
    app.run()
