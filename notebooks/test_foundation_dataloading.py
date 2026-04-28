"""Interactive foundation-model dataset loading notebook."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import ctypes
    import signal
    import sys
    from pathlib import Path

    import ember_core as sk
    import marimo as mo
    import numpy as np
    import torch
    from marimo_3dv import CameraState, Viewer, ViewerState
    from marimo_config_gui import form_gui
    from pydantic import BaseModel, ConfigDict, Field, create_model

    def install_linux_parent_death_signal() -> None:
        """Ensure notebook child processes exit with the parent on Linux."""
        if sys.platform != "linux":
            return

        libc = ctypes.CDLL(None, use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0)

    install_linux_parent_death_signal()


@app.cell
def _():
    class ArtifactConfig(BaseModel):
        artifact_path: Path = Path(
            "/home/schlack/Documents/3DGS_scenes/360/garden/must3r_output"
        )
        image_root: Path = Path(
            "/home/schlack/Documents/3DGS_scenes/360/garden/images"
        )
        apply_horizon_adjustment: bool = True

    class RunConfig(BaseModel):
        image_dir: Path = Path(
            "/home/schlack/Documents/3DGS_scenes/360/garden/images"
        )
        output_dir: Path = Path(
            "/home/schlack/Documents/3DGS_scenes/360/garden/must3r_output"
        )
        checkpoint_repo_id: str = ""
        checkpoint_filename: str = "MUSt3R_512.pth"
        image_size: int = Field(default=512, ge=160, le=1024)
        device: str = "cuda"
        apply_horizon_adjustment: bool = True

    class PointCloudRenderConfig(BaseModel):
        max_points: int = Field(default=50000, ge=1000, le=500000)
        point_radius: int = Field(default=2, ge=1, le=6)
        background_brightness: int = Field(default=18, ge=0, le=255)

    artifact_form = form_gui(
        ArtifactConfig,
        value=ArtifactConfig(),
        label="MUSt3R Artifact",
        live_update=False,
    )
    run_form = form_gui(
        RunConfig,
        value=RunConfig(),
        label="Run MUSt3R",
        live_update=False,
    )
    point_cloud_form = form_gui(
        PointCloudRenderConfig,
        value=PointCloudRenderConfig(),
        label="Point Cloud Render",
        live_update=True,
    )
    return artifact_form, point_cloud_form, run_form


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Test Foundation Dataloading

    "
        "Load a MUSt3R reconstruction artifact or run the MUSt3R adapter, "
        "inspect the predicted point cloud, jump to any recovered frame, and "
        "compare it with the GT image.
    """)
    return


@app.cell
def _():
    source_mode = mo.ui.dropdown(
        ["artifact", "run"],
        value="artifact",
        label="Foundation-model source",
        full_width=True,
    )
    return (source_mode,)


@app.cell
def _(artifact_form, point_cloud_form, run_form, source_mode):
    active_form = artifact_form if source_mode.value == "artifact" else run_form
    mo.vstack([source_mode, active_form, point_cloud_form], gap=1.0)
    return


@app.cell
def _():
    viewer_state = ViewerState(camera_convention="opencv")
    return (viewer_state,)


@app.function
def load_must3r_dataset_from_artifact(load_config):
    """Load an existing MUSt3R artifact."""
    if not load_config.artifact_path.exists():
        return None
    scene_record = sk.load_must3r_scene_record(
        load_config.artifact_path,
        image_root=load_config.image_root,
    )
    if load_config.apply_horizon_adjustment:
        return sk.adjust_scene_record_horizon(
            scene_record,
            sk.HorizonAdjustmentSpec(enabled=True),
        )
    return scene_record


@app.function
def run_must3r_dataset_from_form(load_config):
    """Run the MUSt3R adapter from a notebook form config."""
    if not load_config.image_dir.exists():
        return None
    if not load_config.checkpoint_repo_id:
        return None
    scene_record = sk.run_must3r_scene_record(
        load_config.image_dir,
        output_dir=load_config.output_dir,
        checkpoint_repo_id=load_config.checkpoint_repo_id,
        checkpoint_filename=load_config.checkpoint_filename,
        image_size=load_config.image_size,
        device=load_config.device,
    )
    if load_config.apply_horizon_adjustment:
        return sk.adjust_scene_record_horizon(
            scene_record,
            sk.HorizonAdjustmentSpec(enabled=True),
        )
    return scene_record


@app.cell
def _(artifact_form, run_form, source_mode):
    if source_mode.value == "artifact":
        dataset = load_must3r_dataset_from_artifact(artifact_form.value)
        dataset_status = (
            "Choose an existing MUSt3R artifact path and submit the form."
            if dataset is None
            else None
        )
    else:
        dataset = run_must3r_dataset_from_form(run_form.value)
        if not run_form.value.image_dir.exists():
            dataset_status = "Choose an image directory and submit the form."
        elif not run_form.value.checkpoint_repo_id:
            dataset_status = (
                "Set `checkpoint_repo_id` to enable the MUSt3R run path."
            )
        elif dataset is None:
            dataset_status = "Submit the form to run the MUSt3R adapter."
        else:
            dataset_status = None
    return dataset, dataset_status


@app.cell(hide_code=True)
def _(dataset, dataset_status, source_mode):
    if dataset is None:
        summary = mo.callout(
            dataset_status or "No foundation-model dataset is loaded.",
            kind="warn",
        )
    else:
        num_points = (
            0
            if dataset.point_cloud is None
            else dataset.point_cloud.points.shape[0]
        )
        summary = mo.callout(
            (
                f"Loaded `{source_mode.value}` dataset `{dataset.root_path}` with "
                f"{dataset.num_frames} frames and {num_points} points."
            ),
            kind="success",
        )
    summary
    return


@app.cell
def _(dataset):
    if dataset is None:
        frame_form = None
    else:
        frame_count = dataset.num_frames
        FrameSelectionConfig = create_model(
            "FoundationFrameSelectionConfig",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            frame_index=(
                int,
                Field(default=0, ge=0, le=max(0, frame_count - 1)),
            ),
        )
        frame_form = form_gui(
            FrameSelectionConfig,
            value=FrameSelectionConfig(frame_index=0),
            label="Frame",
            live_update=True,
        )
    return (frame_form,)


@app.cell
def _(frame_form):
    frame_form_ui = (
        mo.callout(
            "Load a foundation-model dataset to enable frame navigation.",
            kind="warn",
        )
        if frame_form is None
        else frame_form
    )
    frame_form_ui
    return


@app.cell
def _(dataset, frame_form):
    if dataset is None or frame_form is None:
        selected_frame = None
        selected_frame_index = None
    else:
        selected_frame_index = frame_form.value.frame_index
        selected_frame = dataset.frames[selected_frame_index]
    return selected_frame, selected_frame_index


@app.function
def dataset_frame_to_viewer_camera(dataset, frame_index):
    """Convert one dataset frame camera into a marimo-3dv camera state."""
    backend_camera = sk.CameraState(
        width=dataset.camera.width[frame_index : frame_index + 1],
        height=dataset.camera.height[frame_index : frame_index + 1],
        fov_degrees=dataset.camera.fov_degrees[frame_index : frame_index + 1],
        cam_to_world=dataset.camera.cam_to_world[frame_index : frame_index + 1],
        intrinsics=(
            None
            if dataset.camera.intrinsics is None
            else dataset.camera.intrinsics[frame_index : frame_index + 1]
        ),
        camera_convention=dataset.camera.camera_convention,
        up_direction=dataset.camera.up_direction,
    )
    return CameraState(
        fov_degrees=float(backend_camera.fov_degrees[0].item()),
        width=int(backend_camera.width[0].item()),
        height=int(backend_camera.height[0].item()),
        cam_to_world=backend_camera.cam_to_world[0].cpu().numpy(),
        camera_convention="opencv",
    )


@app.cell
def _(dataset, selected_frame_index):
    if dataset is None or selected_frame_index is None:
        selected_camera = CameraState.default(camera_convention="opencv")
    else:
        selected_camera = dataset_frame_to_viewer_camera(
            dataset,
            selected_frame_index,
        )
    return (selected_camera,)


@app.cell
def _(selected_camera, viewer_state):
    viewer_state.set_camera(selected_camera)
    return


@app.function
def point_cloud_buffers(dataset, max_points):
    """Move point cloud tensors to the active render device with caching."""
    if dataset is None or dataset.point_cloud is None:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (
        str(dataset.root_path),
        int(dataset.point_cloud.points.shape[0]),
        int(max_points),
        device.type,
    )
    cache = getattr(point_cloud_buffers, "_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    points = dataset.point_cloud.points
    if dataset.point_cloud.colors is None:
        colors = torch.full_like(points, 255.0)
    else:
        colors = torch.clamp(dataset.point_cloud.colors * 255.0, 0.0, 255.0)
    if points.shape[0] > max_points:
        selection = torch.linspace(
            0,
            points.shape[0] - 1,
            steps=max_points,
            dtype=torch.int64,
        )
        points = points[selection]
        colors = colors[selection]
    cache[cache_key] = (
        points.to(device=device, dtype=torch.float32),
        colors.to(device=device, dtype=torch.float32),
    )
    point_cloud_buffers._cache = cache
    return cache[cache_key]


@app.function
def point_cloud_render(
    camera_state,
    dataset,
    render_config,
):
    """Render a sparse point cloud from the current viewer camera."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.full(
        (camera_state.height, camera_state.width, 3),
        render_config.background_brightness,
        dtype=torch.uint8,
        device=device,
    )
    point_cloud = point_cloud_buffers(dataset, render_config.max_points)
    if point_cloud is None:
        return image.cpu().numpy()

    points, colors = point_cloud
    if points.shape[0] == 0:
        return image.cpu().numpy()
    camera_opencv = camera_state.with_convention("opencv")
    world_to_camera = torch.linalg.inv(
        torch.as_tensor(
            camera_opencv.cam_to_world,
            dtype=torch.float32,
            device=device,
        )
    )
    homogeneous_points = torch.cat(
        [
            points,
            torch.ones(
                (points.shape[0], 1), dtype=torch.float32, device=device
            ),
        ],
        dim=1,
    )
    camera_points = (world_to_camera @ homogeneous_points.T).T[:, :3]
    depth = camera_points[:, 2]
    valid = depth > 1e-4
    if not bool(valid.any()):
        return image.cpu().numpy()

    fx = (camera_opencv.width / 2.0) / np.tan(
        np.deg2rad(camera_opencv.fov_degrees) / 2.0
    )
    fy = fx
    cx = camera_opencv.width / 2.0
    cy = camera_opencv.height / 2.0

    x = camera_points[valid, 0]
    y = camera_points[valid, 1]
    z = depth[valid]
    u = torch.round(fx * (x / z) + cx).to(torch.int64)
    v = torch.round(fy * (y / z) + cy).to(torch.int64)
    inside = (
        (u >= 0)
        & (u < camera_opencv.width)
        & (v >= 0)
        & (v < camera_opencv.height)
    )
    if not bool(inside.any()):
        return image.cpu().numpy()

    u_valid = u[inside]
    v_valid = v[inside]
    z_valid = z[inside]
    color_valid = colors[valid][inside].to(torch.uint8)
    order = torch.argsort(z_valid, descending=True)
    radius = render_config.point_radius
    for delta_y in range(-radius + 1, radius):
        for delta_x in range(-radius + 1, radius):
            if delta_x * delta_x + delta_y * delta_y >= radius * radius:
                continue
            draw_u = u_valid[order] + delta_x
            draw_v = v_valid[order] + delta_y
            draw_inside = (
                (draw_u >= 0)
                & (draw_u < camera_opencv.width)
                & (draw_v >= 0)
                & (draw_v < camera_opencv.height)
            )
            if not bool(draw_inside.any()):
                continue
            image[draw_v[draw_inside], draw_u[draw_inside]] = color_valid[
                order
            ][draw_inside]
    return image.cpu().numpy()


@app.cell
def _(dataset, point_cloud_form, viewer_state):
    def render_frame(camera_state):
        return point_cloud_render(
            camera_state,
            dataset,
            point_cloud_form.value,
        )

    viewer = Viewer(render_frame, state=viewer_state)
    return (viewer,)


@app.cell
def _(selected_frame):
    gt_image = (
        mo.callout(
            "Select a frame to show the ground-truth image.", kind="warn"
        )
        if selected_frame is None
        else mo.image(
            selected_frame.image_path,
            width="100%",
            caption=f"GT image: {selected_frame.image_path.name}",
        )
    )
    return (gt_image,)


@app.cell
def _(selected_frame, selected_frame_index):
    frame_info = (
        mo.callout("No frame selected.", kind="warn")
        if selected_frame is None or selected_frame_index is None
        else mo.md(
            f"""
            **Selected frame:** `{selected_frame_index}`  
            **Frame id:** `{selected_frame.frame_id}`  
            **Image:** `{selected_frame.image_path}`
            """
        )
    )
    return (frame_info,)


@app.cell
def _(frame_info, gt_image, viewer):
    layout = mo.hstack(
        [
            mo.vstack([viewer], gap=0.5),
            mo.vstack([frame_info, gt_image], gap=0.75),
        ],
        widths=[2.2, 1.0],
        align="start",
        gap=1.0,
    )
    layout
    return


if __name__ == "__main__":
    app.run()
