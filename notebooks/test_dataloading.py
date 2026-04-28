"""Interactive COLMAP dataset loading notebook."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import ctypes
    import os
    import signal
    import sys
    from pathlib import Path

    import cv2
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
    class DatasetConfig(BaseModel):
        colmap_root: Path = Path(
            os.environ.get(
                "EMBER_COLMAP_ROOT",
                str(sk.get_sample_scene_path()),
            )
        )
        write_undistorted_cache: bool = False
        undistort_output_dir: Path = Path(
            os.environ.get(
                "EMBER_COLMAP_UNDISTORTED",
                str(sk.get_sample_scene_path() / "undistorted"),
            )
        )
        apply_horizon_adjustment: bool = True

    class PointCloudRenderConfig(BaseModel):
        max_points: int = Field(default=50000, ge=1000, le=500000)
        point_radius: int = Field(default=2, ge=1, le=6)
        background_brightness: int = Field(default=18, ge=0, le=255)

    dataset_form = form_gui(
        DatasetConfig,
        value=DatasetConfig(),
        label="COLMAP Dataset",
        live_update=False,
    )
    point_cloud_form = form_gui(
        PointCloudRenderConfig,
        value=PointCloudRenderConfig(),
        label="Point Cloud Render",
        live_update=True,
    )
    return dataset_form, point_cloud_form


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Test Dataloading

    "
        "Load a COLMAP dataset, inspect its sparse point cloud through the "
        "`marimo-3dv` viewer, and jump directly to individual training frames.
    """)
    return


@app.cell
def _(dataset_form, point_cloud_form):
    mo.vstack([dataset_form, point_cloud_form], gap=1.0)
    return


@app.cell
def _():
    viewer_state = ViewerState(camera_convention="opencv")
    return (viewer_state,)


@app.function
def load_colmap_dataset_from_form(load_config):
    """Load a COLMAP dataset from a form config."""
    if not load_config.colmap_root.exists():
        return None
    horizon_adjustment = sk.HorizonAdjustmentSpec(
        enabled=load_config.apply_horizon_adjustment
    )
    undistort_output_dir = (
        load_config.undistort_output_dir
        if load_config.write_undistorted_cache
        else None
    )
    return sk.load_colmap_dataset(
        load_config.colmap_root,
        undistort_output_dir=undistort_output_dir,
        horizon_adjustment=horizon_adjustment,
    )


@app.cell
def _(dataset_form):
    dataset = load_colmap_dataset_from_form(dataset_form.value)
    return (dataset,)


@app.cell(hide_code=True)
def _(dataset):
    if dataset is None:
        summary = mo.callout(
            "Choose a COLMAP dataset root and submit the form to load it.",
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
                f"Loaded `{dataset.root_path}` with {dataset.num_frames} frames "
                f"and {num_points} sparse points."
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
            "FrameSelectionConfig",
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
        mo.callout("Load a dataset to enable frame navigation.", kind="warn")
        if frame_form is None
        else frame_form
    )
    frame_form_ui
    return


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


if __name__ == "__main__":
    app.run()
