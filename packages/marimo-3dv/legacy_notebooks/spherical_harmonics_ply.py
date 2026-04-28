"""PLY-backed spherical harmonics inspection notebook."""

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns")

with app.setup:
    from dataclasses import replace
    from math import isqrt
    from pathlib import Path
    from typing import TypedDict

    import marimo as mo
    import nerfview
    import numpy as np
    import numpy.typing as npt
    import torch
    from plyfile import PlyData
    from pydantic import BaseModel, Field

    from marimo_viser import (
        apply_rotation_to_quaternions,
        apply_rotation_to_sh_coefficients,
        apply_scale_to_log_scales,
        apply_to_points,
        compose_transforms,
        form_gui,
        pca_transform_from_points,
        viser_marimo,
    )
    from marimo_viser.notebooks.spherical_harmonics import render_spheres


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Spherical Harmonics from PLY

    This notebook uses the same Gaussian-splat loading flow as the PLY viewer
    notebook, then lets you click a splat to inspect that Gaussian's spherical
    harmonics in the SH viewer.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Loading
    """)
    return


@app.class_definition
class GaussianSceneData(TypedDict):
    centers: npt.NDArray[np.float32]
    sh_coefficients: npt.NDArray[np.float32]
    degree_to_use: int
    rgbs: npt.NDArray[np.float32]
    opacities: npt.NDArray[np.float32]
    log_scales: npt.NDArray[np.float32]
    rotations: npt.NDArray[np.float32]


@app.cell
def _(
    pick_point_toggle,
    point_server,
    point_status,
    selection_state: SelectionState,
    sh_status,
    sh_view_state,
    sh_viewer,
    viewer_config,
):
    def handle_point_pick(event: object) -> None:
        """Select the Gaussian center nearest to the clicked scene ray."""
        _scene_data = selection_state["scene_data"]
        if _scene_data is None:
            return
        if event.ray_origin is None or event.ray_direction is None:
            return

        selected_index = selected_point_index_from_ray(
            _scene_data["centers"],
            np.asarray(event.ray_origin, dtype=np.float32),
            np.asarray(event.ray_direction, dtype=np.float32),
        )
        if selected_index is None:
            return

        selection_state["selected_index"] = selected_index
        sh_view_state["coefficients"] = torch.as_tensor(
            _scene_data["sh_coefficients"][selected_index],
            dtype=torch.float32,
        )
        sh_view_state["degrees_to_use"] = min(
            _scene_data["degree_to_use"],
            viewer_config.max_sh_degree if viewer_config is not None else 4,
        )

        point_server.scene.add_point_cloud(
            "/ply_gaussians/selected_point",
            points=_scene_data["centers"][[selected_index]],
            colors=np.array([[255, 255, 0]], dtype=np.uint8),
            point_size=selection_marker_size(_scene_data["log_scales"])
            * (
                viewer_config.selection_size
                if viewer_config is not None
                else 4.0
            ),
            point_shape="sparkle",
        )

        _status_markdown = selection_markdown(
            _scene_data,
            selected_index,
            sh_view_state["degrees_to_use"],
        )
        point_status.content = _status_markdown
        sh_status.content = _status_markdown
        sh_viewer.rerender(None)
        pick_point_toggle.value = False

    @pick_point_toggle.on_update
    def _(_event: object) -> None:
        """Enable scene picking only while Gaussian-pick mode is active."""
        if pick_point_toggle.value:
            point_server.scene.on_pointer_event("click")(handle_point_pick)
        elif point_server.scene._scene_pointer_cb is not None:
            point_server.scene.remove_pointer_callback()

    return


@app.class_definition
class LoadConfig(BaseModel):
    source: Path = Field(
        default=Path.cwd() / "point_cloud.ply",
        description="Select a 3DGS-style `.ply` file to load.",
    )
    normalize_scene: bool = Field(
        default=True,
        description="Apply PCA-based scene normalization before rendering.",
    )


@app.class_definition
class ViewerConfig(BaseModel):
    max_sh_degree: int = Field(
        default=3,
        ge=0,
        le=4,
        description="Maximum SH degree to display for the selected Gaussian.",
    )
    selection_size: float = Field(
        default=4.0,
        ge=1.0,
        le=20.0,
        description="Multiplier for the selected Gaussian marker size.",
    )


@app.class_definition
class SelectionState(TypedDict):
    scene_data: GaussianSceneData | None
    selected_index: int | None


@app.function
def infer_sh_degree(num_bases: int) -> int:
    """Infer the SH degree from the number of basis functions."""
    degree = isqrt(num_bases) - 1
    if (degree + 1) ** 2 != num_bases:
        raise ValueError(f"Invalid SH basis count: {num_bases}")
    if not 0 <= degree <= 4:
        raise ValueError(f"Only SH degrees 0-4 are supported, got {degree}")
    return degree


@app.function
def load_ply_file(path: Path) -> GaussianSceneData:
    """Load a 3DGS-style `.ply` into Gaussian centers and SH coefficients."""
    ply_data = PlyData.read(path)
    vertices = ply_data["vertex"]
    property_names = list(vertices.data.dtype.names)

    centers = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
    ).astype(np.float32)
    dc_coefficients = np.stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]],
        axis=1,
    ).astype(np.float32)
    rest_feature_names = sorted(
        [name for name in property_names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    num_rest_coefficients = len(rest_feature_names)
    if num_rest_coefficients % 3 != 0:
        raise ValueError(
            "Expected the number of `f_rest_*` attributes to be divisible by 3."
        )

    num_bases = 1 + num_rest_coefficients // 3
    degree_to_use = infer_sh_degree(num_bases)
    sh_coefficients = np.zeros((centers.shape[0], 25, 3), dtype=np.float32)
    sh_coefficients[:, 0, :] = dc_coefficients
    if rest_feature_names:
        rest_coefficients = np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rest_feature_names
            ],
            axis=1,
        )
        rest_coefficients = rest_coefficients.reshape(
            centers.shape[0],
            3,
            num_bases - 1,
        )
        sh_coefficients[:, 1:num_bases, :] = np.transpose(
            rest_coefficients,
            (0, 2, 1),
        )

    scale_feature_names = sorted(
        [name for name in property_names if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rotation_feature_names = sorted(
        [name for name in property_names if name.startswith("rot")],
        key=lambda name: int(name.split("_")[-1]),
    )
    log_scales = (
        np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in scale_feature_names
            ],
            axis=1,
        )
        if scale_feature_names
        else np.full((centers.shape[0], 3), np.log(0.01), dtype=np.float32)
    )
    rotations = (
        np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rotation_feature_names
            ],
            axis=1,
        )
        if rotation_feature_names
        else np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (centers.shape[0], 1),
        )
    )
    rgbs = (0.5 + 0.28209479177387814 * dc_coefficients).astype(np.float32)
    opacities = (
        1.0
        / (
            1.0
            + np.exp(
                -np.asarray(vertices["opacity"], dtype=np.float32)[:, None]
            )
        )
    ).astype(np.float32)

    return GaussianSceneData(
        centers=centers,
        sh_coefficients=sh_coefficients,
        degree_to_use=degree_to_use,
        rgbs=rgbs,
        opacities=opacities,
        log_scales=log_scales,
        rotations=rotations,
    )


@app.function
def normalize_gaussian_data(scene_data: GaussianSceneData) -> GaussianSceneData:
    """Apply PCA normalization to centers, scales, rotations, and SH bases."""
    raw_centers = np.asarray(scene_data["centers"], dtype=np.float32)
    pca_transform = pca_transform_from_points(raw_centers.astype(np.float64))
    pca_centers = apply_to_points(pca_transform, raw_centers.astype(np.float64))
    point_distances = np.linalg.norm(pca_centers, axis=1)
    positive_distances = point_distances[point_distances > 1e-8]
    scene_scale = (
        1.0 / float(np.median(positive_distances))
        if positive_distances.size > 0
        else 1.0
    )

    scale_transform = np.eye(4, dtype=np.float64)
    scale_transform[:3, :3] *= scene_scale
    normalization_transform = compose_transforms(pca_transform, scale_transform)
    scene_rotation = pca_transform[:3, :3].astype(np.float32)

    return GaussianSceneData(
        centers=apply_to_points(
            normalization_transform,
            raw_centers.astype(np.float64),
        ).astype(np.float32),
        sh_coefficients=apply_rotation_to_sh_coefficients(
            scene_rotation,
            np.asarray(scene_data["sh_coefficients"], dtype=np.float32),
        ).astype(np.float32),
        degree_to_use=int(scene_data["degree_to_use"]),
        rgbs=np.asarray(scene_data["rgbs"], dtype=np.float32),
        opacities=np.asarray(scene_data["opacities"], dtype=np.float32),
        log_scales=apply_scale_to_log_scales(
            scene_scale,
            np.asarray(scene_data["log_scales"], dtype=np.float32),
        ).astype(np.float32),
        rotations=apply_rotation_to_quaternions(
            scene_rotation,
            np.asarray(scene_data["rotations"], dtype=np.float32),
        ).astype(np.float32),
    )


@app.function
def quaternion_to_rotation_matrices(
    quaternions_wxyz: npt.NDArray[np.floating],
) -> npt.NDArray[np.float32]:
    """Convert wxyz quaternions into rotation matrices."""
    normalized_quaternions = np.asarray(quaternions_wxyz, dtype=np.float32)
    norms = np.linalg.norm(normalized_quaternions, axis=1, keepdims=True)
    normalized_quaternions = normalized_quaternions / np.clip(norms, 1e-8, None)
    w = normalized_quaternions[:, 0]
    x = normalized_quaternions[:, 1]
    y = normalized_quaternions[:, 2]
    z = normalized_quaternions[:, 3]

    rotation_matrices = np.empty(
        (normalized_quaternions.shape[0], 3, 3),
        dtype=np.float32,
    )
    rotation_matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation_matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    rotation_matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    rotation_matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    rotation_matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation_matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    rotation_matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    rotation_matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    rotation_matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotation_matrices


@app.function
def gaussian_covariances(
    gaussian_scales: npt.NDArray[np.floating],
    gaussian_rotations: npt.NDArray[np.floating],
    global_scale: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Build Gaussian covariance matrices from axis scales and quaternions."""
    clamped_scale = max(float(global_scale), 1e-6)
    scaled_axes = np.asarray(gaussian_scales, dtype=np.float32) * clamped_scale
    diagonal_variances = np.square(scaled_axes)
    diagonal_matrices = np.zeros((scaled_axes.shape[0], 3, 3), dtype=np.float32)
    diagonal_matrices[:, 0, 0] = diagonal_variances[:, 0]
    diagonal_matrices[:, 1, 1] = diagonal_variances[:, 1]
    diagonal_matrices[:, 2, 2] = diagonal_variances[:, 2]
    rotation_matrices = quaternion_to_rotation_matrices(gaussian_rotations)
    return (
        rotation_matrices
        @ diagonal_matrices
        @ np.transpose(rotation_matrices, (0, 2, 1))
    )


@app.function
def selection_marker_size(log_scales: npt.NDArray[np.floating]) -> float:
    """Choose a stable marker size for the selected Gaussian."""
    return float(max(np.exp(np.median(log_scales)), 0.01))


@app.function
def selected_point_index_from_ray(
    centers: npt.NDArray[np.floating],
    ray_origin: npt.NDArray[np.floating],
    ray_direction: npt.NDArray[np.floating],
) -> int | None:
    """Pick the Gaussian center closest to the click ray."""
    if centers.shape[0] == 0:
        return None

    normalized_direction = ray_direction / max(
        float(np.linalg.norm(ray_direction)),
        1e-6,
    )
    offsets = centers - ray_origin[None, :]
    distances_along_ray = offsets @ normalized_direction
    valid_mask = distances_along_ray > 0.0
    if not np.any(valid_mask):
        return None

    closest_points = (
        ray_origin[None, :]
        + distances_along_ray[:, None] * normalized_direction[None, :]
    )
    squared_distances = np.sum((centers - closest_points) ** 2, axis=1)
    squared_distances[~valid_mask] = np.inf
    return int(np.argmin(squared_distances))


@app.function
def point_view_camera_state(
    centers: npt.NDArray[np.floating],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a stable overview camera pose for the loaded Gaussians."""
    if centers.shape[0] == 0:
        look_at = np.zeros(3, dtype=np.float32)
        position = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        return look_at, position

    bbox_min = centers.min(axis=0)
    bbox_max = centers.max(axis=0)
    extent = float(max(np.linalg.norm(bbox_max - bbox_min), 1.0))
    look_at = centers.mean(axis=0).astype(np.float32)
    position = look_at + np.array([0.0, 0.0, 2.5 * extent], dtype=np.float32)
    return look_at, position


@app.function
def selection_markdown(
    scene_data: GaussianSceneData | None,
    selected_index: int | None,
    degrees_to_use: int | None,
) -> str:
    """Format selection status for the Gaussian and SH viewers."""
    if scene_data is None or selected_index is None:
        return "**Loaded Gaussian file:** none  \n**Selected Gaussian:** none"

    selected_position = scene_data["centers"][selected_index]
    return (
        f"**Selected Gaussian:** `{selected_index}`  \n"
        f"**Position:** `[{selected_position[0]:.3f}, {selected_position[1]:.3f}, {selected_position[2]:.3f}]`  \n"
        f"**SH degree:** `{degrees_to_use}`"
    )


@app.function
def orbit_camera_position(
    position: npt.NDArray[np.floating],
    orbit_radius: float,
) -> np.ndarray:
    """Project a camera offset back onto the fixed orbit sphere."""
    direction = np.asarray(position, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 1.0, 0.6], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
    return direction / max(norm, 1e-6) * orbit_radius


@app.cell
def load_config(load_form):
    load_config = load_form.value
    return (load_config,)


@app.cell
def loaded_scene_data(load_config):
    if load_config is None:
        loaded_scene_data = None
    else:
        scene_data = load_ply_file(load_config.source.expanduser())
        if load_config.normalize_scene:
            scene_data = normalize_gaussian_data(scene_data)
        loaded_scene_data = scene_data
    return (loaded_scene_data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Viewer Settings

    Live settings for Gaussian selection and SH truncation.
    """)
    return


@app.cell
def viewer_config(viewer_form):
    viewer_config = viewer_form.value
    return (viewer_config,)


@app.cell
def _():
    selection_state: SelectionState = {
        "scene_data": None,
        "selected_index": None,
    }
    return (selection_state,)


@app.cell
def _():
    sh_view_state = {
        "coefficients": torch.zeros((25, 3), dtype=torch.float32),
        "degrees_to_use": 0,
        "orbit_radius": 6.0,
        "sphere_radius": 0.9,
    }
    return (sh_view_state,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Gaussian Viewer

    This scene now uses Gaussian splats, like the PLY viewer notebook. Enable
    picking to inspect one Gaussian's SH coefficients.
    """)
    return


@app.cell
def _():
    point_server, point_widget = viser_marimo(height=680)
    fov_slider = point_server.gui.add_slider(
        "Field of view",
        min=20.0,
        max=120.0,
        step=1.0,
        initial_value=45.0,
    )
    pick_point_toggle = point_server.gui.add_checkbox(
        "Pick Gaussian",
        initial_value=False,
    )
    point_status = point_server.gui.add_markdown(
        "**Loaded Gaussian file:** none  \n**Selected Gaussian:** none"
    )
    return (
        fov_slider,
        pick_point_toggle,
        point_server,
        point_status,
        point_widget,
    )


@app.cell
def _(fov_slider, point_server):
    @fov_slider.on_update
    def _(_) -> None:
        fov_radians = float(np.deg2rad(fov_slider.value))
        point_server.initial_camera.fov = fov_radians
        for client in point_server.get_clients().values():
            client.camera.fov = fov_radians

    return


@app.cell
def _(loaded_scene_data, point_server):
    if loaded_scene_data is not None:
        look_at, camera_position = point_view_camera_state(
            loaded_scene_data["centers"]
        )
        point_server.initial_camera.look_at = look_at.astype(np.float64)
        point_server.initial_camera.position = camera_position.astype(
            np.float64
        )
        for client in point_server.get_clients().values():
            client.camera.look_at = look_at.astype(np.float64)
            client.camera.position = camera_position.astype(np.float64)
    return


@app.cell
def _(
    loaded_scene_data,
    point_server,
    point_status,
    selection_state: SelectionState,
    sh_status,
    sh_view_state,
    sh_viewer,
    viewer_config,
):
    if loaded_scene_data is not None:
        selection_state["scene_data"] = loaded_scene_data
        selection_state["selected_index"] = 0
        sh_view_state["coefficients"] = torch.as_tensor(
            loaded_scene_data["sh_coefficients"][0],
            dtype=torch.float32,
        )
        sh_view_state["degrees_to_use"] = min(
            loaded_scene_data["degree_to_use"],
            viewer_config.max_sh_degree if viewer_config is not None else 4,
        )

        point_server.scene.add_gaussian_splats(
            "/ply_gaussians/gaussian_splats",
            centers=loaded_scene_data["centers"],
            rgbs=loaded_scene_data["rgbs"],
            opacities=loaded_scene_data["opacities"],
            covariances=gaussian_covariances(
                np.exp(loaded_scene_data["log_scales"]),
                loaded_scene_data["rotations"],
            ),
        )
        point_server.scene.add_point_cloud(
            "/ply_gaussians/selected_point",
            points=loaded_scene_data["centers"][[0]],
            colors=np.array([[255, 255, 0]], dtype=np.uint8),
            point_size=selection_marker_size(loaded_scene_data["log_scales"])
            * (
                viewer_config.selection_size
                if viewer_config is not None
                else 4.0
            ),
            point_shape="sparkle",
        )

        status_markdown = selection_markdown(
            loaded_scene_data,
            0,
            sh_view_state["degrees_to_use"],
        )
        point_status.content = status_markdown
        sh_status.content = status_markdown
        sh_viewer.rerender(None)
    return


@app.cell
def _(
    point_server,
    point_status,
    selection_state: SelectionState,
    sh_status,
    sh_view_state,
    sh_viewer,
    viewer_config,
):
    _scene_data = selection_state["scene_data"]
    _selected_index = selection_state["selected_index"]
    if (
        _scene_data is not None
        and _selected_index is not None
        and viewer_config is not None
    ):
        sh_view_state["degrees_to_use"] = min(
            _scene_data["degree_to_use"],
            viewer_config.max_sh_degree,
        )
        point_server.scene.add_point_cloud(
            "/ply_gaussians/selected_point",
            points=_scene_data["centers"][[_selected_index]],
            colors=np.array([[255, 255, 0]], dtype=np.uint8),
            point_size=selection_marker_size(_scene_data["log_scales"])
            * viewer_config.selection_size,
            point_shape="sparkle",
        )
        _status_markdown = selection_markdown(
            _scene_data,
            _selected_index,
            sh_view_state["degrees_to_use"],
        )
        point_status.content = _status_markdown
        sh_status.content = _status_markdown
        sh_viewer.rerender(None)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## SH Viewer

    The selected Gaussian's SH coefficients are visualized with the same SH
    sphere renderer as the main SH notebook.
    """)
    return


@app.cell
def _(sh_view_state):
    def render_fn(
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        """Render the SH split-screen sphere view for the selected Gaussian."""
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        camera_to_world = torch.as_tensor(
            camera_state.c2w,
            dtype=torch.float32,
            device=device,
        )
        intrinsics = torch.as_tensor(
            camera_state.get_K([width, height]),
            dtype=torch.float32,
            device=device,
        )
        image = render_spheres(
            camera_to_world,
            intrinsics,
            sh_view_state["coefficients"].to(
                device=device,
                dtype=torch.float32,
            ),
            int(sh_view_state["degrees_to_use"]),
            float(sh_view_state["orbit_radius"]),
            float(sh_view_state["sphere_radius"]),
            width,
            height,
        )
        return image.detach().cpu().numpy()

    return (render_fn,)


@app.cell
def _(render_fn):
    sh_server, sh_viewer, sh_widget = viser_marimo(
        render_fn=render_fn,
        height=680,
    )
    sh_status = sh_server.gui.add_markdown(
        "**Selected Gaussian:** none  \n**SH degree:** `0`"
    )
    return sh_server, sh_status, sh_viewer, sh_widget


@app.cell
def _(sh_server, sh_view_state, sh_viewer, sh_widget):
    @sh_server.on_client_connect
    def _(client: object) -> None:
        """Keep the SH viewer camera on a fixed orbit while preserving zoom as FOV."""
        syncing_camera = False
        client.camera.look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        client.camera.position = np.array(
            [0.0, 0.0, sh_view_state["orbit_radius"]],
            dtype=np.float32,
        )

        def _snap_camera() -> None:
            """Reset translation to the canonical orbit and convert radius changes into FOV."""
            nonlocal syncing_camera
            if syncing_camera:
                return
            look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            relative_position = np.asarray(
                client.camera.position,
                dtype=np.float32,
            ) - np.asarray(client.camera.look_at, dtype=np.float32)
            distance = float(np.linalg.norm(relative_position))
            safe_distance = max(distance, 1e-6)
            zoomed_fov = 2.0 * np.arctan(
                np.tan(float(client.camera.fov) * 0.5)
                * safe_distance
                / float(sh_view_state["orbit_radius"])
            )
            zoomed_fov = float(
                np.clip(zoomed_fov, np.deg2rad(5.0), np.deg2rad(175.0))
            )
            client_id = getattr(
                client,
                "client_id",
                getattr(client, "id", None),
            )
            camera_state = sh_widget.get_camera_state(client_id=client_id)
            syncing_camera = True
            try:
                sh_widget.set_camera_state(
                    replace(
                        camera_state,
                        position=orbit_camera_position(
                            relative_position,
                            float(sh_view_state["orbit_radius"]),
                        ).astype(np.float64),
                        look_at=look_at.astype(np.float64),
                        fov=zoomed_fov,
                    ),
                    client_id=client_id,
                    update_reset_view=True,
                    sync_gui=True,
                )
            finally:
                syncing_camera = False

        _snap_camera()

        @client.camera.on_update
        def _(_camera: object) -> None:
            """Cancel translation changes and rerender after orbit updates."""
            _snap_camera()
            sh_viewer.rerender(None)

        sh_viewer.rerender(None)

    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## GUI and Viewers
    """)
    return


@app.cell(hide_code=True)
def load_form():
    load_form = form_gui(
        LoadConfig,
        value=LoadConfig(),
        label="Gaussian File Loader",
        submit_label="Load File",
    )
    return (load_form,)


@app.cell
def _(load_form):
    load_form
    return


@app.cell(hide_code=True)
def scene_widget(point_widget):
    mo.vstack(
        [
            mo.md(
                "Use the embedded viewer below to inspect the loaded Gaussian splats and pick a Gaussian to inspect."
            ),
            point_widget,
        ]
    )
    return


@app.cell(hide_code=True)
def viewer_form():
    viewer_form = form_gui(
        ViewerConfig,
        value=ViewerConfig(),
        label="Selection Viewer Settings",
    )
    viewer_form
    return (viewer_form,)


@app.cell(hide_code=True)
def _(sh_widget):
    sh_widget
    return


if __name__ == "__main__":
    app.run()
