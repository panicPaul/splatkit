import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns")

with app.setup:
    import time
    from pathlib import Path
    from typing import TypedDict

    import marimo as mo
    import numpy as np
    import numpy.typing as npt
    import viser
    from plyfile import PlyData
    from pydantic import BaseModel, Field

    from marimo_viser import (
        apply_rotation_to_quaternions,
        apply_scale_to_log_scales,
        apply_to_points,
        compose_transforms,
        form_gui,
        pca_transform_from_points,
        viser_marimo,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Loading
    """)
    return


@app.class_definition
class SplatFile(TypedDict):
    centers: npt.NDArray[np.floating]
    rgbs: npt.NDArray[np.floating]
    opacities: npt.NDArray[np.floating]
    log_scales: npt.NDArray[np.floating]
    rotations: npt.NDArray[np.floating]


@app.class_definition
class LoadConfig(BaseModel):
    source: Path = Field(
        default=Path.cwd() / "point_cloud.ply",
        description="Select a .ply or .splat file to load.",
    )
    normalize_scene: bool = Field(
        default=True,
        description="Apply PCA-based scene normalization before rendering.",
    )


@app.function
def load_splat_file(splat_path: Path) -> SplatFile:
    """Load a binary .splat file into raw Gaussian parameters."""
    start_time = time.perf_counter()
    splat_buffer = splat_path.read_bytes()
    bytes_per_gaussian = 3 * 4 + 3 * 4 + 4 + 4
    if len(splat_buffer) % bytes_per_gaussian != 0:
        raise ValueError(f"Unexpected .splat file size for {splat_path}.")

    num_gaussians = len(splat_buffer) // bytes_per_gaussian
    splat_uint8 = np.frombuffer(splat_buffer, dtype=np.uint8).reshape(
        (num_gaussians, bytes_per_gaussian)
    )
    centers = splat_uint8[:, 0:12].copy().view(np.float32)
    scales = splat_uint8[:, 12:24].copy().view(np.float32)
    log_scales = np.log(np.clip(scales, 1e-8, None)).astype(np.float32)
    rotations = (splat_uint8[:, 28:32] / 255.0 * 2.0 - 1.0).astype(np.float32)

    print(
        f"Loaded {num_gaussians} gaussians from {splat_path.name} "
        f"in {time.perf_counter() - start_time:.2f}s"
    )
    return SplatFile(
        centers=centers,
        rgbs=(splat_uint8[:, 24:27] / 255.0).astype(np.float32),
        opacities=(splat_uint8[:, 27:28] / 255.0).astype(np.float32),
        log_scales=log_scales,
        rotations=rotations,
    )


@app.function
def load_ply_file(ply_path: Path) -> SplatFile:
    """Load a 3DGS-style .ply file into raw Gaussian parameters."""
    start_time = time.perf_counter()
    sh_c0 = 0.28209479177387814

    ply_data = PlyData.read(ply_path)
    vertices = ply_data["vertex"]
    scale_feature_names = sorted(
        [
            name
            for name in vertices.data.dtype.names
            if name.startswith("scale_")
        ],
        key=lambda name: int(name.split("_")[-1]),
    )
    rotation_feature_names = sorted(
        [name for name in vertices.data.dtype.names if name.startswith("rot")],
        key=lambda name: int(name.split("_")[-1]),
    )

    centers = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]], axis=-1
    ).astype(np.float32)
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
    rgbs = (
        0.5
        + sh_c0
        * np.stack(
            [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]],
            axis=1,
        )
    ).astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-vertices["opacity"][:, None]))).astype(
        np.float32
    )

    print(
        f"Loaded {len(vertices)} gaussians from {ply_path.name} "
        f"in {time.perf_counter() - start_time:.2f}s"
    )
    return SplatFile(
        centers=centers,
        rgbs=rgbs,
        opacities=opacities,
        log_scales=log_scales,
        rotations=rotations,
    )


@app.function
def load_gaussian_file(path: Path) -> SplatFile:
    """Dispatch to the correct Gaussian loader based on the file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".splat":
        return load_splat_file(path)
    if suffix == ".ply":
        return load_ply_file(path)
    raise ValueError(f"Expected a .ply or .splat file, got {path}.")


@app.function
def normalize_gaussian_data(splat_data: SplatFile) -> SplatFile:
    """Normalize Gaussian centers, scales, and rotations with repo helpers."""
    raw_centers = np.asarray(splat_data["centers"], dtype=np.float32)
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

    return SplatFile(
        centers=apply_to_points(
            normalization_transform,
            raw_centers.astype(np.float64),
        ).astype(np.float32),
        rgbs=np.asarray(splat_data["rgbs"], dtype=np.float32),
        opacities=np.asarray(splat_data["opacities"], dtype=np.float32),
        log_scales=apply_scale_to_log_scales(
            scene_scale,
            np.asarray(splat_data["log_scales"], dtype=np.float32),
        ).astype(np.float32),
        rotations=apply_rotation_to_quaternions(
            scene_rotation,
            np.asarray(splat_data["rotations"], dtype=np.float32),
        ).astype(np.float32),
    )


@app.function
def quaternion_to_rotation_matrices(
    quaternions_wxyz: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Convert wxyz quaternions into rotation matrices."""
    normalized_quaternions = np.asarray(quaternions_wxyz, dtype=np.float32)
    norms = np.linalg.norm(normalized_quaternions, axis=1, keepdims=True)
    normalized_quaternions = normalized_quaternions / np.clip(norms, 1e-8, None)
    w = normalized_quaternions[:, 0]
    x = normalized_quaternions[:, 1]
    y = normalized_quaternions[:, 2]
    z = normalized_quaternions[:, 3]

    rotation_matrices = np.empty(
        (normalized_quaternions.shape[0], 3, 3), dtype=np.float32
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
) -> npt.NDArray[np.floating]:
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


@app.class_definition
class GaussianSceneController:
    """Manage a viser scene and attach loaded Gaussian splats to it."""

    def __init__(self, server: viser.ViserServer) -> None:
        self.server = server
        self.next_index = 0

    def add_file(self, path: Path, *, normalize_scene: bool) -> str:
        """Load one Gaussian file, normalize it if requested, and add it to the scene."""
        resolved_path = path.expanduser()
        if not resolved_path.exists():
            raise FileNotFoundError(resolved_path)

        splat_data = load_gaussian_file(resolved_path)
        if normalize_scene:
            splat_data = normalize_gaussian_data(splat_data)

        gaussian_scales = np.exp(
            np.asarray(splat_data["log_scales"], dtype=np.float32)
        )
        covariances = gaussian_covariances(
            gaussian_scales,
            np.asarray(splat_data["rotations"], dtype=np.float32),
        )
        node_name = f"/splats/{self.next_index}"
        self.next_index += 1

        splat_handle = self.server.scene.add_gaussian_splats(
            f"{node_name}/gaussian_splats",
            centers=np.asarray(splat_data["centers"], dtype=np.float32),
            rgbs=np.asarray(splat_data["rgbs"], dtype=np.float32),
            opacities=np.asarray(splat_data["opacities"], dtype=np.float32),
            covariances=covariances,
        )

        return node_name


@app.cell
def scene_controller():
    scene_server, scene_widget = viser_marimo(height=720)
    fov_slider = scene_server.gui.add_slider(
        "Field of view",
        min=20.0,
        max=120.0,
        step=1.0,
        initial_value=45.0,
    )

    @fov_slider.on_update
    def _(_) -> None:
        fov_radians = float(np.deg2rad(fov_slider.value))
        scene_server.initial_camera.fov = fov_radians
        for client in scene_server.get_clients().values():
            client.camera.fov = fov_radians

    scene_controller = GaussianSceneController(scene_server)
    return scene_controller, scene_widget


@app.cell
def load_config(load_form):
    load_config = load_form.value
    load_config
    return (load_config,)


@app.cell
def load_action(load_config, scene_controller):
    if load_config is None:
        mo.md("Choose a file and submit the form to add it to the viewer.")
    else:
        loaded_node = scene_controller.add_file(
            load_config.source,
            normalize_scene=load_config.normalize_scene,
        )
        mo.status.toast(
            "Gaussian file loaded",
            str(load_config.source.expanduser()),
        )
        mo.md(f"Loaded `{load_config.source.name}` into `{loaded_node}`.")
    return


@app.cell
def _():
    return


@app.cell(column=1)
def load_form():
    load_form = form_gui(
        LoadConfig,
        value=LoadConfig(),
        label="Gaussian File Loader",
        submit_label="Load File",
    )
    load_form
    return (load_form,)


@app.cell(hide_code=True)
def scene_widget(scene_widget):
    mo.vstack(
        [
            mo.md(
                "Use the embedded viewer below to inspect loaded `.ply` or `.splat` files."
            ),
            scene_widget,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
