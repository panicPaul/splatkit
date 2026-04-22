"""COLMAP dataset import helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor

from splatkit.core.contracts import CameraState
from splatkit.data.contracts import (
    CameraSensorDataset,
    DatasetFrame,
    HorizonAdjustmentSpec,
    PathCameraImageSource,
    PointCloudState,
    SceneDataset,
    horizontal_fov_degrees,
)
from splatkit.data.postprocess import adjust_dataset_horizon

_CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}
_CAMERA_MODELS = {
    name: num_params for name, num_params in _CAMERA_MODEL_IDS.values()
}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_COLMAP_CAMERA_SENSOR_ID = "camera"


@dataclass(frozen=True)
class _ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class _ColmapImage:
    image_id: int
    camera_id: int
    name: str
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]


@dataclass(frozen=True)
class _ColmapPoint:
    point_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float


def _read_struct(handle: BinaryIO, fmt: str) -> tuple[object, ...]:
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Unexpected EOF while reading COLMAP binary file.")
    return struct.unpack(fmt, data)


def _read_c_string(handle: BinaryIO) -> str:
    chars: list[bytes] = []
    while True:
        char = handle.read(1)
        if not char:
            raise ValueError("Unexpected EOF while reading COLMAP string.")
        if char == b"\x00":
            return b"".join(chars).decode("utf-8")
        chars.append(char)


def _parse_cameras_text(path: Path) -> dict[int, _ColmapCamera]:
    cameras: dict[int, _ColmapCamera] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        camera_id = int(tokens[0])
        model = tokens[1]
        width = int(tokens[2])
        height = int(tokens[3])
        params = tuple(float(value) for value in tokens[4:])
        cameras[camera_id] = _ColmapCamera(
            camera_id=camera_id,
            model=model,
            width=width,
            height=height,
            params=params,
        )
    return cameras


def _parse_images_text(path: Path) -> dict[int, _ColmapImage]:
    images: dict[int, _ColmapImage] = {}
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        index += 1
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        image_id = int(tokens[0])
        qvec = tuple(float(value) for value in tokens[1:5])
        tvec = tuple(float(value) for value in tokens[5:8])
        camera_id = int(tokens[8])
        name = tokens[9]
        images[image_id] = _ColmapImage(
            image_id=image_id,
            camera_id=camera_id,
            name=name,
            qvec=qvec,  # type: ignore[arg-type]
            tvec=tvec,  # type: ignore[arg-type]
        )
        index += 1
    return images


def _parse_points_text(path: Path) -> dict[int, _ColmapPoint]:
    points: dict[int, _ColmapPoint] = {}
    if not path.exists():
        return points
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        point_id = int(tokens[0])
        points[point_id] = _ColmapPoint(
            point_id=point_id,
            xyz=(float(tokens[1]), float(tokens[2]), float(tokens[3])),
            rgb=(int(tokens[4]), int(tokens[5]), int(tokens[6])),
            error=float(tokens[7]),
        )
    return points


def _parse_cameras_binary(path: Path) -> dict[int, _ColmapCamera]:
    cameras: dict[int, _ColmapCamera] = {}
    with path.open("rb") as handle:
        (num_cameras,) = _read_struct(handle, "<Q")
        for _ in range(num_cameras):
            camera_id, model_id = _read_struct(handle, "<ii")
            width, height = _read_struct(handle, "<QQ")
            model_name, num_params = _CAMERA_MODEL_IDS[model_id]  # type: ignore[index]
            params = _read_struct(handle, "<" + "d" * num_params)
            cameras[int(camera_id)] = _ColmapCamera(
                camera_id=int(camera_id),
                model=model_name,
                width=int(width),
                height=int(height),
                params=tuple(float(param) for param in params),
            )
    return cameras


def _parse_images_binary(path: Path) -> dict[int, _ColmapImage]:
    images: dict[int, _ColmapImage] = {}
    with path.open("rb") as handle:
        (num_images,) = _read_struct(handle, "<Q")
        for _ in range(num_images):
            (image_id,) = _read_struct(handle, "<i")
            qvec = _read_struct(handle, "<dddd")
            tvec = _read_struct(handle, "<ddd")
            (camera_id,) = _read_struct(handle, "<i")
            name = _read_c_string(handle)
            (num_points2d,) = _read_struct(handle, "<Q")
            handle.seek(int(num_points2d) * 24, 1)
            images[int(image_id)] = _ColmapImage(
                image_id=int(image_id),
                camera_id=int(camera_id),
                name=name,
                qvec=tuple(float(value) for value in qvec),  # type: ignore[arg-type]
                tvec=tuple(float(value) for value in tvec),  # type: ignore[arg-type]
            )
    return images


def _parse_points_binary(path: Path) -> dict[int, _ColmapPoint]:
    points: dict[int, _ColmapPoint] = {}
    if not path.exists():
        return points
    with path.open("rb") as handle:
        (num_points,) = _read_struct(handle, "<Q")
        for _ in range(num_points):
            (point_id,) = _read_struct(handle, "<Q")
            xyz = _read_struct(handle, "<ddd")
            rgb = _read_struct(handle, "<BBB")
            (error,) = _read_struct(handle, "<d")
            (track_length,) = _read_struct(handle, "<Q")
            handle.seek(int(track_length) * 8, 1)
            points[int(point_id)] = _ColmapPoint(
                point_id=int(point_id),
                xyz=tuple(float(value) for value in xyz),  # type: ignore[arg-type]
                rgb=tuple(int(value) for value in rgb),  # type: ignore[arg-type]
                error=float(error),
            )
    return points


def _resolve_sparse_paths(path: str | Path) -> tuple[Path, Path, Path]:
    sparse_root = Path(path)
    if sparse_root.is_dir() and (sparse_root / "cameras.bin").exists():
        cameras_path = sparse_root / "cameras.bin"
        images_path = sparse_root / "images.bin"
        points_path = sparse_root / "points3D.bin"
        return cameras_path, images_path, points_path
    if sparse_root.is_dir() and (sparse_root / "cameras.txt").exists():
        cameras_path = sparse_root / "cameras.txt"
        images_path = sparse_root / "images.txt"
        points_path = sparse_root / "points3D.txt"
        return cameras_path, images_path, points_path
    if sparse_root.is_dir() and (sparse_root / "sparse" / "0").exists():
        return _resolve_sparse_paths(sparse_root / "sparse" / "0")
    raise ValueError(f"Could not locate a COLMAP sparse model under {path}.")


def _resolve_image_root(
    dataset_root: Path, image_root: str | Path | None
) -> Path:
    if image_root is not None:
        return Path(image_root)
    if (dataset_root / "images").exists():
        return dataset_root / "images"
    raise ValueError(
        "Could not infer the COLMAP image directory. Pass image_root explicitly."
    )


def _qvec_to_rotation_matrix(
    qvec: tuple[float, float, float, float],
) -> Float[Tensor, " 3 3"]:
    qw, qx, qy, qz = qvec
    return torch.tensor(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
            ],
            [
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qw * qx),
            ],
            [
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=torch.float32,
    )


def _camera_to_intrinsics(camera: _ColmapCamera) -> Float[Tensor, " 3 3"]:
    if camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = camera.params
        fx = focal
        fy = focal
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    elif camera.model in {"SIMPLE_RADIAL", "RADIAL"}:
        focal, cx, cy = camera.params[:3]
        fx = focal
        fy = focal
    elif camera.model == "OPENCV":
        fx, fy, cx, cy = camera.params[:4]
    else:
        raise ValueError(f"Unsupported COLMAP camera model {camera.model!r}.")
    return torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _camera_distortion(
    camera: _ColmapCamera,
) -> Float[Tensor, " distortion"] | None:
    if camera.model in {"SIMPLE_PINHOLE", "PINHOLE"}:
        return None
    if camera.model == "SIMPLE_RADIAL":
        return torch.tensor(
            [camera.params[3], 0.0, 0.0, 0.0], dtype=torch.float32
        )
    if camera.model == "RADIAL":
        return torch.tensor(
            [camera.params[3], camera.params[4], 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
    if camera.model == "OPENCV":
        return torch.tensor(camera.params[4:], dtype=torch.float32)
    raise ValueError(f"Unsupported COLMAP camera model {camera.model!r}.")


def _require_cv2() -> object:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Undistorting COLMAP images requires OpenCV. Install splatkit[data]."
        ) from exc
    return cv2


def undistort_colmap_images(
    *,
    image_root: str | Path,
    output_root: str | Path,
    cameras: dict[int, _ColmapCamera],
    images: dict[int, _ColmapImage],
) -> dict[int, tuple[Path, Float[Tensor, " 3 3"]]]:
    """Undistort images into a cache directory and return updated intrinsics."""
    cv2 = _require_cv2()
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    updated: dict[int, tuple[Path, Float[Tensor, " 3 3"]]] = {}
    for image_id, image in images.items():
        camera = cameras[image.camera_id]
        source_path = Path(image_root) / image.name
        target_path = output_dir / image.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        intrinsics = _camera_to_intrinsics(camera).numpy()
        distortion = _camera_distortion(camera)
        if distortion is None:
            updated[image_id] = (target_path, torch.from_numpy(intrinsics))
            if not target_path.exists():
                target_path.write_bytes(source_path.read_bytes())
            continue
        image_array = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image_array is None:
            raise ValueError(f"Could not load COLMAP image {source_path}.")
        distortion_coeffs = distortion.numpy()
        new_intrinsics, _roi = cv2.getOptimalNewCameraMatrix(
            intrinsics,
            distortion_coeffs,
            (camera.width, camera.height),
            0.0,
            (camera.width, camera.height),
        )
        undistorted = cv2.undistort(
            image_array,
            intrinsics,
            distortion_coeffs,
            newCameraMatrix=new_intrinsics,
        )
        if not cv2.imwrite(str(target_path), undistorted):
            raise ValueError(
                f"Failed to write undistorted image to {target_path}."
            )
        updated[image_id] = (
            target_path,
            torch.from_numpy(new_intrinsics).to(torch.float32),
        )
    return updated


def _build_scene_dataset(
    *,
    image_root: Path,
    source_root: Path,
    cameras: dict[int, _ColmapCamera],
    images: dict[int, _ColmapImage],
    points: dict[int, _ColmapPoint],
    undistorted_images: dict[int, tuple[Path, Float[Tensor, " 3 3"]]] | None,
) -> SceneDataset:
    frames: list[DatasetFrame] = []
    frame_paths: dict[str, Path] = {}
    widths: list[int] = []
    heights: list[int] = []
    fov_degrees: list[float] = []
    intrinsics: list[Float[Tensor, " 3 3"]] = []
    cam_to_worlds: list[Float[Tensor, " 4 4"]] = []

    for camera_index, image in enumerate(
        sorted(images.values(), key=lambda item: item.image_id)
    ):
        camera = cameras[image.camera_id]
        if undistorted_images is None:
            if camera.model not in {"SIMPLE_PINHOLE", "PINHOLE"}:
                raise ValueError(
                    "COLMAP loader requires undistorted pinhole data. "
                    "Pass undistort_output_dir to generate an undistorted cache."
                )
            image_path = image_root / image.name
            intrinsics_matrix = _camera_to_intrinsics(camera)
        else:
            image_path, intrinsics_matrix = undistorted_images[image.image_id]
        widths.append(camera.width)
        heights.append(camera.height)
        fov_degrees.append(
            horizontal_fov_degrees(camera.width, intrinsics_matrix)
        )
        intrinsics.append(intrinsics_matrix)

        world_to_camera = torch.eye(4, dtype=torch.float32)
        world_to_camera[:3, :3] = _qvec_to_rotation_matrix(image.qvec)
        world_to_camera[:3, 3] = torch.tensor(image.tvec, dtype=torch.float32)
        cam_to_worlds.append(torch.linalg.inv(world_to_camera))
        frames.append(
            DatasetFrame(
                frame_id=str(image.image_id),
                sensor_id=_COLMAP_CAMERA_SENSOR_ID,
                camera_index=camera_index,
                width=camera.width,
                height=camera.height,
            )
        )
        frame_paths[str(image.image_id)] = image_path

    point_cloud = None
    if points:
        point_cloud = PointCloudState(
            points=torch.tensor(
                [point.xyz for point in points.values()],
                dtype=torch.float32,
            ),
            colors=torch.tensor(
                [point.rgb for point in points.values()],
                dtype=torch.float32,
            )
            / 255.0,
            confidence=torch.tensor(
                [1.0 / max(point.error, 1e-6) for point in points.values()],
                dtype=torch.float32,
            ),
        )

    camera_sensor = CameraSensorDataset(
        sensor_id=_COLMAP_CAMERA_SENSOR_ID,
        kind="camera",
        frames=tuple(frames),
        timestamps_us=tuple(frame.timestamp_us for frame in frames),
        camera=CameraState(
            width=torch.tensor(widths, dtype=torch.int64),
            height=torch.tensor(heights, dtype=torch.int64),
            fov_degrees=torch.tensor(fov_degrees, dtype=torch.float32),
            cam_to_world=torch.stack(cam_to_worlds, dim=0),
            intrinsics=torch.stack(intrinsics, dim=0),
            camera_convention="opencv",
        ),
        image_source=PathCameraImageSource(frame_paths=frame_paths),
    )

    return SceneDataset(
        sensors=(camera_sensor,),
        source_format="colmap",
        default_camera_sensor_id=_COLMAP_CAMERA_SENSOR_ID,
        source_uris=(str(source_root),),
        point_cloud=point_cloud,
    )


@beartype
def load_colmap_dataset(
    path: str | Path,
    *,
    image_root: str | Path | None = None,
    undistort_output_dir: str | Path | None = None,
    horizon_adjustment: HorizonAdjustmentSpec | None = None,
) -> SceneDataset:
    """Load a COLMAP sparse model into a SceneDataset."""
    source_root = Path(path)
    cameras_path, images_path, points_path = _resolve_sparse_paths(source_root)
    if cameras_path.suffix == ".bin":
        cameras = _parse_cameras_binary(cameras_path)
        images = _parse_images_binary(images_path)
        points = _parse_points_binary(points_path)
    else:
        cameras = _parse_cameras_text(cameras_path)
        images = _parse_images_text(images_path)
        points = _parse_points_text(points_path)

    if not images:
        raise ValueError(f"No COLMAP images found under {path}.")

    resolved_image_root = _resolve_image_root(source_root, image_root)
    undistorted_images = None
    if undistort_output_dir is not None:
        undistorted_images = undistort_colmap_images(
            image_root=resolved_image_root,
            output_root=undistort_output_dir,
            cameras=cameras,
            images=images,
        )
    dataset = _build_scene_dataset(
        image_root=resolved_image_root,
        source_root=source_root,
        cameras=cameras,
        images=images,
        points=points,
        undistorted_images=undistorted_images,
    )
    if horizon_adjustment is not None:
        dataset = adjust_dataset_horizon(dataset, horizon_adjustment)
    return dataset
