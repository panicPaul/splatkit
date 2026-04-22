"""MUSt3R dataset import and runtime adapter helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from beartype import beartype

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

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_MUST3R_CAMERA_SENSOR_ID = "camera"


@dataclass(frozen=True)
class Must3rCheckpointPaths:
    """Resolved MUSt3R checkpoints."""

    checkpoint_path: Path
    retrieval_checkpoint_path: Path | None = None


@runtime_checkable
class Must3rRuntime(Protocol):
    """Runtime interface for MUSt3R inference."""

    def run(
        self,
        *,
        image_dir: Path,
        output_dir: Path,
        checkpoints: Must3rCheckpointPaths,
        image_size: int,
        device: str,
    ) -> Path:
        """Run MUSt3R and return the produced artifact path."""


@dataclass(frozen=True)
class SubprocessMust3rSlamRuntime:
    """Thin wrapper around the upstream `must3r_slam` executable."""

    executable: str = "must3r_slam"
    extra_args: tuple[str, ...] = ()

    def run(
        self,
        *,
        image_dir: Path,
        output_dir: Path,
        checkpoints: Must3rCheckpointPaths,
        image_size: int,
        device: str,
    ) -> Path:
        """Run the upstream MUSt3R SLAM executable and return its artifact path."""
        executable_path = shutil.which(self.executable)
        if executable_path is None:
            raise RuntimeError(
                "MUSt3R runtime is not installed. Install the upstream `must3r` "
                "package or pass a custom Must3rRuntime."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            executable_path,
            "--chkpt",
            str(checkpoints.checkpoint_path),
            "--res",
            str(image_size),
            "--device",
            device,
            "--input",
            str(image_dir),
            "--output",
            str(output_dir),
        ]
        command.extend(self.extra_args)
        subprocess.run(command, check=True)
        dataset_json = output_dir / "dataset.json"
        if dataset_json.exists():
            return dataset_json
        npz_path = output_dir / "all_poses.npz"
        if npz_path.exists():
            return npz_path
        raise RuntimeError(
            "MUSt3R runtime finished without producing a supported artifact."
        )


def _require_huggingface_hub() -> object:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint resolution requires huggingface_hub. Install splatkit[data]."
        ) from exc
    return hf_hub_download


def resolve_must3r_checkpoints(
    *,
    checkpoint_repo_id: str,
    checkpoint_filename: str,
    retrieval_repo_id: str | None = None,
    retrieval_filename: str | None = None,
    cache_dir: str | Path | None = None,
) -> Must3rCheckpointPaths:
    """Resolve MUSt3R checkpoint files from Hugging Face."""
    hf_hub_download = _require_huggingface_hub()
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=checkpoint_repo_id,
            filename=checkpoint_filename,
            cache_dir=cache_dir,
        )
    )
    retrieval_checkpoint_path = None
    if retrieval_repo_id is not None and retrieval_filename is not None:
        retrieval_checkpoint_path = Path(
            hf_hub_download(
                repo_id=retrieval_repo_id,
                filename=retrieval_filename,
                cache_dir=cache_dir,
            )
        )
    return Must3rCheckpointPaths(
        checkpoint_path=checkpoint_path,
        retrieval_checkpoint_path=retrieval_checkpoint_path,
    )


def _resolve_image_paths(
    image_paths: list[str] | None,
    *,
    image_root: str | Path | None,
    num_frames: int,
) -> list[Path]:
    if image_paths is not None:
        root_path = Path(image_root) if image_root is not None else None
        resolved_paths = []
        for path in image_paths:
            path_obj = Path(path)
            if not path_obj.is_absolute() and root_path is not None:
                path_obj = root_path / path_obj
            resolved_paths.append(path_obj)
        return resolved_paths
    if image_root is None:
        raise ValueError(
            "MUSt3R import requires image paths in the artifact or image_root."
        )
    candidates = sorted(
        path
        for path in Path(image_root).iterdir()
        if path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if len(candidates) != num_frames:
        raise ValueError(
            "Could not infer MUSt3R image paths from image_root; frame count "
            f"mismatch ({len(candidates)} files vs {num_frames} poses)."
        )
    return candidates


def _load_point_cloud_from_payload(
    payload: dict[str, Any],
) -> PointCloudState | None:
    points = payload.get("points")
    if points is None:
        points = payload.get("points3d")
    if points is None:
        points = payload.get("xyz")
    if points is None:
        return None
    colors = payload.get("colors")
    if colors is None:
        colors = payload.get("rgb")
    confidence = payload.get("confidence")
    if confidence is None:
        confidence = payload.get("conf")
    return PointCloudState(
        points=torch.as_tensor(points, dtype=torch.float32),
        colors=(
            torch.as_tensor(colors, dtype=torch.float32)
            if colors is not None
            else None
        ),
        confidence=(
            torch.as_tensor(confidence, dtype=torch.float32)
            if confidence is not None
            else None
        ),
    )


def _scene_dataset_from_arrays(
    *,
    cam_to_world: np.ndarray,
    intrinsics: np.ndarray,
    image_paths: list[Path],
    point_cloud: PointCloudState | None,
    source_root: Path,
) -> SceneDataset:
    if cam_to_world.shape[0] != len(image_paths):
        raise ValueError("MUSt3R camera count does not match image path count.")
    if intrinsics.shape[0] != len(image_paths):
        raise ValueError(
            "MUSt3R intrinsics count does not match image path count."
        )
    widths: list[int] = []
    heights: list[int] = []
    fov_degrees: list[float] = []
    frames: list[DatasetFrame] = []
    frame_paths: dict[str, Path] = {}
    tensor_intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)
    for index, (path, intrinsics_matrix) in enumerate(
        zip(image_paths, tensor_intrinsics, strict=True)
    ):
        if not path.exists():
            raise ValueError(f"MUSt3R image path does not exist: {path}.")
        width = round(float(intrinsics_matrix[0, 2] * 2.0))
        height = round(float(intrinsics_matrix[1, 2] * 2.0))
        widths.append(width)
        heights.append(height)
        fov_degrees.append(horizontal_fov_degrees(width, intrinsics_matrix))
        frames.append(
            DatasetFrame(
                frame_id=str(index),
                sensor_id=_MUST3R_CAMERA_SENSOR_ID,
                camera_index=index,
                width=width,
                height=height,
            )
        )
        frame_paths[str(index)] = path
    camera_sensor = CameraSensorDataset(
        sensor_id=_MUST3R_CAMERA_SENSOR_ID,
        kind="camera",
        frames=tuple(frames),
        timestamps_us=tuple(frame.timestamp_us for frame in frames),
        camera=CameraState(
            width=torch.tensor(widths, dtype=torch.int64),
            height=torch.tensor(heights, dtype=torch.int64),
            fov_degrees=torch.tensor(fov_degrees, dtype=torch.float32),
            cam_to_world=torch.as_tensor(cam_to_world, dtype=torch.float32),
            intrinsics=tensor_intrinsics,
            camera_convention="opencv",
        ),
        image_source=PathCameraImageSource(frame_paths=frame_paths),
    )
    return SceneDataset(
        sensors=(camera_sensor,),
        source_format="must3r",
        default_camera_sensor_id=_MUST3R_CAMERA_SENSOR_ID,
        source_uris=(str(source_root),),
        point_cloud=point_cloud,
    )


def _load_must3r_json(
    path: Path,
    *,
    image_root: str | Path | None,
) -> SceneDataset:
    payload = json.loads(path.read_text())
    frames_payload = payload["frames"]
    image_paths = _resolve_image_paths(
        [frame["image_path"] for frame in frames_payload],
        image_root=image_root or path.parent,
        num_frames=len(frames_payload),
    )
    cam_to_world = np.asarray(
        [frame["cam_to_world"] for frame in frames_payload],
        dtype=np.float32,
    )
    intrinsics = np.asarray(
        [frame["intrinsics"] for frame in frames_payload],
        dtype=np.float32,
    )
    point_cloud = None
    point_cloud_path = payload.get("point_cloud_path")
    if point_cloud_path is not None:
        point_cloud_npz = np.load(path.parent / point_cloud_path)
        point_cloud = _load_point_cloud_from_payload(dict(point_cloud_npz))
    return _scene_dataset_from_arrays(
        cam_to_world=cam_to_world,
        intrinsics=intrinsics,
        image_paths=image_paths,
        point_cloud=point_cloud,
        source_root=path.parent,
    )


def _first_present(payload: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    raise ValueError(f"None of the expected MUSt3R keys were found: {names}.")


def _load_must3r_npz(
    path: Path,
    *,
    image_root: str | Path | None,
) -> SceneDataset:
    payload = dict(np.load(path, allow_pickle=True))
    cam_to_world = np.asarray(
        _first_present(payload, ("cam_to_world", "poses", "all_poses")),
        dtype=np.float32,
    )
    intrinsics = np.asarray(
        _first_present(payload, ("intrinsics", "K", "Ks")),
        dtype=np.float32,
    )
    raw_image_paths = payload.get("image_paths")
    if raw_image_paths is not None:
        image_paths = _resolve_image_paths(
            [str(item) for item in raw_image_paths.tolist()],
            image_root=image_root or path.parent,
            num_frames=cam_to_world.shape[0],
        )
    else:
        image_paths = _resolve_image_paths(
            None,
            image_root=image_root,
            num_frames=cam_to_world.shape[0],
        )
    point_cloud = _load_point_cloud_from_payload(payload)
    return _scene_dataset_from_arrays(
        cam_to_world=cam_to_world,
        intrinsics=intrinsics,
        image_paths=image_paths,
        point_cloud=point_cloud,
        source_root=path.parent,
    )


@beartype
def load_must3r_dataset(
    path: str | Path,
    *,
    image_root: str | Path | None = None,
    horizon_adjustment: HorizonAdjustmentSpec | None = None,
) -> SceneDataset:
    """Load MUSt3R outputs into a SceneDataset."""
    artifact_path = Path(path)
    if artifact_path.is_dir():
        dataset_json = artifact_path / "dataset.json"
        if dataset_json.exists():
            artifact_path = dataset_json
        else:
            npz_path = artifact_path / "all_poses.npz"
            if npz_path.exists():
                artifact_path = npz_path
    if artifact_path.suffix == ".json":
        dataset = _load_must3r_json(artifact_path, image_root=image_root)
    elif artifact_path.suffix == ".npz":
        dataset = _load_must3r_npz(artifact_path, image_root=image_root)
    else:
        raise ValueError(
            f"Unsupported MUSt3R artifact format {artifact_path.suffix!r}."
        )
    if horizon_adjustment is not None:
        dataset = adjust_dataset_horizon(dataset, horizon_adjustment)
    return dataset


@beartype
def run_must3r_dataset(
    image_dir: str | Path,
    *,
    output_dir: str | Path,
    checkpoint_repo_id: str,
    checkpoint_filename: str,
    image_size: int = 512,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
    runtime: Must3rRuntime | None = None,
    horizon_adjustment: HorizonAdjustmentSpec | None = None,
) -> SceneDataset:
    """Run MUSt3R through a runtime adapter and import the produced dataset."""
    checkpoints = resolve_must3r_checkpoints(
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_filename=checkpoint_filename,
        cache_dir=cache_dir,
    )
    resolved_runtime = runtime or SubprocessMust3rSlamRuntime()
    artifact_path = resolved_runtime.run(
        image_dir=Path(image_dir),
        output_dir=Path(output_dir),
        checkpoints=checkpoints,
        image_size=image_size,
        device=device,
    )
    return load_must3r_dataset(
        artifact_path,
        image_root=image_dir,
        horizon_adjustment=horizon_adjustment,
    )
