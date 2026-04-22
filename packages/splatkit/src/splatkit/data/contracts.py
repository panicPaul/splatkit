"""Declarative dataset contracts for image-based and multi-sensor data."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import inspect
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlparse, unquote

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, UInt8
from torch import Tensor

from splatkit.core.contracts import CameraState

DatasetSource = Literal["colmap", "must3r", "ncore"]
SensorKind = Literal["camera", "lidar", "radar", "other"]
MaterializationStage = Literal["none", "decoded", "prepared"]
MaterializationMode = Literal["lazy", "eager"]


def horizontal_fov_degrees(
    width: int,
    intrinsics: Float[Tensor, " 3 3"],
) -> float:
    """Compute the horizontal field of view in degrees."""
    fx = intrinsics[0, 0]
    return float(torch.rad2deg(2.0 * torch.atan(width / (2.0 * fx))).item())


def _local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        return Path(uri)
    return None


def _normalize_rgb_array(
    image: np.ndarray,
) -> UInt8[np.ndarray, "height width 3"]:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(
            f"Expected an RGB image with 3 dimensions, got shape {array.shape}."
        )
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(
            f"Expected an RGB image with 3 channels, got shape {array.shape}."
        )
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


@beartype
@dataclass(frozen=True)
class ResizeSpec:
    """Declarative image resizing configuration."""

    width_scale: float | None = None
    width_target: int | None = None
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"

    def __post_init__(self) -> None:
        if self.width_scale is not None and self.width_target is not None:
            raise ValueError(
                "ResizeSpec must use either width_scale or width_target."
            )
        if self.width_scale is None and self.width_target is None:
            raise ValueError("ResizeSpec requires width_scale or width_target.")
        if self.width_scale is not None and self.width_scale <= 0.0:
            raise ValueError("ResizeSpec.width_scale must be > 0.")
        if self.width_target is not None and self.width_target <= 0:
            raise ValueError("ResizeSpec.width_target must be > 0.")


@beartype
@dataclass(frozen=True)
class ImagePreparationSpec:
    """Image preparation settings for lazy dataloading."""

    resize: ResizeSpec | None = None
    normalize: bool = True
    color_space: Literal["rgb"] = "rgb"


@beartype
@dataclass(frozen=True)
class HorizonAdjustmentSpec:
    """Optional pose canonicalization settings."""

    enabled: bool = False
    target_up: Float[Tensor, " 3"] = field(
        default_factory=lambda: torch.tensor([0.0, 1.0, 0.0])
    )
    focus_fallback: Literal["mean_camera_position"] = "mean_camera_position"


@beartype
@dataclass(frozen=True)
class DatasetFrame:
    """Immutable per-frame metadata."""

    frame_id: str
    sensor_id: str
    camera_index: int
    width: int
    height: int
    timestamp_us: int | None = None


@runtime_checkable
class CameraImageSource(Protocol):
    """Camera-backed RGB frame source."""

    def load_rgb(
        self,
        frame: DatasetFrame,
    ) -> UInt8[np.ndarray, "height width 3"]:
        """Load an RGB image for one frame."""


@beartype
@dataclass(frozen=True)
class PathCameraImageSource:
    """Path-backed image source used by file-based dataset loaders."""

    frame_paths: dict[str, Path]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_paths",
            {frame_id: Path(path) for frame_id, path in self.frame_paths.items()},
        )

    def path_for_frame(self, frame: DatasetFrame) -> Path:
        """Return the backing path for a frame."""
        try:
            return self.frame_paths[frame.frame_id]
        except KeyError as exc:
            raise KeyError(
                f"PathCameraImageSource does not contain frame {frame.frame_id!r}."
            ) from exc

    def load_rgb(
        self,
        frame: DatasetFrame,
    ) -> UInt8[np.ndarray, "height width 3"]:
        """Load an RGB image from disk."""
        from PIL import Image

        with Image.open(self.path_for_frame(frame)) as image:
            return _normalize_rgb_array(np.asarray(image.convert("RGB")))


def _invoke_image_reader(
    reader: object,
    frame: DatasetFrame,
) -> np.ndarray:
    method = None
    for name in ("load_rgb", "read_rgb", "get_rgb"):
        candidate = getattr(reader, name, None)
        if callable(candidate):
            method = candidate
            break
    if method is None:
        if callable(reader):
            method = reader
        else:
            raise TypeError(
                "NCore camera readers must provide load_rgb/read_rgb/get_rgb "
                "or be directly callable."
            )

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(frame)
    arg_names = tuple(signature.parameters)
    kwargs: dict[str, object] = {}
    if "frame" in arg_names:
        kwargs["frame"] = frame
    if "frame_id" in arg_names:
        kwargs["frame_id"] = frame.frame_id
    if "frame_index" in arg_names:
        kwargs["frame_index"] = frame.camera_index
    if "camera_index" in arg_names:
        kwargs["camera_index"] = frame.camera_index
    if "sensor_id" in arg_names:
        kwargs["sensor_id"] = frame.sensor_id
    if "timestamp_us" in arg_names:
        kwargs["timestamp_us"] = frame.timestamp_us
    if kwargs:
        return method(**kwargs)
    if not arg_names:
        return method()
    return method(frame)


@beartype
@dataclass(frozen=True)
class NCoreCameraImageSource:
    """Reader-backed image source for optional ncore camera sensors."""

    reader: object

    def load_rgb(
        self,
        frame: DatasetFrame,
    ) -> UInt8[np.ndarray, "height width 3"]:
        """Load an RGB image through an ncore camera reader."""
        return _normalize_rgb_array(_invoke_image_reader(self.reader, frame))


@beartype
@dataclass(frozen=True)
class PointCloudState:
    """Optional point cloud associated with a dataset."""

    points: Float[Tensor, "num_points 3"]
    colors: Float[Tensor, "num_points 3"] | None = None
    confidence: Float[Tensor, " num_points"] | None = None

    def transformed(
        self,
        rotation: Float[Tensor, " 3 3"],
        translation: Float[Tensor, " 3"],
    ) -> PointCloudState:
        """Apply a rigid transform in world space."""
        transformed_points = self.points @ rotation.T + translation
        return replace(self, points=transformed_points)


@beartype
@dataclass(frozen=True, kw_only=True)
class DatasetSensor:
    """Canonical multi-sensor dataset record."""

    sensor_id: str
    kind: SensorKind
    frames: tuple[DatasetFrame, ...]
    timestamps_us: tuple[int | None, ...]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.timestamps_us):
            raise ValueError(
                "DatasetSensor timestamps must align with frames: "
                f"got {len(self.timestamps_us)} timestamps for "
                f"{len(self.frames)} frames."
            )
        for frame, timestamp_us in zip(
            self.frames, self.timestamps_us, strict=True
        ):
            if frame.sensor_id != self.sensor_id:
                raise ValueError(
                    "DatasetSensor frames must reference the owning sensor id: "
                    f"expected {self.sensor_id!r}, got {frame.sensor_id!r}."
                )
            if frame.timestamp_us != timestamp_us:
                raise ValueError(
                    "DatasetSensor timestamps_us must match frame metadata."
                )


@beartype
@dataclass(frozen=True, kw_only=True)
class CameraSensorDataset(DatasetSensor):
    """Camera-backed sensor with batched poses and image loading."""

    camera: CameraState
    image_source: CameraImageSource
    mask_source: object | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "camera":
            raise ValueError(
                "CameraSensorDataset requires kind='camera', got "
                f"{self.kind!r}."
            )
        num_frames = len(self.frames)
        if int(self.camera.width.shape[0]) != num_frames:
            raise ValueError(
                "CameraSensorDataset requires one camera entry per frame: "
                f"got {self.camera.width.shape[0]} cameras for {num_frames} "
                "frames."
            )


@beartype
@dataclass(frozen=True)
class SceneDataset:
    """Immutable scene-level dataset record."""

    sensors: tuple[DatasetSensor, ...]
    source_format: DatasetSource
    default_camera_sensor_id: str | None = None
    source_uris: tuple[str, ...] | None = None
    point_cloud: PointCloudState | None = None
    world_up: Float[Tensor, " 3"] | None = None
    focus_point: Float[Tensor, " 3"] | None = None

    def __post_init__(self) -> None:
        sensor_ids = tuple(sensor.sensor_id for sensor in self.sensors)
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("SceneDataset sensor ids must be unique.")
        camera_sensor_ids = self.available_camera_sensor_ids
        default_camera_sensor_id = self.default_camera_sensor_id
        if default_camera_sensor_id is None and len(camera_sensor_ids) == 1:
            default_camera_sensor_id = camera_sensor_ids[0]
            object.__setattr__(
                self,
                "default_camera_sensor_id",
                default_camera_sensor_id,
            )
        if default_camera_sensor_id is not None and (
            default_camera_sensor_id not in camera_sensor_ids
        ):
            raise ValueError(
                "SceneDataset.default_camera_sensor_id must reference a camera "
                f"sensor, got {default_camera_sensor_id!r}."
            )
        if len(camera_sensor_ids) > 1 and default_camera_sensor_id is None:
            raise ValueError(
                "SceneDataset with multiple camera sensors requires "
                "default_camera_sensor_id."
            )

    @property
    def available_camera_sensor_ids(self) -> tuple[str, ...]:
        """Return the registered camera sensor ids."""
        return tuple(
            sensor.sensor_id
            for sensor in self.sensors
            if isinstance(sensor, CameraSensorDataset)
        )

    @property
    def camera_sensors(self) -> tuple[CameraSensorDataset, ...]:
        """Return all camera sensors."""
        return tuple(
            sensor
            for sensor in self.sensors
            if isinstance(sensor, CameraSensorDataset)
        )

    def resolve_camera_sensor(
        self,
        camera_sensor_id: str | None = None,
    ) -> CameraSensorDataset:
        """Resolve a concrete camera sensor for compatibility or batching."""
        resolved_sensor_id = camera_sensor_id or self.default_camera_sensor_id
        if resolved_sensor_id is not None:
            for sensor in self.camera_sensors:
                if sensor.sensor_id == resolved_sensor_id:
                    return sensor
            raise ValueError(
                f"Unknown camera sensor id {resolved_sensor_id!r}. "
                f"Available camera sensors: {self.available_camera_sensor_ids!r}."
            )
        if not self.camera_sensors:
            raise ValueError("SceneDataset does not contain any camera sensors.")
        if len(self.camera_sensors) == 1:
            return self.camera_sensors[0]
        raise ValueError(
            "SceneDataset contains multiple camera sensors but no "
            "default_camera_sensor_id was configured."
        )

    @property
    def default_camera_sensor(self) -> CameraSensorDataset | None:
        """Return the default camera sensor when one is available."""
        if not self.camera_sensors:
            return None
        return self.resolve_camera_sensor()

    @property
    def frames(self) -> tuple[DatasetFrame, ...]:
        """Compatibility view over the default camera stream."""
        return self.resolve_camera_sensor().frames

    @property
    def camera(self) -> CameraState:
        """Compatibility view over the default camera stream."""
        return self.resolve_camera_sensor().camera

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the default camera stream."""
        return len(self.frames)

    @property
    def root_path(self) -> Path | None:
        """Compatibility view over local source provenance when available."""
        if self.source_uris is None or len(self.source_uris) != 1:
            return None
        return _local_path_from_uri(self.source_uris[0])


@beartype
@dataclass(frozen=True)
class PreparedFrameSample:
    """Prepared sample returned by the Torch dataset adapter."""

    frame: DatasetFrame
    image: Float[Tensor, "height width 3"]
    camera: CameraState

    def to(self, device: torch.device) -> PreparedFrameSample:
        """Move tensor fields to a device."""
        return replace(
            self,
            image=self.image.to(device),
            camera=self.camera.to(device),
        )


@beartype
@dataclass(frozen=True)
class DecodedFrameSample:
    """Decoded sample with canonical in-memory image data."""

    frame: DatasetFrame
    image: UInt8[Tensor, "height width 3"]
    camera: CameraState

    def to(self, device: torch.device) -> DecodedFrameSample:
        """Move tensor fields to a device."""
        return replace(
            self,
            image=self.image.to(device),
            camera=self.camera.to(device),
        )


@beartype
@dataclass(frozen=True)
class PreparedFrameBatch:
    """Batched prepared samples."""

    frames: tuple[DatasetFrame, ...]
    images: Float[Tensor, "batch height width 3"]
    camera: CameraState

    def to(self, device: torch.device) -> PreparedFrameBatch:
        """Move tensor fields to a device."""
        return replace(
            self,
            images=self.images.to(device),
            camera=self.camera.to(device),
        )


class HasCamera(Protocol):
    """Batch capability for camera supervision."""

    camera: CameraState


class HasImages(Protocol):
    """Batch capability for RGB supervision."""

    images: Float[Tensor, "batch height width 3"]


class HasDepth(Protocol):
    """Batch capability for depth supervision."""

    depth: Float[Tensor, "batch height width"]


class HasMask(Protocol):
    """Batch capability for binary masks."""

    mask: Float[Tensor, "batch height width"]
