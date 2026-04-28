"""Optional ncore dataset import helpers."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from beartype import beartype

from ember_core.data.contracts import (
    CameraSensorDataset,
    DatasetFrame,
    DatasetSensor,
    NCoreCameraImageSource,
    PointCloudState,
    SceneRecord,
    SensorKind,
)


def _require_ncore() -> object:
    try:
        return import_module("ncore")
    except ImportError as exc:
        raise RuntimeError(
            "Loading ncore datasets requires the optional `ncore` package."
        ) from exc


_MISSING = object()


def _get_value(obj: object, *names: str, default: object = _MISSING) -> object:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not _MISSING:
        return default
    raise AttributeError(f"Could not resolve any of the fields {names!r}.")


def _maybe_iterable(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _normalize_sensor_kind(value: object) -> SensorKind:
    name = str(value).lower()
    if name in {"camera", "cam", "rgb"}:
        return "camera"
    if name in {"lidar", "laser"}:
        return "lidar"
    if name == "radar":
        return "radar"
    return "other"


def _load_component_groups(
    ncore_module: object,
    component_group_paths: tuple[Path, ...],
) -> tuple[object, ...]:
    if hasattr(ncore_module, "load_component_groups"):
        groups = ncore_module.load_component_groups(component_group_paths)
        return _maybe_iterable(groups)
    if hasattr(ncore_module, "load_component_group"):
        return tuple(
            ncore_module.load_component_group(path)
            for path in component_group_paths
        )
    raise RuntimeError(
        "Unsupported ncore installation: expected load_component_groups "
        "or load_component_group."
    )


def _extract_group_sensors(group: object) -> tuple[object, ...]:
    sensors = _get_value(
        group,
        "sensors",
        "sensor_components",
        "components",
        default=None,
    )
    if sensors is None:
        kind = _get_value(group, "kind", "sensor_kind", default=None)
        if kind is None:
            return ()
        return (group,)
    return _maybe_iterable(sensors)


def _camera_frames_from_sensor(
    sensor_id: str,
    sensor: object,
    *,
    camera_width: torch.Tensor,
    camera_height: torch.Tensor,
) -> tuple[DatasetFrame, ...]:
    frame_payloads = _get_value(
        sensor,
        "frames",
        "frame_metadata",
        "samples",
        default=None,
    )
    frames: list[DatasetFrame] = []
    if frame_payloads is None:
        for index in range(int(camera_width.shape[0])):
            frames.append(
                DatasetFrame(
                    frame_id=str(index),
                    sensor_id=sensor_id,
                    camera_index=index,
                    width=int(camera_width[index].item()),
                    height=int(camera_height[index].item()),
                    timestamp_us=None,
                )
            )
        return tuple(frames)

    for index, frame in enumerate(_maybe_iterable(frame_payloads)):
        timestamp_us = _get_value(
            frame,
            "timestamp_us",
            "timestamp",
            "time_us",
            default=None,
        )
        frames.append(
            DatasetFrame(
                frame_id=str(
                    _get_value(frame, "frame_id", "id", default=index)
                ),
                sensor_id=sensor_id,
                camera_index=index,
                width=int(
                    _get_value(
                        frame,
                        "width",
                        default=int(camera_width[index].item()),
                    )
                ),
                height=int(
                    _get_value(
                        frame,
                        "height",
                        default=int(camera_height[index].item()),
                    )
                ),
                timestamp_us=(
                    None if timestamp_us is None else int(timestamp_us)
                ),
            )
        )
    return tuple(frames)


def _inventory_frames_from_sensor(
    sensor_id: str,
    sensor: object,
) -> tuple[DatasetFrame, ...]:
    frame_payloads = _get_value(
        sensor,
        "frames",
        "frame_metadata",
        "samples",
        default=None,
    )
    if frame_payloads is None:
        return ()
    frames: list[DatasetFrame] = []
    for index, frame in enumerate(_maybe_iterable(frame_payloads)):
        timestamp_us = _get_value(
            frame,
            "timestamp_us",
            "timestamp",
            "time_us",
            default=None,
        )
        frames.append(
            DatasetFrame(
                frame_id=str(
                    _get_value(frame, "frame_id", "id", default=index)
                ),
                sensor_id=sensor_id,
                camera_index=index,
                width=int(_get_value(frame, "width", default=0)),
                height=int(_get_value(frame, "height", default=0)),
                timestamp_us=(
                    None if timestamp_us is None else int(timestamp_us)
                ),
            )
        )
    return tuple(frames)


def _build_camera_sensor(sensor: object) -> CameraSensorDataset:
    sensor_id = str(_get_value(sensor, "sensor_id", "id", "name"))
    camera = _get_value(sensor, "camera", "camera_state")
    if not hasattr(camera, "cam_to_world"):
        raise TypeError(
            f"ncore camera sensor {sensor_id!r} does not expose CameraState."
        )
    reader = _get_value(
        sensor,
        "image_source",
        "reader",
        "camera_reader",
        "rgb_reader",
    )
    frames = _camera_frames_from_sensor(
        sensor_id,
        sensor,
        camera_width=camera.width,
        camera_height=camera.height,
    )
    return CameraSensorDataset(
        sensor_id=sensor_id,
        kind="camera",
        frames=frames,
        timestamps_us=tuple(frame.timestamp_us for frame in frames),
        metadata=_get_value(sensor, "metadata", "attrs", default=None),
        camera=camera,
        image_source=NCoreCameraImageSource(reader=reader),
    )


def _build_inventory_sensor(sensor: object) -> DatasetSensor:
    sensor_id = str(_get_value(sensor, "sensor_id", "id", "name"))
    kind = _normalize_sensor_kind(
        _get_value(sensor, "kind", "sensor_kind", "type", "modality")
    )
    frames = _inventory_frames_from_sensor(sensor_id, sensor)
    return DatasetSensor(
        sensor_id=sensor_id,
        kind=kind,
        frames=frames,
        timestamps_us=tuple(frame.timestamp_us for frame in frames),
        metadata=_get_value(sensor, "metadata", "attrs", default=None),
    )


def _build_point_cloud(value: object | None) -> PointCloudState | None:
    if value is None:
        return None
    points = _get_value(value, "points", "xyz", default=None)
    if points is None:
        return None
    colors = _get_value(value, "colors", "rgb", default=None)
    confidence = _get_value(value, "confidence", "conf", default=None)
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


def _extract_point_cloud(groups: tuple[object, ...]) -> PointCloudState | None:
    for group in groups:
        point_cloud = _build_point_cloud(
            _get_value(
                group,
                "point_cloud",
                "point_source",
                "points",
                default=None,
            )
        )
        if point_cloud is not None:
            return point_cloud
    return None


@beartype
def load_ncore_dataset(
    component_group_paths: tuple[str | Path, ...],
    *,
    camera_sensor_id: str | None = None,
) -> SceneRecord:
    """Load an ncore component-group inventory into a SceneRecord."""
    resolved_paths = tuple(Path(path) for path in component_group_paths)
    ncore_module = _require_ncore()
    groups = _load_component_groups(ncore_module, resolved_paths)
    sensors: list[DatasetSensor] = []
    for group in groups:
        for sensor in _extract_group_sensors(group):
            kind = _normalize_sensor_kind(
                _get_value(
                    sensor,
                    "kind",
                    "sensor_kind",
                    "type",
                    "modality",
                )
            )
            if kind == "camera":
                sensors.append(_build_camera_sensor(sensor))
            else:
                sensors.append(_build_inventory_sensor(sensor))
    if not sensors:
        raise ValueError("ncore loader did not discover any sensors.")
    camera_sensor_ids = tuple(
        sensor.sensor_id
        for sensor in sensors
        if isinstance(sensor, CameraSensorDataset)
    )
    default_camera_sensor_id = camera_sensor_id
    if default_camera_sensor_id is None and len(camera_sensor_ids) == 1:
        default_camera_sensor_id = camera_sensor_ids[0]
    return SceneRecord(
        sensors=tuple(sensors),
        source_format="ncore",
        default_camera_sensor_id=default_camera_sensor_id,
        source_uris=tuple(str(path) for path in resolved_paths),
        point_cloud=_extract_point_cloud(groups),
    )
