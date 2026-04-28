from __future__ import annotations

import json
import struct
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from ember_core.core.contracts import CameraState
from ember_core.data import (
    CameraSensorDataset,
    ColmapSceneConfig,
    DatasetFrame,
    DatasetSensor,
    HorizonAdjustmentSpec,
    HorizonAlignPipeConfig,
    ImagePreparationSpec,
    MaterializationConfig,
    MipNerf360IndoorPreparedFrameDatasetConfig,
    MipNerf360OutdoorPreparedFrameDatasetConfig,
    NCoreSceneConfig,
    PathCameraImageSource,
    PreparedFrameDataset,
    PreparedFrameDatasetConfig,
    ResizeSpec,
    SceneRecord,
    SplitConfig,
    adjust_scene_record_horizon,
    collate_frame_samples,
    load_colmap_scene_record,
    load_must3r_scene_record,
    load_ncore_scene_record,
    load_scene_record,
    prepare_frame_dataset,
    resolve_must3r_checkpoints,
    run_must3r_scene_record,
)
from ember_core.data.adapters import _resolve_materialization_num_workers
from PIL import Image
from torch.utils.data import DataLoader


def _write_rgb_image(
    path: Path, color: tuple[int, int, int], size: tuple[int, int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[..., 0] = color[0]
    image[..., 1] = color[1]
    image[..., 2] = color[2]
    Image.fromarray(image).save(path)


def _camera_line(
    camera_id: int, model: str, width: int, height: int, params: list[float]
) -> str:
    params_str = " ".join(str(param) for param in params)
    return f"{camera_id} {model} {width} {height} {params_str}"


def _image_line(
    image_id: int,
    qvec: tuple[float, float, float, float],
    tvec: tuple[float, float, float],
    camera_id: int,
    name: str,
) -> str:
    q_str = " ".join(str(value) for value in qvec)
    t_str = " ".join(str(value) for value in tvec)
    return f"{image_id} {q_str} {t_str} {camera_id} {name}"


def _write_colmap_text_model(root: Path, *, distorted: bool = False) -> None:
    sparse_dir = root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    model = "SIMPLE_RADIAL" if distorted else "PINHOLE"
    params = [10.0, 10.0, 8.0, 6.0]
    if distorted:
        params = [10.0, 8.0, 6.0, 0.01]
    (sparse_dir / "cameras.txt").write_text(
        "# Camera list\n" + _camera_line(1, model, 16, 12, params) + "\n"
    )
    image_lines = [
        "# Image list",
        _image_line(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 1, "000.png"),
        "",
        _image_line(2, (1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, -1.0), 1, "001.png"),
        "",
    ]
    (sparse_dir / "images.txt").write_text("\n".join(image_lines))
    (sparse_dir / "points3D.txt").write_text(
        "1 0.0 0.0 0.0 255 0 0 0.1 1 0\n2 0.5 0.0 0.0 0 255 0 0.2 1 1\n"
    )


def _write_colmap_binary_model(root: Path) -> None:
    sparse_dir = root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    with (sparse_dir / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, 16, 12))
        handle.write(struct.pack("<dddd", 10.0, 10.0, 8.0, 6.0))
    with (sparse_dir / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<i", 1))
        handle.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))
        handle.write(struct.pack("<ddd", 0.0, 0.0, -1.0))
        handle.write(struct.pack("<i", 1))
        handle.write(b"000.png\x00")
        handle.write(struct.pack("<Q", 0))
    with (sparse_dir / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<ddd", 0.0, 0.0, 0.0))
        handle.write(struct.pack("<BBB", 255, 0, 0))
        handle.write(struct.pack("<d", 0.1))
        handle.write(struct.pack("<Q", 0))


def _write_images(root: Path) -> None:
    image_dir = root / "images"
    _write_rgb_image(image_dir / "000.png", (255, 0, 0), (16, 12))
    _write_rgb_image(image_dir / "001.png", (0, 255, 0), (16, 12))


def _image_path_for_frame(scene_record: SceneRecord, frame: object) -> Path:
    camera_sensor = scene_record.resolve_camera_sensor()
    image_source = camera_sensor.image_source
    assert isinstance(image_source, PathCameraImageSource)
    return image_source.path_for_frame(frame)


def _build_camera_sensor(
    *,
    sensor_id: str,
    frame_colors: tuple[tuple[int, int, int], ...],
    root: Path,
    base_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CameraSensorDataset:
    frame_paths: dict[str, Path] = {}
    frames = []
    cam_to_world = []
    for index, color in enumerate(frame_colors):
        frame_id = f"{sensor_id}_{index}"
        path = root / f"{frame_id}.png"
        _write_rgb_image(path, color, (4, 3))
        frame_paths[frame_id] = path
        frames.append(
            {
                "frame_id": frame_id,
                "sensor_id": sensor_id,
                "camera_index": index,
                "width": 4,
                "height": 3,
                "timestamp_us": index * 10,
            }
        )
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, 3] = torch.tensor(
            [
                base_translation[0] + index,
                base_translation[1],
                base_translation[2],
            ],
            dtype=torch.float32,
        )
        cam_to_world.append(transform)
    dataset_frames = tuple(
        DatasetFrame(
            frame_id=frame["frame_id"],
            sensor_id=frame["sensor_id"],
            camera_index=frame["camera_index"],
            width=frame["width"],
            height=frame["height"],
            timestamp_us=frame["timestamp_us"],
        )
        for frame in frames
    )
    return CameraSensorDataset(
        sensor_id=sensor_id,
        kind="camera",
        frames=dataset_frames,
        timestamps_us=tuple(frame.timestamp_us for frame in dataset_frames),
        camera=CameraState(
            width=torch.tensor([4] * len(dataset_frames), dtype=torch.int64),
            height=torch.tensor([3] * len(dataset_frames), dtype=torch.int64),
            fov_degrees=torch.tensor(
                [60.0] * len(dataset_frames), dtype=torch.float32
            ),
            cam_to_world=torch.stack(cam_to_world, dim=0),
            intrinsics=torch.tensor(
                [[[2.0, 0.0, 2.0], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]]
                * len(dataset_frames),
                dtype=torch.float32,
            ),
        ),
        image_source=PathCameraImageSource(frame_paths=frame_paths),
    )


@dataclass(frozen=True)
class _MemoryImageSource:
    images: dict[str, np.ndarray]

    def load_rgb(self, frame: DatasetFrame) -> np.ndarray:
        return self.images[frame.frame_id]


@dataclass(frozen=True)
class _DummyNCoreReader:
    images: dict[str, np.ndarray]

    def load_rgb(self, *, frame_id: str) -> np.ndarray:
        return self.images[frame_id]


def _make_ncore_sensor(
    *,
    sensor_id: str,
    colors: tuple[tuple[int, int, int], ...],
    base_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, object]:
    images = {
        f"{sensor_id}_{index}": np.full((3, 4, 3), color, dtype=np.uint8)
        for index, color in enumerate(colors)
    }
    frames = [
        {
            "frame_id": frame_id,
            "timestamp_us": index * 100,
            "width": 4,
            "height": 3,
        }
        for index, frame_id in enumerate(images)
    ]
    cam_to_world = []
    for index in range(len(images)):
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, 3] = torch.tensor(
            [
                base_translation[0] + index,
                base_translation[1],
                base_translation[2],
            ],
            dtype=torch.float32,
        )
        cam_to_world.append(transform)
    return {
        "sensor_id": sensor_id,
        "kind": "camera",
        "frames": frames,
        "camera": CameraState(
            width=torch.tensor([4] * len(images), dtype=torch.int64),
            height=torch.tensor([3] * len(images), dtype=torch.int64),
            fov_degrees=torch.tensor([60.0] * len(images), dtype=torch.float32),
            cam_to_world=torch.stack(cam_to_world, dim=0),
            intrinsics=torch.tensor(
                [[[2.0, 0.0, 2.0], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]]
                * len(images),
                dtype=torch.float32,
            ),
        ),
        "reader": _DummyNCoreReader(images=images),
    }


def test_load_colmap_scene_record_from_text(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path)
    scene_record = load_colmap_scene_record(tmp_path)
    assert scene_record.source_format == "colmap"
    assert scene_record.num_frames == 2
    assert scene_record.point_cloud is not None
    assert scene_record.camera.intrinsics is not None
    assert scene_record.default_camera_sensor_id == "camera"
    assert (
        _image_path_for_frame(scene_record, scene_record.frames[0]).name
        == "000.png"
    )
    assert scene_record.camera.cam_to_world.shape == (2, 4, 4)


def test_load_colmap_scene_record_from_binary(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_rgb_image(image_dir / "000.png", (255, 0, 0), (16, 12))
    _write_colmap_binary_model(tmp_path)
    scene_record = load_colmap_scene_record(tmp_path)
    assert scene_record.num_frames == 1
    assert scene_record.point_cloud is not None
    assert int(scene_record.camera.width[0].item()) == 16


def test_load_colmap_scene_record_requires_undistortion_for_distorted_cameras(
    tmp_path: Path,
) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path, distorted=True)
    with pytest.raises(ValueError, match="undistorted pinhole data"):
        load_colmap_scene_record(tmp_path)


def test_load_colmap_scene_record_can_undistort_into_cache(
    tmp_path: Path,
) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path, distorted=True)
    undistorted_dir = tmp_path / "undistorted"
    scene_record = load_colmap_scene_record(
        tmp_path,
        undistort_output_dir=undistorted_dir,
    )
    assert scene_record.num_frames == 2
    image_path = _image_path_for_frame(scene_record, scene_record.frames[0])
    assert image_path.parent == undistorted_dir
    assert image_path.exists()


def test_torch_frame_dataset_resizes_and_collates(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path)
    scene_record = load_colmap_scene_record(tmp_path)
    frame_dataset = PreparedFrameDataset(
        scene_record,
        preparation=ImagePreparationSpec(
            resize=ResizeSpec(width_target=8),
        ),
    )
    sample = frame_dataset[0]
    assert sample.image.shape == (6, 8, 3)
    assert int(sample.camera.width[0].item()) == 8

    loader = DataLoader(
        frame_dataset,
        batch_size=2,
        collate_fn=collate_frame_samples,
    )
    batch = next(iter(loader))
    assert batch.images.shape == (2, 6, 8, 3)
    assert batch.camera.intrinsics is not None
    assert batch.camera.intrinsics.shape == (2, 3, 3)


def test_materialization_config_rejects_single_worker() -> None:
    with pytest.raises(ValueError, match="0, None, or >= 2"):
        MaterializationConfig(num_workers=1)


def test_materialization_num_workers_resolver_rejects_single_worker() -> None:
    with pytest.raises(ValueError, match="0, None, or >= 2"):
        _resolve_materialization_num_workers(1)


def test_horizon_adjustment_applies_consistent_transform(
    tmp_path: Path,
) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path)
    scene_record = adjust_scene_record_horizon(
        load_colmap_scene_record(tmp_path),
        HorizonAdjustmentSpec(enabled=True),
    )
    assert scene_record.world_up is not None
    assert torch.allclose(
        scene_record.world_up,
        torch.tensor([0.0, 1.0, 0.0]),
    )
    assert scene_record.point_cloud is not None
    assert scene_record.point_cloud.points.shape[1] == 3


def test_colmap_scene_config_serializes_source_pipe_kinds() -> None:
    config = ColmapSceneConfig(
        path=Path("scene"),
        source_pipes=(HorizonAlignPipeConfig(),),
    )
    prepared_config = PreparedFrameDatasetConfig(
        image_preparation={"resize_width_target": 640},
    )

    payload = config.model_dump(mode="json")
    prepared_payload = prepared_config.model_dump(mode="json")

    assert payload["kind"] == "colmap"
    assert payload["source_pipes"][0]["kind"] == "horizon_align"
    assert prepared_payload["image_preparation"]["resize_width_target"] == 640
    assert prepared_payload["image_preparation"]["normalize"] is True


def test_colmap_scene_and_prepared_frame_defaults_are_split() -> None:
    scene_config = ColmapSceneConfig(path=Path("scene"))
    prepared_config = PreparedFrameDatasetConfig()

    scene_payload = scene_config.model_dump(mode="json")
    prepared_payload = prepared_config.model_dump(mode="json")

    assert scene_payload["source_pipes"] == [
        {
            "kind": "horizon_align",
            "enabled": True,
            "target_up": [0.0, 1.0, 0.0],
        }
    ]
    assert prepared_payload["camera_sensor_id"] is None
    assert prepared_payload["split"] == {
        "target": "train",
        "every_n": 8,
        "train_ratio": None,
    }
    assert prepared_payload["materialization"] == {
        "stage": "decoded",
        "mode": "eager",
        "num_workers": 0,
    }
    assert prepared_payload["image_preparation"] == {
        "normalize": True,
        "resize_width_scale": None,
        "resize_width_target": None,
        "interpolation": "bicubic",
    }


def test_mipnerf360_indoor_prepared_frame_config_uses_quarter_scale_resize() -> (
    None
):
    config = MipNerf360IndoorPreparedFrameDatasetConfig()

    payload = config.model_dump(mode="json")

    assert payload["image_preparation"] == {
        "normalize": True,
        "resize_width_scale": 0.25,
        "resize_width_target": None,
        "interpolation": "bicubic",
    }


def test_mipnerf360_outdoor_prepared_frame_config_uses_half_scale_resize() -> (
    None
):
    config = MipNerf360OutdoorPreparedFrameDatasetConfig()

    payload = config.model_dump(mode="json")

    assert payload["image_preparation"] == {
        "normalize": True,
        "resize_width_scale": 0.5,
        "resize_width_target": None,
        "interpolation": "bicubic",
    }


def test_explicit_two_step_colmap_flow_returns_prepared_dataset(
    tmp_path: Path,
) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path)

    scene_record = load_scene_record(
        ColmapSceneConfig(
            path=tmp_path,
        )
    )
    dataset = prepare_frame_dataset(
        scene_record,
        PreparedFrameDatasetConfig(
            image_preparation={"resize_width_target": 8},
        ),
    )

    assert isinstance(dataset, PreparedFrameDataset)
    sample = dataset[0]
    assert sample.image.shape == (6, 8, 3)
    assert sample.image.dtype == torch.float32
    assert int(sample.camera.width[0].item()) == 8


def test_load_scene_record_applies_source_pipes_from_config(
    tmp_path: Path,
) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path)

    scene_record = load_scene_record(
        ColmapSceneConfig(
            path=tmp_path,
        )
    )

    assert scene_record.world_up is not None


def test_scene_record_compatibility_properties_use_default_camera_sensor(
    tmp_path: Path,
) -> None:
    front = _build_camera_sensor(
        sensor_id="front",
        frame_colors=((255, 0, 0),),
        root=tmp_path,
    )
    rear = _build_camera_sensor(
        sensor_id="rear",
        frame_colors=((0, 255, 0), (0, 0, 255)),
        root=tmp_path,
        base_translation=(10.0, 0.0, 0.0),
    )
    lidar = DatasetSensor(
        sensor_id="lidar_top",
        kind="lidar",
        frames=(),
        timestamps_us=(),
        metadata={"channels": 64},
    )
    scene_record = SceneRecord(
        sensors=(front, rear, lidar),
        source_format="ncore",
        default_camera_sensor_id="rear",
        source_uris=(str(tmp_path / "group"),),
    )

    assert scene_record.frames == rear.frames
    assert scene_record.camera == rear.camera
    assert scene_record.num_frames == 2
    assert scene_record.available_camera_sensor_ids == ("front", "rear")


def test_scene_record_can_omit_default_camera_sensor_for_multicam_scenes(
    tmp_path: Path,
) -> None:
    front = _build_camera_sensor(
        sensor_id="front",
        frame_colors=((255, 0, 0),),
        root=tmp_path,
    )
    rear = _build_camera_sensor(
        sensor_id="rear",
        frame_colors=((0, 255, 0),),
        root=tmp_path,
    )

    scene_record = SceneRecord(
        sensors=(front, rear),
        source_format="ncore",
        source_uris=(str(tmp_path / "group"),),
    )
    assert scene_record.default_camera_sensor_id is None
    with pytest.raises(
        ValueError,
        match="multiple camera sensors but no default_camera_sensor_id",
    ):
        scene_record.resolve_camera_sensor()


def test_prepared_frame_dataset_supports_non_path_image_sources() -> None:
    frame = DatasetFrame(
        frame_id="memory_0",
        sensor_id="memory_camera",
        camera_index=0,
        width=4,
        height=3,
        timestamp_us=42,
    )
    scene_record = SceneRecord(
        sensors=(
            CameraSensorDataset(
                sensor_id="memory_camera",
                kind="camera",
                frames=(frame,),
                timestamps_us=(42,),
                camera=CameraState(
                    width=torch.tensor([4], dtype=torch.int64),
                    height=torch.tensor([3], dtype=torch.int64),
                    fov_degrees=torch.tensor([60.0], dtype=torch.float32),
                    cam_to_world=torch.eye(4, dtype=torch.float32)[None],
                ),
                image_source=_MemoryImageSource(
                    images={"memory_0": np.full((3, 4, 3), 128, dtype=np.uint8)}
                ),
            ),
        ),
        source_format="ncore",
        default_camera_sensor_id="memory_camera",
    )

    frame_dataset = PreparedFrameDataset(
        scene_record,
        preparation=ImagePreparationSpec(normalize=False),
    )
    sample = frame_dataset[0]

    assert sample.frame.sensor_id == "memory_camera"
    assert sample.image.shape == (3, 4, 3)
    assert torch.equal(
        sample.image.to(torch.uint8),
        torch.full((3, 4, 3), 128, dtype=torch.uint8),
    )


def test_adjust_scene_record_horizon_updates_all_camera_sensors_consistently(
    tmp_path: Path,
) -> None:
    front = _build_camera_sensor(
        sensor_id="front",
        frame_colors=((255, 0, 0),),
        root=tmp_path,
    )
    rear = _build_camera_sensor(
        sensor_id="rear",
        frame_colors=((0, 255, 0),),
        root=tmp_path,
        base_translation=(2.0, 1.0, 0.0),
    )
    scene_record = SceneRecord(
        sensors=(front, rear),
        source_format="ncore",
        default_camera_sensor_id="front",
        source_uris=(str(tmp_path / "group"),),
    )

    adjusted = adjust_scene_record_horizon(
        scene_record,
        HorizonAdjustmentSpec(enabled=True),
    )

    assert adjusted.world_up is not None
    front_transform = adjusted.resolve_camera_sensor(
        "front"
    ).camera.cam_to_world[0] @ torch.linalg.inv(front.camera.cam_to_world[0])
    rear_transform = adjusted.resolve_camera_sensor("rear").camera.cam_to_world[
        0
    ] @ torch.linalg.inv(rear.camera.cam_to_world[0])
    assert torch.allclose(front_transform, rear_transform, atol=1e-5)


def test_load_ncore_scene_record_discovers_camera_and_inventory_sensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(
        load_component_groups=lambda paths: (
            {
                "sensors": [
                    _make_ncore_sensor(
                        sensor_id="front",
                        colors=((255, 0, 0),),
                    ),
                    _make_ncore_sensor(
                        sensor_id="rear",
                        colors=((0, 255, 0),),
                        base_translation=(1.0, 0.0, 0.0),
                    ),
                    {
                        "sensor_id": "lidar_top",
                        "kind": "lidar",
                        "frames": [
                            {
                                "frame_id": "scan_0",
                                "timestamp_us": 100,
                                "width": 0,
                                "height": 0,
                            }
                        ],
                        "metadata": {"channels": 64},
                    },
                ],
                "point_cloud": {
                    "points": [[0.0, 0.0, 0.0]],
                    "colors": [[1.0, 0.0, 0.0]],
                },
            },
        )
    )
    monkeypatch.setattr(
        "ember_core.data.loaders.ncore._require_ncore",
        lambda: fake_module,
    )

    scene_record = load_ncore_scene_record(
        (tmp_path / "group_a",),
    )

    assert scene_record.source_format == "ncore"
    assert scene_record.default_camera_sensor_id is None
    assert scene_record.available_camera_sensor_ids == ("front", "rear")
    assert any(sensor.kind == "lidar" for sensor in scene_record.sensors)
    assert scene_record.point_cloud is not None

    frame_dataset = PreparedFrameDataset(
        scene_record,
        camera_sensor_id="rear",
        preparation=ImagePreparationSpec(normalize=False),
    )
    sample = frame_dataset[0]
    assert sample.frame.sensor_id == "rear"
    assert sample.image.shape == (3, 4, 3)


def test_explicit_two_step_ncore_flow_selects_prepared_camera_sensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(
        load_component_groups=lambda paths: (
            {
                "sensors": [
                    _make_ncore_sensor(
                        sensor_id="front",
                        colors=((255, 0, 0),),
                    ),
                    _make_ncore_sensor(
                        sensor_id="rear",
                        colors=((0, 255, 0),),
                    ),
                ]
            },
        )
    )
    monkeypatch.setattr(
        "ember_core.data.loaders.ncore._require_ncore",
        lambda: fake_module,
    )

    scene_record = load_scene_record(
        NCoreSceneConfig(
            component_group_paths=(tmp_path / "group_a",),
        )
    )
    dataset = prepare_frame_dataset(
        scene_record,
        PreparedFrameDatasetConfig(
            camera_sensor_id="rear",
            split=SplitConfig(target="all", every_n=None, train_ratio=None),
        ),
    )

    assert isinstance(dataset, PreparedFrameDataset)
    assert dataset.scene_record.default_camera_sensor_id is None
    assert dataset[0].frame.sensor_id == "rear"


def test_load_must3r_scene_record_from_json(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_rgb_image(image_dir / "000.png", (255, 0, 0), (16, 12))
    _write_rgb_image(image_dir / "001.png", (0, 255, 0), (16, 12))
    payload = {
        "frames": [
            {
                "image_path": "images/000.png",
                "cam_to_world": np.eye(4).tolist(),
                "intrinsics": [
                    [10.0, 0.0, 8.0],
                    [0.0, 10.0, 6.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            {
                "image_path": "images/001.png",
                "cam_to_world": np.eye(4).tolist(),
                "intrinsics": [
                    [10.0, 0.0, 8.0],
                    [0.0, 10.0, 6.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        ]
    }
    artifact_path = tmp_path / "dataset.json"
    artifact_path.write_text(json.dumps(payload))
    scene_record = load_must3r_scene_record(artifact_path)
    assert scene_record.source_format == "must3r"
    assert scene_record.num_frames == 2


def test_resolve_must3r_checkpoints_uses_huggingface_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloads: list[tuple[str, str]] = []

    def fake_download(
        *, repo_id: str, filename: str, cache_dir: str | Path | None
    ) -> str:
        del cache_dir
        downloads.append((repo_id, filename))
        checkpoint_path = tmp_path / filename
        checkpoint_path.write_bytes(b"weights")
        return str(checkpoint_path)

    monkeypatch.setattr(
        "ember_core.data.loaders.must3r._require_huggingface_hub",
        lambda: fake_download,
    )
    checkpoints = resolve_must3r_checkpoints(
        checkpoint_repo_id="naver/must3r",
        checkpoint_filename="model.pth",
    )
    assert checkpoints.checkpoint_path.name == "model.pth"
    assert downloads == [("naver/must3r", "model.pth")]


def test_run_must3r_scene_record_errors_when_runtime_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"weights")
    monkeypatch.setattr(
        "ember_core.data.loaders.must3r.resolve_must3r_checkpoints",
        lambda **_: type(
            "Checkpoints",
            (),
            {
                "checkpoint_path": checkpoint_path,
                "retrieval_checkpoint_path": None,
            },
        )(),
    )
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="MUSt3R runtime is not installed"):
        run_must3r_scene_record(
            tmp_path,
            output_dir=tmp_path / "must3r_output",
            checkpoint_repo_id="naver/must3r",
            checkpoint_filename="model.pth",
        )


@dataclass(frozen=True)
class _StubMust3rRuntime:
    artifact_path: Path

    def run(
        self,
        *,
        image_dir: Path,
        output_dir: Path,
        checkpoints: object,
        image_size: int,
        device: str,
    ) -> Path:
        del image_dir, output_dir, checkpoints, image_size, device
        return self.artifact_path


def test_run_must3r_scene_record_with_stub_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images"
    _write_rgb_image(image_dir / "000.png", (255, 0, 0), (16, 12))
    payload = {
        "frames": [
            {
                "image_path": "000.png",
                "cam_to_world": np.eye(4).tolist(),
                "intrinsics": [
                    [10.0, 0.0, 8.0],
                    [0.0, 10.0, 6.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        ]
    }
    artifact_path = tmp_path / "dataset.json"
    artifact_path.write_text(json.dumps(payload))
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"weights")
    monkeypatch.setattr(
        "ember_core.data.loaders.must3r.resolve_must3r_checkpoints",
        lambda **_: type(
            "Checkpoints",
            (),
            {
                "checkpoint_path": checkpoint_path,
                "retrieval_checkpoint_path": None,
            },
        )(),
    )
    scene_record = run_must3r_scene_record(
        image_dir,
        output_dir=tmp_path / "unused_output",
        checkpoint_repo_id="naver/must3r",
        checkpoint_filename="model.pth",
        runtime=_StubMust3rRuntime(artifact_path=artifact_path),
    )
    assert isinstance(scene_record, SceneRecord)
    assert scene_record.num_frames == 1
