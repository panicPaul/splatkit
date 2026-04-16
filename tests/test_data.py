from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from splatkit.data import (
    ColmapDatasetConfig,
    FrameDataset,
    HorizonAdjustmentSpec,
    HorizonAlignPipeConfig,
    ImagePreparationSpec,
    MaterializationConfig,
    MipNerf360IndoorDatasetConfig,
    MipNerf360OutdoorDatasetConfig,
    NormalizePipeConfig,
    ResizePipeConfig,
    ResizeSpec,
    SceneDataset,
    collate_frame_samples,
    load_colmap_dataset,
    load_dataset,
    load_must3r_dataset,
    resolve_must3r_checkpoints,
    run_must3r_dataset,
)
from splatkit.data.adapters import _resolve_materialization_num_workers
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


def test_load_colmap_dataset_from_text(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path)
    dataset = load_colmap_dataset(tmp_path)
    assert dataset.source_format == "colmap"
    assert dataset.num_frames == 2
    assert dataset.point_cloud is not None
    assert dataset.camera.intrinsics is not None
    assert dataset.frames[0].image_path.name == "000.png"
    assert dataset.camera.cam_to_world.shape == (2, 4, 4)


def test_load_colmap_dataset_from_binary(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_rgb_image(image_dir / "000.png", (255, 0, 0), (16, 12))
    _write_colmap_binary_model(tmp_path)
    dataset = load_colmap_dataset(tmp_path)
    assert dataset.num_frames == 1
    assert dataset.point_cloud is not None
    assert int(dataset.camera.width[0].item()) == 16


def test_load_colmap_dataset_requires_undistortion_for_distorted_cameras(
    tmp_path: Path,
) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path, distorted=True)
    with pytest.raises(ValueError, match="undistorted pinhole data"):
        load_colmap_dataset(tmp_path)


def test_load_colmap_dataset_can_undistort_into_cache(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path, distorted=True)
    undistorted_dir = tmp_path / "undistorted"
    dataset = load_colmap_dataset(
        tmp_path,
        undistort_output_dir=undistorted_dir,
    )
    assert dataset.num_frames == 2
    assert dataset.frames[0].image_path.parent == undistorted_dir
    assert dataset.frames[0].image_path.exists()


def test_torch_frame_dataset_resizes_and_collates(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path)
    scene_dataset = load_colmap_dataset(tmp_path)
    frame_dataset = FrameDataset(
        scene_dataset,
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
    dataset = load_colmap_dataset(
        tmp_path,
        horizon_adjustment=HorizonAdjustmentSpec(enabled=True),
    )
    assert dataset.world_up is not None
    assert torch.allclose(
        dataset.world_up,
        torch.tensor([0.0, 1.0, 0.0]),
    )
    assert dataset.point_cloud is not None
    assert dataset.point_cloud.points.shape[1] == 3


def test_colmap_dataset_config_serializes_pipe_kinds() -> None:
    config = ColmapDatasetConfig(
        path=Path("scene"),
        source_pipes=(HorizonAlignPipeConfig(),),
        cache_pipes=(ResizePipeConfig(width_target=640),),
        prepare_pipes=(NormalizePipeConfig(),),
    )

    payload = config.model_dump(mode="json")

    assert payload["kind"] == "colmap"
    assert payload["source_pipes"][0]["kind"] == "horizon_align"
    assert payload["cache_pipes"][0]["kind"] == "resize"
    assert payload["prepare_pipes"][0]["kind"] == "normalize"


def test_colmap_dataset_config_exposes_all_default_pipe_phases() -> None:
    config = ColmapDatasetConfig(path=Path("scene"))

    payload = config.model_dump(mode="json")

    assert payload["runtime"] == {
        "split": {"target": "train", "every_n": 8, "train_ratio": None},
        "materialization": {
            "stage": "decoded",
            "mode": "eager",
            "num_workers": 0,
        },
    }
    assert payload["source_pipes"] == [
        {
            "kind": "horizon_align",
            "enabled": True,
            "target_up": [0.0, 1.0, 0.0],
        }
    ]
    assert payload["cache_pipes"] == [
        {
            "kind": "resize",
            "width_scale": None,
            "width_target": 1980,
            "interpolation": "bicubic",
        }
    ]
    assert payload["prepare_pipes"] == [{"kind": "normalize", "enabled": True}]


def test_mipnerf360_indoor_dataset_config_uses_quarter_scale_resize() -> None:
    config = MipNerf360IndoorDatasetConfig(path=Path("scene"))

    payload = config.model_dump(mode="json")

    assert payload["kind"] == "colmap"
    assert payload["cache_pipes"] == [
        {
            "kind": "resize",
            "width_scale": 0.25,
            "width_target": None,
            "interpolation": "bicubic",
        }
    ]


def test_mipnerf360_outdoor_dataset_config_uses_half_scale_resize() -> None:
    config = MipNerf360OutdoorDatasetConfig(path=Path("scene"))

    payload = config.model_dump(mode="json")

    assert payload["kind"] == "colmap"
    assert payload["cache_pipes"] == [
        {
            "kind": "resize",
            "width_scale": 0.5,
            "width_target": None,
            "interpolation": "bicubic",
        }
    ]


def test_load_dataset_from_colmap_config_returns_prepared_dataset(
    tmp_path: Path,
) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path)

    dataset = load_dataset(
        ColmapDatasetConfig(
            path=tmp_path,
            cache_pipes=(ResizePipeConfig(width_target=8),),
        )
    )

    assert isinstance(dataset, FrameDataset)
    sample = dataset[0]
    assert sample.image.shape == (6, 8, 3)
    assert sample.image.dtype == torch.float32
    assert int(sample.camera.width[0].item()) == 8


def test_load_dataset_applies_source_pipes_from_config(tmp_path: Path) -> None:
    _write_images(tmp_path)
    _write_colmap_text_model(tmp_path)

    dataset = load_dataset(
        ColmapDatasetConfig(
            path=tmp_path,
        )
    )

    assert isinstance(dataset, FrameDataset)
    assert dataset.dataset.world_up is not None


def test_load_must3r_dataset_from_json(tmp_path: Path) -> None:
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
    dataset = load_must3r_dataset(artifact_path)
    assert dataset.source_format == "must3r"
    assert dataset.num_frames == 2


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
        "splatkit.data.loaders.must3r._require_huggingface_hub",
        lambda: fake_download,
    )
    checkpoints = resolve_must3r_checkpoints(
        checkpoint_repo_id="naver/must3r",
        checkpoint_filename="model.pth",
    )
    assert checkpoints.checkpoint_path.name == "model.pth"
    assert downloads == [("naver/must3r", "model.pth")]


def test_run_must3r_dataset_errors_when_runtime_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"weights")
    monkeypatch.setattr(
        "splatkit.data.loaders.must3r.resolve_must3r_checkpoints",
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
        run_must3r_dataset(
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


def test_run_must3r_dataset_with_stub_runtime(
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
        "splatkit.data.loaders.must3r.resolve_must3r_checkpoints",
        lambda **_: type(
            "Checkpoints",
            (),
            {
                "checkpoint_path": checkpoint_path,
                "retrieval_checkpoint_path": None,
            },
        )(),
    )
    dataset = run_must3r_dataset(
        image_dir,
        output_dir=tmp_path / "unused_output",
        checkpoint_repo_id="naver/must3r",
        checkpoint_filename="model.pth",
        runtime=_StubMust3rRuntime(artifact_path=artifact_path),
    )
    assert isinstance(dataset, SceneDataset)
    assert dataset.num_frames == 1
