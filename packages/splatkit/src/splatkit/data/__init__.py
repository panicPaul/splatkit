"""Public scene-record and prepared-frame data utilities for splatkit."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

from splatkit.data.adapters import PreparedFrameDataset, collate_frame_samples
from splatkit.data.config import (
    ColmapSceneConfig,
    MipNerf360IndoorPreparedFrameDatasetConfig,
    MipNerf360OutdoorPreparedFrameDatasetConfig,
    NCoreSceneConfig,
)
from splatkit.data.config_contracts import (
    ImagePreparationConfig,
    MaterializationConfig,
    PreparedFrameDatasetConfig,
    SceneLoadConfig,
    SplitConfig,
)
from splatkit.data.contracts import (
    CameraImageSource,
    CameraSensorDataset,
    DatasetFrame,
    DatasetSensor,
    DecodedFrameSample,
    HasCamera,
    HasDepth,
    HasImages,
    HasMask,
    HorizonAdjustmentSpec,
    ImagePreparationSpec,
    MaterializationMode,
    MaterializationStage,
    NCoreCameraImageSource,
    PathCameraImageSource,
    PointCloudState,
    PreparedFrameBatch,
    PreparedFrameSample,
    ResizeSpec,
    SceneRecord,
    SensorKind,
)
from splatkit.data.loaders.colmap import load_colmap_dataset
from splatkit.data.loaders.must3r import (
    Must3rCheckpointPaths,
    Must3rRuntime,
    SubprocessMust3rSlamRuntime,
    load_must3r_dataset,
    resolve_must3r_checkpoints,
    run_must3r_dataset,
)
from splatkit.data.loaders.ncore import load_ncore_dataset
from splatkit.data.pipes import (
    HorizonAlignPipeConfig,
    NormalizePipeConfig,
    ResizePipeConfig,
    SourcePipeConfig,
    apply_source_pipe,
)
from splatkit.data.postprocess import (
    adjust_scene_record_horizon as _adjust_scene_record_horizon,
)
from splatkit.data.samples import (
    get_sample_scene_path,
    resolve_colmap_scene_path,
)

SceneSourceFormat = Literal["colmap", "must3r", "ncore"]


def _infer_scene_source_format(path: Path) -> SceneSourceFormat:
    if path.is_dir() and (
        (path / "sparse").exists()
        or (path / "cameras.bin").exists()
        or (path / "cameras.txt").exists()
    ):
        return "colmap"
    if path.is_dir() and (
        (path / "dataset.json").exists() or (path / "all_poses.npz").exists()
    ):
        return "must3r"
    if path.suffix in {".json", ".npz"}:
        return "must3r"
    raise ValueError(f"Could not infer scene source format from {path}.")


def load_colmap_scene_record(
    path: str | Path,
    *,
    image_root: str | Path | None = None,
    undistort_output_dir: str | Path | None = None,
) -> SceneRecord:
    """Load a COLMAP sparse model into a scene record."""
    return load_colmap_dataset(
        path,
        image_root=image_root,
        undistort_output_dir=undistort_output_dir,
    )


def load_must3r_scene_record(path: str | Path, **kwargs: object) -> SceneRecord:
    """Load MUSt3R outputs into a scene record."""
    return load_must3r_dataset(path, **kwargs)


def run_must3r_scene_record(path: str | Path, **kwargs: object) -> SceneRecord:
    """Run MUSt3R preprocessing and load the resulting scene record."""
    return run_must3r_dataset(path, **kwargs)


def load_ncore_scene_record(
    component_group_paths: tuple[str | Path, ...],
) -> SceneRecord:
    """Load an ncore component-group inventory into a scene record."""
    return load_ncore_dataset(component_group_paths)


def adjust_scene_record_horizon(
    scene_record: SceneRecord,
    adjustment: HorizonAdjustmentSpec,
) -> SceneRecord:
    """Adjust the canonical up direction of a scene record."""
    return _adjust_scene_record_horizon(scene_record, adjustment)


def _load_scene_record_from_config(config: SceneLoadConfig) -> SceneRecord:
    if isinstance(config, ColmapSceneConfig):
        scene_record = load_colmap_scene_record(
            config.path,
            image_root=config.image_root,
            undistort_output_dir=config.undistort_output_dir,
        )
    elif isinstance(config, NCoreSceneConfig):
        scene_record = load_ncore_scene_record(config.component_group_paths)
    else:
        raise ValueError(f"Unsupported scene-load config type {type(config)!r}.")
    for pipe in config.source_pipes:
        scene_record = apply_source_pipe(scene_record, pipe)
    return scene_record


@overload
def load_scene_record(config: SceneLoadConfig) -> SceneRecord: ...


@overload
def load_scene_record(
    path: str | Path,
    *,
    format: SceneSourceFormat | None = None,
    **kwargs: object,
) -> SceneRecord: ...


def load_scene_record(
    path_or_config: SceneLoadConfig | str | Path,
    *,
    format: SceneSourceFormat | None = None,
    **kwargs: object,
) -> SceneRecord:
    """Load a canonical scene record from config or a raw path."""
    if isinstance(path_or_config, SceneLoadConfig):
        return _load_scene_record_from_config(path_or_config)

    resolved_path = Path(path_or_config)
    resolved_format = format or _infer_scene_source_format(resolved_path)
    if resolved_format == "colmap":
        return load_colmap_scene_record(resolved_path, **kwargs)
    if resolved_format == "must3r":
        return load_must3r_scene_record(resolved_path, **kwargs)
    if resolved_format == "ncore":
        raise ValueError(
            "Raw path loading does not support ncore scene records. "
            "Use NCoreSceneConfig instead."
        )
    raise ValueError(f"Unsupported scene source format {resolved_format!r}.")


def prepare_frame_dataset(
    scene_record: SceneRecord,
    config: PreparedFrameDatasetConfig | None = None,
) -> PreparedFrameDataset:
    """Build a prepared frame dataset from a canonical scene record."""
    return PreparedFrameDataset(scene_record, config=config)


__all__ = [
    "CameraImageSource",
    "CameraSensorDataset",
    "ColmapSceneConfig",
    "DatasetFrame",
    "DatasetSensor",
    "DecodedFrameSample",
    "HasCamera",
    "HasDepth",
    "HasImages",
    "HasMask",
    "HorizonAdjustmentSpec",
    "HorizonAlignPipeConfig",
    "ImagePreparationConfig",
    "ImagePreparationSpec",
    "MaterializationConfig",
    "MaterializationMode",
    "MaterializationStage",
    "MipNerf360IndoorPreparedFrameDatasetConfig",
    "MipNerf360OutdoorPreparedFrameDatasetConfig",
    "Must3rCheckpointPaths",
    "Must3rRuntime",
    "NCoreCameraImageSource",
    "NCoreSceneConfig",
    "NormalizePipeConfig",
    "PathCameraImageSource",
    "PointCloudState",
    "PreparedFrameBatch",
    "PreparedFrameDataset",
    "PreparedFrameDatasetConfig",
    "PreparedFrameSample",
    "ResizePipeConfig",
    "ResizeSpec",
    "SceneLoadConfig",
    "SceneRecord",
    "SceneSourceFormat",
    "SensorKind",
    "SourcePipeConfig",
    "SplitConfig",
    "SubprocessMust3rSlamRuntime",
    "adjust_scene_record_horizon",
    "collate_frame_samples",
    "get_sample_scene_path",
    "load_colmap_scene_record",
    "load_must3r_scene_record",
    "load_ncore_scene_record",
    "load_scene_record",
    "prepare_frame_dataset",
    "resolve_colmap_scene_path",
    "resolve_must3r_checkpoints",
    "run_must3r_scene_record",
]
