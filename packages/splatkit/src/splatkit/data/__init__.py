"""Public dataset utilities for splatkit."""

from pathlib import Path
from typing import Literal

from splatkit.data.adapters import FrameDataset, collate_frame_samples
from splatkit.data.contracts import (
    DatasetFrame,
    HasCamera,
    HasDepthTargets,
    HasMaskTargets,
    HasRgbTargets,
    HorizonAdjustmentSpec,
    ImagePreparationSpec,
    PointCloudState,
    PreparedFrameBatch,
    PreparedFrameSample,
    ResizeSpec,
    SceneDataset,
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
from splatkit.data.postprocess import adjust_dataset_horizon

DatasetFormat = Literal["colmap", "must3r"]


def load_dataset(
    path: str | Path,
    *,
    format: DatasetFormat | None = None,
    **kwargs: object,
) -> SceneDataset:
    """Load a dataset from a supported source."""
    resolved_path = Path(path)
    resolved_format = format
    if resolved_format is None:
        if resolved_path.is_dir() and (
            (resolved_path / "sparse").exists()
            or (resolved_path / "cameras.bin").exists()
            or (resolved_path / "cameras.txt").exists()
        ):
            resolved_format = "colmap"
        elif resolved_path.is_dir() and (
            (resolved_path / "dataset.json").exists()
            or (resolved_path / "all_poses.npz").exists()
        ):
            resolved_format = "must3r"
        elif resolved_path.suffix in {".json", ".npz"}:
            resolved_format = "must3r"
        else:
            raise ValueError(f"Could not infer dataset format from {path}.")
    if resolved_format == "colmap":
        return load_colmap_dataset(resolved_path, **kwargs)
    if resolved_format == "must3r":
        return load_must3r_dataset(resolved_path, **kwargs)
    raise ValueError(f"Unsupported dataset format {resolved_format!r}.")


__all__ = [
    "DatasetFormat",
    "DatasetFrame",
    "FrameDataset",
    "HasCamera",
    "HasDepthTargets",
    "HasMaskTargets",
    "HasRgbTargets",
    "HorizonAdjustmentSpec",
    "ImagePreparationSpec",
    "Must3rCheckpointPaths",
    "Must3rRuntime",
    "PointCloudState",
    "PreparedFrameBatch",
    "PreparedFrameSample",
    "ResizeSpec",
    "SceneDataset",
    "SubprocessMust3rSlamRuntime",
    "adjust_dataset_horizon",
    "collate_frame_samples",
    "load_colmap_dataset",
    "load_dataset",
    "load_must3r_dataset",
    "resolve_must3r_checkpoints",
    "run_must3r_dataset",
]
