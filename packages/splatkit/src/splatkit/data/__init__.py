"""Public dataset utilities for splatkit.

The primary user path is:

```python
from splatkit.data import ColmapDatasetConfig, ResizePipeConfig, load_dataset

dataset = load_dataset(
    ColmapDatasetConfig(
        path="scene_dir",
        cache_pipes=(ResizePipeConfig(width_target=1280),),
    )
)
```

Advanced users should generally subclass a concrete dataset config and override
its ordered pipe tuples rather than assembling ad hoc generic pipeline data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

from splatkit.data.adapters import FrameDataset, collate_frame_samples
from splatkit.data.config import (
    ColmapDatasetConfig,
    DatasetConfig,
    DatasetRuntimeConfig,
    FrameDatasetConfig,
    ImagePreparationConfig,
    MaterializationConfig,
    SplitConfig,
)
from splatkit.data.contracts import (
    DatasetFrame,
    DecodedFrameSample,
    HasCamera,
    HasDepth,
    HasImages,
    HasMask,
    HorizonAdjustmentSpec,
    ImagePreparationSpec,
    MaterializationMode,
    MaterializationStage,
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
from splatkit.data.pipes import (
    CachePipeConfig,
    HorizonAlignPipeConfig,
    NormalizePipeConfig,
    PreparePipeConfig,
    ResizePipeConfig,
    SourcePipeConfig,
    apply_source_pipe,
)
from splatkit.data.postprocess import adjust_dataset_horizon

DatasetFormat = Literal["colmap", "must3r"]


def _infer_dataset_format(path: Path) -> DatasetFormat:
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
    raise ValueError(f"Could not infer dataset format from {path}.")


def _load_scene_from_config(config: DatasetConfig) -> SceneDataset:
    if isinstance(config, ColmapDatasetConfig):
        dataset = load_colmap_dataset(
            config.path,
            image_root=config.image_root,
            undistort_output_dir=config.undistort_output_dir,
        )
    else:
        raise ValueError(f"Unsupported dataset config type {type(config)!r}.")
    for pipe in config.source_pipes:
        dataset = apply_source_pipe(dataset, pipe)
    return dataset


def _compile_runtime_frame_config(config: DatasetConfig) -> FrameDatasetConfig:
    image_preparation = ImagePreparationConfig(normalize=False)
    resize_pipe: ResizePipeConfig | None = None
    normalize_pipe: NormalizePipeConfig | None = None

    for pipe in config.cache_pipes:
        if isinstance(pipe, ResizePipeConfig):
            if resize_pipe is not None:
                raise ValueError(
                    "Only one ResizePipeConfig is currently supported."
                )
            resize_pipe = pipe
            image_preparation.resize_width_scale = pipe.width_scale
            image_preparation.resize_width_target = pipe.width_target
            image_preparation.interpolation = pipe.interpolation
        else:
            raise ValueError(f"Unsupported cache pipe {type(pipe)!r}.")

    for pipe in config.prepare_pipes:
        if isinstance(pipe, NormalizePipeConfig):
            if normalize_pipe is not None:
                raise ValueError(
                    "Only one NormalizePipeConfig is currently supported."
                )
            normalize_pipe = pipe
            image_preparation.normalize = pipe.enabled
        else:
            raise ValueError(f"Unsupported prepare pipe {type(pipe)!r}.")

    return FrameDatasetConfig(
        split=config.runtime.split,
        materialization=config.runtime.materialization,
        image_preparation=image_preparation,
    )


@overload
def load_dataset(config: DatasetConfig) -> FrameDataset: ...


@overload
def load_dataset(
    path: str | Path,
    *,
    format: DatasetFormat | None = None,
    **kwargs: object,
) -> SceneDataset: ...


def load_dataset(
    path_or_config: DatasetConfig | str | Path,
    *,
    format: DatasetFormat | None = None,
    **kwargs: object,
) -> FrameDataset | SceneDataset:
    """Load a dataset from config or a raw path.

    Config objects are the preferred API and return prepared datasets. Raw path
    loading is kept as a low-level convenience and returns ``SceneDataset``.
    """
    if isinstance(path_or_config, DatasetConfig):
        scene = _load_scene_from_config(path_or_config)
        runtime_config = _compile_runtime_frame_config(path_or_config)
        return FrameDataset(scene, config=runtime_config)

    resolved_path = Path(path_or_config)
    resolved_format = format or _infer_dataset_format(resolved_path)
    if resolved_format == "colmap":
        return load_colmap_dataset(resolved_path, **kwargs)
    if resolved_format == "must3r":
        return load_must3r_dataset(resolved_path, **kwargs)
    raise ValueError(f"Unsupported dataset format {resolved_format!r}.")


__all__ = [
    "CachePipeConfig",
    "ColmapDatasetConfig",
    "DatasetConfig",
    "DatasetFormat",
    "DatasetFrame",
    "DatasetRuntimeConfig",
    "DecodedFrameSample",
    "FrameDataset",
    "FrameDatasetConfig",
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
    "Must3rCheckpointPaths",
    "Must3rRuntime",
    "NormalizePipeConfig",
    "PointCloudState",
    "PreparePipeConfig",
    "PreparedFrameBatch",
    "PreparedFrameSample",
    "ResizePipeConfig",
    "ResizeSpec",
    "SceneDataset",
    "SourcePipeConfig",
    "SplitConfig",
    "SubprocessMust3rSlamRuntime",
    "adjust_dataset_horizon",
    "collate_frame_samples",
    "load_colmap_dataset",
    "load_dataset",
    "load_must3r_dataset",
    "resolve_must3r_checkpoints",
    "run_must3r_dataset",
]
