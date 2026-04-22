"""Concrete user-facing dataset configuration defaults.

Most users should only need ``ColmapDatasetConfig`` or a preset subclass.
Researchers extending dataset semantics should subclass a concrete dataset
config, register any new pipe specs, and override the ordered phase tuples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from splatkit.data.config_contracts import DatasetConfig, DatasetRuntimeConfig
from splatkit.data.pipes import (
    HorizonAlignPipeConfig,
    NormalizePipeConfig,
    ResizePipeConfig,
)


class ColmapDatasetConfig(DatasetConfig):
    """Default end-to-end COLMAP dataset spec."""

    kind: Literal["colmap"] = Field(
        default="colmap",
        description="Dataset source family handled by this concrete config.",
    )
    path: Path = Field(
        description="Root path containing the COLMAP scene and sparse model.",
    )
    image_root: Path | None = Field(
        default=None,
        description=(
            "Optional image directory override relative to or separate from the "
            "COLMAP scene root."
        ),
    )
    undistort_output_dir: Path | None = Field(
        default=None,
        description=(
            "Optional output directory used to cache undistorted images when "
            "the COLMAP source contains distorted cameras."
        ),
    )
    runtime: DatasetRuntimeConfig = Field(
        default_factory=DatasetRuntimeConfig,
        description=(
            "Runtime split and materialization settings applied after the scene "
            "is loaded."
        ),
    )
    source_pipes: tuple[HorizonAlignPipeConfig] = Field(
        default=(HorizonAlignPipeConfig(),),
        description=(
            "Ordered source-phase pipes applied to the raw loaded scene before "
            "prepared dataset construction. The default COLMAP pipeline aligns "
            "the scene horizon to a canonical up direction."
        ),
    )
    cache_pipes: tuple[ResizePipeConfig] = Field(
        default=(ResizePipeConfig(width_target=1980),),
        description=(
            "Ordered cache-phase pipes compiled into cached sample "
            "materialization behavior. The default COLMAP pipeline resizes "
            "images to a width target of 1980 pixels."
        ),
    )
    prepare_pipes: tuple[NormalizePipeConfig] = Field(
        default=(NormalizePipeConfig(),),
        description=(
            "Ordered prepare-phase pipes applied to training-facing samples "
            "after loading and cache-time transforms. The default COLMAP "
            "pipeline normalizes RGB intensities."
        ),
    )


class MipNerf360IndoorDatasetConfig(ColmapDatasetConfig):
    """COLMAP preset tuned for MipNeRF360-style indoor scenes."""

    cache_pipes: tuple[ResizePipeConfig] = Field(
        default=(ResizePipeConfig(width_scale=0.25, width_target=None),),
        description=(
            "Default cache-phase pipeline for indoor MipNeRF360 scenes, using "
            "quarter-resolution resizing."
        ),
    )


class MipNerf360OutdoorDatasetConfig(ColmapDatasetConfig):
    """COLMAP preset tuned for MipNeRF360-style outdoor scenes."""

    cache_pipes: tuple[ResizePipeConfig] = Field(
        default=(ResizePipeConfig(width_scale=0.5, width_target=None),),
        description=(
            "Default cache-phase pipeline for outdoor MipNeRF360 scenes, using "
            "half-resolution resizing."
        ),
    )


class NCoreDatasetConfig(DatasetConfig):
    """Default end-to-end ncore dataset spec."""

    kind: Literal["ncore"] = Field(
        default="ncore",
        description="Dataset source family handled by this concrete config.",
    )
    component_group_paths: tuple[Path, ...] = Field(
        description=(
            "Ordered ncore component-group paths used to discover sensors "
            "and optional point-cloud sources."
        )
    )
    camera_sensor_id: str | None = Field(
        default=None,
        description=(
            "Optional default camera sensor id to bind at load time when the "
            "ncore source contains multiple camera streams."
        ),
    )
    runtime: DatasetRuntimeConfig = Field(
        default_factory=DatasetRuntimeConfig,
        description=(
            "Runtime split, camera selection, and materialization settings "
            "applied after the scene is loaded."
        ),
    )
    source_pipes: tuple[HorizonAlignPipeConfig] = Field(
        default=(HorizonAlignPipeConfig(),),
        description=(
            "Ordered source-phase pipes applied to the raw loaded scene before "
            "prepared dataset construction."
        ),
    )
    cache_pipes: tuple[ResizePipeConfig] = Field(
        default=(ResizePipeConfig(width_target=1980),),
        description=(
            "Ordered cache-phase pipes compiled into cached sample "
            "materialization behavior."
        ),
    )
    prepare_pipes: tuple[NormalizePipeConfig] = Field(
        default=(NormalizePipeConfig(),),
        description=(
            "Ordered prepare-phase pipes applied to training-facing samples "
            "after loading and cache-time transforms."
        ),
    )


__all__ = [
    "ColmapDatasetConfig",
    "MipNerf360IndoorDatasetConfig",
    "MipNerf360OutdoorDatasetConfig",
    "NCoreDatasetConfig",
]
