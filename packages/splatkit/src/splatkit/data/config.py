"""Concrete user-facing scene-load and frame-preparation defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from splatkit.data.config_contracts import (
    ImagePreparationConfig,
    PreparedFrameDatasetConfig,
    SceneLoadConfig,
)
from splatkit.data.pipes import HorizonAlignPipeConfig


class ColmapSceneConfig(SceneLoadConfig):
    """Default COLMAP scene-record loading spec."""

    kind: Literal["colmap"] = Field(
        default="colmap",
        description="Scene-record source family handled by this config.",
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
    source_pipes: tuple[HorizonAlignPipeConfig] = Field(
        default=(HorizonAlignPipeConfig(),),
        description=(
            "Ordered source-phase pipes applied while loading the scene "
            "record. The default COLMAP pipeline aligns the scene horizon to "
            "a canonical up direction."
        ),
    )


class NCoreSceneConfig(SceneLoadConfig):
    """Default ncore scene-record loading spec."""

    kind: Literal["ncore"] = Field(
        default="ncore",
        description="Scene-record source family handled by this config.",
    )
    component_group_paths: tuple[Path, ...] = Field(
        description=(
            "Ordered ncore component-group paths used to discover sensors "
            "and optional point-cloud sources."
        )
    )
    source_pipes: tuple[HorizonAlignPipeConfig] = Field(
        default=(HorizonAlignPipeConfig(),),
        description=(
            "Ordered source-phase pipes applied while loading the scene "
            "record."
        ),
    )


class MipNerf360IndoorPreparedFrameDatasetConfig(PreparedFrameDatasetConfig):
    """Prepared-frame defaults for MipNeRF360-style indoor scenes."""

    image_preparation: ImagePreparationConfig | None = Field(
        default_factory=lambda: ImagePreparationConfig(
            resize_width_scale=0.25,
            resize_width_target=None,
            normalize=True,
            interpolation="bicubic",
        )
    )


class MipNerf360OutdoorPreparedFrameDatasetConfig(PreparedFrameDatasetConfig):
    """Prepared-frame defaults for MipNeRF360-style outdoor scenes."""

    image_preparation: ImagePreparationConfig | None = Field(
        default_factory=lambda: ImagePreparationConfig(
            resize_width_scale=0.5,
            resize_width_target=None,
            normalize=True,
            interpolation="bicubic",
        )
    )


__all__ = [
    "ColmapSceneConfig",
    "MipNerf360IndoorPreparedFrameDatasetConfig",
    "MipNerf360OutdoorPreparedFrameDatasetConfig",
    "NCoreSceneConfig",
]
