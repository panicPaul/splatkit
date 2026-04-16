"""Serializable dataset configuration models.

Most users should only need ``ColmapDatasetConfig`` or a preset subclass.
Researchers extending dataset semantics should subclass a concrete dataset
config, register any new pipe specs, and override the ordered phase tuples.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SerializeAsAny, model_validator

from splatkit.data.contracts import MaterializationMode, MaterializationStage
from splatkit.data.pipes import (
    CachePipeConfig,
    NormalizePipeConfig,
    PreparePipeConfig,
    SourcePipeConfig,
)


class DataConfigBase(BaseModel):
    """Base config with strict validation."""

    model_config = {
        "extra": "forbid",
    }


class SplitConfig(DataConfigBase):
    """Frame selection policy for train/validation datasets."""

    target: Literal["train", "val", "all"] = Field(
        default="train",
        description="Which subset of frames this dataset instance should expose.",
    )
    every_n: int | None = Field(
        default=8,
        ge=1,
        description=(
            "Use every Nth frame for validation when split mode is 'every_n'."
        ),
    )
    train_ratio: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description=(
            "Proportion of frames to use for training when split mode is "
            "'ratio'. The rest will be used for validation."
        ),
    )

    @property
    def mode(self) -> Literal["every_n", "ratio", "none"]:
        """Determine split mode based on which parameters are set."""
        if self.target == "all":
            return "none"
        if self.every_n is not None and self.train_ratio is not None:
            raise ValueError(
                "SplitConfig must use either every_n or train_ratio, not both."
            )
        if self.every_n is not None:
            return "every_n"
        if self.train_ratio is not None:
            return "ratio"
        raise ValueError("SplitConfig requires either every_n or train_ratio.")

    @model_validator(mode="after")
    def _validate_split(self) -> SplitConfig:
        if self.target == "all":
            if self.every_n is not None or self.train_ratio is not None:
                raise ValueError(
                    "SplitConfig with target 'all' cannot use every_n or "
                    "train_ratio."
                )
        else:
            if self.every_n is None and self.train_ratio is None:
                raise ValueError(
                    "SplitConfig with target 'train' or 'val' requires "
                    "every_n or train_ratio."
                )
            if self.every_n is not None and self.train_ratio is not None:
                raise ValueError(
                    "SplitConfig must use either every_n or train_ratio, not both."
                )
        return self


class MaterializationConfig(DataConfigBase):
    """Caching policy for frame dataset materialization."""

    stage: MaterializationStage = Field(
        default="decoded",
        description="Which stage of the dataset pipeline should be cached.",
    )
    mode: MaterializationMode = Field(
        default="eager",
        description="Whether to build caches eagerly or on first access.",
    )
    num_workers: int | None = Field(
        default=0,
        description=(
            "Number of worker threads for eager decoded materialization. "
            "Use 0 for serial materialization, None to pick a heuristic, "
            "or 2+ for explicit parallel materialization."
        ),
    )

    @model_validator(mode="after")
    def _validate_num_workers(self) -> MaterializationConfig:
        if self.num_workers is not None and self.num_workers == 1:
            raise ValueError(
                "MaterializationConfig.num_workers must be 0, None, or >= 2."
            )
        return self


class DatasetRuntimeConfig(DataConfigBase):
    """Runtime knobs for prepared dataset construction."""

    split: SplitConfig | None = Field(
        default_factory=SplitConfig,
        description=(
            "Frame split policy for this dataset instance. Use None to "
            "disable splitting and expose all frames."
        ),
    )
    materialization: MaterializationConfig = Field(
        default_factory=MaterializationConfig
    )


class ImagePreparationConfig(DataConfigBase):
    """Legacy image preprocessing policy for prepared frame datasets."""

    normalize: bool = Field(
        default=True,
        description="Whether to scale image intensities into the [0, 1] range.",
    )
    resize_width_scale: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Scale factor applied to image width while preserving aspect ratio."
        ),
    )
    resize_width_target: int | None = Field(
        default=None,
        ge=1,
        description="Target image width used when resizing.",
    )
    interpolation: Literal["nearest", "bilinear", "bicubic"] = Field(
        default="bicubic",
        description="Interpolation mode used when resizing images.",
    )

    @model_validator(mode="after")
    def _validate_resize(self) -> ImagePreparationConfig:
        if (
            self.resize_width_scale is not None
            and self.resize_width_target is not None
        ):
            raise ValueError(
                "ImagePreparationConfig must use either resize_width_scale "
                "or resize_width_target, not both."
            )
        return self


class DatasetConfig(DataConfigBase, ABC):
    """Abstract end-to-end dataset spec.

    Concrete subclasses define source-specific fields and the ordered pipe
    tuples for ``source``, ``cache``, and ``prepare`` phases.
    """

    runtime: DatasetRuntimeConfig = Field(default_factory=DatasetRuntimeConfig)
    source_pipes: tuple[SerializeAsAny[SourcePipeConfig], ...] = Field(
        default_factory=tuple
    )
    cache_pipes: tuple[SerializeAsAny[CachePipeConfig], ...] = Field(
        default_factory=tuple
    )
    prepare_pipes: tuple[SerializeAsAny[PreparePipeConfig], ...] = Field(
        default_factory=tuple
    )


class ColmapDatasetConfig(DatasetConfig):
    """Default end-to-end COLMAP dataset spec."""

    kind: Literal["colmap"] = "colmap"
    path: Path
    image_root: Path | None = None
    undistort_output_dir: Path | None = None
    prepare_pipes: tuple[SerializeAsAny[PreparePipeConfig], ...] = Field(
        default_factory=lambda: (NormalizePipeConfig(),)
    )


class FrameDatasetConfig(DataConfigBase):
    """Legacy dataset-side configuration for prepared frame loading."""

    split: SplitConfig | None = Field(
        default_factory=SplitConfig,
        description=(
            "Frame split policy for this dataset instance. Use None to "
            "disable splitting and expose all frames."
        ),
    )
    materialization: MaterializationConfig | None = Field(
        default_factory=MaterializationConfig,
        description=(
            "Caching policy for decoded or prepared frame samples. Use None "
            "to keep the default decoded eager serial behavior."
        ),
    )
    image_preparation: ImagePreparationConfig | None = Field(
        default_factory=ImagePreparationConfig,
        description=(
            "Image preprocessing policy applied before batching. Use None to "
            "keep images at source resolution with default normalization."
        ),
    )


__all__ = [
    "ColmapDatasetConfig",
    "DataConfigBase",
    "DatasetConfig",
    "DatasetRuntimeConfig",
    "FrameDatasetConfig",
    "ImagePreparationConfig",
    "MaterializationConfig",
    "SplitConfig",
]
