"""Declarative dataset contracts for image-based training data."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol

import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor

from splatkit.core.contracts import CameraState

DatasetSource = Literal["colmap", "must3r"]


def horizontal_fov_degrees(
    width: int,
    intrinsics: Float[Tensor, " 3 3"],
) -> float:
    """Compute the horizontal field of view in degrees."""
    fx = intrinsics[0, 0]
    return float(torch.rad2deg(2.0 * torch.atan(width / (2.0 * fx))).item())


@beartype
@dataclass(frozen=True)
class ResizeSpec:
    """Declarative image resizing configuration."""

    width: int | None = None
    height: int | None = None
    max_long_edge: int | None = None
    interpolation: Literal["nearest", "bilinear", "bicubic", "lanczos"] = (
        "lanczos"
    )

    def __post_init__(self) -> None:
        has_exact_shape = self.width is not None or self.height is not None
        if has_exact_shape and self.max_long_edge is not None:
            raise ValueError(
                "ResizeSpec must use either exact width/height or max_long_edge."
            )
        if self.width is not None and self.height is None:
            raise ValueError("ResizeSpec.height is required when width is set.")
        if self.height is not None and self.width is None:
            raise ValueError("ResizeSpec.width is required when height is set.")
        if (
            self.width is None
            and self.height is None
            and self.max_long_edge is None
        ):
            raise ValueError(
                "ResizeSpec requires width/height or max_long_edge."
            )


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
    image_path: Path
    camera_index: int
    width: int
    height: int


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
@dataclass(frozen=True)
class SceneDataset:
    """Immutable scene-level dataset record."""

    frames: tuple[DatasetFrame, ...]
    camera: CameraState
    source_format: DatasetSource
    root_path: Path | None = None
    point_cloud: PointCloudState | None = None
    world_up: Float[Tensor, " 3"] | None = None
    focus_point: Float[Tensor, " 3"] | None = None

    def __post_init__(self) -> None:
        num_frames = len(self.frames)
        if int(self.camera.width.shape[0]) != num_frames:
            raise ValueError(
                "SceneDataset requires one camera entry per frame: "
                f"got {self.camera.width.shape[0]} cameras for {num_frames} frames."
            )

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the dataset."""
        return len(self.frames)


@beartype
@dataclass(frozen=True)
class PreparedFrameSample:
    """Prepared sample returned by the Torch dataset adapter."""

    frame: DatasetFrame
    image: Float[Tensor, "3 height width"]
    camera: CameraState


@beartype
@dataclass(frozen=True)
class PreparedFrameBatch:
    """Batched prepared samples."""

    frames: tuple[DatasetFrame, ...]
    images: Float[Tensor, "batch 3 height width"]
    camera: CameraState


class HasCamera(Protocol):
    """Batch capability for camera supervision."""

    camera: CameraState


class HasRgbTargets(Protocol):
    """Batch capability for RGB supervision."""

    images: Float[Tensor, "batch 3 height width"]


class HasDepthTargets(Protocol):
    """Batch capability for depth supervision."""

    depth: Float[Tensor, "batch height width"]


class HasMaskTargets(Protocol):
    """Batch capability for binary masks."""

    mask: Float[Tensor, "batch height width"]
