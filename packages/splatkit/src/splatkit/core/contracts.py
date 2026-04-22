"""Shared contracts for splatkit scenes, cameras, and render options."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Literal, NamedTuple, Self

import torch
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor

from splatkit.core.sparse_voxel import (
    SVRasterBackendName,
    svraster_build_grid_points_link,
    svraster_octpath_decoding,
)

CameraConvention = Literal["opencv", "opengl", "blender", "colmap"]
BackendName = str
OutputName = Literal[
    "alpha",
    "depth",
    "gaussian_impact_score",
    "normals",
    "2d_projections",
    "projective_intersection_transforms",
]
SceneFamily = Literal["gaussian", "sparse_voxel"]


class CameraParams(NamedTuple):
    """Camera intrinsic parameters."""

    width: Int[Tensor, " num_cams"]
    height: Int[Tensor, " num_cams"]
    fov_degrees: Float[Tensor, " num_cams"]


def camera_params_to_intrinsics(
    width: Int[Tensor, " num_cams"],
    height: Int[Tensor, " num_cams"],
    fov_degrees: Float[Tensor, " num_cams"],
) -> Float[Tensor, "num_cams 3 3"]:
    """Compute batched 3x3 intrinsics from camera parameters."""
    fov_rad = torch.deg2rad(fov_degrees)
    fx = (width / 2.0) / torch.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    return torch.stack(
        [
            torch.stack([fx, zeros, cx], dim=-1),
            torch.stack([zeros, fx, cy], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )


def intrinsics_to_camera_params(
    intrinsics: Float[Tensor, "num_cams 3 3"],
) -> CameraParams:
    """Extract camera parameters from batched 3x3 intrinsics matrices."""
    fx = intrinsics[:, 0, 0]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    width = (cx * 2.0).long()
    height = (cy * 2.0).long()
    fov_degrees = torch.rad2deg(2.0 * torch.atan(cx / fx))
    return CameraParams(width=width, height=height, fov_degrees=fov_degrees)


class Scene(ABC):
    """Nominal base class for scene-family contracts."""

    @property
    @abstractmethod
    def scene_family(self) -> SceneFamily:
        """Return the scene family tag."""

    @abstractmethod
    def to(self, device: torch.device) -> Self:
        """Move scene tensors to a target device."""


@beartype
@dataclass(frozen=True)
class CameraState:
    """Batched camera state."""

    width: Int[Tensor, " num_cams"]
    height: Int[Tensor, " num_cams"]
    fov_degrees: Float[Tensor, " num_cams"]
    cam_to_world: Float[Tensor, "num_cams 4 4"]
    intrinsics: Float[Tensor, "num_cams 3 3"] | None = None
    camera_convention: CameraConvention = "opencv"
    up_direction: Literal["up", "down"] = "up"

    def to(self, device: torch.device) -> Self:
        """Move state to a device."""
        return replace(
            self,
            width=self.width.to(device),
            height=self.height.to(device),
            fov_degrees=self.fov_degrees.to(device),
            intrinsics=(
                self.intrinsics.to(device)
                if self.intrinsics is not None
                else None
            ),
            cam_to_world=self.cam_to_world.to(device),
        )

    def get_intrinsics(self) -> Float[Tensor, "num_cams 3 3"]:
        """Compute batched intrinsics."""
        if self.intrinsics is not None:
            return self.intrinsics
        return camera_params_to_intrinsics(
            self.width,
            self.height,
            self.fov_degrees,
        )


@beartype
@dataclass(frozen=True)
class GaussianScene(Scene, ABC):
    """Shared Gaussian-scene contract."""

    center_position: Float[Tensor, "num_splats spatial_dims"]
    log_scales: Float[Tensor, "num_splats spatial_dims"]
    quaternion_orientation: Float[Tensor, "num_splats 4"]
    logit_opacity: Float[Tensor, " num_splats"]
    feature: (
        Float[Tensor, "num_splats feature_dim"]
        | Float[Tensor, "num_splats sh_coeffs 3"]
    )
    sh_degree: int

    @property
    def scene_family(self) -> SceneFamily:
        """Return the Gaussian scene family tag."""
        return "gaussian"

    @property
    @abstractmethod
    def spatial_dims(self) -> int:
        """Return the Gaussian scale dimensionality."""

    def __post_init__(self) -> None:
        if self.log_scales.shape[-1] != self.spatial_dims:
            raise ValueError(
                "Gaussian scene scale dimensionality mismatch: expected "
                f"{self.spatial_dims}, got {self.log_scales.shape[-1]}."
            )

    def to(self, device: torch.device) -> Self:
        """Move scene tensors to a device."""
        return replace(
            self,
            center_position=self.center_position.to(device),
            log_scales=self.log_scales.to(device),
            quaternion_orientation=self.quaternion_orientation.to(device),
            logit_opacity=self.logit_opacity.to(device),
            feature=self.feature.to(device),
        )


@beartype
@dataclass(frozen=True)
class GaussianScene3D(GaussianScene):
    """Canonical 3D Gaussian-splat scene contract."""

    @property
    def spatial_dims(self) -> int:
        """Return the Gaussian scale dimensionality."""
        return 3


@beartype
@dataclass(frozen=True)
class GaussianScene2D(GaussianScene):
    """Canonical 2D Gaussian scene contract."""

    @property
    def spatial_dims(self) -> int:
        """Return the Gaussian scale dimensionality."""
        return 2


@beartype
@dataclass(frozen=True)
class SparseVoxelScene(Scene):
    """Declarative sparse-voxel scene derived from an SV Raster checkpoint."""

    backend_name: SVRasterBackendName
    active_sh_degree: int
    max_num_levels: int
    scene_center: Float[Tensor, " 3"]
    scene_extent: Float[Tensor, " 1"]
    inside_extent: Float[Tensor, " 1"]
    octpath: Int[Tensor, "num_voxels 1"]
    octlevel: Int[Tensor, "num_voxels 1"]
    geo_grid_pts: Float[Tensor, "num_grid_points 1"]
    sh0: Float[Tensor, "num_voxels 3"]
    shs: Float[Tensor, "num_voxels sh_coeffs 3"]

    @property
    def scene_family(self) -> SceneFamily:
        """Return the sparse-voxel scene family tag."""
        return "sparse_voxel"

    def __post_init__(self) -> None:
        if self.max_num_levels < 1:
            raise ValueError("SparseVoxelScene.max_num_levels must be >= 1.")
        if self.octpath.shape[0] != self.octlevel.shape[0]:
            raise ValueError("SparseVoxelScene octpath/octlevel size mismatch.")
        if self.sh0.shape[0] != self.octpath.shape[0]:
            raise ValueError(
                "SparseVoxelScene sh0 size must match voxel count."
            )
        if self.shs.shape[0] != self.octpath.shape[0]:
            raise ValueError(
                "SparseVoxelScene shs size must match voxel count."
            )
        if self.active_sh_degree < 0:
            raise ValueError(
                "SparseVoxelScene.active_sh_degree must be non-negative."
            )

    def to(self, device: torch.device) -> Self:
        """Move scene tensors to a device."""
        return replace(
            self,
            scene_center=self.scene_center.to(device),
            scene_extent=self.scene_extent.to(device),
            inside_extent=self.inside_extent.to(device),
            octpath=self.octpath.to(device),
            octlevel=self.octlevel.to(device),
            geo_grid_pts=self.geo_grid_pts.to(device),
            sh0=self.sh0.to(device),
            shs=self.shs.to(device),
        )

    @property
    def num_voxels(self) -> int:
        """Return the number of voxels in the sparse grid."""
        return int(self.octpath.shape[0])

    @property
    def num_grid_points(self) -> int:
        """Return the number of unique sparse-grid corner points."""
        return int(self.geo_grid_pts.shape[0])

    @cached_property
    def vox_center(self) -> Float[Tensor, "num_voxels 3"]:
        """Return voxel centers in world coordinates."""
        center, _size = svraster_octpath_decoding(
            self.octpath,
            self.octlevel,
            self.scene_center,
            self.scene_extent,
            backend_name=self.backend_name,
            max_num_levels=self.max_num_levels,
        )
        return center

    @cached_property
    def vox_size(self) -> Float[Tensor, "num_voxels 1"]:
        """Return voxel side lengths in world coordinates."""
        _center, size = svraster_octpath_decoding(
            self.octpath,
            self.octlevel,
            self.scene_center,
            self.scene_extent,
            backend_name=self.backend_name,
            max_num_levels=self.max_num_levels,
        )
        return size

    @cached_property
    def grid_pts_key(self) -> Int[Tensor, "num_grid_points 3"]:
        """Return integer grid-point keys at the finest octree level."""
        grid_pts_key, _vox_key = svraster_build_grid_points_link(
            self.octpath,
            self.octlevel,
            backend_name=self.backend_name,
            max_num_levels=self.max_num_levels,
        )
        return grid_pts_key

    @cached_property
    def vox_key(self) -> Int[Tensor, "num_voxels 8"]:
        """Return the eight corner-point indices for each voxel."""
        _grid_pts_key, vox_key = svraster_build_grid_points_link(
            self.octpath,
            self.octlevel,
            backend_name=self.backend_name,
            max_num_levels=self.max_num_levels,
        )
        return vox_key

    @cached_property
    def voxel_geometries(self) -> Float[Tensor, "num_voxels 8"]:
        """Return per-voxel trilinear density corner values."""
        return self.geo_grid_pts[self.vox_key].squeeze(-1)


@beartype
@dataclass(frozen=True)
class RenderOutput:
    """Base render output shared by all backends."""

    render: Float[Tensor, "num_cams height width 3"]


@beartype
@dataclass(frozen=True)
class RenderOptions:
    """Shared render configuration."""

    background_color: Float[Tensor, " 3"] = field(
        default_factory=lambda: torch.zeros(3)
    )
