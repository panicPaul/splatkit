"""Shared contracts for Gaussian splatting backends."""

from dataclasses import dataclass, field, replace
from typing import Literal, NamedTuple, Self

import torch
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor

CameraConvention = Literal["opencv", "opengl", "blender", "colmap"]
BackendName = str
OutputName = Literal["alpha", "depth", "2d_projections"]


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


@beartype
@dataclass(frozen=True)
class CameraState:
    """Batched camera state."""

    width: Int[Tensor, " num_cams"]
    height: Int[Tensor, " num_cams"]
    fov_degrees: Float[Tensor, " num_cams"]
    cam_to_world: Float[Tensor, "num_cams 4 4"]
    camera_convention: CameraConvention = "opencv"
    up_direction: Literal["up", "down"] = "up"

    def to(self, device: torch.device) -> Self:
        """Move state to a device."""
        return replace(
            self,
            width=self.width.to(device),
            height=self.height.to(device),
            fov_degrees=self.fov_degrees.to(device),
            cam_to_world=self.cam_to_world.to(device),
        )

    def get_intrinsics(self) -> Float[Tensor, "num_cams 3 3"]:
        """Compute batched intrinsics."""
        return camera_params_to_intrinsics(
            self.width, self.height, self.fov_degrees
        )


@beartype
@dataclass(frozen=True)
class GaussianScene:
    """Canonical scene contract shared by all backends."""

    center_position: Float[Tensor, "num_splats 3"]
    log_scales: Float[Tensor, "num_splats 3"] | Float[Tensor, "num_splats 2"]
    quaternion_orientation: Float[Tensor, "num_splats 4"]
    logit_opacity: Float[Tensor, " num_splats"]
    feature: (
        Float[Tensor, "num_splats feature_dim"]
        | Float[Tensor, "num_splats sh_coeffs 3"]
    )
    sh_degree: int

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
