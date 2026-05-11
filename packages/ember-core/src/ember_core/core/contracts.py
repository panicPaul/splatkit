"""Shared contracts for ember-core scenes, cameras, and render options."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal, NamedTuple, Self, cast

import torch
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor, nn

from ember_core.core.keys import SceneFamilyKey
from ember_core.core.sparse_voxel import (
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
SceneFamily = str | SceneFamilyKey


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


class Scene(nn.Module, ABC):
    """Base class for parameter-owning scene-family contracts."""

    parameter_field_names: ClassVar[tuple[str, ...]] = ()
    buffer_field_names: ClassVar[tuple[str, ...]] = ()
    metadata_field_names: ClassVar[tuple[str, ...]] = ()
    topology_field_names: ClassVar[tuple[str, ...]] = ()

    @property
    @abstractmethod
    def scene_family(self) -> SceneFamily:
        """Return the scene family tag."""

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return all public scene field names."""
        return (
            self.parameter_field_names
            + self.buffer_field_names
            + self.metadata_field_names
        )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> Self:
        """Apply ``nn.Module.to`` transforms to registered and transient tensors."""
        super()._apply(fn, recurse=recurse)
        for name in self.parameter_field_names + self.buffer_field_names:
            if name in self._parameters or name in self._buffers:
                continue
            value = self.__dict__.get(name)
            if isinstance(value, Tensor):
                self.__dict__[name] = fn(value)
        return self

    def replace_fields_(self, **updates: Tensor | None) -> dict[str, Tensor]:
        """Replace owned parameter/buffer fields in-place.

        Returns the installed tensor objects so optimizer bindings can update
        references to newly registered parameters.
        """
        valid_names = set(self.parameter_field_names) | set(
            self.buffer_field_names
        )
        unknown_names = sorted(set(updates) - valid_names)
        if unknown_names:
            raise KeyError(f"Unknown scene tensor field(s): {unknown_names!r}.")

        installed: dict[str, Tensor] = {}
        for name, value in updates.items():
            self._clear_unregistered_field(name)
            if name in self.parameter_field_names:
                if value is None:
                    self.register_parameter(name, None)
                    continue
                parameter = nn.Parameter(
                    value.detach(),
                    requires_grad=value.requires_grad,
                )
                self.register_parameter(name, parameter)
                installed[name] = parameter
                continue
            if value is None:
                self.register_buffer(name, None)
                continue
            self.register_buffer(name, value.detach())
            installed[name] = cast("Tensor", getattr(self, name))
        self._validate()
        return installed

    def _clear_unregistered_field(self, name: str) -> None:
        if name not in self.__dict__:
            return
        if name in self._parameters or name in self._buffers:
            return
        del self.__dict__[name]

    def with_fields(self, **updates: Tensor | int | str | None) -> Self:
        """Return a shallow render-time scene variant preserving autograd."""
        return self._copy_with_fields(
            updates,
            detach=False,
            device=None,
            register_parameters=False,
        )

    def detached_copy(self, device: torch.device | None = None) -> Self:
        """Return a detached copy suitable for snapshots and checkpoints."""
        return self._copy_with_fields(
            {},
            detach=True,
            device=device,
            register_parameters=True,
        )

    def _copy_with_fields(
        self,
        updates: Mapping[str, Tensor | int | str | None],
        *,
        detach: bool,
        device: torch.device | None,
        register_parameters: bool,
    ) -> Self:
        unknown_names = sorted(set(updates) - set(self.field_names))
        if unknown_names:
            raise KeyError(f"Unknown scene field(s): {unknown_names!r}.")

        copied = self.__class__.__new__(self.__class__)
        nn.Module.__init__(copied)

        for name in self.metadata_field_names:
            setattr(copied, name, updates.get(name, getattr(self, name)))

        for name in self.parameter_field_names:
            value = updates.get(name, getattr(self, name))
            if value is None:
                copied.register_parameter(name, None)
                continue
            if not isinstance(value, Tensor):
                raise TypeError(f"Scene field {name!r} must be a Tensor.")
            source_value = getattr(self, name)
            requires_grad = (
                source_value.requires_grad
                if isinstance(source_value, Tensor)
                else value.requires_grad
            )
            copied_value = self._copy_tensor(
                value,
                detach=detach,
                device=device,
            )
            if register_parameters:
                copied.register_parameter(
                    name,
                    nn.Parameter(copied_value, requires_grad=requires_grad),
                )
            else:
                setattr(copied, name, copied_value)

        for name in self.buffer_field_names:
            value = updates.get(name, getattr(self, name))
            if value is None:
                copied.register_buffer(name, None)
                continue
            if not isinstance(value, Tensor):
                raise TypeError(f"Scene field {name!r} must be a Tensor.")
            copied_value = self._copy_tensor(
                value,
                detach=detach,
                device=device,
            )
            if register_parameters:
                copied.register_buffer(name, copied_value)
            else:
                setattr(copied, name, copied_value)

        copied._validate()
        return cast("Self", copied)

    @staticmethod
    def _copy_tensor(
        value: Tensor,
        *,
        detach: bool,
        device: torch.device | None,
    ) -> Tensor:
        copied = value.detach().clone() if detach else value
        if device is not None:
            copied = copied.to(device)
        return copied

    @abstractmethod
    def _validate(self) -> None:
        """Validate scene field consistency."""


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

    def to(self, device: torch.device, non_blocking: bool = False) -> Self:
        """Move state to a device."""
        return replace(
            self,
            width=self.width.to(device, non_blocking=non_blocking),
            height=self.height.to(device, non_blocking=non_blocking),
            fov_degrees=self.fov_degrees.to(
                device, non_blocking=non_blocking
            ),
            intrinsics=(
                self.intrinsics.to(device, non_blocking=non_blocking)
                if self.intrinsics is not None
                else None
            ),
            cam_to_world=self.cam_to_world.to(
                device, non_blocking=non_blocking
            ),
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


class GaussianScene(Scene, ABC):
    """Shared Gaussian-scene contract."""

    parameter_field_names: ClassVar[tuple[str, ...]] = (
        "center_position",
        "log_scales",
        "quaternion_orientation",
        "logit_opacity",
        "feature",
    )
    metadata_field_names: ClassVar[tuple[str, ...]] = ("sh_degree",)
    topology_field_names: ClassVar[tuple[str, ...]] = parameter_field_names

    def __init__(
        self,
        *,
        center_position: Float[Tensor, "num_splats spatial_dims"],
        log_scales: Float[Tensor, "num_splats spatial_dims"],
        quaternion_orientation: Float[Tensor, "num_splats 4"],
        logit_opacity: Float[Tensor, " num_splats"],
        feature: (
            Float[Tensor, "num_splats feature_dim"]
            | Float[Tensor, "num_splats sh_coeffs 3"]
        ),
        sh_degree: int,
    ) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        self.register_parameter(
            "center_position",
            self._to_parameter(center_position),
        )
        self.register_parameter("log_scales", self._to_parameter(log_scales))
        self.register_parameter(
            "quaternion_orientation",
            self._to_parameter(quaternion_orientation),
        )
        self.register_parameter(
            "logit_opacity",
            self._to_parameter(logit_opacity),
        )
        self.register_parameter("feature", self._to_parameter(feature))
        self._validate()

    @property
    def scene_family(self) -> SceneFamily:
        """Return the Gaussian scene family tag."""
        return "gaussian"

    @property
    @abstractmethod
    def spatial_dims(self) -> int:
        """Return the Gaussian scale dimensionality."""

    @staticmethod
    def _to_parameter(value: Tensor) -> nn.Parameter:
        if isinstance(value, nn.Parameter):
            return value
        return nn.Parameter(value, requires_grad=value.requires_grad)

    def _validate(self) -> None:
        if self.log_scales.shape[-1] != self.spatial_dims:
            raise ValueError(
                "Gaussian scene scale dimensionality mismatch: expected "
                f"{self.spatial_dims}, got {self.log_scales.shape[-1]}."
            )
        if self.quaternion_orientation.shape[-1] != 4:
            raise ValueError(
                "Gaussian scene quaternion_orientation must have shape "
                f"(num_splats, 4); got {tuple(self.quaternion_orientation.shape)}."
            )
        if self.logit_opacity.ndim != 1:
            raise ValueError(
                "Gaussian scene logit_opacity must have shape "
                f"(num_splats,); got {tuple(self.logit_opacity.shape)}."
            )
        if self.feature.ndim not in (2, 3):
            raise ValueError(
                "Gaussian scene feature must be rank 2 or 3; got "
                f"{tuple(self.feature.shape)}."
            )


class GaussianScene3D(GaussianScene):
    """Canonical 3D Gaussian-splat scene contract."""

    @property
    def spatial_dims(self) -> int:
        """Return the Gaussian scale dimensionality."""
        return 3


class GaussianScene2D(GaussianScene):
    """Canonical 2D Gaussian scene contract."""

    @property
    def spatial_dims(self) -> int:
        """Return the Gaussian scale dimensionality."""
        return 2


class SparseVoxelScene(Scene):
    """Declarative sparse-voxel scene derived from an SV Raster checkpoint."""

    parameter_field_names: ClassVar[tuple[str, ...]] = (
        "geo_grid_pts",
        "sh0",
        "shs",
        "subdivision_priority",
    )
    buffer_field_names: ClassVar[tuple[str, ...]] = (
        "scene_center",
        "scene_extent",
        "inside_extent",
        "octpath",
        "octlevel",
    )
    metadata_field_names: ClassVar[tuple[str, ...]] = (
        "backend_name",
        "active_sh_degree",
        "max_num_levels",
    )
    topology_field_names: ClassVar[tuple[str, ...]] = (
        "octpath",
        "octlevel",
        "geo_grid_pts",
        "sh0",
        "shs",
        "subdivision_priority",
    )

    def __init__(
        self,
        *,
        backend_name: SVRasterBackendName,
        active_sh_degree: int,
        max_num_levels: int,
        scene_center: Float[Tensor, " 3"],
        scene_extent: Float[Tensor, " 1"],
        inside_extent: Float[Tensor, " 1"],
        octpath: Int[Tensor, "num_voxels 1"],
        octlevel: Int[Tensor, "num_voxels 1"],
        geo_grid_pts: Float[Tensor, "num_grid_points 1"],
        sh0: Float[Tensor, "num_voxels 3"],
        shs: Float[Tensor, "num_voxels sh_coeffs 3"],
        subdivision_priority: Float[Tensor, "num_voxels 1"] | None = None,
    ) -> None:
        super().__init__()
        self.backend_name = backend_name
        self.active_sh_degree = active_sh_degree
        self.max_num_levels = max_num_levels
        self.register_buffer("scene_center", scene_center)
        self.register_buffer("scene_extent", scene_extent)
        self.register_buffer("inside_extent", inside_extent)
        self.register_buffer("octpath", octpath)
        self.register_buffer("octlevel", octlevel)
        self.register_parameter(
            "geo_grid_pts",
            self._to_parameter(geo_grid_pts),
        )
        self.register_parameter("sh0", self._to_parameter(sh0))
        self.register_parameter("shs", self._to_parameter(shs))
        self.register_parameter(
            "subdivision_priority",
            (
                None
                if subdivision_priority is None
                else self._to_parameter(subdivision_priority)
            ),
        )
        self._validate()

    @property
    def scene_family(self) -> SceneFamily:
        """Return the sparse-voxel scene family tag."""
        return "sparse_voxel"

    @staticmethod
    def _to_parameter(value: Tensor) -> nn.Parameter:
        if isinstance(value, nn.Parameter):
            return value
        return nn.Parameter(value, requires_grad=value.requires_grad)

    def _validate(self) -> None:
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
        if (
            self.subdivision_priority is not None
            and self.subdivision_priority.shape != self.octpath.shape
        ):
            raise ValueError(
                "SparseVoxelScene subdivision_priority must have shape "
                f"{tuple(self.octpath.shape)}, got "
                f"{tuple(self.subdivision_priority.shape)}."
            )
        if self.active_sh_degree < 0:
            raise ValueError(
                "SparseVoxelScene.active_sh_degree must be non-negative."
            )

    @property
    def num_voxels(self) -> int:
        """Return the number of voxels in the sparse grid."""
        return int(self.octpath.shape[0])

    @property
    def num_grid_points(self) -> int:
        """Return the number of unique sparse-grid corner points."""
        return int(self.geo_grid_pts.shape[0])

    @property
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

    @property
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

    @property
    def grid_pts_key(self) -> Int[Tensor, "num_grid_points 3"]:
        """Return integer grid-point keys at the finest octree level."""
        grid_pts_key, _vox_key = svraster_build_grid_points_link(
            self.octpath,
            self.octlevel,
            backend_name=self.backend_name,
            max_num_levels=self.max_num_levels,
        )
        return grid_pts_key

    @property
    def vox_key(self) -> Int[Tensor, "num_voxels 8"]:
        """Return the eight corner-point indices for each voxel."""
        _grid_pts_key, vox_key = svraster_build_grid_points_link(
            self.octpath,
            self.octlevel,
            backend_name=self.backend_name,
            max_num_levels=self.max_num_levels,
        )
        return vox_key

    @property
    def voxel_geometries(self) -> Float[Tensor, "num_voxels 8"]:
        """Return per-voxel trilinear density corner values."""
        return self.geo_grid_pts[self.vox_key].squeeze(-1)

    @property
    def resolved_subdivision_priority(self) -> Float[Tensor, "num_voxels 1"]:
        """Return persistent or default per-voxel subdivision priority."""
        if self.subdivision_priority is not None:
            return self.subdivision_priority
        return torch.ones(
            (self.num_voxels, 1),
            dtype=self.geo_grid_pts.dtype,
            device=self.geo_grid_pts.device,
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
