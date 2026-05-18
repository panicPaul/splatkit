"""Shared contracts for ember-core scenes, cameras, and render options."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal, NamedTuple, Self, cast

import torch
from beartype import beartype
from jaxtyping import Float, Int, UInt
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
            fov_degrees=self.fov_degrees.to(device, non_blocking=non_blocking),
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


class RadFoamScene(Scene):
    """Canonical Radiant Foam scene contract."""

    min_points: ClassVar[int] = 32
    parameter_field_names: ClassVar[tuple[str, ...]] = (
        "primal_points",
        "density",
        "att_dc",
        "att_sh",
    )
    buffer_field_names: ClassVar[tuple[str, ...]] = (
        "point_adjacency",
        "point_adjacency_offsets",
    )
    metadata_field_names: ClassVar[tuple[str, ...]] = (
        "sh_degree",
        "activation_scale",
    )
    topology_field_names: ClassVar[tuple[str, ...]] = (
        "primal_points",
        "density",
        "att_dc",
        "att_sh",
        "point_adjacency",
        "point_adjacency_offsets",
    )

    def __init__(
        self,
        *,
        primal_points: Float[Tensor, "num_points 3"],
        density: Float[Tensor, "num_points 1"],
        att_dc: Float[Tensor, "num_points 3"],
        att_sh: Float[Tensor, "num_points sh_coeffs"],
        point_adjacency: (
            Int[Tensor, " num_adjacency"]
            | UInt[Tensor, " num_adjacency"]
        ),
        point_adjacency_offsets: (
            Int[Tensor, " adjacency_offsets"]
            | UInt[Tensor, " adjacency_offsets"]
        ),
        sh_degree: int,
        activation_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        self.activation_scale = activation_scale
        self.register_parameter(
            "primal_points",
            self._to_parameter(primal_points),
        )
        self.register_parameter("density", self._to_parameter(density))
        self.register_parameter("att_dc", self._to_parameter(att_dc))
        self.register_parameter("att_sh", self._to_parameter(att_sh))
        self.register_buffer("point_adjacency", point_adjacency)
        self.register_buffer(
            "point_adjacency_offsets",
            point_adjacency_offsets,
        )
        self._validate()

    @property
    def scene_family(self) -> SceneFamily:
        """Return the Radiant Foam scene family tag."""
        return "radfoam"

    @staticmethod
    def _to_parameter(value: Tensor) -> nn.Parameter:
        if isinstance(value, nn.Parameter):
            return value
        return nn.Parameter(value, requires_grad=value.requires_grad)

    def _validate(self) -> None:
        if self.sh_degree < 0:
            raise ValueError("RadFoamScene.sh_degree must be non-negative.")
        if self.activation_scale <= 0.0:
            raise ValueError(
                "RadFoamScene.activation_scale must be positive."
            )
        if self.primal_points.ndim != 2 or self.primal_points.shape[-1] != 3:
            raise ValueError(
                "RadFoamScene.primal_points must have shape "
                f"(num_points, 3); got {tuple(self.primal_points.shape)}."
            )
        num_points = int(self.primal_points.shape[0])
        if num_points < self.min_points:
            raise ValueError(
                "RadFoamScene.primal_points must contain at least "
                f"{self.min_points} points for RADFOAM topology; got "
                f"{num_points}."
            )
        if self.density.shape != (num_points, 1):
            raise ValueError(
                "RadFoamScene.density must have shape "
                f"({num_points}, 1); got {tuple(self.density.shape)}."
            )
        if self.att_dc.shape != (num_points, 3):
            raise ValueError(
                "RadFoamScene.att_dc must have shape "
                f"({num_points}, 3); got {tuple(self.att_dc.shape)}."
            )
        expected_sh_coeffs = 3 * ((1 + self.sh_degree) ** 2 - 1)
        if self.att_sh.shape != (num_points, expected_sh_coeffs):
            raise ValueError(
                "RadFoamScene.att_sh must have shape "
                f"({num_points}, {expected_sh_coeffs}); got "
                f"{tuple(self.att_sh.shape)}."
            )
        if self.point_adjacency.ndim != 1:
            raise ValueError(
                "RadFoamScene.point_adjacency must be rank 1; got "
                f"{tuple(self.point_adjacency.shape)}."
            )
        if self.point_adjacency_offsets.shape != (num_points + 1,):
            raise ValueError(
                "RadFoamScene.point_adjacency_offsets must have shape "
                f"({num_points + 1},); got "
                f"{tuple(self.point_adjacency_offsets.shape)}."
            )
        if self.point_adjacency.dtype not in (
            torch.int32,
            torch.int64,
            torch.uint32,
        ):
            raise ValueError(
                "RadFoamScene.point_adjacency must have int32, int64, or "
                f"uint32 dtype; got {self.point_adjacency.dtype}."
            )
        if self.point_adjacency_offsets.dtype not in (
            torch.int32,
            torch.int64,
            torch.uint32,
        ):
            raise ValueError(
                "RadFoamScene.point_adjacency_offsets must have int32, "
                f"int64, or uint32 dtype; got "
                f"{self.point_adjacency_offsets.dtype}."
            )


class PowerFoamScene(Scene):
    """Canonical PowerFoam scene contract."""

    parameter_field_names: ClassVar[tuple[str, ...]] = (
        "points",
        "radii",
        "quaternions",
        "density",
        "texel_sites",
        "texel_sv_axis",
        "texel_sv_rgb",
        "texel_height",
    )
    buffer_field_names: ClassVar[tuple[str, ...]] = (
        "adjacency",
        "adjacency_offsets",
    )
    metadata_field_names: ClassVar[tuple[str, ...]] = (
        "sv_dof",
        "num_texel_sites",
        "render_objective",
        "attr_dtype",
    )
    topology_field_names: ClassVar[tuple[str, ...]] = (
        "points",
        "radii",
        "quaternions",
        "density",
        "texel_sites",
        "texel_sv_axis",
        "texel_sv_rgb",
        "texel_height",
        "adjacency",
        "adjacency_offsets",
    )

    def __init__(
        self,
        *,
        points: Float[Tensor, "num_points 3"],
        radii: Float[Tensor, " num_points"],
        quaternions: Float[Tensor, "num_points 4"],
        density: Float[Tensor, " num_points"],
        texel_sites: Float[Tensor, "num_points num_texel_sites 2"],
        texel_sv_axis: Float[
            Tensor,
            "num_points num_texel_sites sv_axis_coeffs",
        ],
        texel_sv_rgb: Float[
            Tensor,
            "num_points num_texel_sites sv_rgb_coeffs",
        ],
        texel_height: Float[Tensor, "num_points num_texel_sites"],
        adjacency: Int[Tensor, " num_adjacency"] | UInt[Tensor, " num_adjacency"],
        adjacency_offsets: (
            Int[Tensor, " adjacency_offsets"]
            | UInt[Tensor, " adjacency_offsets"]
        ),
        sv_dof: int,
        num_texel_sites: int,
        render_objective: Literal["volume", "surface"] = "volume",
        attr_dtype: Literal["float", "half"] = "float",
    ) -> None:
        super().__init__()
        self.sv_dof = sv_dof
        self.num_texel_sites = num_texel_sites
        self.render_objective = render_objective
        self.attr_dtype = attr_dtype
        self.register_parameter("points", self._to_parameter(points))
        self.register_parameter("radii", self._to_parameter(radii))
        self.register_parameter("quaternions", self._to_parameter(quaternions))
        self.register_parameter("density", self._to_parameter(density))
        self.register_parameter("texel_sites", self._to_parameter(texel_sites))
        self.register_parameter(
            "texel_sv_axis",
            self._to_parameter(texel_sv_axis),
        )
        self.register_parameter(
            "texel_sv_rgb",
            self._to_parameter(texel_sv_rgb),
        )
        self.register_parameter(
            "texel_height",
            self._to_parameter(texel_height),
        )
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("adjacency_offsets", adjacency_offsets)
        self._validate()

    @property
    def scene_family(self) -> SceneFamily:
        """Return the PowerFoam method tag."""
        return "powerfoam"

    @staticmethod
    def _to_parameter(value: Tensor) -> nn.Parameter:
        if isinstance(value, nn.Parameter):
            return value
        return nn.Parameter(value, requires_grad=value.requires_grad)

    def _validate(self) -> None:
        if self.sv_dof <= 0:
            raise ValueError("PowerFoamScene.sv_dof must be positive.")
        if self.num_texel_sites <= 0:
            raise ValueError(
                "PowerFoamScene.num_texel_sites must be positive."
            )
        if self.points.ndim != 2 or self.points.shape[-1] != 3:
            raise ValueError(
                "PowerFoamScene.points must have shape "
                f"(num_points, 3); got {tuple(self.points.shape)}."
            )
        num_points = int(self.points.shape[0])
        if self.radii.shape != (num_points,):
            raise ValueError(
                "PowerFoamScene.radii must have shape "
                f"({num_points},); got {tuple(self.radii.shape)}."
            )
        if self.quaternions.shape != (num_points, 4):
            raise ValueError(
                "PowerFoamScene.quaternions must have shape "
                f"({num_points}, 4); got {tuple(self.quaternions.shape)}."
            )
        if self.density.shape != (num_points,):
            raise ValueError(
                "PowerFoamScene.density must have shape "
                f"({num_points},); got {tuple(self.density.shape)}."
            )
        expected_texel_shape = (num_points, self.num_texel_sites)
        if self.texel_sites.shape != (*expected_texel_shape, 2):
            raise ValueError(
                "PowerFoamScene.texel_sites must have shape "
                f"({num_points}, {self.num_texel_sites}, 2); got "
                f"{tuple(self.texel_sites.shape)}."
            )
        expected_sv_shape = (*expected_texel_shape, 3 * self.sv_dof)
        if self.texel_sv_axis.shape != expected_sv_shape:
            raise ValueError(
                "PowerFoamScene.texel_sv_axis must have shape "
                f"{expected_sv_shape}; got {tuple(self.texel_sv_axis.shape)}."
            )
        if self.texel_sv_rgb.shape != expected_sv_shape:
            raise ValueError(
                "PowerFoamScene.texel_sv_rgb must have shape "
                f"{expected_sv_shape}; got {tuple(self.texel_sv_rgb.shape)}."
            )
        if self.texel_height.shape != expected_texel_shape:
            raise ValueError(
                "PowerFoamScene.texel_height must have shape "
                f"{expected_texel_shape}; got {tuple(self.texel_height.shape)}."
            )
        if self.adjacency.ndim != 1:
            raise ValueError(
                "PowerFoamScene.adjacency must be rank 1; got "
                f"{tuple(self.adjacency.shape)}."
            )
        if self.adjacency_offsets.shape != (num_points + 1,):
            raise ValueError(
                "PowerFoamScene.adjacency_offsets must have shape "
                f"({num_points + 1},); got "
                f"{tuple(self.adjacency_offsets.shape)}."
            )
        if self.adjacency.dtype not in (
            torch.int32,
            torch.int64,
            torch.uint32,
        ):
            raise ValueError(
                "PowerFoamScene.adjacency must have int32, int64, or "
                f"uint32 dtype; got {self.adjacency.dtype}."
            )
        if self.adjacency_offsets.dtype not in (
            torch.int32,
            torch.int64,
            torch.uint32,
        ):
            raise ValueError(
                "PowerFoamScene.adjacency_offsets must have int32, int64, "
                f"or uint32 dtype; got {self.adjacency_offsets.dtype}."
            )


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
        self._svraster_derived_cache: dict[str, Any] = {}
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

    @staticmethod
    def _tensor_cache_signature(value: Tensor) -> tuple[Any, ...]:
        return (
            id(value),
            value.device,
            value.dtype,
            tuple(value.shape),
            int(value._version),
        )

    def _cache(self) -> dict[str, Any]:
        cache = self.__dict__.get("_svraster_derived_cache")
        if not isinstance(cache, dict):
            cache = {}
            self.__dict__["_svraster_derived_cache"] = cache
        return cache

    def _voxel_geometry_signature(self) -> tuple[Any, ...]:
        return (
            self.backend_name,
            self.max_num_levels,
            self._tensor_cache_signature(self.octpath),
            self._tensor_cache_signature(self.octlevel),
            self._tensor_cache_signature(self.scene_center),
            self._tensor_cache_signature(self.scene_extent),
        )

    def _grid_link_signature(self) -> tuple[Any, ...]:
        return (
            self.backend_name,
            self.max_num_levels,
            self._tensor_cache_signature(self.octpath),
            self._tensor_cache_signature(self.octlevel),
        )

    def _voxel_geometry(
        self,
    ) -> tuple[
        Float[Tensor, "num_voxels 3"],
        Float[Tensor, "num_voxels 1"],
    ]:
        signature = self._voxel_geometry_signature()
        cache = self._cache()
        cached = cache.get("voxel_geometry")
        if cached is not None and cached[0] == signature:
            return cast(
                "tuple[Float[Tensor, 'num_voxels 3'], Float[Tensor, 'num_voxels 1']]",
                cached[1],
            )
        value = svraster_octpath_decoding(
            self.octpath,
            self.octlevel,
            self.scene_center,
            self.scene_extent,
            backend_name=self.backend_name,
            max_num_levels=self.max_num_levels,
        )
        cache["voxel_geometry"] = (signature, value)
        return value

    def _grid_link(
        self,
    ) -> tuple[
        Int[Tensor, "num_grid_points 3"],
        Int[Tensor, "num_voxels 8"],
    ]:
        signature = self._grid_link_signature()
        cache = self._cache()
        cached = cache.get("grid_link")
        if cached is not None and cached[0] == signature:
            return cast(
                "tuple[Int[Tensor, 'num_grid_points 3'], Int[Tensor, 'num_voxels 8']]",
                cached[1],
            )
        value = svraster_build_grid_points_link(
            self.octpath,
            self.octlevel,
            backend_name=self.backend_name,
            max_num_levels=self.max_num_levels,
        )
        cache["grid_link"] = (signature, value)
        return value

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
        center, _size = self._voxel_geometry()
        return center

    @property
    def vox_size(self) -> Float[Tensor, "num_voxels 1"]:
        """Return voxel side lengths in world coordinates."""
        _center, size = self._voxel_geometry()
        return size

    @property
    def grid_pts_key(self) -> Int[Tensor, "num_grid_points 3"]:
        """Return integer grid-point keys at the finest octree level."""
        grid_pts_key, _vox_key = self._grid_link()
        return grid_pts_key

    @property
    def vox_key(self) -> Int[Tensor, "num_voxels 8"]:
        """Return the eight corner-point indices for each voxel."""
        _grid_pts_key, vox_key = self._grid_link()
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
