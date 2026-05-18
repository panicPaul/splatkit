"""Tensor stages shared by PowerFoam render and training code."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from ember_core.core.contracts import PowerFoamScene
from jaxtyping import Float
from torch import Tensor

from ember_native_powerfoam.powerfoam.runtime.ops import interpenetration_op


def powerfoam_density(
    scene: PowerFoamScene,
    *,
    beta: float = 100.0,
) -> Float[Tensor, " num_points"]:
    """Return activated PowerFoam densities."""
    return F.softplus(scene.density, beta=beta)


def powerfoam_radii(
    scene: PowerFoamScene,
    *,
    beta: float = 100.0,
) -> Float[Tensor, " num_points"]:
    """Return activated PowerFoam radii."""
    return F.softplus(scene.radii, beta=beta)


def powerfoam_normals(scene: PowerFoamScene) -> Float[Tensor, "num_points 3"]:
    """Return PowerFoam primitive normals from quaternion orientation."""
    quaternions = scene.quaternions / scene.quaternions.norm(
        dim=-1,
        keepdim=True,
    )
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    normals = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ],
        dim=-1,
    )
    return normals / normals.norm(dim=-1, keepdim=True)


def powerfoam_tangents(
    scene: PowerFoamScene,
) -> tuple[Float[Tensor, "num_points 3"], Float[Tensor, "num_points 3"]]:
    """Return tangent and bitangent axes from quaternion orientation."""
    quaternions = scene.quaternions / scene.quaternions.norm(
        dim=-1,
        keepdim=True,
    )
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    tangents = torch.stack(
        [
            2 * (x * y + z * w),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - x * w),
        ],
        dim=-1,
    )
    tangents = tangents / tangents.norm(dim=-1, keepdim=True)
    bitangents = torch.stack(
        [
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    bitangents = bitangents / bitangents.norm(dim=-1, keepdim=True)
    return tangents, bitangents


def powerfoam_texel_world_sites(
    scene: PowerFoamScene,
    radii: Float[Tensor, " num_points"],
) -> Float[Tensor, "num_points num_texel_sites 3"]:
    """Return world-space texel sites on each primitive disk."""
    tangents, bitangents = powerfoam_tangents(scene)
    offsets = scene.texel_sites * radii[:, None, None]
    offsets = (
        offsets[..., 0:1] * tangents[:, None, :]
        + offsets[..., 1:2] * bitangents[:, None, :]
    )
    return scene.points[:, None, :] + offsets


def powerfoam_att_sv(
    scene: PowerFoamScene,
) -> tuple[
    Float[Tensor, "sv_dof texel_count 3"],
    Float[Tensor, "sv_dof texel_count 3"],
    Float[Tensor, "sv_dof texel_count"],
]:
    """Return PowerFoam's spherical-Voronoi structure-of-arrays attributes."""
    sv_dof = scene.sv_dof
    sv_grid_dim = scene.num_texel_sites
    axis = scene.texel_sv_axis.view(-1, sv_grid_dim, sv_dof, 3)
    temp = axis.norm(dim=-1)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    axis = axis.view(-1, sv_grid_dim, sv_dof, 3)
    axis = axis.permute(2, 0, 1, 3).reshape(sv_dof, -1, 3).contiguous()
    rgb = scene.texel_sv_rgb.view(-1, sv_grid_dim, sv_dof, 3)
    rgb = rgb.permute(2, 0, 1, 3).reshape(sv_dof, -1, 3).contiguous()
    temp = temp.view(-1, sv_grid_dim, sv_dof)
    temp = temp.permute(2, 0, 1).reshape(sv_dof, -1).contiguous()
    return axis, rgb, temp


def powerfoam_interpenetration(
    scene: PowerFoamScene,
    *,
    radii_beta: float = 100.0,
) -> Float[Tensor, " num_points"]:
    """Return PowerFoam's interpenetration regularizer signal."""
    return interpenetration_op(
        scene.points,
        powerfoam_radii(scene, beta=radii_beta),
        scene.adjacency,
        scene.adjacency_offsets,
    )
