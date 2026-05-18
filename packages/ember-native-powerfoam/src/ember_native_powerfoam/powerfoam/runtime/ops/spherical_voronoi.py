"""Spherical-Voronoi custom ops for the PowerFoam Warp runtime."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Literal

import torch
import warp as wp
from torch import Tensor

from ember_native_powerfoam.powerfoam.native.warp.camera import TorchCamera
from ember_native_powerfoam.powerfoam.native.warp.color_fn import (
    _FOV_BUFFER_RAD,
    SphericalVoronoi,
)


def _spherical_voronoi(
    *,
    device: torch.device,
    sv_dof: int,
    attr_dtype: str,
) -> SphericalVoronoi:
    args = SimpleNamespace(sv_dof=int(sv_dof))
    return SphericalVoronoi(args, device, attr_dtype)


def _camera_forward(camera_up: Tensor, camera_right: Tensor) -> Tensor:
    forward = torch.cross(camera_up, camera_right, dim=-1)
    return forward / torch.norm(forward, dim=-1, keepdim=True)


def _camera_fov_cos_cutoff(camera_up: Tensor, camera_right: Tensor) -> float:
    right_norm = float(torch.linalg.norm(camera_right).item())
    up_norm = float(torch.linalg.norm(camera_up).item())
    tan_diag = math.sqrt(right_norm**2 + up_norm**2)
    return math.cos(math.atan(tan_diag) + _FOV_BUFFER_RAD)


@torch.library.custom_op(
    "powerfoam::spherical_voronoi_fwd",
    mutates_args=(),
)
def spherical_voronoi_fwd_op(
    points: Tensor,
    camera_origin: Tensor,
    camera_forward: Tensor,
    att_sites: Tensor,
    att_values: Tensor,
    att_temps: Tensor,
    fov_cos_cutoff: float,
    sv_dof: int,
    attr_dtype: str,
) -> tuple[Tensor, Tensor]:
    """Evaluate PowerFoam spherical Voronoi colors with vendored Warp kernels."""
    kernel_owner = _spherical_voronoi(
        device=points.device,
        sv_dof=sv_dof,
        attr_dtype=attr_dtype,
    )
    with wp.ScopedDevice(str(points.device)):
        torch_stream = torch.cuda.current_stream()
        wp_stream = wp.stream_from_torch(torch_stream)
        wp.set_stream(wp_stream)

        num_points = points.shape[0]
        values_out = torch.empty(
            (num_points, 3),
            device=points.device,
            dtype=kernel_owner.tscalar,
        )
        weights_sum_out = torch.empty(
            (num_points,),
            device=points.device,
            dtype=kernel_owner.tscalar,
        )

        wp.launch(
            kernel=kernel_owner.spherical_voronoi_fwd_kernel,
            dim=num_points,
            inputs=[
                points.detach(),
                camera_origin,
                camera_forward,
                fov_cos_cutoff,
                att_sites.detach(),
                att_values.detach(),
                att_temps.detach(),
                num_points,
                values_out,
                weights_sum_out,
            ],
            block_dim=256,
        )
        return values_out, weights_sum_out


@spherical_voronoi_fwd_op.register_fake
def _spherical_voronoi_fwd_fake(
    points: Tensor,
    camera_origin: Tensor,
    camera_forward: Tensor,
    att_sites: Tensor,
    att_values: Tensor,
    att_temps: Tensor,
    fov_cos_cutoff: float,
    sv_dof: int,
    attr_dtype: str,
) -> tuple[Tensor, Tensor]:
    del (
        camera_origin,
        camera_forward,
        att_sites,
        att_values,
        att_temps,
        fov_cos_cutoff,
        sv_dof,
        attr_dtype,
    )
    return (
        torch.empty((points.shape[0], 3), dtype=points.dtype, device=points.device),
        torch.empty((points.shape[0],), dtype=points.dtype, device=points.device),
    )


@torch.library.custom_op(
    "powerfoam::spherical_voronoi_bwd",
    mutates_args=(),
)
def spherical_voronoi_bwd_op(
    points: Tensor,
    camera_origin: Tensor,
    camera_forward: Tensor,
    att_sites: Tensor,
    att_values: Tensor,
    att_temps: Tensor,
    values: Tensor,
    weights_sum: Tensor,
    grad_values: Tensor,
    fov_cos_cutoff: float,
    sv_dof: int,
    attr_dtype: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Differentiate PowerFoam spherical Voronoi colors with Warp kernels."""
    kernel_owner = _spherical_voronoi(
        device=points.device,
        sv_dof=sv_dof,
        attr_dtype=attr_dtype,
    )
    with wp.ScopedDevice(str(points.device)):
        torch_stream = torch.cuda.current_stream()
        wp_stream = wp.stream_from_torch(torch_stream)
        wp.set_stream(wp_stream)

        num_points = points.shape[0]
        grad_points = torch.zeros_like(points)
        grad_att_sites = torch.zeros_like(att_sites)
        grad_att_values = torch.zeros_like(att_values)
        grad_att_temps = torch.zeros_like(att_temps)

        wp.launch(
            kernel=kernel_owner.spherical_voronoi_bwd_kernel,
            dim=num_points,
            inputs=[
                points.detach(),
                camera_origin,
                camera_forward,
                fov_cos_cutoff,
                att_sites.detach(),
                att_values.detach(),
                att_temps.detach(),
                weights_sum.detach(),
                values.detach(),
                grad_values.detach(),
                num_points,
                grad_points,
                grad_att_sites,
                grad_att_values,
                grad_att_temps,
            ],
            block_dim=256,
        )
        return (
            grad_points,
            grad_att_sites,
            grad_att_values,
            grad_att_temps,
        )


@spherical_voronoi_bwd_op.register_fake
def _spherical_voronoi_bwd_fake(
    points: Tensor,
    camera_origin: Tensor,
    camera_forward: Tensor,
    att_sites: Tensor,
    att_values: Tensor,
    att_temps: Tensor,
    values: Tensor,
    weights_sum: Tensor,
    grad_values: Tensor,
    fov_cos_cutoff: float,
    sv_dof: int,
    attr_dtype: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del (
        camera_origin,
        camera_forward,
        values,
        weights_sum,
        grad_values,
        fov_cos_cutoff,
        sv_dof,
        attr_dtype,
    )
    return (
        torch.empty_like(points),
        torch.empty_like(att_sites),
        torch.empty_like(att_values),
        torch.empty_like(att_temps),
    )


def _spherical_voronoi_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, Tensor],
) -> None:
    values, weights_sum = output
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        values,
        weights_sum,
    )
    ctx.fov_cos_cutoff = inputs[6]
    ctx.sv_dof = inputs[7]
    ctx.attr_dtype = inputs[8]


def _spherical_voronoi_backward(
    ctx: Any,
    grad_values: Tensor | None,
    grad_weights_sum: Tensor | None,
) -> tuple[Tensor | None, ...]:
    del grad_weights_sum
    (
        points,
        camera_origin,
        camera_forward,
        att_sites,
        att_values,
        att_temps,
        values,
        weights_sum,
    ) = ctx.saved_tensors
    resolved_grad_values = (
        torch.zeros_like(values) if grad_values is None else grad_values
    )
    _grad_points, grad_att_sites, grad_att_values, grad_att_temps = (
        spherical_voronoi_bwd_op(
            points,
            camera_origin,
            camera_forward,
            att_sites,
            att_values,
            att_temps,
            values,
            weights_sum,
            resolved_grad_values,
            ctx.fov_cos_cutoff,
            ctx.sv_dof,
            ctx.attr_dtype,
        )
    )
    return (
        None,
        None,
        None,
        grad_att_sites,
        grad_att_values,
        grad_att_temps,
        None,
        None,
        None,
    )


spherical_voronoi_fwd_op.register_autograd(
    _spherical_voronoi_backward,
    setup_context=_spherical_voronoi_setup_context,
)


def spherical_voronoi_colors(
    points: Tensor,
    camera: TorchCamera,
    att_sites: Tensor,
    att_values: Tensor,
    att_temps: Tensor,
    *,
    sv_dof: int,
    attr_dtype: Literal["float", "half"],
) -> Tensor:
    """Return per-texel spherical Voronoi colors from the custom op stage."""
    camera_forward = _camera_forward(camera.up, camera.right)
    fov_cos_cutoff = _camera_fov_cos_cutoff(camera.up, camera.right)
    values, _weights_sum = spherical_voronoi_fwd_op(
        points,
        camera.eye,
        camera_forward,
        att_sites,
        att_values,
        att_temps,
        fov_cos_cutoff,
        sv_dof,
        attr_dtype,
    )
    return values


__all__ = [
    "spherical_voronoi_bwd_op",
    "spherical_voronoi_colors",
    "spherical_voronoi_fwd_op",
]
