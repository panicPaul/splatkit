"""Interpenetration custom ops for the PowerFoam Warp runtime."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp
from torch import Tensor

from ember_native_powerfoam.powerfoam.native.warp.geometry import (
    interpenetration_kernel,
)


@torch.library.custom_op("powerfoam::interpenetration_fwd", mutates_args=())
def interpenetration_fwd_op(
    points: Tensor,
    radii: Tensor,
    adjacency: Tensor,
    adjacency_offsets: Tensor,
) -> Tensor:
    """Compute PowerFoam interpenetration areas with vendored Warp kernels."""
    with wp.ScopedDevice(str(points.device)):
        num_points = points.shape[0]
        spheres = torch.cat([points, radii[:, None]], dim=-1).to(torch.float32)
        warp_spheres = wp.from_torch(
            spheres,
            dtype=wp.vec4f,
            requires_grad=True,
        )
        areas = wp.zeros(
            num_points,
            dtype=wp.float32,
            device=str(points.device),
            requires_grad=True,
        )
        wp.launch(
            kernel=interpenetration_kernel,
            dim=num_points,
            inputs=[
                warp_spheres,
                adjacency.to(torch.int32).contiguous(),
                adjacency_offsets.to(torch.int32).contiguous(),
            ],
            outputs=[areas],
            block_dim=128,
        )
        return wp.to_torch(areas)


@interpenetration_fwd_op.register_fake
def _interpenetration_fwd_fake(
    points: Tensor,
    radii: Tensor,
    adjacency: Tensor,
    adjacency_offsets: Tensor,
) -> Tensor:
    del radii, adjacency, adjacency_offsets
    return torch.empty((points.shape[0],), dtype=torch.float32, device=points.device)


@torch.library.custom_op("powerfoam::interpenetration_bwd", mutates_args=())
def interpenetration_bwd_op(
    points: Tensor,
    radii: Tensor,
    adjacency: Tensor,
    adjacency_offsets: Tensor,
    grad_areas: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute PowerFoam interpenetration gradients with Warp adjoints."""
    with wp.ScopedDevice(str(points.device)):
        num_points = points.shape[0]
        spheres = torch.cat([points, radii[:, None]], dim=-1).to(torch.float32)
        warp_spheres = wp.from_torch(
            spheres,
            dtype=wp.vec4f,
            requires_grad=True,
        )
        areas = wp.zeros(
            num_points,
            dtype=wp.float32,
            device=str(points.device),
            requires_grad=True,
        )

        # Replay the forward launch so Warp can evaluate the adjoint program.
        wp.launch(
            kernel=interpenetration_kernel,
            dim=num_points,
            inputs=[
                warp_spheres,
                adjacency.to(torch.int32).contiguous(),
                adjacency_offsets.to(torch.int32).contiguous(),
            ],
            outputs=[areas],
            block_dim=128,
        )
        areas.grad = wp.from_torch(
            grad_areas.contiguous(),
            dtype=wp.float32,
        )
        wp.launch(
            kernel=interpenetration_kernel,
            dim=num_points,
            inputs=[
                warp_spheres,
                adjacency.to(torch.int32).contiguous(),
                adjacency_offsets.to(torch.int32).contiguous(),
            ],
            outputs=[areas],
            adj_inputs=[
                warp_spheres.grad,
                None,
                None,
            ],
            adj_outputs=[areas.grad],
            block_dim=128,
            adjoint=True,
        )
        sphere_gradients = wp.to_torch(warp_spheres.grad)
        return sphere_gradients[..., :3].clone(), sphere_gradients[..., 3].clone()


@interpenetration_bwd_op.register_fake
def _interpenetration_bwd_fake(
    points: Tensor,
    radii: Tensor,
    adjacency: Tensor,
    adjacency_offsets: Tensor,
    grad_areas: Tensor,
) -> tuple[Tensor, Tensor]:
    del adjacency, adjacency_offsets, grad_areas
    return torch.empty_like(points), torch.empty_like(radii)


@torch.library.custom_op("powerfoam::interpenetration", mutates_args=())
def interpenetration_op(
    points: Tensor,
    radii: Tensor,
    adjacency: Tensor,
    adjacency_offsets: Tensor,
) -> Tensor:
    """Autograd-enabled PowerFoam interpenetration op."""
    return interpenetration_fwd_op(points, radii, adjacency, adjacency_offsets)


@interpenetration_op.register_fake
def _interpenetration_fake(*args: Any) -> Tensor:
    return _interpenetration_fwd_fake(*args)


def _interpenetration_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: Tensor,
) -> None:
    del output
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], inputs[3])


def _interpenetration_backward(
    ctx: Any,
    grad_areas: Tensor | None,
) -> tuple[Tensor | None, ...]:
    points, radii, adjacency, adjacency_offsets = ctx.saved_tensors
    resolved_grad_areas = (
        torch.zeros_like(radii) if grad_areas is None else grad_areas
    )
    grad_points, grad_radii = interpenetration_bwd_op(
        points,
        radii,
        adjacency,
        adjacency_offsets,
        resolved_grad_areas,
    )
    return grad_points, grad_radii, None, None


interpenetration_op.register_autograd(
    _interpenetration_backward,
    setup_context=_interpenetration_setup_context,
)


__all__ = [
    "interpenetration_bwd_op",
    "interpenetration_fwd_op",
    "interpenetration_op",
]
