"""Tracing custom ops for the RADFOAM native runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ember_native_radfoam.radfoam.runtime.ops._common import pipeline

TraceOutput = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
TraceBackwardOutput = tuple[Tensor, Tensor, Tensor, Tensor]


def _empty_contribution(points: Tensor, attributes: Tensor) -> Tensor:
    return torch.empty(
        (0, 1),
        dtype=attributes.dtype,
        device=points.device,
    )


def _empty_ray_error(rays: Tensor, attributes: Tensor) -> Tensor:
    return torch.empty(
        rays.shape[:-1],
        dtype=attributes.dtype,
        device=rays.device,
    )


@torch.library.custom_op("radfoam::trace_fwd", mutates_args=())
def trace_fwd_op(
    points: Tensor,
    attributes: Tensor,
    point_adjacency: Tensor,
    point_adjacency_offsets: Tensor,
    rays: Tensor,
    start_point: Tensor,
    depth_quantiles: Tensor,
    return_contribution: bool,
    sh_degree: int,
    weight_threshold: float,
    max_intersections: int,
) -> TraceOutput:
    """Low-level RADFOAM trace forward op."""
    has_depth = depth_quantiles.shape[-1] > 0
    results = pipeline(sh_degree, attributes.dtype).trace_forward(
        points.contiguous(),
        attributes.contiguous(),
        point_adjacency.contiguous().to(torch.uint32),
        point_adjacency_offsets.contiguous().to(torch.uint32),
        rays.contiguous(),
        start_point.contiguous().to(torch.uint32),
        depth_quantiles=depth_quantiles.contiguous() if has_depth else None,
        weight_threshold=weight_threshold,
        max_intersections=max_intersections,
        return_contribution=return_contribution,
    )
    depth = results.get(
        "depth",
        torch.empty(
            (*rays.shape[:-1], 0),
            dtype=torch.float32,
            device=rays.device,
        ),
    )
    depth_indices = results.get(
        "depth_indices",
        torch.empty(
            (*rays.shape[:-1], 0),
            dtype=torch.uint32,
            device=rays.device,
        ),
    )
    contribution = results.get(
        "contribution",
        _empty_contribution(points, attributes),
    )
    return (
        results["rgba"],
        depth,
        depth_indices,
        contribution,
        results["num_intersections"],
    )


@trace_fwd_op.register_fake
def _trace_fwd_fake(
    points: Tensor,
    attributes: Tensor,
    point_adjacency: Tensor,
    point_adjacency_offsets: Tensor,
    rays: Tensor,
    start_point: Tensor,
    depth_quantiles: Tensor,
    return_contribution: bool,
    sh_degree: int,
    weight_threshold: float,
    max_intersections: int,
) -> TraceOutput:
    del (
        point_adjacency,
        point_adjacency_offsets,
        start_point,
        sh_degree,
        weight_threshold,
        max_intersections,
    )
    depth_shape = (*rays.shape[:-1], depth_quantiles.shape[-1])
    contribution = (
        torch.empty(
            (points.shape[0], 1),
            dtype=attributes.dtype,
            device=points.device,
        )
        if return_contribution
        else _empty_contribution(points, attributes)
    )
    return (
        torch.empty((*rays.shape[:-1], 4), dtype=attributes.dtype, device=rays.device),
        torch.empty(depth_shape, dtype=torch.float32, device=rays.device),
        torch.empty(depth_shape, dtype=torch.uint32, device=rays.device),
        contribution,
        torch.empty((*rays.shape[:-1], 1), dtype=torch.uint32, device=rays.device),
    )


@torch.library.custom_op("radfoam::trace_bwd", mutates_args=())
def trace_bwd_op(
    points: Tensor,
    attributes: Tensor,
    point_adjacency: Tensor,
    point_adjacency_offsets: Tensor,
    rays: Tensor,
    start_point: Tensor,
    rgba: Tensor,
    grad_rgba: Tensor,
    depth_quantiles: Tensor,
    depth_indices: Tensor,
    grad_depth: Tensor,
    ray_error: Tensor,
    has_depth: bool,
    has_ray_error: bool,
    sh_degree: int,
    weight_threshold: float,
    max_intersections: int,
) -> TraceBackwardOutput:
    """Low-level RADFOAM trace backward op."""
    results = pipeline(sh_degree, attributes.dtype).trace_backward(
        points.contiguous(),
        attributes.contiguous(),
        point_adjacency.contiguous().to(torch.uint32),
        point_adjacency_offsets.contiguous().to(torch.uint32),
        rays.contiguous(),
        start_point.contiguous().to(torch.uint32),
        rgba.contiguous(),
        grad_rgba.contiguous(),
        depth_quantiles.contiguous() if has_depth else None,
        depth_indices.contiguous().to(torch.uint32) if has_depth else None,
        grad_depth.contiguous() if has_depth else None,
        ray_error.contiguous() if has_ray_error else None,
        weight_threshold=weight_threshold,
        max_intersections=max_intersections,
    )
    point_error = results.get(
        "point_error",
        torch.empty((0, 1), dtype=attributes.dtype, device=points.device),
    )
    points_grad = results["points_grad"]
    attr_grad = results["attr_grad"]
    points_grad = torch.where(points_grad.isfinite(), points_grad, 0.0)
    attr_grad = torch.where(attr_grad.isfinite(), attr_grad, 0.0)
    return points_grad, attr_grad, results["ray_grad"], point_error


@trace_bwd_op.register_fake
def _trace_bwd_fake(
    points: Tensor,
    attributes: Tensor,
    point_adjacency: Tensor,
    point_adjacency_offsets: Tensor,
    rays: Tensor,
    start_point: Tensor,
    rgba: Tensor,
    grad_rgba: Tensor,
    depth_quantiles: Tensor,
    depth_indices: Tensor,
    grad_depth: Tensor,
    ray_error: Tensor,
    has_depth: bool,
    has_ray_error: bool,
    sh_degree: int,
    weight_threshold: float,
    max_intersections: int,
) -> TraceBackwardOutput:
    del (
        point_adjacency,
        point_adjacency_offsets,
        start_point,
        rgba,
        grad_rgba,
        depth_quantiles,
        depth_indices,
        grad_depth,
        ray_error,
        has_depth,
        has_ray_error,
        sh_degree,
        weight_threshold,
        max_intersections,
    )
    return (
        torch.empty_like(points),
        torch.empty_like(attributes),
        torch.empty_like(rays),
        torch.empty((0, 1), dtype=attributes.dtype, device=points.device),
    )


def _trace_impl(
    points: Tensor,
    attributes: Tensor,
    point_adjacency: Tensor,
    point_adjacency_offsets: Tensor,
    rays: Tensor,
    start_point: Tensor,
    depth_quantiles: Tensor,
    return_contribution: bool,
    sh_degree: int,
    weight_threshold: float,
    max_intersections: int,
) -> TraceOutput:
    """Autograd-enabled RADFOAM trace op."""
    return trace_fwd_op(
        points,
        attributes,
        point_adjacency,
        point_adjacency_offsets,
        rays,
        start_point,
        depth_quantiles,
        return_contribution,
        sh_degree,
        weight_threshold,
        max_intersections,
    )


@torch.library.custom_op("radfoam::trace", mutates_args=())
def trace_op(
    points: Tensor,
    attributes: Tensor,
    point_adjacency: Tensor,
    point_adjacency_offsets: Tensor,
    rays: Tensor,
    start_point: Tensor,
    depth_quantiles: Tensor,
    return_contribution: bool,
    sh_degree: int,
    weight_threshold: float,
    max_intersections: int,
) -> TraceOutput:
    """Autograd-enabled RADFOAM trace op."""
    return _trace_impl(
        points,
        attributes,
        point_adjacency,
        point_adjacency_offsets,
        rays,
        start_point,
        depth_quantiles,
        return_contribution,
        sh_degree,
        weight_threshold,
        max_intersections,
    )


@trace_op.register_fake
def _trace_fake(*args: Any) -> TraceOutput:
    return _trace_fwd_fake(*args)


def _trace_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: TraceOutput,
) -> None:
    rgba, depth, depth_indices, _contribution, _num_intersections = output
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        rgba,
        depth,
        depth_indices,
    )
    ctx.return_contribution = inputs[7]
    ctx.sh_degree = inputs[8]
    ctx.weight_threshold = inputs[9]
    ctx.max_intersections = inputs[10]


def _trace_backward(
    ctx: Any,
    grad_rgba: Tensor | None,
    grad_depth: Tensor | None,
    grad_depth_indices: Tensor | None,
    grad_contribution: Tensor | None,
    grad_num_intersections: Tensor | None,
) -> tuple[Tensor | None, ...]:
    del grad_depth_indices, grad_contribution, grad_num_intersections
    (
        points,
        attributes,
        point_adjacency,
        point_adjacency_offsets,
        rays,
        start_point,
        depth_quantiles,
        rgba,
        depth,
        depth_indices,
    ) = ctx.saved_tensors
    resolved_grad_rgba = (
        torch.zeros_like(rgba) if grad_rgba is None else grad_rgba
    )
    has_depth = depth_quantiles.shape[-1] > 0
    resolved_grad_depth = (
        torch.zeros_like(depth) if grad_depth is None else grad_depth
    )
    points_grad, attr_grad, ray_grad, _point_error = trace_bwd_op(
        points,
        attributes,
        point_adjacency,
        point_adjacency_offsets,
        rays,
        start_point,
        rgba,
        resolved_grad_rgba,
        depth_quantiles,
        depth_indices,
        resolved_grad_depth,
        _empty_ray_error(rays, attributes),
        has_depth,
        False,
        ctx.sh_degree,
        ctx.weight_threshold,
        ctx.max_intersections,
    )
    return (
        points_grad,
        attr_grad,
        None,
        None,
        ray_grad,
        None,
        None,
        None,
        None,
        None,
        None,
    )


trace_op.register_autograd(
    _trace_backward,
    setup_context=_trace_setup_context,
)


__all__ = [
    "TraceBackwardOutput",
    "TraceOutput",
    "trace_bwd_op",
    "trace_fwd_op",
    "trace_op",
]
