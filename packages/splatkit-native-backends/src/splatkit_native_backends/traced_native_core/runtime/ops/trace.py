"""Trace-stage custom ops for traced native backends."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from splatkit_native_backends.traced_native_core.reuse.factories import (
    register_trace_family,
)
from splatkit_native_backends.traced_native_core.runtime.ops._common import (
    tracer_wrapper,
)


@torch.library.custom_op("traced_native_core::trace_fwd", mutates_args=())
def trace_fwd_op(
    state_token: Tensor,
    ray_to_world: Tensor,
    ray_ori: Tensor,
    ray_dir: Tensor,
    particle_density: Tensor,
    particle_radiance: Tensor,
    render_opts: int,
    sph_degree: int,
    min_transmittance: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Low-level native trace forward op."""
    return tracer_wrapper(state_token).trace(
        0,
        ray_to_world,
        ray_ori,
        ray_dir,
        particle_density,
        particle_radiance,
        render_opts,
        sph_degree,
        min_transmittance,
    )


@trace_fwd_op.register_fake
def _trace_fwd_fake(
    state_token: Tensor,
    ray_to_world: Tensor,
    ray_ori: Tensor,
    ray_dir: Tensor,
    particle_density: Tensor,
    particle_radiance: Tensor,
    render_opts: int,
    sph_degree: int,
    min_transmittance: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del state_token, ray_to_world, ray_dir, render_opts, sph_degree, min_transmittance
    batch_size, height, width, _ = ray_ori.shape
    device = ray_ori.device
    dtype = ray_ori.dtype
    num_particles = int(particle_density.shape[0])
    radiance_dim = int(particle_radiance.shape[1])
    del radiance_dim
    return (
        torch.empty((batch_size, height, width, 3), device=device, dtype=dtype),
        torch.empty((batch_size, height, width, 1), device=device, dtype=dtype),
        torch.empty((batch_size, height, width, 2), device=device, dtype=dtype),
        torch.empty((batch_size, height, width, 3), device=device, dtype=dtype),
        torch.empty((batch_size, height, width, 1), device=device, dtype=dtype),
        torch.empty((num_particles, 1), device=device, dtype=dtype),
        torch.empty((num_particles, 1), device=device, dtype=dtype),
        torch.empty((batch_size, height, width, 16), device=device, dtype=dtype),
    )


@torch.library.custom_op("traced_native_core::trace_bwd", mutates_args=())
def trace_bwd_op(
    state_token: Tensor,
    grad_radiance: Tensor,
    grad_density: Tensor,
    grad_hit: Tensor,
    grad_normals: Tensor,
    ray_to_world: Tensor,
    ray_ori: Tensor,
    ray_dir: Tensor,
    particle_density: Tensor,
    particle_radiance: Tensor,
    radiance: Tensor,
    density: Tensor,
    hit: Tensor,
    sample_cache: Tensor,
    normals: Tensor,
    render_opts: int,
    sph_degree: int,
    min_transmittance: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Low-level native trace backward op."""
    return tracer_wrapper(state_token).trace_bwd(
        0,
        ray_to_world,
        ray_ori,
        ray_dir,
        radiance,
        density,
        hit,
        sample_cache,
        normals,
        particle_density,
        particle_radiance,
        grad_radiance,
        grad_density,
        grad_hit,
        grad_normals,
        render_opts,
        sph_degree,
        min_transmittance,
    )


@trace_bwd_op.register_fake
def _trace_bwd_fake(
    state_token: Tensor,
    grad_radiance: Tensor,
    grad_density: Tensor,
    grad_hit: Tensor,
    grad_normals: Tensor,
    ray_to_world: Tensor,
    ray_ori: Tensor,
    ray_dir: Tensor,
    particle_density: Tensor,
    particle_radiance: Tensor,
    radiance: Tensor,
    density: Tensor,
    hit: Tensor,
    sample_cache: Tensor,
    normals: Tensor,
    render_opts: int,
    sph_degree: int,
    min_transmittance: float,
) -> tuple[Tensor, Tensor, Tensor]:
    del (
        state_token,
        grad_radiance,
        grad_density,
        grad_hit,
        grad_normals,
        ray_to_world,
        ray_ori,
        ray_dir,
        radiance,
        density,
        hit,
        sample_cache,
        normals,
        render_opts,
        sph_degree,
        min_transmittance,
    )
    return (
        torch.empty_like(particle_density),
        torch.empty_like(particle_radiance),
        torch.empty(
            (particle_density.shape[0], 1),
            device=particle_density.device,
            dtype=particle_density.dtype,
        ),
    )


def _trace_impl(
    state_token: Tensor,
    ray_to_world: Tensor,
    ray_ori: Tensor,
    ray_dir: Tensor,
    particle_density: Tensor,
    particle_radiance: Tensor,
    render_opts: int,
    sph_degree: int,
    min_transmittance: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Autograd-enabled trace op."""
    return trace_fwd_op(
        state_token,
        ray_to_world,
        ray_ori,
        ray_dir,
        particle_density,
        particle_radiance,
        render_opts,
        sph_degree,
        min_transmittance,
    )


def _trace_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, ...],
) -> None:
    ctx.save_for_backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        output[0],
        output[1],
        output[2],
        output[7],
        output[3],
    )
    ctx.render_opts = inputs[6]
    ctx.sph_degree = inputs[7]
    ctx.min_transmittance = inputs[8]


def _trace_backward(
    ctx: Any,
    grad_radiance: Tensor,
    grad_density: Tensor,
    grad_hit: Tensor,
    grad_normals: Tensor,
    grad_hitcounts: Tensor,
    grad_visibility: Tensor,
    grad_weights: Tensor,
    grad_sample_cache: Tensor,
) -> tuple[Tensor | None, ...]:
    del grad_hitcounts, grad_visibility, grad_weights, grad_sample_cache
    (
        state_token,
        ray_to_world,
        ray_ori,
        ray_dir,
        particle_density,
        particle_radiance,
        radiance,
        density,
        hit,
        sample_cache,
        normals,
    ) = ctx.saved_tensors
    grad_particle_density, grad_particle_radiance, _grad_ray_vis = trace_bwd_op(
        state_token,
        grad_radiance,
        grad_density,
        grad_hit,
        grad_normals,
        ray_to_world,
        ray_ori,
        ray_dir,
        particle_density,
        particle_radiance,
        radiance,
        density,
        hit,
        sample_cache,
        normals,
        ctx.render_opts,
        ctx.sph_degree,
        ctx.min_transmittance,
    )
    return (
        None,
        None,
        None,
        None,
        grad_particle_density,
        grad_particle_radiance,
        None,
        None,
        None,
    )


trace_op = register_trace_family(
    op_name="traced_native_core::trace",
    forward_impl=_trace_impl,
    fake_impl=_trace_fwd_fake,
    setup_context=_trace_setup_context,
    backward_impl=_trace_backward,
)


__all__ = [
    "trace_bwd_op",
    "trace_fwd_op",
    "trace_op",
]
