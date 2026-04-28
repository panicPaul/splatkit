"""Render-stage custom ops for traced native backends."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ember_native_3dgrt.core.reuse.factories import (
    register_render_family,
)
from ember_native_3dgrt.core.runtime.ops.trace import (
    _trace_backward,
    _trace_fwd_fake,
    _trace_setup_context,
    trace_bwd_op,
    trace_fwd_op,
)


def render_fwd_op(
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
    """Low-level native render forward op."""
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


render_fwd_op = torch.library.custom_op(
    "core::render_fwd",
    mutates_args=(),
)(render_fwd_op)
render_fwd_op.register_fake(_trace_fwd_fake)


def render_bwd_op(
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
    """Low-level native render backward op."""
    return trace_bwd_op(
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
        render_opts,
        sph_degree,
        min_transmittance,
    )


render_bwd_op = torch.library.custom_op(
    "core::render_bwd",
    mutates_args=(),
)(render_bwd_op)


@render_bwd_op.register_fake
def _render_bwd_fake(
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


def _render_impl(
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
    """Autograd-enabled render op."""
    return render_fwd_op(
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


def _render_backward(
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
    return _trace_backward(
        ctx,
        grad_radiance,
        grad_density,
        grad_hit,
        grad_normals,
        grad_hitcounts,
        grad_visibility,
        grad_weights,
        grad_sample_cache,
    )


render_op = register_render_family(
    op_name="core::render",
    forward_impl=_render_impl,
    fake_impl=_trace_fwd_fake,
    setup_context=_trace_setup_context,
    backward_impl=_render_backward,
)


__all__ = [
    "render_bwd_op",
    "render_fwd_op",
    "render_op",
]
