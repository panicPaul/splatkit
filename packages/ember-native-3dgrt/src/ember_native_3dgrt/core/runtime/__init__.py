"""Public staged runtime API for traced native backends."""

from __future__ import annotations

from torch import Tensor

from ember_native_3dgrt.core.runtime.ops import (
    build_acc_op,
    destroy_acc_op,
    render_op,
    trace_op,
    update_acc_op,
)
from ember_native_3dgrt.core.runtime.packing import (
    make_render_result,
    make_trace_result,
    pack_particle_density,
)
from ember_native_3dgrt.core.runtime.state import (
    TraceStateConfig,
    acquire_state_token,
    destroy_state_token,
)
from ember_native_3dgrt.core.runtime.types import (
    RenderResult,
    TraceResult,
)


def build_acc(
    state_token: Tensor,
    particle_density: Tensor,
) -> Tensor:
    """Build the traced acceleration structure."""
    (updated_token,) = build_acc_op(state_token, particle_density)
    return updated_token


def update_acc(
    state_token: Tensor,
    particle_density: Tensor,
) -> Tensor:
    """Update the traced acceleration structure."""
    (updated_token,) = update_acc_op(state_token, particle_density)
    return updated_token


def destroy_acc(state_token: Tensor) -> Tensor:
    """Destroy the traced acceleration structure."""
    (released_token,) = destroy_acc_op(state_token)
    return released_token


def trace(
    state_token: Tensor,
    ray_to_world: Tensor,
    ray_ori: Tensor,
    ray_dir: Tensor,
    particle_density: Tensor,
    particle_radiance: Tensor,
    *,
    render_opts: int = 0,
    sph_degree: int,
    min_transmittance: float,
) -> TraceResult:
    """Run the traced forward/backward stage."""
    return make_trace_result(
        trace_op(
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
    )


def render(
    state_token: Tensor,
    ray_to_world: Tensor,
    ray_ori: Tensor,
    ray_dir: Tensor,
    particle_density: Tensor,
    particle_radiance: Tensor,
    bg_color: Tensor,
    *,
    render_opts: int = 0,
    sph_degree: int,
    min_transmittance: float,
) -> RenderResult:
    """Run the full traced render stage."""
    return make_render_result(
        render_op(
            state_token,
            ray_to_world,
            ray_ori,
            ray_dir,
            particle_density,
            particle_radiance,
            render_opts,
            sph_degree,
            min_transmittance,
        ),
        bg_color,
    )


__all__ = [
    "RenderResult",
    "TraceResult",
    "TraceStateConfig",
    "acquire_state_token",
    "build_acc",
    "destroy_acc",
    "destroy_state_token",
    "pack_particle_density",
    "render",
    "trace",
    "update_acc",
]
