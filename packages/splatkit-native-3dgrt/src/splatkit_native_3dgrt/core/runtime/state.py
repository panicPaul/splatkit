"""Process-local acceleration-state management for traced native backends."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch
from torch import Tensor

from splatkit_native_3dgrt.native_build.stoch3dgs import (
    Stoch3DGSPluginConfig,
    get_cuda_home,
    load_stoch3dgs_optix_tracer_runtime,
)


@dataclass(frozen=True)
class TraceStateConfig:
    """Configuration for a traced OptiX state instance."""

    pipeline_type: str = "fullStochastic"
    backward_pipeline_type: str = "fullStochasticBwd"
    primitive_type: str = "instances"
    particle_kernel_degree: int = 4
    particle_kernel_density_clamping: bool = True
    particle_kernel_min_response: float = 0.0113
    particle_kernel_min_alpha: float = 1.0 / 255.0
    particle_kernel_max_alpha: float = 0.99
    particle_radiance_sph_degree: int = 0
    min_transmittance: float = 0.001
    enable_normals: bool = True
    enable_hitcounts: bool = True
    max_consecutive_bvh_update: int = 2

    def to_plugin_config(self) -> Stoch3DGSPluginConfig:
        """Project the state config onto the plugin specialization config."""
        return Stoch3DGSPluginConfig(
            pipeline_type=self.pipeline_type,
            backward_pipeline_type=self.backward_pipeline_type,
            primitive_type=self.primitive_type,
            particle_kernel_degree=self.particle_kernel_degree,
            particle_kernel_min_response=self.particle_kernel_min_response,
            particle_kernel_density_clamping=self.particle_kernel_density_clamping,
            particle_kernel_min_alpha=self.particle_kernel_min_alpha,
            particle_kernel_max_alpha=self.particle_kernel_max_alpha,
            particle_radiance_sph_degree=self.particle_radiance_sph_degree,
            enable_normals=self.enable_normals,
            enable_hitcounts=self.enable_hitcounts,
        )


@dataclass
class _TraceState:
    """Mutable root-managed traced state."""

    tracer_wrapper: Any
    config: TraceStateConfig
    num_update_bvh: int = 0


_STATE_LOCK = Lock()
_NEXT_STATE_ID = 1
_STATES: dict[int, _TraceState] = {}


def _make_tracer_wrapper(config: TraceStateConfig) -> Any:
    """Instantiate a configured vendored OptiX tracer wrapper."""
    runtime = load_stoch3dgs_optix_tracer_runtime(
        config.to_plugin_config()
    )
    torch.zeros(1, device="cuda")
    return runtime.tracer_class(
        runtime.source_root,
        get_cuda_home(),
        config.pipeline_type,
        config.backward_pipeline_type,
        config.primitive_type,
        config.particle_kernel_degree,
        config.particle_kernel_min_response,
        config.particle_kernel_density_clamping,
        config.particle_radiance_sph_degree,
        config.enable_normals,
        config.enable_hitcounts,
    )


def acquire_state_token(
    config: TraceStateConfig,
    device: torch.device,
) -> Tensor:
    """Create a new traced state and return its token tensor."""
    global _NEXT_STATE_ID
    tracer_wrapper = _make_tracer_wrapper(config)
    with _STATE_LOCK:
        state_id = _NEXT_STATE_ID
        _NEXT_STATE_ID += 1
        _STATES[state_id] = _TraceState(
            tracer_wrapper=tracer_wrapper,
            config=config,
        )
    return torch.tensor(state_id, dtype=torch.int64, device=device)


def get_state(state_token: Tensor) -> _TraceState:
    """Resolve a traced state token into the managed state object."""
    state_id = int(state_token.item())
    with _STATE_LOCK:
        state = _STATES.get(state_id)
    if state is None:
        raise KeyError(f"Unknown traced state token {state_id}.")
    return state


def destroy_state_token(state_token: Tensor) -> None:
    """Release a traced state token and its associated native resources."""
    state_id = int(state_token.item())
    with _STATE_LOCK:
        _STATES.pop(state_id, None)


def _split_particle_density(particle_density: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split packed particle density into the pieces expected by the OptiX tracer."""
    return (
        particle_density[:, 0:3].contiguous(),
        particle_density[:, 4:8].contiguous(),
        particle_density[:, 8:11].contiguous(),
        particle_density[:, 3:4].contiguous(),
    )


def build_or_update_acc(
    state_token: Tensor,
    particle_density: Tensor,
    *,
    force_rebuild: bool,
) -> Tensor:
    """Run the vendored BVH build/update path and return the input token."""
    state = get_state(state_token)
    allow_update = (
        state.config.max_consecutive_bvh_update > 1
        and not state.config.particle_kernel_density_clamping
    )
    rebuild = (
        force_rebuild
        or state.config.particle_kernel_density_clamping
        or state.num_update_bvh >= state.config.max_consecutive_bvh_update
    )
    positions, rotations, scales, densities = _split_particle_density(
        particle_density.contiguous()
    )
    state.tracer_wrapper.build_bvh(
        positions,
        rotations,
        scales,
        densities,
        rebuild,
        allow_update,
    )
    state.num_update_bvh = 0 if rebuild else state.num_update_bvh + 1
    return state_token


__all__ = [
    "TraceStateConfig",
    "acquire_state_token",
    "build_or_update_acc",
    "destroy_state_token",
    "get_state",
]
