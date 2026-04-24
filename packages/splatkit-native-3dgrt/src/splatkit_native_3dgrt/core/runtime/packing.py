"""Helpers for packing traced runtime inputs and outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as torch_f
from torch import Tensor

from splatkit_native_3dgrt.core.runtime.types import (
    RawRenderOutputs,
    RawTraceOutputs,
    RenderResult,
    TraceResult,
)


def pack_particle_density(
    positions: Tensor,
    densities: Tensor,
    rotations: Tensor,
    scales: Tensor,
) -> Tensor:
    """Pack activated particle data into the tracer-native density layout."""
    zeros = torch.zeros_like(densities)
    return torch.cat((positions, densities, rotations, scales, zeros), dim=1)


def make_trace_result(outputs: RawTraceOutputs) -> TraceResult:
    """Convert raw trace outputs into the public typed result."""
    radiance, density, hit, normals, hitcounts, visibility, weights, _sample_cache = (
        outputs
    )
    return TraceResult(
        radiance=radiance,
        density=density.squeeze(-1),
        depth=hit[..., 0],
        normals=torch_f.normalize(normals, dim=3),
        hitcounts=hitcounts.squeeze(-1),
        visibility=visibility,
        weights=weights,
    )


def make_render_result(outputs: RawRenderOutputs, bg_color: Tensor) -> RenderResult:
    """Convert raw render outputs into the public typed result."""
    trace_result = make_trace_result(outputs)
    bg = bg_color.to(device=trace_result.radiance.device, dtype=trace_result.radiance.dtype)
    render = trace_result.radiance + bg.view(1, 1, 1, 3) * (
        1.0 - trace_result.density.unsqueeze(-1)
    )
    return RenderResult(
        render=render,
        alphas=trace_result.density,
        depth=trace_result.depth,
        normals=trace_result.normals,
        hitcounts=trace_result.hitcounts,
        visibility=trace_result.visibility,
        weights=trace_result.weights,
    )


__all__ = [
    "make_render_result",
    "make_trace_result",
    "pack_particle_density",
]
