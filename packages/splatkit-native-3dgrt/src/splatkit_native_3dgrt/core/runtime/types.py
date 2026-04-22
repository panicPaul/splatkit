"""Typed runtime containers for traced native backends."""

from __future__ import annotations

from dataclasses import dataclass

from beartype import beartype
from jaxtyping import Float
from torch import Tensor


@beartype
@dataclass(frozen=True)
class TraceResult:
    """Public traced-stage outputs."""

    radiance: Float[Tensor, " num_cams height width 3"]
    density: Float[Tensor, " num_cams height width"]
    depth: Float[Tensor, " num_cams height width"]
    normals: Float[Tensor, " num_cams height width 3"]
    hitcounts: Float[Tensor, " num_cams height width"]
    visibility: Float[Tensor, " num_splats 1"]
    weights: Float[Tensor, " num_splats 1"]


@beartype
@dataclass(frozen=True)
class RenderResult:
    """Public render-stage outputs."""

    render: Float[Tensor, " num_cams height width 3"]
    alphas: Float[Tensor, " num_cams height width"]
    depth: Float[Tensor, " num_cams height width"]
    normals: Float[Tensor, " num_cams height width 3"]
    hitcounts: Float[Tensor, " num_cams height width"]
    visibility: Float[Tensor, " num_splats 1"]
    weights: Float[Tensor, " num_splats 1"]


RawTraceOutputs = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


RawRenderOutputs = RawTraceOutputs


__all__ = [
    "RawRenderOutputs",
    "RawTraceOutputs",
    "RenderResult",
    "TraceResult",
]
