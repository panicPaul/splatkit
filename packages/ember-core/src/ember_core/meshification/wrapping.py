"""Primitive-agnostic wrapping meshification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from beartype import beartype
from jaxtyping import Float
from torch import Tensor

from ember_core.meshification.contracts import (
    MeshArtifact,
    MeshificationOptions,
    MeshificationRequest,
    MeshificationResult,
    WrappingSurfaceProvider,
)
from ember_core.meshification.registry import register_meshifier


@beartype
@dataclass(frozen=True)
class WrappingMeshificationOptions(MeshificationOptions):
    """Configuration for the lightweight wrapping meshifier."""

    padding_fraction: float = 0.05
    min_extent: float = 1.0e-3
    inside_threshold: float = 0.5


def _box_mesh_from_bounds(
    bounds_min: Float[Tensor, " 3"],
    bounds_max: Float[Tensor, " 3"],
) -> MeshArtifact:
    x0, y0, z0 = bounds_min.unbind(dim=0)
    x1, y1, z1 = bounds_max.unbind(dim=0)
    vertices = torch.stack(
        (
            torch.stack((x0, y0, z0)),
            torch.stack((x1, y0, z0)),
            torch.stack((x1, y1, z0)),
            torch.stack((x0, y1, z0)),
            torch.stack((x0, y0, z1)),
            torch.stack((x1, y0, z1)),
            torch.stack((x1, y1, z1)),
            torch.stack((x0, y1, z1)),
        ),
        dim=0,
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=torch.int64,
        device=bounds_min.device,
    )
    return MeshArtifact(
        vertices=vertices,
        faces=faces,
        metadata={"primitive": "axis_aligned_bounds"},
    )


@register_meshifier(
    name="wrapping",
    default_options=WrappingMeshificationOptions(),
    required_trait=WrappingSurfaceProvider,
)
def extract_wrapping_mesh(
    request: MeshificationRequest,
    *,
    options: MeshificationOptions | None,
    provider: object | None,
) -> MeshificationResult:
    """Extract a first-pass wrapping mesh from backend surface samples."""
    if not isinstance(provider, WrappingSurfaceProvider):
        raise TypeError(
            "The wrapping meshifier requires a WrappingSurfaceProvider."
        )
    if options is None:
        resolved_options = WrappingMeshificationOptions()
    elif isinstance(options, WrappingMeshificationOptions):
        resolved_options = options
    else:
        raise TypeError(
            "wrapping meshifier options must be WrappingMeshificationOptions."
        )

    samples = provider.sample_surface_points(request)
    if samples.points.numel() == 0:
        raise ValueError("Wrapping meshification received no samples.")
    query = provider.query_wrapping_field(request, samples.points)
    selected_points = samples.points
    if query.inside is not None and bool(query.inside.any().item()):
        selected_points = samples.points[
            query.values >= resolved_options.inside_threshold
        ]
    if selected_points.numel() == 0:
        selected_points = samples.points

    bounds_min = selected_points.min(dim=0).values
    bounds_max = selected_points.max(dim=0).values
    extents = (bounds_max - bounds_min).clamp_min(resolved_options.min_extent)
    padding = extents * resolved_options.padding_fraction
    center = (bounds_min + bounds_max) * 0.5
    half_extent = extents * 0.5 + padding
    mesh = _box_mesh_from_bounds(center - half_extent, center + half_extent)
    return MeshificationResult(
        mesh=mesh,
        diagnostics={
            "meshifier": "wrapping",
            "num_samples": int(samples.points.shape[0]),
            "num_selected_samples": int(selected_points.shape[0]),
            "inside_threshold": resolved_options.inside_threshold,
        },
    )
