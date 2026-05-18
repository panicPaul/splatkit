"""Public RADFOAM native runtime facade."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int, UInt
from torch import Tensor

from ember_native_radfoam.radfoam.runtime.ops import (
    build_aabb_tree_op,
    farthest_neighbor_op,
    nearest_neighbor_op,
    trace_op,
    triangulate_op,
)
from ember_native_radfoam.radfoam.runtime.types import (
    RadFoamTopology,
    TraceResult,
)

MIN_RADFOAM_POINTS = 32


def _validate_min_points(points: Tensor) -> None:
    if points.shape[0] < MIN_RADFOAM_POINTS:
        raise ValueError(
            "RADFOAM native topology requires at least "
            f"{MIN_RADFOAM_POINTS} points; got {points.shape[0]}."
        )


def build_aabb_tree(
    points: Float[Tensor, "num_points 3"],
) -> Float[Tensor, "tree_nodes 2 3"]:
    """Build RADFOAM's AABB tree over primal points."""
    return build_aabb_tree_op(points.contiguous())


def nearest_neighbor(
    points: Float[Tensor, "num_points 3"],
    aabb_tree: Float[Tensor, "tree_nodes 2 3"],
    queries: Float[Tensor, "... 3"],
) -> UInt[Tensor, " ..."]:
    """Find the nearest RADFOAM point for each query."""
    _validate_min_points(points)
    return nearest_neighbor_op(
        points.contiguous(),
        aabb_tree.contiguous(),
        queries.contiguous(),
    )


def farthest_neighbor(
    points: Float[Tensor, "num_points 3"],
    point_adjacency: Int[Tensor, " num_adjacency"]
    | UInt[Tensor, " num_adjacency"],
    point_adjacency_offsets: Int[Tensor, " adjacency_offsets"]
    | UInt[Tensor, " adjacency_offsets"],
) -> tuple[UInt[Tensor, " num_points"], Float[Tensor, " num_points"]]:
    """Find each point's farthest adjacent point and cell radius."""
    return farthest_neighbor_op(
        points.contiguous(),
        point_adjacency.contiguous().to(torch.uint32),
        point_adjacency_offsets.contiguous().to(torch.uint32),
    )


def build_radfoam_topology(
    points: Float[Tensor, "num_points 3"],
) -> RadFoamTopology:
    """Triangulate points and return RADFOAM topology buffers."""
    _validate_min_points(points)
    permutation, point_adjacency, point_adjacency_offsets = triangulate_op(
        points.contiguous()
    )
    ordered_points = points[permutation].contiguous()
    aabb_tree = build_aabb_tree(ordered_points)
    return RadFoamTopology(
        permutation=permutation,
        point_adjacency=point_adjacency,
        point_adjacency_offsets=point_adjacency_offsets,
        aabb_tree=aabb_tree,
    )


def trace(
    points: Float[Tensor, "num_points 3"],
    attributes: Float[Tensor, "num_points attributes"],
    point_adjacency: Int[Tensor, " num_adjacency"]
    | UInt[Tensor, " num_adjacency"],
    point_adjacency_offsets: Int[Tensor, " adjacency_offsets"]
    | UInt[Tensor, " adjacency_offsets"],
    rays: Float[Tensor, "... 6"],
    start_point: UInt[Tensor, " ..."] | Int[Tensor, " ..."],
    *,
    depth_quantiles: Float[Tensor, "... quantiles"] | None = None,
    return_contribution: bool = False,
    sh_degree: int = 3,
    weight_threshold: float = 0.001,
    max_intersections: int = 1024,
) -> TraceResult:
    """Trace world-space rays through a RADFOAM scene."""
    _validate_min_points(points)
    resolved_depth_quantiles = (
        torch.empty(
            (*rays.shape[:-1], 0),
            dtype=torch.float32,
            device=rays.device,
        )
        if depth_quantiles is None
        else depth_quantiles
    )
    rgba, depth, depth_indices, contribution, num_intersections = trace_op(
        points.contiguous(),
        attributes.contiguous(),
        point_adjacency.contiguous().to(torch.uint32),
        point_adjacency_offsets.contiguous().to(torch.uint32),
        rays.contiguous(),
        start_point.contiguous().to(torch.uint32),
        resolved_depth_quantiles.contiguous(),
        return_contribution,
        sh_degree,
        weight_threshold,
        max_intersections,
    )
    return TraceResult(
        rgba=rgba,
        depth=depth,
        depth_indices=depth_indices,
        contribution=contribution,
        num_intersections=num_intersections,
    )


__all__ = [
    "MIN_RADFOAM_POINTS",
    "build_aabb_tree",
    "build_radfoam_topology",
    "farthest_neighbor",
    "nearest_neighbor",
    "trace",
]
