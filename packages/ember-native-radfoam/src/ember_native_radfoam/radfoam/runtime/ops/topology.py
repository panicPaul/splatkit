"""Topology custom ops for the RADFOAM native runtime."""

from __future__ import annotations

import torch
from torch import Tensor

from ember_native_radfoam.radfoam.runtime.ops._common import (
    backend,
    pow2_round_up,
)


@torch.library.custom_op("radfoam::triangulate", mutates_args=())
def triangulate_op(points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Low-level RADFOAM Delaunay triangulation op."""
    triangulation = backend().Triangulation(points.contiguous())
    return (
        triangulation.permutation().to(torch.long).clone(),
        triangulation.point_adjacency().clone(),
        triangulation.point_adjacency_offsets().clone(),
    )


@triangulate_op.register_fake
def _triangulate_fake(points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty((points.shape[0],), dtype=torch.long, device=points.device),
        torch.empty((0,), dtype=torch.uint32, device=points.device),
        torch.empty((points.shape[0] + 1,), dtype=torch.uint32, device=points.device),
    )


@torch.library.custom_op("radfoam::build_aabb_tree", mutates_args=())
def build_aabb_tree_op(points: Tensor) -> Tensor:
    """Low-level RADFOAM AABB tree build op."""
    return backend().build_aabb_tree(points.contiguous())


@build_aabb_tree_op.register_fake
def _build_aabb_tree_fake(points: Tensor) -> Tensor:
    return torch.empty(
        (pow2_round_up(int(points.shape[0])), 2, 3),
        dtype=points.dtype,
        device=points.device,
    )


@torch.library.custom_op("radfoam::nearest_neighbor", mutates_args=())
def nearest_neighbor_op(
    points: Tensor,
    aabb_tree: Tensor,
    queries: Tensor,
) -> Tensor:
    """Low-level RADFOAM nearest-neighbor query op."""
    return backend().nn(
        points.contiguous(),
        aabb_tree.contiguous(),
        queries.contiguous(),
    )


@nearest_neighbor_op.register_fake
def _nearest_neighbor_fake(
    points: Tensor,
    aabb_tree: Tensor,
    queries: Tensor,
) -> Tensor:
    del points, aabb_tree
    return torch.empty(
        queries.shape[:-1],
        dtype=torch.uint32,
        device=queries.device,
    )


@torch.library.custom_op("radfoam::farthest_neighbor", mutates_args=())
def farthest_neighbor_op(
    points: Tensor,
    point_adjacency: Tensor,
    point_adjacency_offsets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Low-level RADFOAM farthest-neighbor query op."""
    return backend().farthest_neighbor(
        points.contiguous(),
        point_adjacency.contiguous(),
        point_adjacency_offsets.contiguous(),
    )


@farthest_neighbor_op.register_fake
def _farthest_neighbor_fake(
    points: Tensor,
    point_adjacency: Tensor,
    point_adjacency_offsets: Tensor,
) -> tuple[Tensor, Tensor]:
    del point_adjacency, point_adjacency_offsets
    return (
        torch.empty(
            (points.shape[0],),
            dtype=torch.uint32,
            device=points.device,
        ),
        torch.empty(
            (points.shape[0],),
            dtype=torch.float32,
            device=points.device,
        ),
    )
