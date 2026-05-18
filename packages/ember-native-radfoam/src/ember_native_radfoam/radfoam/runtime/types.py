"""Typed stage outputs for the RADFOAM native runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from jaxtyping import Float, Int, UInt
from torch import Tensor


@dataclass(frozen=True)
class RadFoamTopology:
    """RADFOAM topology buffers built from primal points."""

    permutation: Int[Tensor, " num_points"]
    point_adjacency: UInt[Tensor, " num_adjacency"]
    point_adjacency_offsets: UInt[Tensor, " adjacency_offsets"]
    aabb_tree: Float[Tensor, " aabb_nodes 2 3"]


@dataclass(frozen=True)
class TraceResult:
    """Output of the RADFOAM trace stage."""

    rgba: Float[Tensor, "... 4"]
    depth: Float[Tensor, "... num_depth_quantiles"]
    depth_indices: UInt[Tensor, "... num_depth_quantiles"]
    contribution: Float[Tensor, " contribution_points 1"]
    num_intersections: UInt[Tensor, "... 1"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build a trace result from raw op outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, ...]:
        """Return raw tensor outputs for custom-op composition."""
        return (
            self.rgba,
            self.depth,
            self.depth_indices,
            self.contribution,
            self.num_intersections,
        )
