"""Typed stage outputs for the PowerFoam runtime."""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Int
from torch import Tensor


@dataclass(frozen=True)
class PowerFoamTopology:
    """PowerFoam Cech adjacency buffers."""

    adjacency: Int[Tensor, " num_adjacency"]
    adjacency_offsets: Int[Tensor, " adjacency_offsets"]

