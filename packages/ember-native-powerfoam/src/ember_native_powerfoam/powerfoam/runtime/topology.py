"""Topology stages for PowerFoam Cech adjacency."""

from __future__ import annotations

import torch
from ember_core.core.contracts import PowerFoamScene
from jaxtyping import Float
from torch import Tensor

from ember_native_powerfoam.powerfoam.runtime.ops import build_topology_op
from ember_native_powerfoam.powerfoam.runtime.scene_math import powerfoam_radii
from ember_native_powerfoam.powerfoam.runtime.types import PowerFoamTopology


def build_powerfoam_topology(
    points: Float[Tensor, "num_points 3"],
    radii: Float[Tensor, " num_points"],
) -> PowerFoamTopology:
    """Build PowerFoam's Cech adjacency buffers."""
    if not points.is_cuda:
        raise ValueError("PowerFoam topology requires CUDA points.")
    adjacency, adjacency_offsets = build_topology_op(
        points.contiguous(),
        radii.contiguous(),
    )
    return PowerFoamTopology(
        adjacency=adjacency.to(torch.int32),
        adjacency_offsets=adjacency_offsets.to(torch.int32),
    )


def rebuild_powerfoam_topology(
    scene: PowerFoamScene,
    *,
    radii_beta: float = 100.0,
) -> PowerFoamTopology:
    """Build PowerFoam topology from an Ember PowerFoam scene."""
    return build_powerfoam_topology(
        scene.points,
        powerfoam_radii(scene, beta=radii_beta),
    )
