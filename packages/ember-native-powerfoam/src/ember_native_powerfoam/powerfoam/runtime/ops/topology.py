"""Topology custom ops for the PowerFoam Warp runtime."""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

from ember_native_powerfoam.powerfoam.native.warp.bvh import AABBTree


@torch.library.custom_op("powerfoam::build_topology", mutates_args=())
def build_topology_op(
    points: Tensor,
    radii: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build PowerFoam Cech adjacency buffers with vendored Warp kernels."""
    wp.init()
    tree = AABBTree(points.device)
    tree.update(points.detach(), radii.detach())
    adjacency, adjacency_offsets = tree.build_cech_complex()
    return adjacency.to(torch.int32), adjacency_offsets.to(torch.int32)


@build_topology_op.register_fake
def _build_topology_fake(
    points: Tensor,
    radii: Tensor,
) -> tuple[Tensor, Tensor]:
    del radii
    return (
        torch.empty((0,), dtype=torch.int32, device=points.device),
        torch.empty((points.shape[0] + 1,), dtype=torch.int32, device=points.device),
    )


__all__ = ["build_topology_op"]
