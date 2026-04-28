"""Preprocess-stage custom ops for the SVRaster native runtime."""

from __future__ import annotations

import torch
from torch import Tensor

from ember_native_svraster.core.runtime.ops._common import backend


@torch.library.custom_op("svraster::preprocess", mutates_args=())
def preprocess_op(
    image_width: int,
    image_height: int,
    tanfovx: float,
    tanfovy: float,
    cx: float,
    cy: float,
    w2c_matrix: Tensor,
    c2w_matrix: Tensor,
    near: float,
    octree_paths: Tensor,
    vox_centers: Tensor,
    vox_lengths: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run the native preprocess stage."""
    return backend().rasterize_preprocess(
        image_width,
        image_height,
        tanfovx,
        tanfovy,
        cx,
        cy,
        w2c_matrix,
        c2w_matrix,
        near,
        octree_paths,
        vox_centers,
        vox_lengths,
        False,
    )


@preprocess_op.register_fake
def _preprocess_fake(
    image_width: int,
    image_height: int,
    tanfovx: float,
    tanfovy: float,
    cx: float,
    cy: float,
    w2c_matrix: Tensor,
    c2w_matrix: Tensor,
    near: float,
    octree_paths: Tensor,
    vox_centers: Tensor,
    vox_lengths: Tensor,
) -> tuple[Tensor, Tensor]:
    del (
        image_width,
        image_height,
        tanfovx,
        tanfovy,
        cx,
        cy,
        w2c_matrix,
        c2w_matrix,
        near,
        vox_centers,
        vox_lengths,
    )
    device = octree_paths.device
    num_voxels = int(octree_paths.numel())
    return (
        torch.empty((num_voxels,), device=device, dtype=torch.int32),
        torch.empty((0,), device=device, dtype=torch.uint8),
    )
