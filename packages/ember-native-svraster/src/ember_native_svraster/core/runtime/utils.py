"""Utility wrappers for the SVRaster native runtime."""

from __future__ import annotations

from torch import Tensor

from ember_native_svraster.core.runtime._extension import load_extension


def max_num_levels() -> int:
    """Return the native SVRaster finest octree level."""
    return int(load_extension().MAX_NUM_LEVELS)


def voxel_order_rank(octree_paths: Tensor) -> Tensor:
    """Compute the eight possible voxel rendering orders."""
    return load_extension().voxel_order_rank(octree_paths)


def ijk_2_octpath(ijk: Tensor, octlevel: Tensor) -> Tensor:
    """Encode integer voxel coordinates into octree paths."""
    return load_extension().ijk_2_octpath(ijk, octlevel)


def octpath_2_ijk(octpath: Tensor, octlevel: Tensor) -> Tensor:
    """Decode octree paths into integer voxel coordinates."""
    return load_extension().octpath_2_ijk(octpath, octlevel)


def is_in_cone(
    tanfovx: float,
    tanfovy: float,
    near: float,
    w2c_matrix: Tensor,
    pts: Tensor,
) -> Tensor:
    """Test whether points fall inside the SVRaster camera cone."""
    return load_extension().is_in_cone(
        tanfovx,
        tanfovy,
        near,
        w2c_matrix,
        pts,
    )


def compute_rd(
    width: int,
    height: int,
    cx: float,
    cy: float,
    tanfovx: float,
    tanfovy: float,
    c2w_matrix: Tensor,
) -> Tensor:
    """Compute world-space ray directions for an image plane."""
    return load_extension().compute_rd(
        width,
        height,
        cx,
        cy,
        tanfovx,
        tanfovy,
        c2w_matrix,
    )


def depth2pts(
    width: int,
    height: int,
    cx: float,
    cy: float,
    tanfovx: float,
    tanfovy: float,
    c2w_matrix: Tensor,
    depth: Tensor,
) -> Tensor:
    """Project an SVRaster depth buffer back to world-space points."""
    return load_extension().depth2pts(
        width,
        height,
        cx,
        cy,
        tanfovx,
        tanfovy,
        c2w_matrix,
        depth,
    )
