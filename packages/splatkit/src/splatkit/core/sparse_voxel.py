"""Sparse-voxel helper functions shared by I/O and backend adapters."""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor

SVRasterBackendName = Literal[
    "new_cuda",
    "new_cuda_aa",
    "new_cuda_cont",
    "new_cuda_spline",
]
SUPPORTED_SVRASTER_BACKENDS = frozenset({"new_cuda"})
DEFAULT_SVRASTER_MAX_NUM_LEVELS = 16
_SUBTREE_SHIFTS = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=torch.int64,
)
_SH_ZERO_SCALE = 0.28209479177387814


def _get_backend_utils(
    backend_name: SVRasterBackendName,
) -> object | None:
    """Return backend utility bindings when the native backend is installed."""
    if backend_name != "new_cuda":
        return None
    try:
        import new_svraster_cuda
    except ImportError:
        return None
    return new_svraster_cuda.utils


def _get_backend_max_num_levels(
    backend_name: SVRasterBackendName,
    fallback_max_num_levels: int,
) -> int:
    """Return the backend-native finest octree level."""
    if backend_name != "new_cuda":
        return fallback_max_num_levels
    try:
        import new_svraster_cuda
    except ImportError:
        return fallback_max_num_levels
    return int(new_svraster_cuda.meta.MAX_NUM_LEVELS)


def svraster_rgb_to_sh_zero(
    rgb: Float[Tensor, "... 3"],
) -> Float[Tensor, "... 3"]:
    """Convert RGB values in [0, 1] to SV Raster SH-zero coefficients."""
    return (rgb - 0.5) / _SH_ZERO_SCALE


def svraster_sh_zero_to_rgb(
    sh0: Float[Tensor, "... 3"],
) -> Float[Tensor, "... 3"]:
    """Convert SV Raster SH-zero coefficients to RGB values in [0, 1]."""
    return sh0 * _SH_ZERO_SCALE + 0.5


def svraster_level_to_voxel_size(
    scene_extent: Float[Tensor, " 1"],
    octlevel: Int[Tensor, "num_voxels 1"],
) -> Float[Tensor, "num_voxels 1"]:
    """Compute voxel sizes from octree levels."""
    return scene_extent * torch.pow(2.0, -octlevel.to(torch.float32))


def svraster_octpath_to_ijk(
    octpath: Int[Tensor, "num_voxels 1"],
    octlevel: Int[Tensor, "num_voxels 1"],
    *,
    backend_name: SVRasterBackendName | None = None,
    max_num_levels: int,
) -> Int[Tensor, "num_voxels 3"]:
    """Decode octree paths into integer voxel coordinates at each voxel level."""
    if backend_name is not None:
        backend_utils = _get_backend_utils(backend_name)
        if backend_utils is not None and octpath.device.type == "cuda":
            return backend_utils.octpath_2_ijk(
                octpath.reshape(-1, 1),
                octlevel.reshape(-1, 1).to(torch.int8),
            )

    squeezed_path = octpath.reshape(-1)
    squeezed_level = octlevel.reshape(-1).to(torch.int64)
    ijk_at_max = torch.zeros(
        (squeezed_path.shape[0], 3),
        dtype=torch.int64,
        device=octpath.device,
    )
    for level in range(1, max_num_levels + 1):
        bit_shift = 3 * (max_num_levels - level)
        subtree_id = (squeezed_path >> bit_shift) & 0b111
        ijk_at_max[:, 0] |= ((subtree_id >> 2) & 1) << (max_num_levels - level)
        ijk_at_max[:, 1] |= ((subtree_id >> 1) & 1) << (max_num_levels - level)
        ijk_at_max[:, 2] |= (subtree_id & 1) << (max_num_levels - level)
    level_shift = (max_num_levels - squeezed_level).clamp_min(0)
    return ijk_at_max >> level_shift[:, None]


def svraster_octpath_decoding(
    octpath: Int[Tensor, "num_voxels 1"],
    octlevel: Int[Tensor, "num_voxels 1"],
    scene_center: Float[Tensor, " 3"],
    scene_extent: Float[Tensor, " 1"],
    *,
    backend_name: SVRasterBackendName | None = None,
    max_num_levels: int,
) -> tuple[
    Float[Tensor, "num_voxels 3"],
    Float[Tensor, "num_voxels 1"],
]:
    """Compute voxel centers and sizes from sparse-voxel checkpoint state."""
    vox_size = svraster_level_to_voxel_size(scene_extent, octlevel)
    scene_min = scene_center - 0.5 * scene_extent
    ijk = svraster_octpath_to_ijk(
        octpath,
        octlevel,
        backend_name=backend_name,
        max_num_levels=max_num_levels,
    ).to(vox_size.dtype)
    vox_center = scene_min + (ijk + 0.5) * vox_size
    return vox_center, vox_size


def svraster_build_grid_points_link(
    octpath: Int[Tensor, "num_voxels 1"],
    octlevel: Int[Tensor, "num_voxels 1"],
    *,
    backend_name: SVRasterBackendName | None = None,
    max_num_levels: int,
) -> tuple[
    Int[Tensor, "num_grid_points 3"],
    Int[Tensor, "num_voxels 8"],
]:
    """Build the voxel-corner lookup used by SV Raster checkpoints."""
    ijk = svraster_octpath_to_ijk(
        octpath,
        octlevel,
        backend_name=backend_name,
        max_num_levels=max_num_levels,
    )
    finest_level = (
        _get_backend_max_num_levels(backend_name, max_num_levels)
        if backend_name is not None
        else max_num_levels
    )
    level_shift = (
        finest_level - octlevel.to(torch.int64)
    ).reshape(-1, 1, 1)
    subtree_shifts = _SUBTREE_SHIFTS.to(octpath.device)
    base_grid_ijk = (ijk << level_shift.reshape(-1, 1)).reshape(-1, 1, 3)
    grid_points = base_grid_ijk + (subtree_shifts << level_shift)
    grid_points_key, vox_key = grid_points.reshape(-1, 3).unique(
        dim=0,
        return_inverse=True,
    )
    return grid_points_key.contiguous(), vox_key.reshape(-1, 8).contiguous()
