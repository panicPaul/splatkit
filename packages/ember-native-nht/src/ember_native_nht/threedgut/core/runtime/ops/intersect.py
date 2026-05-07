"""Tile-intersection stage native ops for the NHT runtime."""

from __future__ import annotations

import math

from ember_native_nht.threedgut.core.runtime.ops._common import backend
from ember_native_nht.threedgut.core.runtime.packing import (
    parse_intersection_outputs,
)
from ember_native_nht.threedgut.core.runtime.types import IntersectionResult
from torch import Tensor


def intersect(
    *,
    projected_means: Tensor,
    radii: Tensor,
    primitive_depths: Tensor,
    num_cameras: int,
    image_width: int,
    image_height: int,
    tile_size: int,
) -> IntersectionResult:
    """Map projected Gaussians to sorted tile intersections."""
    tile_width = math.ceil(image_width / float(tile_size))
    tile_height = math.ceil(image_height / float(tile_size))
    tiles_per_gaussian, tile_intersection_ids, flattened_gaussian_ids = (
        backend().intersect_tiles_fwd(
            projected_means.contiguous(),
            radii.contiguous(),
            primitive_depths.contiguous(),
            None,
            None,
            num_cameras,
            tile_size,
            tile_width,
            tile_height,
            True,
            False,
        )
    )
    tile_offsets = backend().intersect_offsets_fwd(
        tile_intersection_ids.contiguous(),
        num_cameras,
        tile_width,
        tile_height,
    )
    return parse_intersection_outputs(
        (
            tiles_per_gaussian,
            tile_intersection_ids,
            flattened_gaussian_ids,
            tile_offsets,
        )
    )
