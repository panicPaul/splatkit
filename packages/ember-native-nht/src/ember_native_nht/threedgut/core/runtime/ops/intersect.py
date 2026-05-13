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
    num_cameras: int,
    image_width: int,
    image_height: int,
    tile_size: int,
    primitive_depth: Tensor | None = None,
    primitive_depths: Tensor | None = None,
) -> IntersectionResult:
    """Map projected Gaussians to sorted tile intersections."""
    if primitive_depth is None:
        if primitive_depths is None:
            raise TypeError("intersect requires primitive_depth.")
        primitive_depth = primitive_depths
    tile_width = math.ceil(image_width / float(tile_size))
    tile_height = math.ceil(image_height / float(tile_size))
    num_touched_tiles, intersection_ids, instance_primitive_indices = (
        backend().intersect_fwd(
            projected_means.contiguous(),
            radii.contiguous(),
            primitive_depth.contiguous(),
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
        intersection_ids.contiguous(),
        num_cameras,
        tile_width,
        tile_height,
    )
    return parse_intersection_outputs(
        (
            num_touched_tiles,
            intersection_ids,
            instance_primitive_indices,
            tile_offsets,
        )
    )
