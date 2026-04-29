"""Torch custom ops for FasterGS native training utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from ember_native_faster_gs.faster_gs.training.runtime._extension import (
    load_extension,
)


@torch.library.custom_op("faster_gs_training::morton_codes", mutates_args=())
def morton_codes_op(
    positions: Tensor,
    scene_min: Tensor,
    scene_extent: float,
) -> Tensor:
    """Encode 3D positions into 30-bit Morton codes."""
    return load_extension().morton_codes(
        positions,
        scene_min,
        scene_extent,
    )


@morton_codes_op.register_fake
def _morton_codes_fake(
    positions: Tensor,
    scene_min: Tensor,
    scene_extent: float,
) -> Tensor:
    del scene_min, scene_extent
    return torch.empty(
        (positions.shape[0],),
        dtype=torch.int64,
        device=positions.device,
    )


__all__ = ["morton_codes_op"]
