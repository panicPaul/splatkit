"""Gaussian/splatting buffer refs."""

from __future__ import annotations

from typing import Final

from ember_core.core.buffers import BufferRef
from ember_core.core.keys import BufferKey
from torch import Tensor

GAUSSIAN_CENTER: Final[BufferRef[Tensor]] = BufferRef(
    BufferKey("gaussian", "center_position"),
    doc="Gaussian centers, shape [capacity, 3].",
)
GAUSSIAN_LOG_SCALES: Final[BufferRef[Tensor]] = BufferRef(
    BufferKey("gaussian", "log_scales"),
)
GAUSSIAN_ROTATION: Final[BufferRef[Tensor]] = BufferRef(
    BufferKey("gaussian", "quaternion_orientation"),
)
GAUSSIAN_OPACITY: Final[BufferRef[Tensor]] = BufferRef(
    BufferKey("gaussian", "logit_opacity"),
)
GAUSSIAN_FEATURE: Final[BufferRef[Tensor]] = BufferRef(
    BufferKey("gaussian", "feature"),
)
GAUSSIAN_DENSIFY_STATS: Final[BufferRef[Tensor]] = BufferRef(
    BufferKey("gaussian", "densify_stats"),
)
GAUSSIAN_CLONE_MASK: Final[BufferRef[Tensor]] = BufferRef(
    BufferKey("gaussian", "clone_mask"),
)
GAUSSIAN_SPLIT_MASK: Final[BufferRef[Tensor]] = BufferRef(
    BufferKey("gaussian", "split_mask"),
)
GAUSSIAN_PRUNE_MASK: Final[BufferRef[Tensor]] = BufferRef(
    BufferKey("gaussian", "prune_mask"),
)
