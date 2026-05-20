"""Contracts for per-ray compaction loss maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from beartype import beartype
from jaxtyping import Bool, Float
from torch import Tensor


@beartype
@dataclass(frozen=True)
class RayCompactionTargets:
    """Optional supervision targets for ray-compaction losses."""

    rgb: Float[Tensor, "num_cams height width 3"] | None = None
    depth: Float[Tensor, "num_cams height width"] | None = None
    mask: (
        Float[Tensor, "num_cams height width"]
        | Bool[Tensor, "num_cams height width"]
        | None
    ) = None


@beartype
@dataclass(frozen=True)
class RayCompactionLossMaps:
    """Unreduced per-pixel ray-compaction losses."""

    color_l2: Float[Tensor, "num_cams height width"] | None = None
    depth_l2: Float[Tensor, "num_cams height width"] | None = None
    feature_l2: Float[Tensor, "num_cams height width"] | None = None
    weight_sum: Float[Tensor, "num_cams height width"] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

