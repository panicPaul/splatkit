"""Typed stage outputs for the FasterGS depth native runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from jaxtyping import Float, Int
from torch import Tensor


@dataclass(frozen=True)
class BlendResult:
    """Output of the depth-aware blend stage."""

    image: Float[Tensor, " 3 height width"]
    depth: Float[Tensor, " height width"]
    tile_final_transmittances: Float[Tensor, " num_tile_pixels"]
    tile_max_n_processed: Int[Tensor, " num_tiles"]
    tile_n_processed: Int[Tensor, " num_tile_pixels"]
    bucket_tile_index: Int[Tensor, " num_buckets"]
    bucket_color_transmittance: Float[Tensor, " num_bucket_pixels 4"]
    bucket_depth_prefix: Float[Tensor, " num_bucket_pixels"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build a blend result from the raw op outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, ...]:
        """Return the raw tensor tuple for custom-op composition."""
        return (
            self.image,
            self.depth,
            self.tile_final_transmittances,
            self.tile_max_n_processed,
            self.tile_n_processed,
            self.bucket_tile_index,
            self.bucket_color_transmittance,
            self.bucket_depth_prefix,
        )


@dataclass(frozen=True)
class RenderResult:
    """Output of the depth-aware render stage."""

    image: Float[Tensor, " 3 height width"]
    depth: Float[Tensor, " height width"]
