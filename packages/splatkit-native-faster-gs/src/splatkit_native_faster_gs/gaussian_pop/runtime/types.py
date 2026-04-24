"""Typed runtime containers for the GaussianPOP native backend."""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float, Int
from torch import Tensor


@dataclass(frozen=True)
class BlendResult:
    """Output of the GaussianPOP blend stage."""

    image: Float[Tensor, " 3 height width"]
    tile_final_transmittances: Float[Tensor, " num_tile_pixels"]
    tile_max_n_processed: Int[Tensor, " num_tiles"]
    tile_n_processed: Int[Tensor, " num_tile_pixels"]
    bucket_tile_index: Int[Tensor, " num_buckets"]
    bucket_color_transmittance: Float[Tensor, " num_bucket_pixels 4"]
    bucket_depth_prefix: Float[Tensor, " num_bucket_pixels"] | None
    gaussian_impact_score: Float[Tensor, " num_splats"] | None
    depth: Float[Tensor, " height width"] | None = None


@dataclass(frozen=True)
class RenderResult:
    """Output of the GaussianPOP runtime render path."""

    image: Float[Tensor, " 3 height width"]
    gaussian_impact_score: Float[Tensor, " num_splats"] | None
    depth: Float[Tensor, " height width"] | None = None
