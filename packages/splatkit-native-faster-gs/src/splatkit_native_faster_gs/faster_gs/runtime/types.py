"""Typed stage outputs for the FasterGS native runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from jaxtyping import Float, Int, UInt
from torch import Tensor


@dataclass(frozen=True)
class PreprocessResult:
    """Output of the preprocess stage."""

    projected_means: Float[Tensor, " num_primitives 2"]
    conic_opacity: Float[Tensor, " num_primitives 4"]
    colors_rgb: Float[Tensor, " num_primitives 3"]
    primitive_depth: Float[Tensor, " num_primitives"]
    depth_keys: Int[Tensor, " num_primitives"]
    primitive_indices: Int[Tensor, " num_primitives"]
    num_touched_tiles: Int[Tensor, " num_primitives"]
    screen_bounds: UInt[Tensor, " num_primitives 4"]
    visible_count: Int[Tensor, " 1"]
    instance_count: Int[Tensor, " 1"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build a preprocess result from the raw op outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, ...]:
        """Return the raw tensor tuple for custom-op composition."""
        return (
            self.projected_means,
            self.conic_opacity,
            self.colors_rgb,
            self.primitive_depth,
            self.depth_keys,
            self.primitive_indices,
            self.num_touched_tiles,
            self.screen_bounds,
            self.visible_count,
            self.instance_count,
        )


@dataclass(frozen=True)
class SortResult:
    """Output of the sort stage."""

    instance_primitive_indices: Int[Tensor, " num_instances"]
    tile_instance_ranges: Int[Tensor, " num_tiles 2"]
    tile_bucket_offsets: Int[Tensor, " num_tiles"]
    bucket_count: Int[Tensor, " 1"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build a sort result from the raw op outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, ...]:
        """Return the raw tensor tuple for custom-op composition."""
        return (
            self.instance_primitive_indices,
            self.tile_instance_ranges,
            self.tile_bucket_offsets,
            self.bucket_count,
        )


@dataclass(frozen=True)
class BlendResult:
    """Output of the blend stage."""

    image: Float[Tensor, " 3 height width"]
    tile_final_transmittances: Float[Tensor, " num_tile_pixels"]
    tile_max_n_processed: Int[Tensor, " num_tiles"]
    tile_n_processed: Int[Tensor, " num_tile_pixels"]
    bucket_tile_index: Int[Tensor, " num_buckets"]
    bucket_color_transmittance: Float[Tensor, " num_bucket_pixels 4"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build a blend result from the raw op outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, ...]:
        """Return the raw tensor tuple for custom-op composition."""
        return (
            self.image,
            self.tile_final_transmittances,
            self.tile_max_n_processed,
            self.tile_n_processed,
            self.bucket_tile_index,
            self.bucket_color_transmittance,
        )


@dataclass(frozen=True)
class RenderResult:
    """Output of the full render stage."""

    image: Float[Tensor, " 3 height width"]
