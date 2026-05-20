"""Typed stage outputs for the native NHT runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from jaxtyping import Float, Int
from torch import Tensor


@dataclass(frozen=True)
class ProjectionResult:
    """Output of the 3DGUT projection stage."""

    radii: Int[Tensor, " num_cams num_splats 2"]
    projected_means: Float[Tensor, " num_cams num_splats 2"]
    primitive_depth: Float[Tensor, " num_cams num_splats"]
    conics: Float[Tensor, " num_cams num_splats 3"]
    mip_splatting_screen_filter_compensations: (
        Float[Tensor, " num_cams num_splats"] | None
    )

    @classmethod
    def from_tensors(cls, *tensors: Tensor | None) -> Self:
        """Build a projection result from raw native outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor | None, ...]:
        """Return the raw tensor tuple for stage composition."""
        return (
            self.radii,
            self.projected_means,
            self.primitive_depth,
            self.conics,
            self.mip_splatting_screen_filter_compensations,
        )

    @property
    def primitive_depths(self) -> Float[Tensor, " num_cams num_splats"]:
        """Compatibility alias for earlier NHT runtime callers."""
        return self.primitive_depth


@dataclass(frozen=True)
class IntersectionResult:
    """Output of the tile-intersection stage."""

    num_touched_tiles: Int[Tensor, " num_cams num_splats"]
    intersection_ids: Int[Tensor, " num_intersections"]
    instance_primitive_indices: Int[Tensor, " num_intersections"]
    tile_offsets: Int[Tensor, " num_cams tile_height tile_width"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build an intersection result from raw native outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, ...]:
        """Return the raw tensor tuple for stage composition."""
        return (
            self.num_touched_tiles,
            self.intersection_ids,
            self.instance_primitive_indices,
            self.tile_offsets,
        )

    @property
    def tiles_per_gaussian(self) -> Int[Tensor, " num_cams num_splats"]:
        """Compatibility alias for earlier NHT runtime callers."""
        return self.num_touched_tiles

    @property
    def tile_intersection_ids(self) -> Int[Tensor, " num_intersections"]:
        """Compatibility alias for earlier NHT runtime callers."""
        return self.intersection_ids

    @property
    def flattened_gaussian_ids(self) -> Int[Tensor, " num_intersections"]:
        """Compatibility alias for earlier NHT runtime callers."""
        return self.instance_primitive_indices


@dataclass(frozen=True)
class FeatureRasterizationResult:
    """Output of the NHT feature rasterization stage."""

    features: Float[Tensor, " num_cams height width feature_dim"]
    alphas: Float[Tensor, " num_cams height width 1"]
    feature_square_sums: Float[Tensor, " num_cams height width feature_dim"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build a feature rasterization result from raw native outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return the raw tensor tuple for stage composition."""
        return (self.features, self.alphas, self.feature_square_sums)


@dataclass(frozen=True)
class DepthRasterizationResult:
    """Output of the eval3d depth rasterization stage."""

    depths: Float[Tensor, " num_cams height width 1"]
    alphas: Float[Tensor, " num_cams height width 1"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build a depth rasterization result from raw native outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, Tensor]:
        """Return the raw tensor tuple for stage composition."""
        return (self.depths, self.alphas)


@dataclass(frozen=True)
class RenderResult:
    """Output of the composed native NHT render stage."""

    renders: Float[Tensor, " num_cams height width channels"]
    alphas: Float[Tensor, " num_cams height width 1"]
    feature_square_sums: Float[Tensor, " num_cams height width feature_dim"]
    projection: ProjectionResult
    intersections: IntersectionResult
