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
    primitive_depths: Float[Tensor, " num_cams num_splats"]
    conics: Float[Tensor, " num_cams num_splats 3"]
    compensations: Float[Tensor, " num_cams num_splats"] | None

    @classmethod
    def from_tensors(cls, *tensors: Tensor | None) -> Self:
        """Build a projection result from raw native outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor | None, ...]:
        """Return the raw tensor tuple for stage composition."""
        return (
            self.radii,
            self.projected_means,
            self.primitive_depths,
            self.conics,
            self.compensations,
        )


@dataclass(frozen=True)
class IntersectionResult:
    """Output of the tile-intersection stage."""

    tiles_per_gaussian: Int[Tensor, " num_cams num_splats"]
    tile_intersection_ids: Int[Tensor, " num_intersections"]
    flattened_gaussian_ids: Int[Tensor, " num_intersections"]
    tile_offsets: Int[Tensor, " num_cams tile_height tile_width"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build an intersection result from raw native outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, ...]:
        """Return the raw tensor tuple for stage composition."""
        return (
            self.tiles_per_gaussian,
            self.tile_intersection_ids,
            self.flattened_gaussian_ids,
            self.tile_offsets,
        )


@dataclass(frozen=True)
class FeatureRasterizationResult:
    """Output of the NHT feature rasterization stage."""

    features: Float[Tensor, " num_cams height width feature_dim"]
    alphas: Float[Tensor, " num_cams height width 1"]

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build a feature rasterization result from raw native outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, Tensor]:
        """Return the raw tensor tuple for stage composition."""
        return (self.features, self.alphas)


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
    projection: ProjectionResult
    intersections: IntersectionResult
