"""Typed stage outputs for the SVRaster native runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from jaxtyping import Float, Int
from torch import Tensor


@dataclass(frozen=True)
class PreprocessResult:
    """Output of the preprocess stage."""

    n_duplicates: Int[Tensor, " num_voxels"]
    geom_buffer: Tensor

    @classmethod
    def from_tensors(cls, *tensors: Tensor) -> Self:
        """Build a preprocess result from the raw op outputs."""
        return cls(*tensors)

    def as_tensors(self) -> tuple[Tensor, ...]:
        """Return the raw tensor tuple for stage composition."""
        return (self.n_duplicates, self.geom_buffer)


@dataclass(frozen=True)
class RasterizeResult:
    """Public output of the rasterize stage."""

    color: Float[Tensor, " 3 height width"]
    depth: Tensor
    normal: Tensor
    transmittance: Tensor
    max_weight: Tensor


@dataclass(frozen=True)
class ParsedRasterizeOutputs:
    """Internal parsed view over rasterize op outputs."""

    num_rendered: Int[Tensor, " 1"]
    binning_buffer: Tensor
    image_buffer: Tensor
    result: RasterizeResult

