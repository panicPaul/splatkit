"""Gaussian splatting densification utilities."""

from ember_splatting_training.densification.mcmc import (
    GaussianMCMC,
    add_noise,
    relocation_adjustment,
)
from ember_splatting_training.fastergs import (
    GaussianMipSplatting3DFilter,
    GaussianMortonOrdering,
)

__all__ = [
    "GaussianMCMC",
    "GaussianMipSplatting3DFilter",
    "GaussianMortonOrdering",
    "add_noise",
    "relocation_adjustment",
]
