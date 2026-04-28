"""Gaussian splatting densification utilities."""

from ember_splatting_training.densification.mcmc import (
    GaussianMCMC,
    add_noise,
    relocation_adjustment,
)

__all__ = [
    "GaussianMCMC",
    "add_noise",
    "relocation_adjustment",
]
