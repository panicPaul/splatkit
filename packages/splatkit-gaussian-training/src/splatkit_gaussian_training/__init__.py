"""Optional Gaussian-specific training utilities for splatkit."""

from importlib.metadata import PackageNotFoundError, version

from splatkit_gaussian_training.densification import (
    GaussianMCMC,
    add_noise,
    relocation_adjustment,
)
from splatkit_gaussian_training.optim import FusedAdam

try:
    from splatkit_gaussian_training._version import __version__
except ImportError:
    try:
        __version__ = version("splatkit-gaussian-training")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "FusedAdam",
    "GaussianMCMC",
    "add_noise",
    "relocation_adjustment",
]
