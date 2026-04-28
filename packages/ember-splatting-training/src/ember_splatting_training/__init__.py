"""Optional splatting training utilities for Ember."""

from importlib.metadata import PackageNotFoundError, version

try:
    from ember_splatting_training._version import __version__
except ImportError:
    try:
        __version__ = version("ember-splatting-training")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "FusedAdam",
    "GaussianMCMC",
    "add_noise",
    "relocation_adjustment",
]


def __getattr__(name: str) -> object:
    """Load optional FasterGS-backed exports only when requested."""
    if name == "FusedAdam":
        from ember_splatting_training.optim import FusedAdam

        return FusedAdam
    if name in {"GaussianMCMC", "add_noise", "relocation_adjustment"}:
        from ember_splatting_training import densification

        return getattr(densification, name)
    raise AttributeError(name)
