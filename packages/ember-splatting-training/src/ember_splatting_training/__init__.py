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
    "Gaussian3DGSOptimizationRecipe",
    "GaussianMCMC",
    "GaussianMipSplattingAntialiasing",
    "GaussianMortonOrdering",
    "active_sh_bases_for_step",
    "add_noise",
    "dssim_loss",
    "fastergs_training_backend_options",
    "gaussian_3dgs_optimization_config",
    "gaussian_3dgs_parameter_groups",
    "relocation_adjustment",
    "rgb_l1_dssim_loss",
    "ssim_score",
]


def __getattr__(name: str) -> object:
    """Load optional FasterGS-backed exports only when requested."""
    if name == "FusedAdam":
        from ember_splatting_training.optim import FusedAdam

        return FusedAdam
    if name in {"GaussianMCMC", "add_noise", "relocation_adjustment"}:
        from ember_splatting_training import densification

        return getattr(densification, name)
    if name in {
        "GaussianMipSplattingAntialiasing",
        "GaussianMortonOrdering",
        "active_sh_bases_for_step",
        "fastergs_training_backend_options",
    }:
        from ember_splatting_training import fastergs

        return getattr(fastergs, name)
    if name in {"dssim_loss", "rgb_l1_dssim_loss", "ssim_score"}:
        from ember_splatting_training import losses

        return getattr(losses, name)
    if name in {
        "Gaussian3DGSOptimizationRecipe",
        "gaussian_3dgs_optimization_config",
        "gaussian_3dgs_parameter_groups",
    }:
        from ember_splatting_training import recipes

        return getattr(recipes, name)
    raise AttributeError(name)
