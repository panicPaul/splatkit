"""Optional SVRaster training utilities for Ember."""

from importlib.metadata import PackageNotFoundError, version

try:
    from ember_svraster_training._version import __version__
except ImportError:
    try:
        __version__ = version("ember-svraster-training")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "SVRasterOptimizationRecipe",
    "SVRasterSparseAdam",
    "SVRasterTVDensityHook",
    "apply_total_variation_density_grad",
    "svraster_optimization_config",
    "svraster_parameter_groups",
]


def __getattr__(name: str) -> object:
    """Load optional SVRaster training exports only when requested."""
    match name:
        case "SVRasterSparseAdam":
            from ember_svraster_training.optim import SVRasterSparseAdam

            return SVRasterSparseAdam
        case "SVRasterTVDensityHook" | "apply_total_variation_density_grad":
            from ember_svraster_training import regularization

            return getattr(regularization, name)
        case (
            "SVRasterOptimizationRecipe"
            | "svraster_optimization_config"
            | "svraster_parameter_groups"
        ):
            from ember_svraster_training import recipes

            return getattr(recipes, name)
        case _:
            raise AttributeError(name)
