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
    "FastGSDensificationRecipe",
    "FastGSFinalPruneMode",
    "FusedAdam",
    "Gaussian3DGSOptimizationRecipe",
    "GaussianFastGS",
    "GaussianMCMC",
    "GaussianMipSplatting3DFilter",
    "GaussianMortonOrdering",
    "TrainingViewerConfig",
    "TrainingViewerHandle",
    "TrainingViewerHook",
    "TrainingViewerSnapshot",
    "TrainingViserViewerConfig",
    "active_sh_bases_for_step",
    "add_noise",
    "checkpoint_logs_dir",
    "create_training_viewer",
    "dssim_loss",
    "empty_scalar_frame",
    "fastergs_training_backend_options",
    "fastgs_l1_metric_map",
    "fastgs_normalize_score",
    "filter_scalars",
    "find_event_files",
    "gaussian_3dgs_optimization_config",
    "gaussian_3dgs_parameter_groups",
    "morton_codes",
    "morton_order",
    "read_scalar_records",
    "read_scalars",
    "relocation_adjustment",
    "rgb_l1_dssim_loss",
    "scalar_line_chart",
    "scalar_tags",
    "ssim_score",
]


def __getattr__(name: str) -> object:
    """Load optional FasterGS-backed exports only when requested."""
    match name:
        case "FusedAdam":
            from ember_splatting_training.optim import FusedAdam

            return FusedAdam
        case "GaussianMCMC" | "add_noise" | "relocation_adjustment":
            from ember_splatting_training import densification

            return getattr(densification, name)
        case (
            "FastGSFinalPruneMode"
            | "GaussianFastGS"
            | "GaussianMipSplatting3DFilter"
            | "GaussianMortonOrdering"
            | "active_sh_bases_for_step"
            | "fastgs_l1_metric_map"
            | "fastgs_normalize_score"
            | "fastergs_training_backend_options"
            | "morton_codes"
            | "morton_order"
        ):
            from ember_splatting_training import fastergs

            return getattr(fastergs, name)
        case "dssim_loss" | "rgb_l1_dssim_loss" | "ssim_score":
            from ember_splatting_training import losses

            return getattr(losses, name)
        case (
            "TrainingViewerConfig"
            | "TrainingViewerHandle"
            | "TrainingViewerHook"
            | "TrainingViewerSnapshot"
            | "TrainingViserViewerConfig"
            | "create_training_viewer"
        ):
            from ember_splatting_training import training_viewer

            return getattr(training_viewer, name)
        case (
            "checkpoint_logs_dir"
            | "empty_scalar_frame"
            | "filter_scalars"
            | "find_event_files"
            | "read_scalar_records"
            | "read_scalars"
            | "scalar_line_chart"
            | "scalar_tags"
        ):
            from ember_splatting_training import tensorboard_analysis

            return getattr(tensorboard_analysis, name)
        case "FastGSDensificationRecipe":
            from ember_splatting_training import typed_recipes

            return getattr(typed_recipes, name)
        case (
            "Gaussian3DGSOptimizationRecipe"
            | "gaussian_3dgs_optimization_config"
            | "gaussian_3dgs_parameter_groups"
        ):
            from ember_splatting_training import recipes

            return getattr(recipes, name)
        case _:
            raise AttributeError(name)
