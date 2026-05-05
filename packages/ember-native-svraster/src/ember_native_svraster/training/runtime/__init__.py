"""Public runtime API for native SVRaster training utilities."""

from __future__ import annotations

from jaxtyping import Float, Int
from torch import Tensor

from ember_native_svraster.training.runtime._extension import load_extension


def unbiased_adam_step(
    *,
    sparse: bool,
    parameter: Float[Tensor, " *shape"],
    gradient: Float[Tensor, " *shape"],
    exponential_average: Float[Tensor, " *shape"],
    exponential_average_squared: Float[Tensor, " *shape"],
    step: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> None:
    """Run one native SVRaster unbiased Adam update."""
    load_extension().unbiased_adam_step(
        bool(sparse),
        parameter,
        gradient,
        exponential_average,
        exponential_average_squared,
        float(step),
        float(learning_rate),
        float(beta1),
        float(beta2),
        float(epsilon),
    )


def biased_adam_step(
    *,
    sparse: bool,
    parameter: Float[Tensor, " *shape"],
    gradient: Float[Tensor, " *shape"],
    exponential_average: Float[Tensor, " *shape"],
    exponential_average_squared: Float[Tensor, " *shape"],
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> None:
    """Run one native SVRaster biased Adam update."""
    load_extension().biased_adam_step(
        bool(sparse),
        parameter,
        gradient,
        exponential_average,
        exponential_average_squared,
        float(learning_rate),
        float(beta1),
        float(beta2),
        float(epsilon),
    )


def apply_total_variation_density_grad(
    *,
    grid_points: Float[Tensor, " num_grid_points channels"],
    voxel_keys: Int[Tensor, " num_voxels 8"],
    weight: float,
    voxel_size_inverse: Float[Tensor, " num_voxels 1"],
    grid_points_grad: Float[Tensor, " num_grid_points channels"],
    no_tv_s: bool = True,
    tv_sparse: bool = False,
) -> None:
    """Accumulate the native SVRaster TV-density gradient in-place."""
    load_extension().total_variation_bw(
        grid_points,
        voxel_keys,
        float(weight),
        voxel_size_inverse,
        bool(no_tv_s),
        bool(tv_sparse),
        grid_points_grad,
    )


__all__ = [
    "apply_total_variation_density_grad",
    "biased_adam_step",
    "load_extension",
    "unbiased_adam_step",
]
