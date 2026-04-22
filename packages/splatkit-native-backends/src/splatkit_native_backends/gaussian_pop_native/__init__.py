"""GaussianPOP native backend."""

from splatkit_native_backends.gaussian_pop_native.renderer import (
    GaussianPopNativeDepthGaussianImpactScoreRenderOutput,
    GaussianPopNativeDepthRenderOutput,
    GaussianPopNativeGaussianImpactScoreRenderOutput,
    GaussianPopNativeRenderOptions,
    GaussianPopNativeRenderOutput,
    register,
    render_gaussian_pop_native,
)

__all__ = [
    "GaussianPopNativeDepthGaussianImpactScoreRenderOutput",
    "GaussianPopNativeDepthRenderOutput",
    "GaussianPopNativeGaussianImpactScoreRenderOutput",
    "GaussianPopNativeRenderOptions",
    "GaussianPopNativeRenderOutput",
    "register",
    "render_gaussian_pop_native",
]
