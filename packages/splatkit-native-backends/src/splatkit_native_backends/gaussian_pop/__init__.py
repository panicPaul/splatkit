"""GaussianPOP native backend."""

from splatkit_native_backends.gaussian_pop.renderer import (
    GaussianPopNativeDepthGaussianImpactScoreRenderOutput,
    GaussianPopNativeDepthRenderOutput,
    GaussianPopNativeGaussianImpactScoreRenderOutput,
    GaussianPopNativeRenderOptions,
    GaussianPopNativeRenderOutput,
    register,
    render_gaussian_pop,
)

__all__ = [
    "GaussianPopNativeDepthGaussianImpactScoreRenderOutput",
    "GaussianPopNativeDepthRenderOutput",
    "GaussianPopNativeGaussianImpactScoreRenderOutput",
    "GaussianPopNativeRenderOptions",
    "GaussianPopNativeRenderOutput",
    "register",
    "render_gaussian_pop",
]
