"""FastGS backend package for Ember."""

from ember_adapter_backends.fastgs.renderer import (
    FastGSGaussianMetricAttribution,
    FastGSRenderOptions,
    FastGSRenderOutput,
    register,
    render_fastgs,
)

__all__ = [
    "FastGSGaussianMetricAttribution",
    "FastGSRenderOptions",
    "FastGSRenderOutput",
    "register",
    "render_fastgs",
]
