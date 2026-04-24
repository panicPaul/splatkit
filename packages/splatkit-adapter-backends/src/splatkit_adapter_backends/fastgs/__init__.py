"""FastGS backend package for splatkit."""

from splatkit_adapter_backends.fastgs.renderer import (
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
