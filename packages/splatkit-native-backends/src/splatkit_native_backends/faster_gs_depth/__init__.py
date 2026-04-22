"""Splatkit-native FasterGS depth proof backend."""

from splatkit_native_backends.faster_gs_depth.renderer import (
    FasterGSDepthNativeDepthRenderOutput,
    FasterGSDepthNativeRenderOptions,
    FasterGSDepthNativeRenderOutput,
    register,
    render_faster_gs_depth,
)

__all__ = [
    "FasterGSDepthNativeDepthRenderOutput",
    "FasterGSDepthNativeRenderOptions",
    "FasterGSDepthNativeRenderOutput",
    "register",
    "render_faster_gs_depth",
]
