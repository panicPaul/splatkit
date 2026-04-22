"""Splatkit-native FasterGS depth proof backend."""

from splatkit_native_backends.faster_gs_depth_native.renderer import (
    FasterGSDepthNativeDepthRenderOutput,
    FasterGSDepthNativeRenderOptions,
    FasterGSDepthNativeRenderOutput,
    register,
    render_faster_gs_depth_native,
)

__all__ = [
    "FasterGSDepthNativeDepthRenderOutput",
    "FasterGSDepthNativeRenderOptions",
    "FasterGSDepthNativeRenderOutput",
    "register",
    "render_faster_gs_depth_native",
]
