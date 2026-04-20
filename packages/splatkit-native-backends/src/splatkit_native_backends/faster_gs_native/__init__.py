"""Splatkit-native FasterGS backend."""

from splatkit_native_backends.faster_gs_native.renderer import (
    FasterGSNativeRenderOptions,
    FasterGSNativeRenderOutput,
    register,
    render_faster_gs_native,
)

__all__ = [
    "FasterGSNativeRenderOptions",
    "FasterGSNativeRenderOutput",
    "register",
    "render_faster_gs_native",
]
