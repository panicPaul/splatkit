"""Splatkit-native FasterGS backend."""

from splatkit_native_faster_gs.faster_gs.renderer import (
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
