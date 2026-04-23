"""Splatkit-native FasterGS Mojo backend."""

from splatkit_native_faster_gs_mojo.core.renderer import (
    FasterGSMojoRenderOptions,
    FasterGSMojoRenderOutput,
    register,
    render_faster_gs_mojo,
)

__all__ = [
    "FasterGSMojoRenderOptions",
    "FasterGSMojoRenderOutput",
    "register",
    "render_faster_gs_mojo",
]
