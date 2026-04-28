"""Ember-native FasterGS Mojo backend."""

from ember_native_faster_gs_mojo.core.renderer import (
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
