"""FasterGS backend package for Ember."""

from ember_adapter_backends.fastergs.renderer import (
    FasterGSDensificationRenderOutput,
    FasterGSRenderOptions,
    FasterGSRenderOutput,
    register,
    render_fastergs,
)

__all__ = [
    "FasterGSDensificationRenderOutput",
    "FasterGSRenderOptions",
    "FasterGSRenderOutput",
    "register",
    "render_fastergs",
]
