"""FasterGS backend package for Ember."""

from ember_adapter_backends.fastergs.renderer import (
    FasterGSRenderOptions,
    FasterGSRenderOutput,
    register,
    render_fastergs,
)

__all__ = [
    "FasterGSRenderOptions",
    "FasterGSRenderOutput",
    "register",
    "render_fastergs",
]
