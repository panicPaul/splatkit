"""FasterGS backend package for splatkit."""

from splatkit_backends.fastergs.renderer import (
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
