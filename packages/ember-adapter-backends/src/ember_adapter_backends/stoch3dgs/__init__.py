"""Stoch3DGS backend package for Ember."""

from ember_adapter_backends.stoch3dgs.renderer import (
    Stoch3DGSAlphaRenderOutput,
    Stoch3DGSRenderOptions,
    Stoch3DGSRenderOutput,
    register,
    render_stoch3dgs,
)

__all__ = [
    "Stoch3DGSAlphaRenderOutput",
    "Stoch3DGSRenderOptions",
    "Stoch3DGSRenderOutput",
    "register",
    "render_stoch3dgs",
]
