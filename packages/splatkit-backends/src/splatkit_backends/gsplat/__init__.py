"""Gsplat backend package for splatkit."""

from splatkit_backends.gsplat.renderer import (
    GsplatAlphaProjectionRenderOutput,
    GsplatAlphaRenderOutput,
    GsplatProjectionRenderOutput,
    GsplatRenderOptions,
    GsplatRenderOutput,
    register,
    render_gsplat,
)

register()

__all__ = [
    "GsplatAlphaProjectionRenderOutput",
    "GsplatAlphaRenderOutput",
    "GsplatProjectionRenderOutput",
    "GsplatRenderOptions",
    "GsplatRenderOutput",
    "register",
    "render_gsplat",
]
