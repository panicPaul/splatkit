"""Gsplat backend package for splatkit."""

from splatkit_backends.gsplat.renderer import (
    GsplatAlphaIntersectionRenderOutput,
    GsplatAlphaProjectionRenderOutput,
    GsplatAlphaRenderOutput,
    GsplatIntersectionRenderOutput,
    GsplatProjectionRenderOutput,
    GsplatRenderOptions,
    GsplatRenderOutput,
    register,
    render_gsplat,
    render_gsplat_2dgs,
)

register()

__all__ = [
    "GsplatAlphaIntersectionRenderOutput",
    "GsplatAlphaProjectionRenderOutput",
    "GsplatAlphaRenderOutput",
    "GsplatIntersectionRenderOutput",
    "GsplatProjectionRenderOutput",
    "GsplatRenderOptions",
    "GsplatRenderOutput",
    "register",
    "render_gsplat",
    "render_gsplat_2dgs",
]
