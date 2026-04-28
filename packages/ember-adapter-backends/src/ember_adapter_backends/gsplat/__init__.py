"""Gsplat backend package for Ember."""

from ember_adapter_backends.gsplat.renderer import (
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
