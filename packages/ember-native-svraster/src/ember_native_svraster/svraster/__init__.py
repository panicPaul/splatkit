"""SVRaster backend package for Ember."""

from ember_native_svraster.svraster.renderer import (
    SVRasterDepthRenderOutput,
    SVRasterRenderOptions,
    SVRasterRenderOutput,
    register,
    render_svraster,
)

register()

__all__ = [
    "SVRasterDepthRenderOutput",
    "SVRasterRenderOptions",
    "SVRasterRenderOutput",
    "register",
    "render_svraster",
]
