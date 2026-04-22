"""SV Raster backend package for splatkit."""

from splatkit_adapter_backends.svraster.renderer import (
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
