"""Inria backend package for splatkit."""

from diff_gaussian_rasterization.wrapper import (
    InriaDepthRenderOutput,
    InriaRenderOptions,
    InriaRenderOutput,
    register,
    render_inria,
)

__all__ = [
    "InriaDepthRenderOutput",
    "InriaRenderOptions",
    "InriaRenderOutput",
    "register",
    "render_inria",
]
