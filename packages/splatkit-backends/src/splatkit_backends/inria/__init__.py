"""Inria backend package for splatkit."""

from .renderer import (
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
