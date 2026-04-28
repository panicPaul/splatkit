"""Inria backend package for Ember."""

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
