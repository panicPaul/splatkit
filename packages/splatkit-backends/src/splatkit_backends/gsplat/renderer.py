"""Gsplat backend registration surface."""

from splatkit.core.contracts import GaussianScene2D, GaussianScene3D
from splatkit.core.registry import register_backend

from splatkit_backends.gsplat.renderer_2d import render_gsplat_2dgs
from splatkit_backends.gsplat.renderer_3d import render_gsplat
from splatkit_backends.gsplat.shared import (
    SUPPORTED_OUTPUTS_2D,
    SUPPORTED_OUTPUTS_3D,
    GsplatAlphaIntersectionRenderOutput,
    GsplatAlphaProjectionRenderOutput,
    GsplatAlphaRenderOutput,
    GsplatIntersectionRenderOutput,
    GsplatProjectionRenderOutput,
    GsplatRenderOptions,
    GsplatRenderOutput,
)


def register() -> None:
    """Register the gsplat backends in the global splatkit registry."""
    register_backend(
        name="gsplat",
        default_options=GsplatRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=SUPPORTED_OUTPUTS_3D,
    )(render_gsplat)
    register_backend(
        name="gsplat_2dgs",
        default_options=GsplatRenderOptions(),
        accepted_scene_types=(GaussianScene2D,),
        supported_outputs=SUPPORTED_OUTPUTS_2D,
    )(render_gsplat_2dgs)


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
