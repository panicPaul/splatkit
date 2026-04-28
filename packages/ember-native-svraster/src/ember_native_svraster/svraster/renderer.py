"""SVRaster backend adapter built on the SVRaster family core."""

from __future__ import annotations

from ember_core.core.contracts import SparseVoxelScene
from ember_core.core.registry import register_backend

from ember_native_svraster.core import (
    SVRasterCoreDepthRenderOutput,
    SVRasterCoreRenderOptions,
    SVRasterCoreRenderOutput,
    render_svraster_core,
)

_SUPPORTED_OUTPUTS = frozenset({"depth"})

SVRasterRenderOutput = SVRasterCoreRenderOutput
SVRasterDepthRenderOutput = SVRasterCoreDepthRenderOutput
SVRasterRenderOptions = SVRasterCoreRenderOptions
render_svraster = render_svraster_core


def register() -> None:
    """Register the SVRaster backend in the global ember-core registry."""
    register_backend(
        name="svraster.core",
        default_options=SVRasterRenderOptions(),
        accepted_scene_types=(SparseVoxelScene,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_svraster)
