"""SVRaster backend adapter built on the SVRaster family core."""

from __future__ import annotations

from ember_core.core.contracts import SparseVoxelScene
from ember_core.core.registry import output_set, register_backend

from ember_native_svraster.core import (
    SVRasterCoreDepthRenderOutput,
    SVRasterCoreRenderOptions,
    SVRasterCoreRenderOutput,
    SVRasterCoreTrainingRenderOutput,
    render_svraster_core,
)

_SUPPORTED_OUTPUTS = output_set("depth", "normals")

SVRasterRenderOutput = SVRasterCoreRenderOutput
SVRasterDepthRenderOutput = SVRasterCoreDepthRenderOutput
SVRasterTrainingRenderOutput = SVRasterCoreTrainingRenderOutput
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
