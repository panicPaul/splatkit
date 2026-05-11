"""Typed backend refs for ember-native-svraster."""

from __future__ import annotations

from typing import Final

from ember_core.core.backend_refs import BackendRef
from ember_core.core.contracts import SparseVoxelScene
from ember_core.core.keys import BackendId

from ember_native_svraster.core import (
    SVRasterCoreRenderOptions,
    SVRasterCoreRenderOutput,
)

SVRASTER_CORE: Final[
    BackendRef[
        SparseVoxelScene,
        SVRasterCoreRenderOptions,
        SVRasterCoreRenderOutput,
    ]
] = BackendRef(
    id=BackendId("svraster", "core"),
    scene_type=SparseVoxelScene,
    options_type=SVRasterCoreRenderOptions,
    output_type=SVRasterCoreRenderOutput,
)
