"""Typed backend refs for ember-native-radfoam."""

from __future__ import annotations

from typing import Final

from ember_core.core.backend_refs import BackendRef
from ember_core.core.contracts import RadFoamScene
from ember_core.core.keys import BackendId

from ember_native_radfoam.radfoam.renderer import (
    RadFoamNativeRenderOptions,
    RadFoamNativeRenderOutput,
)

RADFOAM_CORE: Final[
    BackendRef[
        RadFoamScene,
        RadFoamNativeRenderOptions,
        RadFoamNativeRenderOutput,
    ]
] = BackendRef(
    id=BackendId("radfoam", "core"),
    scene_type=RadFoamScene,
    options_type=RadFoamNativeRenderOptions,
    output_type=RadFoamNativeRenderOutput,
)

__all__ = ["RADFOAM_CORE"]
