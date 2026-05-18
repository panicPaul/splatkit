"""Typed backend refs for ember-native-powerfoam."""

from __future__ import annotations

from typing import Final

from ember_core import BackendId, BackendRef
from ember_core.core.contracts import PowerFoamScene

from ember_native_powerfoam.powerfoam.renderer import (
    PowerFoamNativeRenderOptions,
    PowerFoamNativeRenderOutput,
)

POWERFOAM_RASTERIZE: Final[
    BackendRef[
        PowerFoamScene,
        PowerFoamNativeRenderOptions,
        PowerFoamNativeRenderOutput,
    ]
] = BackendRef(
    id=BackendId("powerfoam", "rasterize"),
    scene_type=PowerFoamScene,
    options_type=PowerFoamNativeRenderOptions,
    output_type=PowerFoamNativeRenderOutput,
)

__all__ = ["POWERFOAM_RASTERIZE"]

