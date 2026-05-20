"""Typed backend refs for ember-native-faster-gs."""

from __future__ import annotations

from typing import Final

from ember_core.core.backend_refs import BackendRef
from ember_core.core.contracts import GaussianScene3D
from ember_core.core.keys import BackendId

from ember_native_faster_gs.faster_gs.renderer import (
    FasterGSNativeRenderOptions,
    FasterGSNativeRenderOutput,
)
from ember_native_faster_gs.faster_gs_depth.renderer import (
    FasterGSDepthNativeRenderOptions,
    FasterGSDepthNativeRenderOutput,
)
from ember_native_faster_gs.fastgs.renderer import (
    FastGSNativeRenderOptions,
    FastGSNativeRenderOutput,
)
from ember_native_faster_gs.gaussian_pop.renderer import (
    GaussianPopNativeRenderOptions,
    GaussianPopNativeRenderOutput,
)
from ember_native_faster_gs.gaussian_wrapping.renderer import (
    GaussianWrappingNativeRenderOptions,
    GaussianWrappingNativeRenderOutput,
)

FASTER_GS_CORE: Final[
    BackendRef[
        GaussianScene3D,
        FasterGSNativeRenderOptions,
        FasterGSNativeRenderOutput,
    ]
] = BackendRef(
    id=BackendId("faster_gs", "core"),
    scene_type=GaussianScene3D,
    options_type=FasterGSNativeRenderOptions,
    output_type=FasterGSNativeRenderOutput,
)

FASTER_GS_FASTGS: Final[
    BackendRef[
        GaussianScene3D,
        FastGSNativeRenderOptions,
        FastGSNativeRenderOutput,
    ]
] = BackendRef(
    id=BackendId("faster_gs", "fastgs"),
    scene_type=GaussianScene3D,
    options_type=FastGSNativeRenderOptions,
    output_type=FastGSNativeRenderOutput,
)

FASTER_GS_DEPTH: Final[
    BackendRef[
        GaussianScene3D,
        FasterGSDepthNativeRenderOptions,
        FasterGSDepthNativeRenderOutput,
    ]
] = BackendRef(
    id=BackendId("faster_gs", "depth"),
    scene_type=GaussianScene3D,
    options_type=FasterGSDepthNativeRenderOptions,
    output_type=FasterGSDepthNativeRenderOutput,
)

FASTER_GS_GAUSSIAN_POP: Final[
    BackendRef[
        GaussianScene3D,
        GaussianPopNativeRenderOptions,
        GaussianPopNativeRenderOutput,
    ]
] = BackendRef(
    id=BackendId("faster_gs", "gaussian_pop"),
    scene_type=GaussianScene3D,
    options_type=GaussianPopNativeRenderOptions,
    output_type=GaussianPopNativeRenderOutput,
)

FASTER_GS_GAUSSIAN_WRAPPING: Final[
    BackendRef[
        GaussianScene3D,
        GaussianWrappingNativeRenderOptions,
        GaussianWrappingNativeRenderOutput,
    ]
] = BackendRef(
    id=BackendId("faster_gs", "gaussian_wrapping"),
    scene_type=GaussianScene3D,
    options_type=GaussianWrappingNativeRenderOptions,
    output_type=GaussianWrappingNativeRenderOutput,
)
