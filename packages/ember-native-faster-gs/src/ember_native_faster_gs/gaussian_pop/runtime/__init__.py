"""Public runtime API for the GaussianPOP native backend."""

from ember_native_faster_gs.gaussian_pop.runtime.blend import blend
from ember_native_faster_gs.gaussian_pop.runtime.render import render
from ember_native_faster_gs.gaussian_pop.runtime.types import (
    BlendResult,
    RenderResult,
)

__all__ = ["BlendResult", "RenderResult", "blend", "render"]
