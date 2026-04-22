"""Public runtime API for the GaussianPOP native backend."""

from splatkit_native_backends.gaussian_pop.runtime.blend import blend
from splatkit_native_backends.gaussian_pop.runtime.render import render
from splatkit_native_backends.gaussian_pop.runtime.types import (
    BlendResult,
    RenderResult,
)

__all__ = ["BlendResult", "RenderResult", "blend", "render"]
