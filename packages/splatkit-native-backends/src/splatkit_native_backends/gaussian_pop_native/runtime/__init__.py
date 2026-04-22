"""Public runtime API for the GaussianPOP native backend."""

from splatkit_native_backends.gaussian_pop_native.runtime.blend import blend
from splatkit_native_backends.gaussian_pop_native.runtime.render import render
from splatkit_native_backends.gaussian_pop_native.runtime.types import (
    BlendResult,
    RenderResult,
)

__all__ = ["BlendResult", "RenderResult", "blend", "render"]
