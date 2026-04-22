"""FastGS backend package for splatkit."""

from splatkit_adapter_backends.fastgs.renderer import (
    FastGSRenderOptions,
    FastGSRenderOutput,
    register,
    render_fastgs,
)

__all__ = [
    "FastGSRenderOptions",
    "FastGSRenderOutput",
    "register",
    "render_fastgs",
]
