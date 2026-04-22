"""Splatkit-native Stoch3DGS backend."""

from splatkit_native_3dgrt.stoch3dgs.renderer import (
    Stoch3DGSNativeRenderOptions,
    Stoch3DGSNativeRenderOutput,
    register,
    render_stoch3dgs_native,
)

__all__ = [
    "Stoch3DGSNativeRenderOptions",
    "Stoch3DGSNativeRenderOutput",
    "register",
    "render_stoch3dgs_native",
]
