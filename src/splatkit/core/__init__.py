"""Public package exports for splatkit.core."""

from splatkit.core.capabilities import (
    Has2DProjections,
    HasAlpha,
    HasDepth,
    RenderWith2DProjections,
    RenderWithAlpha,
    RenderWithAlpha2DProjections,
    RenderWithAlphaDepth,
    RenderWithAlphaDepth2DProjections,
    RenderWithDepth,
    RenderWithDepth2DProjections,
)
from splatkit.core.contracts import (
    CameraConvention,
    CameraParams,
    CameraState,
    GaussianScene,
    OutputName,
    RenderOptions,
    RenderOutput,
    camera_params_to_intrinsics,
    intrinsics_to_camera_params,
)
from splatkit.core.registry import (
    BACKEND_REGISTRY,
    RegisteredBackend,
    register_backend,
    render,
)

__all__ = [
    "BACKEND_REGISTRY",
    "CameraConvention",
    "CameraParams",
    "CameraState",
    "GaussianScene",
    "Has2DProjections",
    "HasAlpha",
    "HasDepth",
    "OutputName",
    "RegisteredBackend",
    "RenderOptions",
    "RenderOutput",
    "RenderWith2DProjections",
    "RenderWithAlpha",
    "RenderWithAlpha2DProjections",
    "RenderWithAlphaDepth",
    "RenderWithAlphaDepth2DProjections",
    "RenderWithDepth",
    "RenderWithDepth2DProjections",
    "camera_params_to_intrinsics",
    "intrinsics_to_camera_params",
    "register_backend",
    "render",
]
