"""Public package exports for splatkit."""

from importlib.metadata import PackageNotFoundError, version

from splatkit.core import (
    BACKEND_REGISTRY,
    CameraConvention,
    CameraParams,
    CameraState,
    GaussianScene,
    GaussianScene2D,
    GaussianScene3D,
    Has2DProjections,
    HasAlpha,
    HasDepth,
    OutputName,
    RegisteredBackend,
    RenderOptions,
    RenderOutput,
    Scene,
    SceneFamily,
    RenderWith2DProjections,
    RenderWithAlpha,
    RenderWithAlpha2DProjections,
    RenderWithAlphaDepth,
    RenderWithAlphaDepth2DProjections,
    RenderWithDepth,
    RenderWithDepth2DProjections,
    SparseVoxelScene,
    camera_params_to_intrinsics,
    intrinsics_to_camera_params,
    register_backend,
    render,
)
from splatkit.io import (
    load_gaussian_ply,
    load_scene,
    load_svraster_checkpoint,
    save_gaussian_ply,
    save_scene,
    save_svraster_checkpoint,
)

try:
    from splatkit._version import __version__
except ImportError:
    try:
        __version__ = version("splatkit")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "BACKEND_REGISTRY",
    "CameraConvention",
    "CameraParams",
    "CameraState",
    "GaussianScene",
    "GaussianScene2D",
    "GaussianScene3D",
    "Has2DProjections",
    "HasAlpha",
    "HasDepth",
    "OutputName",
    "RegisteredBackend",
    "RenderOptions",
    "RenderOutput",
    "Scene",
    "SceneFamily",
    "RenderWith2DProjections",
    "RenderWithAlpha",
    "RenderWithAlpha2DProjections",
    "RenderWithAlphaDepth",
    "RenderWithAlphaDepth2DProjections",
    "RenderWithDepth",
    "RenderWithDepth2DProjections",
    "SparseVoxelScene",
    "camera_params_to_intrinsics",
    "intrinsics_to_camera_params",
    "load_gaussian_ply",
    "load_scene",
    "load_svraster_checkpoint",
    "register_backend",
    "render",
    "save_gaussian_ply",
    "save_scene",
    "save_svraster_checkpoint",
    "__version__",
]
