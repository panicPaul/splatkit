"""Viser-backed marimo viewer helpers using Ember camera contracts."""

from importlib.metadata import PackageNotFoundError, version

try:
    from marimo_viser._version import __version__
except ImportError:
    try:
        __version__ = version("marimo-viser")
    except PackageNotFoundError:
        __version__ = "0.0.0"

from marimo_viser.viewer import (
    NoopViserViewer,
    ViserConnectionInfo,
    ViserControlsConfig,
    ViserControlsHandle,
    ViserRenderConfig,
    ViserRenderState,
    ViserServerConfig,
    ViserViewer,
    ViserViewerState,
    apply_viser_config,
    connection_info,
    viser_controls_gui,
    viser_controls_handle,
    viser_viewer,
)

__all__ = [
    "NoopViserViewer",
    "ViserConnectionInfo",
    "ViserControlsConfig",
    "ViserControlsHandle",
    "ViserRenderConfig",
    "ViserRenderState",
    "ViserServerConfig",
    "ViserViewer",
    "ViserViewerState",
    "__version__",
    "apply_viser_config",
    "connection_info",
    "viser_controls_gui",
    "viser_controls_handle",
    "viser_viewer",
]
