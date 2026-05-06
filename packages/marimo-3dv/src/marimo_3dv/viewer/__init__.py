"""Viewer runtimes and shared viewer primitives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import marimo as mo

from marimo_3dv.viewer.defaults import (
    ViewerCameraConfig,
    ViewerControlsConfig,
    ViewerControlsHandle,
    ViewerInteractionConfig,
    ViewerNavigationConfig,
    ViewerOriginConfig,
    ViewerOverlayConfig,
    ViewerRenderConfig,
    ViewerRotationConfig,
    ViewerTransformConfig,
    apply_viewer_config,
    viewer_controls_config,
    viewer_controls_gui,
    viewer_controls_handle,
)
from marimo_3dv.viewer.link import ViewerStateLink, link_viewer_states
from marimo_3dv.viewer.widget import (
    CameraState,
    LinkedViewerStateField,
    MarimoViewer,
    ViewerClick,
    ViewerState,
    marimo_viewer,
)


class NoopViewer:
    """Non-rendering placeholder returned outside live marimo runtimes."""

    def __init__(self, *, state: ViewerState) -> None:
        self.state = state
        self._closed = False

    def close(self) -> None:
        """Mark the placeholder as closed."""
        self._closed = True

    def rerender(self, *, interactive: bool = False, wait: bool = False) -> None:
        """No-op render scheduling method matching ``MarimoViewer``."""
        del interactive, wait

    def set_camera_state(
        self, camera_state: CameraState, *, wait: bool = False
    ) -> None:
        """Update the placeholder state without rendering."""
        del wait
        self.state.set_camera(camera_state)

    def get_camera_state(self) -> CameraState:
        """Return the current placeholder camera state."""
        return self.state.camera_state

    def get_last_click(self) -> ViewerClick | None:
        """Return the last click stored on the placeholder state."""
        return self.state.last_click

    def anywidget(self) -> Any:
        """Raise because no browser widget exists in script mode."""
        raise RuntimeError("No marimo widget is available outside a notebook.")

    def get_snapshot(self) -> Any:
        """Raise because no frame is rendered in script mode."""
        raise RuntimeError("No rendered frame is available outside a notebook.")


def Viewer(
    render_fn: Callable[[CameraState], Any],
    *,
    state: ViewerState | None = None,
    title: str = "marimo-3dv viewer",
    controls: Any | None = None,
) -> MarimoViewer | NoopViewer:
    """Create the appropriate viewer backend for the current runtime.

    In notebook runtimes this returns a marimo-backed viewer widget. In script
    runtimes it returns a non-rendering placeholder so notebooks remain
    executable without opening a desktop window.
    """
    del title, controls
    viewer_state = state or ViewerState()
    if mo.running_in_notebook():
        return marimo_viewer(render_fn, state=viewer_state)
    return NoopViewer(state=viewer_state)


__all__ = [
    "CameraState",
    "LinkedViewerStateField",
    "MarimoViewer",
    "NoopViewer",
    "Viewer",
    "ViewerCameraConfig",
    "ViewerClick",
    "ViewerControlsConfig",
    "ViewerControlsHandle",
    "ViewerInteractionConfig",
    "ViewerNavigationConfig",
    "ViewerOriginConfig",
    "ViewerOverlayConfig",
    "ViewerRenderConfig",
    "ViewerRotationConfig",
    "ViewerState",
    "ViewerStateLink",
    "ViewerTransformConfig",
    "apply_viewer_config",
    "link_viewer_states",
    "marimo_viewer",
    "viewer_controls_config",
    "viewer_controls_gui",
    "viewer_controls_handle",
]
