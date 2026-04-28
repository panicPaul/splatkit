from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from pydantic import BaseModel
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from marimo_3dv.viewer.controls import DesktopPydanticControls
from marimo_3dv.viewer.desktop import DesktopViewer
from marimo_3dv.viewer.widget import ViewerState


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_desktop_viewer_run_raises_render_errors(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    monkeypatch.setattr(qapp, "exec", lambda: 0)

    viewer = DesktopViewer(
        lambda camera_state: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        viewer.run()


def test_desktop_viewer_updates_camera_size_with_canvas_resize(
    qapp: QApplication,
) -> None:
    del qapp
    state = ViewerState(internal_render_max_side=None)
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
        width=320,
        height=240,
    )

    viewer._on_canvas_resize(640, 480)

    assert viewer.get_camera_state().width == 640
    assert viewer.get_camera_state().height == 480


def test_desktop_viewer_caps_framebuffer_size_with_internal_render_max_side(
    qapp: QApplication,
) -> None:
    del qapp
    state = ViewerState(internal_render_max_side=3840)
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
        width=1280,
        height=720,
    )

    viewer._on_canvas_resize(1280, 720)

    assert viewer.get_camera_state().width == 1280
    assert viewer.get_camera_state().height == 720


def test_desktop_viewer_uses_device_pixel_ratio_for_framebuffer_size(
    qapp: QApplication,
) -> None:
    del qapp
    state = ViewerState(internal_render_max_side=None)
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
        width=640,
        height=480,
    )

    viewer._on_canvas_resize(640, 480)
    viewer._canvas.devicePixelRatioF = lambda: 2.0  # type: ignore[method-assign]
    viewer._sync_camera_size_from_framebuffer()

    assert viewer.get_camera_state().width == 1280
    assert viewer.get_camera_state().height == 960


def test_desktop_viewer_caps_hidpi_framebuffer_size_with_internal_max_side(
    qapp: QApplication,
) -> None:
    del qapp
    state = ViewerState(internal_render_max_side=1000)
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
        width=640,
        height=480,
    )

    viewer._on_canvas_resize(640, 480)
    viewer._canvas.devicePixelRatioF = lambda: 2.0  # type: ignore[method-assign]
    viewer._sync_camera_size_from_framebuffer()

    assert viewer.get_camera_state().width == 1000
    assert viewer.get_camera_state().height == 750


def test_desktop_viewer_uses_interactive_max_side_while_dragging(
    qapp: QApplication,
) -> None:
    del qapp
    state = ViewerState(
        internal_render_max_side=3840,
        interactive_max_side=1200,
    )
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
        width=1280,
        height=720,
    )

    viewer._on_canvas_resize(1280, 720)
    viewer._canvas.devicePixelRatioF = lambda: 2.0  # type: ignore[method-assign]
    viewer._sync_camera_size_from_framebuffer()
    viewer._input.mode = "orbit"

    render_camera = viewer._render_camera_state()

    assert render_camera.width == 1200
    assert render_camera.height == 675


def test_desktop_viewer_uses_full_display_size_when_settled(
    qapp: QApplication,
) -> None:
    del qapp
    state = ViewerState(
        internal_render_max_side=3840,
        interactive_max_side=1200,
    )
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
        width=1280,
        height=720,
    )

    viewer._on_canvas_resize(1280, 720)
    viewer._canvas.devicePixelRatioF = lambda: 2.0  # type: ignore[method-assign]
    viewer._sync_camera_size_from_framebuffer()

    render_camera = viewer._render_camera_state()

    assert render_camera.width == 2560
    assert render_camera.height == 1440


def test_desktop_viewer_get_snapshot_returns_copy(
    qapp: QApplication,
) -> None:
    del qapp
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        )
    )
    viewer._latest_frame = np.full((4, 5, 3), 17, dtype=np.uint8)

    snapshot = viewer.get_snapshot()
    snapshot[0, 0, 0] = 99

    assert viewer._latest_frame[0, 0, 0] == 17


def test_desktop_viewer_orbit_drag_right_moves_camera_right(
    qapp: QApplication,
) -> None:
    del qapp
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        )
    )

    viewer._apply_orbit(10, 0)

    assert viewer.get_camera_state().position[0] > 0.0


def test_desktop_viewer_orbit_drag_up_moves_camera_up(
    qapp: QApplication,
) -> None:
    del qapp
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        )
    )

    viewer._apply_orbit(0, -10)

    assert viewer.get_camera_state().position[1] > 0.0


def test_desktop_viewer_orbit_inversion_flags_flip_drag_direction(
    qapp: QApplication,
) -> None:
    del qapp
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=ViewerState(orbit_invert_x=True, orbit_invert_y=True),
    )

    viewer._apply_orbit(10, -10)

    assert viewer.get_camera_state().position[0] < 0.0
    assert viewer.get_camera_state().position[1] < 0.0


def test_desktop_viewer_defaults_stats_on_without_explicit_state(
    qapp: QApplication,
) -> None:
    del qapp
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        )
    )

    assert viewer._state.show_stats is True


def test_desktop_controls_attach_creates_dock_with_tabs(
    qapp: QApplication,
) -> None:
    del qapp

    class _InnerConfig(BaseModel):
        threshold: float = 0.5

    class _OuterConfig(BaseModel):
        viewer: _InnerConfig = _InnerConfig()
        pipeline: _InnerConfig = _InnerConfig()

    controls = DesktopPydanticControls(_OuterConfig, label="Viewer Controls")
    window = QMainWindow()

    controls.attach(window)

    assert controls.dock_widget is not None
    assert controls.dock_widget.windowTitle() == "Viewer Controls"
    tab_widget = controls.dock_widget.findChild(QTabWidget)
    assert tab_widget is not None
    assert tab_widget.count() == 2


def test_desktop_viewer_expands_window_width_for_controls(
    qapp: QApplication,
) -> None:
    del qapp

    class _ControlsConfig(BaseModel):
        enabled: bool = True

    controls = DesktopPydanticControls(_ControlsConfig)
    viewer = DesktopViewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        controls=controls,
        width=640,
        height=480,
    )

    assert viewer._window.width() == 640 + controls.panel_width
