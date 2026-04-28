"""Desktop offline viewer backed by Qt with docked controls support."""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from marimo_3dv.viewer.controls import DesktopPydanticControls
from marimo_3dv.viewer.widget import (
    CameraState,
    ViewerClick,
    ViewerState,
    _look_at_cam_to_world,
    _normalize,
    _normalize_frame,
)

_ORBIT_SENSITIVITY = 0.008
_SCROLL_ZOOM_SENSITIVITY = 0.0015
_MIN_ORBIT_DISTANCE = 0.05
_MAX_ORBIT_DISTANCE = 1e5
_CLICK_THRESHOLD_PIXELS = 4.0


@dataclass
class _InputState:
    """Mutable per-frame input state."""

    mode: str | None = None
    keys_held: set[int] = field(default_factory=set)
    drag_start: tuple[int, int] | None = None
    drag_exceeded_click_threshold: bool = False


class _ViewerCanvas(QWidget):
    """Central Qt widget that displays frames and forwards interactions."""

    def __init__(
        self, viewer: DesktopViewer, *, width: int, height: int
    ) -> None:
        super().__init__()
        self._viewer = viewer
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.resize(width, height)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        self._viewer._paint_canvas(self)

    def resizeEvent(self, event: QResizeEvent) -> None:
        size = event.size()
        width, height = size.width(), size.height()
        self._viewer._on_canvas_resize(width, height)
        super().resizeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._viewer._on_mouse_press(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._viewer._on_mouse_release(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._viewer._on_mouse_move(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._viewer._on_wheel(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        self._viewer._on_key_press(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        self._viewer._on_key_release(event)


class _ViewerMainWindow(QMainWindow):
    """Main window that hosts the canvas and optional controls dock."""

    def __init__(self, viewer: DesktopViewer, *, title: str) -> None:
        super().__init__()
        self._viewer = viewer
        self.setWindowTitle(title)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._viewer._running = False
        super().closeEvent(event)


class DesktopViewer:
    """Blocking desktop viewer window backed by Qt."""

    def __init__(
        self,
        render_fn: Callable[[CameraState], Any],
        *,
        state: ViewerState | None = None,
        controls: DesktopPydanticControls[Any] | None = None,
        width: int = 1280,
        height: int = 720,
        title: str = "marimo-3dv desktop viewer",
    ) -> None:
        self._render_fn = render_fn
        if state is None:
            self._state = ViewerState(show_stats=True)
        else:
            self._state = state
        self._controls = controls
        self._logical_window_size = (width, height)

        camera = self._state.camera_state
        if camera.width != width or camera.height != height:
            self._state.camera_state = camera.with_size(width, height)
        self._sync_camera_tracking(self._state.camera_state)

        self._latest_frame: np.ndarray | None = None
        self._render_error: Exception | None = None
        self._render_error_traceback: str | None = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._input = _InputState()
        self._last_render_ms = 0.0
        self._last_viewer_fps = 0.0
        self._draw_frame_times: list[float] = []
        self._last_render_fps = 0.0
        self._render_frame_times: list[float] = []
        self._last_render_size: tuple[int, int] = (
            self._state.camera_state.width,
            self._state.camera_state.height,
        )
        self._last_mouse_position: tuple[int, int] | None = None

        self._state._reset_camera_callback = self._on_camera_set
        self._app = _qt_application()
        self._window = _ViewerMainWindow(self, title=title)
        self._canvas = _ViewerCanvas(self, width=width, height=height)
        self._window.setCentralWidget(self._canvas)
        total_width = width
        if self._controls is not None:
            total_width += self._controls.panel_width
        self._window.resize(total_width, height)
        if self._controls is not None:
            self._controls.attach(self._window)

        self._tick_timer = QTimer()
        self._tick_timer.timeout.connect(self._on_tick)
        self._last_tick_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _paint_canvas(self, canvas: QWidget) -> None:
        now = time.perf_counter()
        self._draw_frame_times.append(now)
        cutoff = now - 1.0
        self._draw_frame_times = [
            timestamp
            for timestamp in self._draw_frame_times
            if timestamp > cutoff
        ]
        self._last_viewer_fps = float(len(self._draw_frame_times))

        painter = QPainter(canvas)
        painter.fillRect(canvas.rect(), Qt.GlobalColor.black)
        with self._frame_lock:
            frame = (
                None
                if self._latest_frame is None
                else self._latest_frame.copy()
            )
        if frame is not None:
            pixmap = _pixmap_from_frame(frame)
            painter.drawPixmap(canvas.rect(), pixmap)
        if self._state.show_stats:
            self._draw_stats_overlay(painter, canvas.rect())
        painter.end()

    def _draw_stats_overlay(
        self, painter: QPainter, canvas_rect: QRect
    ) -> None:
        stats_rect = QRect(16, 16, 292, 126)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_qt_color(248, 251, 255, 236))
        painter.drawRoundedRect(stats_rect, 10, 10)
        painter.setPen(QPen(_qt_color(15, 23, 42, 255)))
        cam = self._state.camera_state
        framebuffer_width, framebuffer_height = self._get_framebuffer_size()
        stats_text = (
            f"Viewer {self._last_viewer_fps:.0f}fps\n"
            f"Render {self._last_render_ms:.0f}ms {self._last_render_fps:.0f}fps\n"
            f"Window {canvas_rect.width()}x{canvas_rect.height()}\n"
            f"Display {framebuffer_width}x{framebuffer_height}\n"
            f"Render {self._last_render_size[0]}x{self._last_render_size[1]}"
        )
        painter.drawText(
            stats_rect.adjusted(14, 14, -14, -14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            stats_text,
        )

    # ------------------------------------------------------------------
    # Camera math
    # ------------------------------------------------------------------

    def _get_window_size(self) -> tuple[int, int]:
        """Return the live canvas size used for interaction coordinates."""
        return self._logical_window_size

    def _get_framebuffer_size(self) -> tuple[int, int]:
        """Return the drawable canvas size in device pixels."""
        logical_width, logical_height = self._logical_window_size
        device_pixel_ratio = self._canvas.devicePixelRatioF()
        return (
            max(1, round(logical_width * device_pixel_ratio)),
            max(1, round(logical_height * device_pixel_ratio)),
        )

    def _sync_camera_size_from_framebuffer(self) -> None:
        """Match render resolution to the drawable framebuffer size."""
        framebuffer_width, framebuffer_height = self._get_framebuffer_size()
        max_side = self._state.internal_render_max_side
        target_width = framebuffer_width
        target_height = framebuffer_height
        if max_side is not None:
            larger_axis = max(framebuffer_width, framebuffer_height)
            if larger_axis > max_side:
                scale = max_side / larger_axis
                target_width = max(1, round(framebuffer_width * scale))
                target_height = max(1, round(framebuffer_height * scale))
        camera = self._state.camera_state
        if camera.width != target_width or camera.height != target_height:
            self._state.camera_state = camera.with_size(
                target_width,
                target_height,
            )

    def _camera_state_with_max_side(
        self, camera_state: CameraState, max_side: int | None
    ) -> CameraState:
        """Return a camera state with its larger axis capped to ``max_side``."""
        if max_side is None:
            return camera_state
        larger_axis = max(camera_state.width, camera_state.height)
        if larger_axis <= max_side:
            return camera_state
        scale = max_side / larger_axis
        target_width = max(1, round(camera_state.width * scale))
        target_height = max(1, round(camera_state.height * scale))
        return camera_state.with_size(target_width, target_height)

    def _is_interacting(self) -> bool:
        """Return whether the user is actively manipulating the view."""
        return self._input.mode is not None or bool(self._input.keys_held)

    def _render_camera_state(self) -> CameraState:
        """Return the effective render camera state for the current mode."""
        camera_state = self._camera_state_with_max_side(
            self._state.camera_state,
            self._state.internal_render_max_side,
        )
        if self._is_interacting():
            camera_state = self._camera_state_with_max_side(
                camera_state,
                self._state.interactive_max_side,
            )
        return camera_state

    def _camera_axes(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (position, right, up, forward) from current camera."""
        c2w = self._state.camera_state.cam_to_world
        position = c2w[:3, 3].copy()
        right = c2w[:3, 0].copy()
        up = c2w[:3, 1].copy()
        forward = c2w[:3, 2].copy()
        return position, right, up, forward

    def _viewer_frame_rotation(self) -> np.ndarray:
        """Return the viewer-frame rotation derived from state rotation sliders."""
        return _rotation_matrix_xyz(
            self._state.viewer_rotation_x_degrees,
            self._state.viewer_rotation_y_degrees,
            self._state.viewer_rotation_z_degrees,
        )

    def _viewer_up_vector(self) -> np.ndarray:
        """Return the desktop viewer up vector matching the notebook viewer."""
        return _normalize(
            self._viewer_frame_rotation() @ np.array([0.0, -1.0, 0.0])
        )

    def _sync_camera_tracking(self, camera_state: CameraState) -> None:
        """Update tracked position/target/orbit distance from a camera state."""
        self._position = camera_state.cam_to_world[:3, 3].copy()
        forward = camera_state.cam_to_world[:3, 2].copy()
        fallback_distance = max(
            _MIN_ORBIT_DISTANCE, float(np.linalg.norm(self._position))
        )
        if not hasattr(self, "_orbit_distance"):
            self._orbit_distance = max(3.0, fallback_distance)
        self._target = self._position + forward * self._orbit_distance
        self._orbit_distance = max(
            _MIN_ORBIT_DISTANCE,
            min(
                _MAX_ORBIT_DISTANCE,
                float(np.linalg.norm(self._target - self._position)),
            ),
        )

    def _set_camera_pose(
        self, position: np.ndarray, target: np.ndarray
    ) -> None:
        """Rebuild cam_to_world from tracked position/target and viewer up."""
        cam_to_world = _look_at_cam_to_world(
            position,
            target,
            self._viewer_up_vector(),
        )
        cam = self._state.camera_state
        next_camera = CameraState(
            fov_degrees=cam.fov_degrees,
            width=cam.width,
            height=cam.height,
            cam_to_world=cam_to_world,
            camera_convention=cam.camera_convention,
        )
        self._position = position.copy()
        self._target = target.copy()
        self._orbit_distance = max(
            _MIN_ORBIT_DISTANCE,
            min(
                _MAX_ORBIT_DISTANCE,
                float(np.linalg.norm(self._target - self._position)),
            ),
        )
        self._state.camera_state = next_camera

    def _apply_orbit(self, dx: int, dy: int) -> None:
        """Orbit around the tracked target using JS-equivalent math."""
        viewer_up = self._viewer_up_vector()
        offset = self._position - self._target
        orbit_dx = dx if self._state.orbit_invert_x else -dx
        orbit_dy = dy if self._state.orbit_invert_y else -dy
        yaw_rotation = _rot_axis(viewer_up, orbit_dx * _ORBIT_SENSITIVITY)
        yawed_offset = yaw_rotation @ offset
        yawed_forward = _normalize(-yawed_offset)
        pitch_axis = np.cross(yawed_forward, viewer_up)
        if np.linalg.norm(pitch_axis) <= 1e-8:
            return
        pitch_axis = _normalize(pitch_axis)
        pitch_rotation = _rot_axis(
            pitch_axis,
            orbit_dy * _ORBIT_SENSITIVITY,
        )
        orbited_offset = pitch_rotation @ yawed_offset
        new_position = self._target + orbited_offset
        self._set_camera_pose(new_position, self._target)

    def _apply_pan(self, dx: int, dy: int) -> None:
        """Pan in the image plane using orbit-distance/FOV scaling."""
        _position, right, up, _forward = self._camera_axes()
        cam = self._state.camera_state
        _window_width, window_height = self._get_window_size()
        pan_dx = -dx if self._state.pan_invert_x else dx
        pan_dy = -dy if self._state.pan_invert_y else dy
        fov_radians = np.deg2rad(cam.fov_degrees)
        pan_scale = (
            max(_MIN_ORBIT_DISTANCE, self._orbit_distance)
            * np.tan(fov_radians / 2.0)
            / max(1, window_height)
            * 2.0
        )
        delta = -pan_dx * pan_scale * right + pan_dy * pan_scale * up
        self._set_camera_pose(self._position + delta, self._target + delta)

    def _apply_dolly(self, scroll_y: float) -> None:
        """Zoom by scaling the orbit distance exponentially around the target."""
        zoom_factor = float(np.exp(-scroll_y * _SCROLL_ZOOM_SENSITIVITY))
        offset = self._position - self._target
        direction = _normalize(offset)
        self._orbit_distance = max(
            _MIN_ORBIT_DISTANCE,
            min(_MAX_ORBIT_DISTANCE, self._orbit_distance * zoom_factor),
        )
        self._set_camera_pose(
            self._target + direction * self._orbit_distance,
            self._target,
        )

    def _apply_move(self, dt: float) -> None:
        """Apply WASD/QE keyboard movement each tick."""
        keys = self._input.keys_held
        if not keys:
            return
        position, right, up, forward = self._camera_axes()
        speed = self._state.keyboard_move_speed * dt * 60.0
        if Qt.Key.Key_Shift in keys:
            speed *= self._state.keyboard_sprint_multiplier
        delta = np.zeros(3)
        if Qt.Key.Key_W in keys:
            delta += forward * speed
        if Qt.Key.Key_S in keys:
            delta -= forward * speed
        if Qt.Key.Key_A in keys:
            delta -= right * speed
        if Qt.Key.Key_D in keys:
            delta += right * speed
        if Qt.Key.Key_Q in keys:
            delta -= up * speed
        if Qt.Key.Key_E in keys:
            delta += up * speed
        if np.linalg.norm(delta) > 0:
            self._set_camera_pose(position + delta, self._target + delta)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _on_canvas_resize(self, width: int, height: int) -> None:
        self._logical_window_size = (max(1, width), max(1, height))
        self._sync_camera_size_from_framebuffer()

    def _on_mouse_press(self, event: QMouseEvent) -> None:
        position = event.position().toPoint()
        x, y = position.x(), position.y()
        if event.button() == Qt.MouseButton.LeftButton:
            self._input.mode = "orbit"
            self._input.drag_start = (x, y)
            self._input.drag_exceeded_click_threshold = False
        elif event.button() == Qt.MouseButton.RightButton:
            self._input.mode = "pan"
            self._input.drag_start = (x, y)
            self._input.drag_exceeded_click_threshold = False
        self._last_mouse_position = (x, y)

    def _on_mouse_release(self, event: QMouseEvent) -> None:
        position = event.position().toPoint()
        x, y = position.x(), position.y()
        if event.button() == Qt.MouseButton.LeftButton:
            should_emit_click = (
                self._input.drag_start is not None
                and not self._input.drag_exceeded_click_threshold
            )
            if should_emit_click:
                win_w, win_h = self._get_window_size()
                framebuffer_w, framebuffer_h = self._get_framebuffer_size()
                scale_x = framebuffer_w / max(1, win_w)
                scale_y = framebuffer_h / max(1, win_h)
                self._state.last_click = ViewerClick(
                    x=round(x * scale_x),
                    y=round((win_h - 1 - y) * scale_y),
                    width=framebuffer_w,
                    height=framebuffer_h,
                    camera_state=self._state.camera_state,
                )
        if event.button() in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
        }:
            self._input.mode = None
            self._input.drag_start = None
            self._input.drag_exceeded_click_threshold = False
        self._last_mouse_position = (x, y)

    def _on_mouse_move(self, event: QMouseEvent) -> None:
        position = event.position().toPoint()
        x, y = position.x(), position.y()
        if self._last_mouse_position is None:
            self._last_mouse_position = (x, y)
            return
        last_x, last_y = self._last_mouse_position
        dx = x - last_x
        dy = y - last_y
        self._last_mouse_position = (x, y)
        if self._input.drag_start is not None:
            drag_distance = float(
                np.hypot(
                    x - self._input.drag_start[0],
                    y - self._input.drag_start[1],
                )
            )
            if drag_distance > _CLICK_THRESHOLD_PIXELS:
                self._input.drag_exceeded_click_threshold = True
        if self._input.mode == "orbit":
            self._apply_orbit(dx, dy)
        elif self._input.mode == "pan":
            self._apply_pan(dx, -dy)

    def _on_wheel(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y() / 120.0
        self._apply_dolly(delta)

    def _on_key_press(self, event: QKeyEvent) -> None:
        self._input.keys_held.add(event.key())
        if event.key() == Qt.Key.Key_R:
            self._state.reset_camera()
        elif event.key() == Qt.Key.Key_Escape:
            self._running = False
            self._window.close()

    def _on_key_release(self, event: QKeyEvent) -> None:
        self._input.keys_held.discard(event.key())

    # ------------------------------------------------------------------
    # Viewer state callbacks
    # ------------------------------------------------------------------

    def _on_camera_set(self, camera_state: CameraState) -> None:
        self._state.camera_state = camera_state
        self._sync_camera_tracking(camera_state)

    # ------------------------------------------------------------------
    # Render + tick loops
    # ------------------------------------------------------------------

    def _render_once(self, camera_state: CameraState) -> np.ndarray:
        """Render and normalize a single frame, surfacing backend failures."""
        raw = self._render_fn(camera_state)
        return _normalize_frame(raw)

    def _render_loop(self) -> None:
        """Background thread: render frames continuously."""
        while self._running:
            try:
                start = time.perf_counter()
                camera_state = self._render_camera_state()
                frame = self._render_once(camera_state)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                now = time.perf_counter()
                with self._frame_lock:
                    self._latest_frame = frame
                    self._last_render_ms = elapsed_ms
                    self._last_render_size = (
                        camera_state.width,
                        camera_state.height,
                    )
                    self._render_frame_times.append(now)
                    cutoff = now - 1.0
                    self._render_frame_times = [
                        timestamp
                        for timestamp in self._render_frame_times
                        if timestamp > cutoff
                    ]
                    self._last_render_fps = float(len(self._render_frame_times))
            except Exception as exception:
                self._render_error = exception
                self._render_error_traceback = "".join(
                    traceback.format_exception(
                        type(exception),
                        exception,
                        exception.__traceback__,
                    )
                )
                self._running = False
                QTimer.singleShot(0, self._app.quit)
                return

    def _on_tick(self) -> None:
        """Main-thread tick: apply held-key movement and repaint."""
        now = time.perf_counter()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        self._apply_move(dt)
        self._canvas.update()
        if self._render_error is not None:
            self._app.quit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the render loop and show the window. Blocks until closed."""
        self._window.show()
        self._app.processEvents()
        self._logical_window_size = (
            max(1, self._canvas.width()),
            max(1, self._canvas.height()),
        )
        self._sync_camera_size_from_framebuffer()
        initial_camera_state = self._state.camera_state
        initial_render_camera_state = self._render_camera_state()
        initial_frame = self._render_once(initial_render_camera_state)
        with self._frame_lock:
            self._latest_frame = initial_frame
            self._last_render_ms = 0.0
            self._last_render_size = (
                initial_render_camera_state.width,
                initial_render_camera_state.height,
            )

        self._running = True
        self._last_tick_time = time.perf_counter()
        render_thread = threading.Thread(target=self._render_loop, daemon=True)
        render_thread.start()
        self._tick_timer.start(16)
        self._app.exec()
        self._tick_timer.stop()
        self._running = False
        self._window.hide()
        render_thread.join(timeout=2.0)
        if self._controls is not None:
            self._controls.shutdown()
        if self._render_error is not None:
            raise RuntimeError(
                "Desktop viewer render loop failed.\n"
                f"{self._render_error_traceback}"
            ) from self._render_error

    def get_camera_state(self) -> CameraState:
        """Return the current desktop camera state."""
        return self._state.camera_state

    def get_last_click(self) -> ViewerClick | None:
        """Return the last click captured by the desktop backend."""
        return self._state.last_click

    def get_snapshot(self) -> np.ndarray:
        """Return the latest rendered desktop frame."""
        if self._latest_frame is None:
            raise RuntimeError("No rendered frame is available yet.")
        with self._frame_lock:
            assert self._latest_frame is not None
            return self._latest_frame.copy()


def _qt_application() -> QApplication:
    """Return a shared Qt application instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _pixmap_from_frame(frame: np.ndarray) -> QPixmap:
    """Convert a uint8 RGB frame into a Qt pixmap."""
    from PySide6.QtGui import QImage

    rgb = np.ascontiguousarray(frame)
    image = QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(image)


def _qt_color(red: int, green: int, blue: int, alpha: int) -> QColor:
    """Return a Qt color value."""
    return QColor(red, green, blue, alpha)


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for rotating around ``axis``."""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _rotation_matrix_xyz(
    x_degrees: float,
    y_degrees: float,
    z_degrees: float,
) -> np.ndarray:
    """Return the XYZ Euler rotation matrix used by the desktop viewer."""
    x_radians, y_radians, z_radians = np.radians(
        [x_degrees, y_degrees, z_degrees]
    )
    cx, cy, cz = np.cos([x_radians, y_radians, z_radians])
    sx, sy, sz = np.sin([x_radians, y_radians, z_radians])
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rotation_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotation_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def desktop_viewer(
    render_fn: Callable[[CameraState], Any],
    *,
    state: ViewerState | None = None,
    controls: DesktopPydanticControls[Any] | None = None,
    width: int = 1280,
    height: int = 720,
    title: str = "marimo-3dv desktop viewer",
) -> DesktopViewer:
    """Create and return a ``DesktopViewer`` instance."""
    return DesktopViewer(
        render_fn,
        state=state,
        controls=controls,
        width=width,
        height=height,
        title=title,
    )
