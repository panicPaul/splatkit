"""Native image-based 3D viewer widget for marimo notebooks."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import gc
import json
import secrets
import signal
import socket
import threading
import time
import traceback
import warnings
import weakref
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anywidget
import cv2
import numpy as np
import torch
import traitlets
import uvicorn
from jaxtyping import Float, UInt8
from marimo._plugins.ui._core.ui_element import UIElement
from marimo._plugins.ui._impl.comm import MarimoComm
from marimo._plugins.ui._impl.from_anywidget import (
    AnyWidgetState,
    ModelIdRef,
    get_anywidget_model_id,
    get_anywidget_state,
)
from marimo._plugins.ui._impl.from_anywidget import (
    anywidget as BaseMarimoAnyWidget,
)
from marimo._runtime.context import ContextNotInitializedError, get_context
from marimo._runtime.virtual_file import VirtualFile
from marimo._utils.code import hash_code
from PIL import Image
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

CameraConvention = Literal["opencv", "opengl", "blender", "colmap"]
ViewerTransportMode = Literal["comm", "websocket"]
LinkedViewerStateField = Literal[
    "camera_state",
    "show_axes",
    "show_horizon",
    "show_origin",
    "show_stats",
]
_ViewerStateChangeListener = Callable[[LinkedViewerStateField], None]
_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_ACTIVE_MARIMO_VIEWERS: dict[int, weakref.ReferenceType[MarimoViewer]] = {}
_PROCESS_CLEANUP_REGISTERED = False


def _best_effort_cuda_cleanup() -> None:
    """Release CUDA cache without failing teardown on a poisoned context."""
    if not torch.cuda.is_available():
        return

    cleanup_steps = (
        ("empty_cache", torch.cuda.empty_cache),
        ("ipc_collect", torch.cuda.ipc_collect),
    )
    for step_name, cleanup_step in cleanup_steps:
        try:
            cleanup_step()
        except Exception as error:
            warnings.warn(
                "CUDA cleanup during viewer teardown failed while calling "
                f"`torch.cuda.{step_name}()`: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            break


def _cleanup_active_marimo_viewers() -> None:
    """Release notebook viewer resources during process teardown."""
    viewer_refs = list(_ACTIVE_MARIMO_VIEWERS.values())
    _ACTIVE_MARIMO_VIEWERS.clear()
    for viewer_ref in viewer_refs:
        viewer = viewer_ref()
        if viewer is None:
            continue
        with contextlib.suppress(Exception):
            viewer.close()

    gc.collect()

    _best_effort_cuda_cleanup()


def _register_process_cleanup_handlers() -> None:
    """Install best-effort cleanup hooks for normal process termination."""
    global _PROCESS_CLEANUP_REGISTERED
    if _PROCESS_CLEANUP_REGISTERED:
        return
    _PROCESS_CLEANUP_REGISTERED = True

    atexit.register(_cleanup_active_marimo_viewers)

    def _make_handler(
        previous_handler: object,
    ) -> Callable[[int, object | None], None]:
        def _handler(signum: int, frame: object | None) -> None:
            _cleanup_active_marimo_viewers()

            if callable(previous_handler):
                previous_handler(signum, frame)
                return
            if previous_handler == signal.SIG_IGN:
                return
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + signum)

        return _handler

    cleanup_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        cleanup_signals.append(signal.SIGHUP)

    for cleanup_signal in cleanup_signals:
        with contextlib.suppress(ValueError):
            previous_handler = signal.getsignal(cleanup_signal)
            signal.signal(
                cleanup_signal,
                _make_handler(previous_handler),
            )


def _find_free_port(start: int = 8765, attempts: int = 64) -> int:
    """Return the first free TCP port in [start, start + attempts)."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"Could not find a free port in range {start}-{start + attempts}."
    )


@dataclass
class _ClientSender:
    """Latest-only async sender for one WebSocket client."""

    websocket: WebSocket
    _packet: bytes | None = field(default=None, repr=False)
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, repr=False)

    def push(self, packet: bytes) -> None:
        """Replace the pending packet and wake the sender task."""
        self._packet = packet
        self._event.set()

    def start(self) -> None:
        """Start the background sender task."""
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        """Cancel the sender task and wait for it to finish."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        while True:
            await self._event.wait()
            self._event.clear()
            packet = self._packet
            if packet is None:
                continue
            try:
                await self.websocket.send_bytes(packet)
            except Exception:
                return


@dataclass
class _FrameStreamState:
    """Mutable per-viewer stream state for the local websocket server."""

    token: str
    latest_packet: bytes | None = None
    senders: dict[WebSocket, _ClientSender] | None = None

    def __post_init__(self) -> None:
        if self.senders is None:
            self.senders = {}


class _FrameStreamServer:
    """Local websocket server for native viewer frame streaming."""

    def __init__(self) -> None:
        self.port = _find_free_port()
        self._streams: dict[str, _FrameStreamState] = {}
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._app = Starlette(
            routes=[
                WebSocketRoute("/streams/{stream_id}", self._websocket_endpoint)
            ]
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError(
                "Timed out starting native viewer stream server."
            )

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        self._loop.run_until_complete(server.serve())

    async def _websocket_endpoint(self, websocket: WebSocket) -> None:
        stream_id = str(websocket.path_params["stream_id"])
        token = websocket.query_params.get("token", "")
        with self._lock:
            stream_state = self._streams.get(stream_id)
            is_authorized = stream_state is not None and secrets.compare_digest(
                token, stream_state.token
            )
        if not is_authorized:
            await websocket.close(code=1008)
            return

        await websocket.accept()
        assert stream_state is not None
        sender = _ClientSender(websocket=websocket)
        sender.start()
        with self._lock:
            stream_state.senders[websocket] = sender
            latest_packet = stream_state.latest_packet
        if latest_packet is not None:
            sender.push(latest_packet)

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message["type"] != "websocket.receive":
                    continue
                if "text" not in message or message["text"] is None:
                    continue
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") != "clock_sync_ping":
                    continue
                ping_id = payload.get("ping_id")
                client_sent_at_ms = payload.get("client_sent_at_ms")
                if not isinstance(ping_id, int | float) or not isinstance(
                    client_sent_at_ms, int | float
                ):
                    continue
                try:
                    await websocket.send_json(
                        {
                            "type": "clock_sync_pong",
                            "ping_id": int(ping_id),
                            "client_sent_at_ms": float(client_sent_at_ms),
                            "server_received_at_ms": (
                                time.perf_counter() * 1000.0
                            ),
                        }
                    )
                except RuntimeError as exc:
                    if "close message has been sent" in str(exc):
                        break
                    raise
        except WebSocketDisconnect:
            pass
        finally:
            with self._lock:
                active_stream = self._streams.get(stream_id)
                if active_stream is not None:
                    active_stream.senders.pop(websocket, None)
            await sender.stop()

    def register_stream(self) -> tuple[str, str]:
        """Create a new stream and return `(stream_id, token)`."""
        stream_id = secrets.token_urlsafe(12)
        token = secrets.token_urlsafe(24)
        with self._lock:
            self._streams[stream_id] = _FrameStreamState(token=token)
        return stream_id, token

    def unregister_stream(self, stream_id: str) -> None:
        """Remove a stream and close any connected websocket clients."""
        with self._lock:
            stream_state = self._streams.pop(stream_id, None)
        if stream_state is None or self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._close_stream(stream_state), self._loop
        )

    def publish(
        self, stream_id: str, packet: bytes, *, scheduled_at: float
    ) -> concurrent.futures.Future[dict[str, float]] | None:
        """Publish a packet to a stream and fan it out to connected clients."""
        with self._lock:
            stream_state = self._streams[stream_id]
            stream_state.latest_packet = packet
        if self._loop is None:
            return None
        return asyncio.run_coroutine_threadsafe(
            self._broadcast(stream_id, packet, scheduled_at), self._loop
        )

    async def _broadcast(
        self, stream_id: str, packet: bytes, scheduled_at: float
    ) -> dict[str, float]:
        started_at = time.perf_counter()
        with self._lock:
            stream_state = self._streams.get(stream_id)
            senders = (
                []
                if stream_state is None
                else list(stream_state.senders.values())
            )
        for sender in senders:
            sender.push(packet)
        finished_at = time.perf_counter()
        return {
            "stream_queue_time_ms": (started_at - scheduled_at) * 1000.0,
            "stream_send_time_ms": (finished_at - started_at) * 1000.0,
        }

    async def _close_stream(self, stream_state: _FrameStreamState) -> None:
        """Close all websocket clients associated with a stream."""
        senders = list(stream_state.senders.values())
        websockets = list(stream_state.senders.keys())
        stream_state.latest_packet = None
        stream_state.senders.clear()
        for websocket in websockets:
            with contextlib.suppress(Exception):
                await websocket.close(code=1001)
        for sender in senders:
            with contextlib.suppress(Exception):
                await sender.stop()


_FRAME_STREAM_SERVER: _FrameStreamServer | None = None
_FRAME_STREAM_SERVER_LOCK = threading.Lock()


def _get_frame_stream_server() -> _FrameStreamServer:
    """Return the shared local websocket frame stream server."""
    global _FRAME_STREAM_SERVER
    with _FRAME_STREAM_SERVER_LOCK:
        if _FRAME_STREAM_SERVER is None:
            _FRAME_STREAM_SERVER = _FrameStreamServer()
        return _FRAME_STREAM_SERVER


def _pack_frame_packet(metadata: dict[str, object], payload: bytes) -> bytes:
    """Pack frame metadata and payload into a binary websocket packet."""
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return len(header).to_bytes(4, "big") + header + payload


_CONVENTION_TO_INTERNAL_ROTATION: dict[CameraConvention, np.ndarray] = {
    "opencv": np.diag([1.0, -1.0, 1.0]),
    "opengl": np.diag([1.0, 1.0, -1.0]),
    "blender": np.diag([1.0, 1.0, -1.0]),
    "colmap": np.diag([1.0, -1.0, 1.0]),
}


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Return a normalized copy of the input vector."""
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _camera_basis_from_cam_to_world(
    cam_to_world: Float[np.ndarray, "4 4"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract position, forward, and up vectors from a camera transform."""
    matrix = np.asarray(cam_to_world, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"cam_to_world must have shape (4, 4), got {matrix.shape}."
        )
    position = matrix[:3, 3]
    rotation = matrix[:3, :3]
    forward = _normalize(rotation[:, 2])
    up = _normalize(rotation[:, 1])
    return position, forward, up


def _look_at_cam_to_world(
    position: np.ndarray,
    look_at: np.ndarray,
    up_direction: np.ndarray,
) -> Float[np.ndarray, "4 4"]:
    """Construct a cam-to-world transform from look-at parameters."""
    forward = _normalize(look_at - position)
    right = np.cross(forward, up_direction)
    if np.linalg.norm(right) <= 1e-8:
        fallback_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
    right = _normalize(right)
    up = _normalize(np.cross(forward, right))

    cam_to_world = np.eye(4, dtype=np.float64)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = forward
    cam_to_world[:3, 3] = position
    return cam_to_world


def _roll_cam_to_world_180(
    cam_to_world: Float[np.ndarray, "4 4"],
) -> Float[np.ndarray, "4 4"]:
    """Apply a 180-degree roll around the camera forward axis."""
    rolled = np.asarray(cam_to_world, dtype=np.float64).copy()
    roll = np.diag([-1.0, -1.0, 1.0, 1.0])
    rolled[:3, :3] = rolled[:3, :3] @ roll[:3, :3]
    return rolled


def _convention_transform_matrix(
    camera_convention: CameraConvention,
) -> Float[np.ndarray, "4 4"]:
    """Return the camera-basis transform into the viewer's internal basis."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _CONVENTION_TO_INTERNAL_ROTATION[camera_convention]
    return transform


def _convert_cam_to_world_between_conventions(
    cam_to_world: Float[np.ndarray, "4 4"],
    *,
    source_convention: CameraConvention,
    target_convention: CameraConvention,
) -> Float[np.ndarray, "4 4"]:
    """Convert a camera transform between supported camera conventions."""
    matrix = np.asarray(cam_to_world, dtype=np.float64)
    internal = matrix @ _convention_transform_matrix(source_convention)
    target = internal @ _convention_transform_matrix(target_convention)
    return target


@dataclass(frozen=True)
class CameraState:
    """Serializable native viewer camera state."""

    fov_degrees: float
    width: int
    height: int
    cam_to_world: Float[np.ndarray, "4 4"]
    camera_convention: CameraConvention = "opencv"

    def __post_init__(self) -> None:
        """Validate dimensions and camera matrix shape."""
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}.")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}.")
        if self.fov_degrees <= 0.0:
            raise ValueError(
                f"fov_degrees must be positive, got {self.fov_degrees}."
            )
        if self.fov_degrees >= 180.0:
            raise ValueError(
                "fov_degrees must be less than 180 degrees, "
                f"got {self.fov_degrees}."
            )
        if self.camera_convention not in {
            "opencv",
            "opengl",
            "blender",
            "colmap",
        }:
            raise ValueError(
                "camera_convention must be one of "
                "'opencv', 'opengl', 'blender', or 'colmap', "
                f"got {self.camera_convention!r}."
            )
        matrix = np.asarray(self.cam_to_world, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"cam_to_world must have shape (4, 4), got {matrix.shape}."
            )
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0])):
            raise ValueError("cam_to_world bottom row must be [0, 0, 0, 1].")
        object.__setattr__(self, "cam_to_world", matrix.copy())

    @classmethod
    def default(
        cls,
        *,
        width: int = 800,
        height: int = 600,
        fov_degrees: float = 60.0,
        camera_convention: CameraConvention = "opencv",
    ) -> CameraState:
        """Create a default forward-facing camera state.

        The field of view is expressed in degrees.
        """
        position = np.array([0.0, 0.0, 3.0], dtype=np.float64)
        look_at = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return cls(
            fov_degrees=fov_degrees,
            width=width,
            height=height,
            cam_to_world=_roll_cam_to_world_180(
                _look_at_cam_to_world(position, look_at, up_direction)
            ),
            camera_convention=camera_convention,
        )

    @property
    def position(self) -> np.ndarray:
        """Return the camera position."""
        return self.cam_to_world[:3, 3].copy()

    @property
    def forward(self) -> np.ndarray:
        """Return the normalized camera forward vector."""
        return _normalize(self.cam_to_world[:3, 2])

    @property
    def up_direction(self) -> np.ndarray:
        """Return the normalized camera up vector."""
        return _normalize(self.cam_to_world[:3, 1])

    def with_size(self, width: int, height: int) -> CameraState:
        """Return a copy with updated output dimensions."""
        return CameraState(
            fov_degrees=self.fov_degrees,
            width=width,
            height=height,
            cam_to_world=self.cam_to_world,
            camera_convention=self.camera_convention,
        )

    def with_convention(
        self, camera_convention: CameraConvention
    ) -> CameraState:
        """Return a copy converted into a different camera convention."""
        if camera_convention == self.camera_convention:
            return self
        return CameraState(
            fov_degrees=self.fov_degrees,
            width=self.width,
            height=self.height,
            cam_to_world=_convert_cam_to_world_between_conventions(
                self.cam_to_world,
                source_convention=self.camera_convention,
                target_convention=camera_convention,
            ),
            camera_convention=camera_convention,
        )

    def to_json(self) -> str:
        """Serialize the camera state into a stable JSON string.

        The field of view is expressed in degrees.
        """
        return json.dumps(
            {
                "fov_degrees": self.fov_degrees,
                "width": self.width,
                "height": self.height,
                "cam_to_world": self.cam_to_world.tolist(),
                "camera_convention": self.camera_convention,
            }
        )

    @classmethod
    def from_json(cls, value: str) -> CameraState:
        """Deserialize a camera state from JSON."""
        payload = json.loads(value)
        return cls(
            fov_degrees=float(payload["fov_degrees"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            cam_to_world=np.asarray(payload["cam_to_world"], dtype=np.float64),
            camera_convention=payload.get("camera_convention", "opencv"),
        )


@dataclass(frozen=True)
class ViewerClick:
    """Serializable click event captured from the native viewer."""

    x: int
    y: int
    width: int
    height: int
    camera_state: CameraState

    def __post_init__(self) -> None:
        """Validate click coordinates and viewport dimensions."""
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}.")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}.")
        if not 0 <= self.x < self.width:
            raise ValueError(f"x must be in [0, {self.width}), got {self.x}.")
        if not 0 <= self.y < self.height:
            raise ValueError(f"y must be in [0, {self.height}), got {self.y}.")

    def to_json(self) -> str:
        """Serialize the click into a stable JSON string."""
        return json.dumps(
            {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
                "camera_state": json.loads(self.camera_state.to_json()),
            }
        )

    @classmethod
    def from_json(cls, value: str) -> ViewerClick:
        """Deserialize a click from JSON."""
        payload = json.loads(value)
        return cls(
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            camera_state=CameraState.from_json(
                json.dumps(payload["camera_state"])
            ),
        )


@dataclass
class ViewerState:
    """Explicit persistent state for a viewer session.

    Reuse the same state object across reruns to preserve the current camera
    state and last click even when `render_fn` is recreated.
    """

    camera_state: CameraState
    initial_camera_state: CameraState
    camera_convention: CameraConvention
    aspect_ratio: float
    interactive_quality: int
    settled_quality: Literal["jpeg_95", "jpeg_100", "png"]
    internal_render_max_side: int | None
    interactive_max_side: int | None
    raise_on_error: bool
    transport_mode: ViewerTransportMode
    last_click: ViewerClick | None = None
    show_axes: bool = True
    show_horizon: bool = False
    show_origin: bool = False
    show_stats: bool = False
    viewer_rotation_x_degrees: float = 0.0
    viewer_rotation_y_degrees: float = 0.0
    viewer_rotation_z_degrees: float = 0.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    keyboard_move_speed: float = 0.125
    keyboard_sprint_multiplier: float = 4.0
    orbit_invert_x: bool = False
    orbit_invert_y: bool = False
    pan_invert_x: bool = False
    pan_invert_y: bool = False
    _reset_camera_callback: Callable[[CameraState], None] | None = None
    _viewer_rotation_callback: Callable[[float, float, float], None] | None = (
        None
    )
    _show_axes_callback: Callable[[bool], None] | None = None
    _show_horizon_callback: Callable[[bool], None] | None = None
    _show_origin_callback: Callable[[bool], None] | None = None
    _show_stats_callback: Callable[[bool], None] | None = None
    _origin_callback: Callable[[float, float, float], None] | None = None
    _keyboard_navigation_callback: Callable[[float, float], None] | None = None
    _pointer_controls_callback: (
        Callable[[bool, bool, bool, bool], None] | None
    ) = None
    _field_listeners: list[_ViewerStateChangeListener] = field(
        init=False,
        repr=False,
        default_factory=list,
    )
    _active_marimo_viewer_ref: weakref.ReferenceType[MarimoViewer] | None = None

    def __init__(
        self,
        camera_state: CameraState | None = None,
        camera_convention: CameraConvention = "opencv",
        aspect_ratio: float = 16.3 / 9.0,
        interactive_quality: int = 50,
        settled_quality: Literal["jpeg_95", "jpeg_100", "png"] = "jpeg_100",
        internal_render_max_side: int | None = 3840,
        interactive_max_side: int | None = 1980,
        raise_on_error: bool = True,
        transport_mode: ViewerTransportMode = "comm",
        last_click: ViewerClick | None = None,
        show_axes: bool = True,
        show_horizon: bool = False,
        show_origin: bool = False,
        show_stats: bool = True,
        keyboard_move_speed: float = 0.125,
        keyboard_sprint_multiplier: float = 4.0,
        orbit_invert_x: bool = False,
        orbit_invert_y: bool = False,
        pan_invert_x: bool = False,
        pan_invert_y: bool = False,
    ) -> None:
        resolved_camera_state = camera_state or CameraState.default(
            camera_convention=camera_convention
        )
        if resolved_camera_state.camera_convention != camera_convention:
            resolved_camera_state = resolved_camera_state.with_convention(
                camera_convention
            )
        if aspect_ratio <= 0.0:
            raise ValueError(
                f"aspect_ratio must be positive, got {aspect_ratio}."
            )
        if not 1 <= interactive_quality <= 100:
            raise ValueError(
                "interactive_quality must be in [1, 100], "
                f"got {interactive_quality}."
            )
        if (
            internal_render_max_side is not None
            and internal_render_max_side <= 0
        ):
            raise ValueError(
                "internal_render_max_side must be None or a positive integer, "
                f"got {internal_render_max_side}."
            )
        if interactive_max_side is not None and interactive_max_side <= 0:
            raise ValueError(
                "interactive_max_side must be None or a positive integer, "
                f"got {interactive_max_side}."
            )
        if transport_mode not in {"comm", "websocket"}:
            raise ValueError(
                "transport_mode must be 'comm' or 'websocket', "
                f"got {transport_mode!r}."
            )
        if keyboard_move_speed <= 0.0:
            raise ValueError(
                "keyboard_move_speed must be positive, "
                f"got {keyboard_move_speed}."
            )
        if keyboard_sprint_multiplier < 1.0:
            raise ValueError(
                "keyboard_sprint_multiplier must be at least 1.0, "
                f"got {keyboard_sprint_multiplier}."
            )
        self.camera_state = resolved_camera_state
        self.initial_camera_state = resolved_camera_state
        self.camera_convention = resolved_camera_state.camera_convention
        self.aspect_ratio = aspect_ratio
        self.interactive_quality = interactive_quality
        self.settled_quality = settled_quality
        self.internal_render_max_side = internal_render_max_side
        self.interactive_max_side = interactive_max_side
        self.raise_on_error = raise_on_error
        self.transport_mode = transport_mode
        self.last_click = last_click
        self.show_axes = show_axes
        self.show_horizon = show_horizon
        self.show_origin = show_origin
        self.show_stats = show_stats
        self.viewer_rotation_x_degrees = 0.0
        self.viewer_rotation_y_degrees = 0.0
        self.viewer_rotation_z_degrees = 0.0
        self.origin = (0.0, 0.0, 0.0)
        self.keyboard_move_speed = keyboard_move_speed
        self.keyboard_sprint_multiplier = keyboard_sprint_multiplier
        self.orbit_invert_x = orbit_invert_x
        self.orbit_invert_y = orbit_invert_y
        self.pan_invert_x = pan_invert_x
        self.pan_invert_y = pan_invert_y
        self._reset_camera_callback = None
        self._viewer_rotation_callback = None
        self._show_axes_callback = None
        self._show_horizon_callback = None
        self._show_origin_callback = None
        self._show_stats_callback = None
        self._origin_callback = None
        self._keyboard_navigation_callback = None
        self._pointer_controls_callback = None
        self._field_listeners = []

    def _register_field_listener(
        self,
        listener: _ViewerStateChangeListener,
    ) -> None:
        """Register a private state-change listener."""
        self._field_listeners.append(listener)

    def _unregister_field_listener(
        self,
        listener: _ViewerStateChangeListener,
    ) -> None:
        """Remove a previously registered private state-change listener."""
        with contextlib.suppress(ValueError):
            self._field_listeners.remove(listener)

    def _notify_field_listeners(self, field: LinkedViewerStateField) -> None:
        """Notify private state-change listeners about one field update."""
        for listener in tuple(self._field_listeners):
            listener(field)

    def _set_camera_internal(
        self,
        camera_state: CameraState,
        *,
        push_to_viewer: bool,
        notify_listeners: bool,
    ) -> ViewerState:
        """Update the current camera state with optional viewer push."""
        self.camera_state = camera_state
        self.camera_convention = camera_state.camera_convention
        if push_to_viewer and self._reset_camera_callback is not None:
            self._reset_camera_callback(self.camera_state)
        if notify_listeners:
            self._notify_field_listeners("camera_state")
        return self

    def _set_show_axes_internal(
        self,
        show_axes: bool,
        *,
        push_to_viewer: bool,
        notify_listeners: bool,
    ) -> ViewerState:
        """Update axis-gizmo visibility with optional viewer push."""
        self.show_axes = show_axes
        if push_to_viewer and self._show_axes_callback is not None:
            self._show_axes_callback(show_axes)
        if notify_listeners:
            self._notify_field_listeners("show_axes")
        return self

    def _set_show_horizon_internal(
        self,
        show_horizon: bool,
        *,
        push_to_viewer: bool,
        notify_listeners: bool,
    ) -> ViewerState:
        """Update horizon visibility with optional viewer push."""
        self.show_horizon = show_horizon
        if push_to_viewer and self._show_horizon_callback is not None:
            self._show_horizon_callback(show_horizon)
        if notify_listeners:
            self._notify_field_listeners("show_horizon")
        return self

    def _set_show_origin_internal(
        self,
        show_origin: bool,
        *,
        push_to_viewer: bool,
        notify_listeners: bool,
    ) -> ViewerState:
        """Update origin visibility with optional viewer push."""
        self.show_origin = show_origin
        if push_to_viewer and self._show_origin_callback is not None:
            self._show_origin_callback(show_origin)
        if notify_listeners:
            self._notify_field_listeners("show_origin")
        return self

    def _set_show_stats_internal(
        self,
        show_stats: bool,
        *,
        push_to_viewer: bool,
        notify_listeners: bool,
    ) -> ViewerState:
        """Update stats visibility with optional viewer push."""
        self.show_stats = show_stats
        if push_to_viewer and self._show_stats_callback is not None:
            self._show_stats_callback(show_stats)
        if notify_listeners:
            self._notify_field_listeners("show_stats")
        return self

    def reset_camera(self) -> ViewerState:
        """Reset the current camera state back to the initial camera state."""
        return self._set_camera_internal(
            self.initial_camera_state,
            push_to_viewer=True,
            notify_listeners=True,
        )

    def set_camera(self, camera_state: CameraState) -> ViewerState:
        """Set the current camera state and push it into the live viewer."""
        return self._set_camera_internal(
            camera_state,
            push_to_viewer=True,
            notify_listeners=True,
        )

    def set_fov_degrees(
        self,
        fov_degrees: float,
        *,
        push_to_viewer: bool = True,
    ) -> ViewerState:
        """Set the field of view on the current and initial camera state."""
        current_camera = CameraState(
            fov_degrees=fov_degrees,
            width=self.camera_state.width,
            height=self.camera_state.height,
            cam_to_world=self.camera_state.cam_to_world,
            camera_convention=self.camera_state.camera_convention,
        )
        initial_camera = CameraState(
            fov_degrees=fov_degrees,
            width=self.initial_camera_state.width,
            height=self.initial_camera_state.height,
            cam_to_world=self.initial_camera_state.cam_to_world,
            camera_convention=self.initial_camera_state.camera_convention,
        )
        self.initial_camera_state = initial_camera
        if push_to_viewer:
            return self.set_camera(current_camera)
        return self._set_camera_internal(
            current_camera,
            push_to_viewer=False,
            notify_listeners=True,
        )

    def set_viewer_rotation(
        self,
        x_degrees: float,
        y_degrees: float,
        z_degrees: float,
    ) -> ViewerState:
        """Set the persistent viewer-frame rotation used by controls/overlays."""
        self.viewer_rotation_x_degrees = x_degrees
        self.viewer_rotation_y_degrees = y_degrees
        self.viewer_rotation_z_degrees = z_degrees
        if self._viewer_rotation_callback is not None:
            self._viewer_rotation_callback(x_degrees, y_degrees, z_degrees)
        return self

    def set_show_axes(self, show_axes: bool) -> ViewerState:
        """Set the axis-gizmo visibility and push it into the live viewer."""
        return self._set_show_axes_internal(
            show_axes,
            push_to_viewer=True,
            notify_listeners=True,
        )

    def set_show_horizon(self, show_horizon: bool) -> ViewerState:
        """Set the horizon overlay visibility and push it into the live viewer."""
        return self._set_show_horizon_internal(
            show_horizon,
            push_to_viewer=True,
            notify_listeners=True,
        )

    def set_show_origin(self, show_origin: bool) -> ViewerState:
        """Set the origin marker visibility and push it into the live viewer."""
        return self._set_show_origin_internal(
            show_origin,
            push_to_viewer=True,
            notify_listeners=True,
        )

    def set_show_stats(self, show_stats: bool) -> ViewerState:
        """Set the stats overlay visibility and push it into the live viewer."""
        return self._set_show_stats_internal(
            show_stats,
            push_to_viewer=True,
            notify_listeners=True,
        )

    def set_origin(self, x: float, y: float, z: float) -> ViewerState:
        """Set the world-space origin marker position in the live viewer."""
        self.origin = (x, y, z)
        if self._origin_callback is not None:
            self._origin_callback(x, y, z)
        return self

    def set_keyboard_navigation(
        self,
        move_speed: float,
        sprint_multiplier: float,
    ) -> ViewerState:
        """Set keyboard navigation speed and sprint multiplier."""
        if move_speed <= 0.0:
            raise ValueError(f"move_speed must be positive, got {move_speed}.")
        if sprint_multiplier < 1.0:
            raise ValueError(
                "sprint_multiplier must be at least 1.0, "
                f"got {sprint_multiplier}."
            )
        self.keyboard_move_speed = move_speed
        self.keyboard_sprint_multiplier = sprint_multiplier
        if self._keyboard_navigation_callback is not None:
            self._keyboard_navigation_callback(
                move_speed,
                sprint_multiplier,
            )
        return self

    def set_pointer_controls(
        self,
        orbit_invert_x: bool,
        orbit_invert_y: bool,
        pan_invert_x: bool,
        pan_invert_y: bool,
    ) -> ViewerState:
        """Set pointer drag inversion controls for orbit and pan."""
        self.orbit_invert_x = orbit_invert_x
        self.orbit_invert_y = orbit_invert_y
        self.pan_invert_x = pan_invert_x
        self.pan_invert_y = pan_invert_y
        if self._pointer_controls_callback is not None:
            self._pointer_controls_callback(
                orbit_invert_x,
                orbit_invert_y,
                pan_invert_x,
                pan_invert_y,
            )
        return self

    def copy(self) -> ViewerState:
        """Return a shallow copy of the current viewer configuration/state."""
        copied = ViewerState(
            camera_state=self.camera_state,
            camera_convention=self.camera_convention,
            aspect_ratio=self.aspect_ratio,
            interactive_quality=self.interactive_quality,
            settled_quality=self.settled_quality,
            internal_render_max_side=self.internal_render_max_side,
            interactive_max_side=self.interactive_max_side,
            raise_on_error=self.raise_on_error,
            transport_mode=self.transport_mode,
            last_click=self.last_click,
            show_axes=self.show_axes,
            show_horizon=self.show_horizon,
            show_origin=self.show_origin,
            show_stats=self.show_stats,
            keyboard_move_speed=self.keyboard_move_speed,
            keyboard_sprint_multiplier=self.keyboard_sprint_multiplier,
            orbit_invert_x=self.orbit_invert_x,
            orbit_invert_y=self.orbit_invert_y,
            pan_invert_x=self.pan_invert_x,
            pan_invert_y=self.pan_invert_y,
        )
        copied.initial_camera_state = self.initial_camera_state
        copied.viewer_rotation_x_degrees = self.viewer_rotation_x_degrees
        copied.viewer_rotation_y_degrees = self.viewer_rotation_y_degrees
        copied.viewer_rotation_z_degrees = self.viewer_rotation_z_degrees
        copied.origin = self.origin
        return copied


def _normalize_frame(
    frame: np.ndarray | torch.Tensor,
) -> UInt8[np.ndarray, "height width 3"]:
    """Convert an RGB numpy or torch frame to contiguous uint8."""
    array: np.ndarray
    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"Expected frame shape (height, width, 3), got {array.shape}."
        )

    if np.issubdtype(array.dtype, np.floating):
        working = np.asarray(array, dtype=np.float32)
        if (
            float(np.nanmax(working)) <= 1.0
            and float(np.nanmin(working)) >= 0.0
        ):
            working = working * 255.0
        array = np.clip(working, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


class _WidgetValueProxy(MutableMapping[str, object]):
    """Live mapping view over synced anywidget traits."""

    def __init__(self, widget: _NativeViewerAnyWidget) -> None:
        self._widget = widget

    def _state(self) -> dict[str, object]:
        return self._widget.get_state()

    def __getitem__(self, key: str) -> object:
        return self._state()[key]

    def __setitem__(self, key: str, value: object) -> None:
        if not self._widget.has_trait(key):
            raise KeyError(key)
        setattr(self._widget, key, value)

    def __delitem__(self, key: str) -> None:
        raise TypeError(
            "Deleting widget traits through .value is not supported."
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._state())

    def __len__(self) -> int:
        return len(self._state())


class _StableMarimoAnyWidget(BaseMarimoAnyWidget):
    """Wrap an anywidget with a stable virtual-file URL for marimo."""

    def __init__(self, widget: anywidget.AnyWidget) -> None:
        self.widget = widget
        self._initialized = False

        js = getattr(widget, "_esm", "")
        js_filename = "native_viewer_widget.js"
        if isinstance(js, Path):
            js_filename = js.name
            js = js.read_text(encoding="utf-8")
        if not isinstance(js, str):
            raise TypeError(
                "_StableMarimoAnyWidget expects widget._esm to be a string or Path."
            )

        js_hash = hash_code(js)
        _ = widget.comm
        model_id = get_anywidget_model_id(widget)
        js_url = self._create_js_url(
            js=js,
            js_filename=js_filename,
            js_hash=js_hash,
        )

        UIElement.__init__(
            self,
            component_name="marimo-anywidget",
            initial_value=ModelIdRef(model_id=model_id),
            label=None,
            args={
                "js-url": js_url,
                "js-hash": js_hash,
                "model-id": model_id,
            },
            on_change=None,
        )

    @staticmethod
    def _create_js_url(*, js: str, js_filename: str, js_hash: str) -> str:
        """Create a JS URL that prefers marimo's virtual-file registry."""
        if not js:
            return ""

        js_buffer = js.encode("utf-8")
        suffix = Path(js_filename).suffix or ".js"
        stem = Path(js_filename).stem
        versioned_filename = f"{stem}-{js_hash}{suffix}"

        try:
            context = get_context()
        except ContextNotInitializedError:
            context = None

        if context is None or not context.virtual_files_supported:
            return VirtualFile(
                filename=versioned_filename,
                buffer=js_buffer,
                as_data_url=True,
            ).url

        for attempt in range(4):
            candidate_filename = versioned_filename
            if attempt > 0:
                candidate_filename = (
                    f"{stem}-{js_hash}-{secrets.token_hex(4)}{suffix}"
                )
            virtual_file = VirtualFile(
                filename=candidate_filename,
                buffer=js_buffer,
            )
            try:
                context.virtual_file_registry.add(virtual_file, context)
            except FileExistsError:
                continue
            return virtual_file.url

        return VirtualFile(
            filename=f"{stem}-{js_hash}-{secrets.token_hex(8)}{suffix}",
            buffer=js_buffer,
            as_data_url=True,
        ).url

    def _initialize(
        self,
        initialization_args: Any,
    ) -> None:
        super()._initialize(initialization_args)
        comm = self.widget.comm
        if isinstance(comm, MarimoComm):
            comm.ui_element_id = self._id

    def _convert_value(
        self, value: ModelIdRef | AnyWidgetState
    ) -> AnyWidgetState:
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value)}")

        model_id = value.get("model_id")
        if model_id and len(value) == 1:
            return {}

        self.widget.set_state(value)
        return value

    @property
    def value(self) -> AnyWidgetState:
        """Return the synced anywidget state."""
        return get_anywidget_state(self.widget)

    @value.setter
    def value(self, value: AnyWidgetState) -> None:
        del value
        raise RuntimeError("Setting the value of a UIElement is not allowed.")

    def __setattr__(self, name: str, value: Any) -> None:
        if self._initialized:
            if hasattr(self.widget, name):
                setattr(self.widget, name, value)
                return
            super().__setattr__(name, value)
            return
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name in ("widget", "_initialized"):
            try:
                return self.__getattribute__(name)
            except AttributeError:
                return None
        return getattr(self.widget, name)

    def __getitem__(self, key: Any) -> Any:
        return self.widget[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.widget


class _LatestOnlyRenderer:
    """Background latest-only renderer for camera-driven frames."""

    def __init__(
        self,
        render_fn: Callable[[CameraState], np.ndarray | torch.Tensor],
        publish_frame: Callable[
            [int, CameraState, np.ndarray, float, bool], None
        ],
        publish_error: Callable[[int, Exception, str], None],
        set_rendering: Callable[[bool], None],
    ) -> None:
        self._render_fn = render_fn
        self._publish_frame = publish_frame
        self._publish_error = publish_error
        self._set_rendering = set_rendering
        self._condition = threading.Condition()
        self._latest_revision = -1
        self._pending_revision = -1
        self._pending_state: CameraState | None = None
        self._pending_interaction_active = False
        self._pending_requested_at = 0.0
        self._closed = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def request(
        self, revision: int, camera_state: CameraState, interaction_active: bool
    ) -> None:
        """Request a render for the most recent camera state."""
        with self._condition:
            if self._closed:
                return
            self._latest_revision = revision
            self._pending_revision = revision
            self._pending_state = camera_state
            self._pending_interaction_active = interaction_active
            self._pending_requested_at = time.perf_counter()
            self._set_rendering(True)
            self._condition.notify()

    def close(self) -> None:
        """Stop the background render worker."""
        with self._condition:
            self._closed = True
            self._pending_state = None
            self._condition.notify_all()
        self._worker.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending_state is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                revision = self._pending_revision
                camera_state = self._pending_state
                interaction_active = self._pending_interaction_active
                requested_at = self._pending_requested_at
                self._pending_state = None

            assert camera_state is not None
            try:
                render_started_at = time.perf_counter()
                render_queue_time_ms = (
                    render_started_at - requested_at
                ) * 1000.0
                rendered_frame = self._render_fn(camera_state)
                render_time_ms = (
                    time.perf_counter() - render_started_at
                ) * 1000.0
                frame = _normalize_frame(rendered_frame)
            except Exception as exception:
                message = "".join(
                    traceback.format_exception(
                        exception.__class__,
                        exception,
                        exception.__traceback__,
                    )
                ).rstrip()
                with self._condition:
                    is_latest = revision == self._latest_revision
                if is_latest:
                    self._publish_error(revision, exception, message)
                    self._set_rendering(False)
                continue

            with self._condition:
                is_latest = revision == self._latest_revision
                superseded_by_interaction = (
                    not interaction_active and self._pending_interaction_active
                )

            if not is_latest or superseded_by_interaction:
                continue

            try:
                self._publish_frame(
                    revision,
                    camera_state,
                    frame,
                    render_queue_time_ms,
                    render_time_ms,
                    interaction_active,
                )
            except Exception as exception:
                message = "".join(
                    traceback.format_exception(
                        exception.__class__,
                        exception,
                        exception.__traceback__,
                    )
                ).rstrip()
                self._publish_error(revision, exception, message)
            finally:
                self._set_rendering(False)


class _NativeViewerAnyWidget(anywidget.AnyWidget):
    """Internal anywidget for the native camera-controlled viewer."""

    _css = _ASSET_DIR / "native_viewer.css"

    camera_state_json = traitlets.Unicode("").tag(sync=True)
    aspect_ratio = traitlets.Float(16.3 / 9.0).tag(sync=True)
    stream_port = traitlets.Int(0).tag(sync=True)
    stream_path = traitlets.Unicode("").tag(sync=True)
    stream_token = traitlets.Unicode("").tag(sync=True)
    transport_mode = traitlets.Unicode("comm").tag(sync=True)
    frame_packet = traitlets.Bytes(b"").tag(sync=True)
    _camera_revision = traitlets.Int(0).tag(sync=True)
    render_revision = traitlets.Int(0).tag(sync=True)
    interaction_active = traitlets.Bool(False).tag(sync=True)
    latency_ms = traitlets.Float(0.0).tag(sync=True)
    latency_sample_ms = traitlets.Float(0.0).tag(sync=True)
    render_time_ms = traitlets.Float(0.0).tag(sync=True)
    render_queue_time_ms = traitlets.Float(0.0).tag(sync=True)
    encode_time_ms = traitlets.Float(0.0).tag(sync=True)
    stream_queue_time_ms = traitlets.Float(0.0).tag(sync=True)
    stream_send_time_ms = traitlets.Float(0.0).tag(sync=True)
    backend_to_browser_time_ms = traitlets.Float(0.0).tag(sync=True)
    packet_size_bytes = traitlets.Int(0).tag(sync=True)
    browser_receive_queue_ms = traitlets.Float(0.0).tag(sync=True)
    browser_post_receive_ms = traitlets.Float(0.0).tag(sync=True)
    browser_decode_time_ms = traitlets.Float(0.0).tag(sync=True)
    browser_draw_time_ms = traitlets.Float(0.0).tag(sync=True)
    browser_present_wait_ms = traitlets.Float(0.0).tag(sync=True)
    render_fps = traitlets.Float(0.0).tag(sync=True)
    last_click_json = traitlets.Unicode("").tag(sync=True)
    is_rendering = traitlets.Bool(False).tag(sync=True)
    error_text = traitlets.Unicode("").tag(sync=True)
    show_axes = traitlets.Bool(False).tag(sync=True)
    show_horizon = traitlets.Bool(False).tag(sync=True)
    show_origin = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    viewer_rotation_x_degrees = traitlets.Float(0.0).tag(sync=True)
    viewer_rotation_y_degrees = traitlets.Float(0.0).tag(sync=True)
    viewer_rotation_z_degrees = traitlets.Float(0.0).tag(sync=True)
    origin_x = traitlets.Float(0.0).tag(sync=True)
    origin_y = traitlets.Float(0.0).tag(sync=True)
    origin_z = traitlets.Float(0.0).tag(sync=True)
    keyboard_move_speed = traitlets.Float(0.125).tag(sync=True)
    keyboard_sprint_multiplier = traitlets.Float(4.0).tag(sync=True)
    orbit_invert_x = traitlets.Bool(False).tag(sync=True)
    orbit_invert_y = traitlets.Bool(False).tag(sync=True)
    pan_invert_x = traitlets.Bool(False).tag(sync=True)
    pan_invert_y = traitlets.Bool(False).tag(sync=True)
    controls_hint = traitlets.Unicode(
        "Orbit: drag | Pan: right-drag | Move: WASDQE | Zoom: wheel"
    ).tag(sync=True)

    _esm = _ASSET_DIR / "native_viewer.js"

    def __init__(
        self,
        *,
        camera_state: CameraState,
        aspect_ratio: float,
        show_axes: bool,
        show_horizon: bool,
        show_origin: bool,
        show_stats: bool,
        viewer_rotation_x_degrees: float,
        viewer_rotation_y_degrees: float,
        viewer_rotation_z_degrees: float,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        keyboard_move_speed: float = 0.125,
        keyboard_sprint_multiplier: float = 4.0,
        orbit_invert_x: bool = False,
        orbit_invert_y: bool = False,
        pan_invert_x: bool = False,
        pan_invert_y: bool = False,
        stream_port: int,
        stream_path: str,
        stream_token: str,
        transport_mode: ViewerTransportMode,
    ) -> None:
        super().__init__(
            camera_state_json=camera_state.to_json(),
            aspect_ratio=aspect_ratio,
            show_axes=show_axes,
            show_horizon=show_horizon,
            show_origin=show_origin,
            show_stats=show_stats,
            viewer_rotation_x_degrees=viewer_rotation_x_degrees,
            viewer_rotation_y_degrees=viewer_rotation_y_degrees,
            viewer_rotation_z_degrees=viewer_rotation_z_degrees,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
            keyboard_move_speed=keyboard_move_speed,
            keyboard_sprint_multiplier=keyboard_sprint_multiplier,
            orbit_invert_x=orbit_invert_x,
            orbit_invert_y=orbit_invert_y,
            pan_invert_x=pan_invert_x,
            pan_invert_y=pan_invert_y,
            stream_port=stream_port,
            stream_path=stream_path,
            stream_token=stream_token,
            transport_mode=transport_mode,
        )


class MarimoViewer(_StableMarimoAnyWidget):
    """Marimo-reactive native viewer widget."""

    def __init__(
        self,
        anywidget_instance: _NativeViewerAnyWidget,
        render_fn: Callable[[CameraState], np.ndarray | torch.Tensor],
        interactive_quality: int,
        settled_quality: Literal["jpeg_95", "jpeg_100", "png"],
        internal_render_max_side: int | None,
        interactive_max_side: int | None,
        state: ViewerState | None,
        raise_on_error: bool,
    ) -> None:
        super().__init__(anywidget_instance)
        self._latest_frame_array: np.ndarray | None = None
        self._interactive_quality = interactive_quality
        self._settled_quality = settled_quality
        self._internal_render_max_side = internal_render_max_side
        self._interactive_max_side = interactive_max_side
        self._raise_on_error = raise_on_error
        self._state = state
        self._stream_server = _get_frame_stream_server()
        self._stream_path = anywidget_instance.stream_path
        self._stream_id = self._stream_path.removeprefix("/streams/")
        self._last_debug_sample_at: float | None = None
        self._smoothed_debug_metrics: dict[str, float] = {}
        self._render_frame_timestamps: list[float] = []
        self._render_completion_condition = threading.Condition()
        self._completed_revisions: dict[int, Exception | None] = {}
        self._closed = False
        _register_process_cleanup_handlers()
        _ACTIVE_MARIMO_VIEWERS[id(self)] = weakref.ref(self)
        try:
            self._main_loop: asyncio.AbstractEventLoop | None = (
                asyncio.get_running_loop()
            )
        except RuntimeError:
            self._main_loop = None
        self._renderer = _LatestOnlyRenderer(
            render_fn=render_fn,
            publish_frame=self._publish_frame,
            publish_error=self._publish_error,
            set_rendering=self._set_rendering,
        )
        self.widget.observe(
            self._on_camera_revision_change, names=["_camera_revision"]
        )
        self.widget.observe(
            self._on_camera_state_json_change, names=["camera_state_json"]
        )
        self.widget.observe(
            self._on_last_click_json_change, names=["last_click_json"]
        )
        self.widget.observe(self._on_show_axes_change, names=["show_axes"])
        self.widget.observe(
            self._on_show_horizon_change, names=["show_horizon"]
        )
        self.widget.observe(self._on_show_origin_change, names=["show_origin"])
        self.widget.observe(self._on_show_stats_change, names=["show_stats"])
        if self._state is not None:
            self._state._reset_camera_callback = self.set_camera_state
            self._state._viewer_rotation_callback = self.set_viewer_rotation
            self._state._show_axes_callback = self.set_show_axes
            self._state._show_horizon_callback = self.set_show_horizon
            self._state._show_origin_callback = self.set_show_origin
            self._state._show_stats_callback = self.set_show_stats
            self._state._origin_callback = self.set_origin
            self._state._keyboard_navigation_callback = (
                self.set_keyboard_navigation
            )
            self._state._pointer_controls_callback = self.set_pointer_controls
        self.rerender()

    def close(self) -> None:
        """Release background resources held by this viewer instance."""
        if self._closed:
            return
        self._closed = True
        _ACTIVE_MARIMO_VIEWERS.pop(id(self), None)
        self.widget.unobserve(
            self._on_camera_revision_change, names=["_camera_revision"]
        )
        self.widget.unobserve(
            self._on_camera_state_json_change, names=["camera_state_json"]
        )
        self.widget.unobserve(
            self._on_last_click_json_change, names=["last_click_json"]
        )
        self.widget.unobserve(self._on_show_axes_change, names=["show_axes"])
        self.widget.unobserve(
            self._on_show_horizon_change, names=["show_horizon"]
        )
        self.widget.unobserve(
            self._on_show_origin_change, names=["show_origin"]
        )
        self.widget.unobserve(self._on_show_stats_change, names=["show_stats"])
        if self._state is not None:
            self._clear_state_callback("_reset_camera_callback")
            self._clear_state_callback("_viewer_rotation_callback")
            self._clear_state_callback("_show_axes_callback")
            self._clear_state_callback("_show_horizon_callback")
            self._clear_state_callback("_show_origin_callback")
            self._clear_state_callback("_show_stats_callback")
            self._clear_state_callback("_origin_callback")
            self._clear_state_callback("_keyboard_navigation_callback")
            self._clear_state_callback("_pointer_controls_callback")
            active_ref = self._state._active_marimo_viewer_ref
            active_viewer = None if active_ref is None else active_ref()
            if active_viewer is self:
                self._state._active_marimo_viewer_ref = None
        self._renderer.close()
        self._stream_server.unregister_stream(self._stream_id)

    def _clear_state_callback(self, attribute_name: str) -> None:
        """Clear a ViewerState callback only when it points at this viewer."""
        if self._state is None:
            return
        callback = getattr(self._state, attribute_name)
        if getattr(callback, "__self__", None) is self:
            setattr(self._state, attribute_name, None)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def _update_smoothed_debug_metrics(
        self, **samples: float
    ) -> dict[str, float]:
        """Update smoothed backend timing metrics with idle reset."""
        now = time.perf_counter()
        should_reset = (
            self._last_debug_sample_at is None
            or now - self._last_debug_sample_at > 1.0
        )
        for key, sample in samples.items():
            previous = self._smoothed_debug_metrics.get(key)
            if should_reset or previous is None:
                self._smoothed_debug_metrics[key] = sample
            else:
                self._smoothed_debug_metrics[key] = (
                    previous * 0.85 + sample * 0.15
                )
        self._last_debug_sample_at = now
        return {key: self._smoothed_debug_metrics[key] for key in samples}

    def anywidget(self) -> _NativeViewerAnyWidget:
        """Return the underlying raw anywidget instance."""
        return self.widget

    @property
    def value(self) -> MutableMapping[str, object]:
        """Return a live mapping of synced widget traits."""
        return _WidgetValueProxy(self.widget)

    def get_camera_state(self) -> CameraState:
        """Return the current synced camera state."""
        return CameraState.from_json(self.widget.camera_state_json)

    def get_last_click(self) -> ViewerClick | None:
        """Return the last primary-button click, if any."""
        value = self.widget.last_click_json
        if not value:
            return None
        return ViewerClick.from_json(value)

    def get_debug_info(self) -> dict[str, float | str]:
        """Return the current synced timing diagnostics.

        The returned dictionary includes raw synced metrics such as
        `latency_ms`, `latency_sample_ms`, encoder and stream timings, and
        browser-side timings.

        It also includes two derived accounting views:

        - `leaf`: sums the non-overlapping leaf stages
          (`render_queue`, `render`, `encode`, `stream_queue`,
          `stream_send`, `browser_receive_queue`,
          `browser_decode`, `browser_draw`, `browser_present_wait`).
        - `coarse`: uses the coarser browser bucket
          `browser_post_receive_ms` instead of the decode/draw/present
          leaf stages.

        The `unaccounted_*` fields are residuals against either the
        smoothed average latency (`latency_ms`) or the raw per-frame
        latency sample (`latency_sample_ms`).
        """
        debug_info: dict[str, float | str] = {
            "error_text": str(self.widget.error_text),
            "latency_ms": float(self.widget.latency_ms),
            "latency_sample_ms": float(self.widget.latency_sample_ms),
            "render_time_ms": float(self.widget.render_time_ms),
            "render_queue_time_ms": float(self.widget.render_queue_time_ms),
            "encode_time_ms": float(self.widget.encode_time_ms),
            "stream_queue_time_ms": float(self.widget.stream_queue_time_ms),
            "stream_send_time_ms": float(self.widget.stream_send_time_ms),
            "backend_to_browser_time_ms": float(
                self.widget.backend_to_browser_time_ms
            ),
            "packet_size_bytes": int(self.widget.packet_size_bytes),
            "browser_receive_queue_ms": float(
                self.widget.browser_receive_queue_ms
            ),
            "browser_post_receive_ms": float(
                self.widget.browser_post_receive_ms
            ),
            "browser_decode_time_ms": float(self.widget.browser_decode_time_ms),
            "browser_draw_time_ms": float(self.widget.browser_draw_time_ms),
            "browser_present_wait_ms": float(
                self.widget.browser_present_wait_ms
            ),
        }
        accounted_leaf_latency_ms = (
            float(debug_info["render_queue_time_ms"])
            + float(debug_info["render_time_ms"])
            + float(debug_info["encode_time_ms"])
            + float(debug_info["stream_queue_time_ms"])
            + float(debug_info["stream_send_time_ms"])
            + float(debug_info["browser_receive_queue_ms"])
            + float(debug_info["browser_decode_time_ms"])
            + float(debug_info["browser_draw_time_ms"])
            + float(debug_info["browser_present_wait_ms"])
        )
        accounted_coarse_latency_ms = (
            float(debug_info["render_queue_time_ms"])
            + float(debug_info["render_time_ms"])
            + float(debug_info["encode_time_ms"])
            + float(debug_info["stream_queue_time_ms"])
            + float(debug_info["stream_send_time_ms"])
            + float(debug_info["browser_receive_queue_ms"])
            + float(debug_info["browser_post_receive_ms"])
        )
        debug_info["accounted_leaf_latency_ms"] = accounted_leaf_latency_ms
        debug_info["unaccounted_leaf_latency_ms"] = (
            float(debug_info["latency_ms"]) - accounted_leaf_latency_ms
        )
        debug_info["unaccounted_leaf_latency_sample_ms"] = (
            float(debug_info["latency_sample_ms"]) - accounted_leaf_latency_ms
        )
        debug_info["accounted_coarse_latency_ms"] = accounted_coarse_latency_ms
        debug_info["unaccounted_coarse_latency_ms"] = (
            float(debug_info["latency_ms"]) - accounted_coarse_latency_ms
        )
        debug_info["unaccounted_coarse_latency_sample_ms"] = (
            float(debug_info["latency_sample_ms"]) - accounted_coarse_latency_ms
        )
        return debug_info

    def get_snapshot(self) -> Image.Image:
        """Return the latest rendered frame as a PIL image."""
        if self._latest_frame_array is None:
            raise RuntimeError("No rendered frame is available yet.")
        return Image.fromarray(self._latest_frame_array.copy(), mode="RGB")

    def _run_on_main_loop(self, callback: Callable[[], None]) -> None:
        """Run a callback on the main asyncio loop when available."""
        if self._main_loop is not None and self._main_loop.is_running():
            self._main_loop.call_soon_threadsafe(callback)
            return
        callback()

    def _on_camera_state_json_change(self, change: dict[str, object]) -> None:
        """Persist the current camera state into the shared state object."""
        new_value = change.get("new")
        if isinstance(new_value, str) and self._state is not None:
            self._state._set_camera_internal(
                CameraState.from_json(new_value),
                push_to_viewer=False,
                notify_listeners=True,
            )

    def _on_last_click_json_change(self, change: dict[str, object]) -> None:
        """Persist the last click into the shared state object."""
        new_value = change.get("new")
        if not isinstance(new_value, str) or self._state is None:
            return
        self._state.last_click = (
            None if not new_value else ViewerClick.from_json(new_value)
        )

    def _on_show_axes_change(self, change: dict[str, object]) -> None:
        """Persist the axis-gizmo visibility into the shared state object."""
        new_value = change.get("new")
        if not isinstance(new_value, bool) or self._state is None:
            return
        self._state._set_show_axes_internal(
            new_value,
            push_to_viewer=False,
            notify_listeners=True,
        )

    def _on_show_horizon_change(self, change: dict[str, object]) -> None:
        """Persist the horizon visibility into the shared state object."""
        new_value = change.get("new")
        if not isinstance(new_value, bool) or self._state is None:
            return
        self._state._set_show_horizon_internal(
            new_value,
            push_to_viewer=False,
            notify_listeners=True,
        )

    def _on_show_origin_change(self, change: dict[str, object]) -> None:
        """Persist the origin visibility into the shared state object."""
        new_value = change.get("new")
        if not isinstance(new_value, bool) or self._state is None:
            return
        self._state._set_show_origin_internal(
            new_value,
            push_to_viewer=False,
            notify_listeners=True,
        )

    def _on_show_stats_change(self, change: dict[str, object]) -> None:
        """Persist the stats visibility into the shared state object."""
        new_value = change.get("new")
        if not isinstance(new_value, bool) or self._state is None:
            return
        self._state._set_show_stats_internal(
            new_value,
            push_to_viewer=False,
            notify_listeners=True,
        )

    def set_camera_state(self, camera_state: CameraState) -> None:
        """Apply a camera state and request a fresh render."""
        self.widget.camera_state_json = camera_state.to_json()
        self.widget.error_text = ""
        self.widget._camera_revision += 1
        self._wait_for_revision(self.widget._camera_revision)

    def set_viewer_rotation(
        self,
        x_degrees: float,
        y_degrees: float,
        z_degrees: float,
    ) -> None:
        """Update the viewer-frame rotation used by controls and overlays."""
        self.widget.viewer_rotation_x_degrees = x_degrees
        self.widget.viewer_rotation_y_degrees = y_degrees
        self.widget.viewer_rotation_z_degrees = z_degrees
        self.widget.send_state(
            [
                "viewer_rotation_x_degrees",
                "viewer_rotation_y_degrees",
                "viewer_rotation_z_degrees",
            ]
        )

    def set_show_axes(self, show_axes: bool) -> None:
        """Update axis-gizmo visibility in the live viewer."""
        self.widget.show_axes = show_axes
        self.widget.send_state("show_axes")

    def set_show_horizon(self, show_horizon: bool) -> None:
        """Update horizon visibility in the live viewer."""
        self.widget.show_horizon = show_horizon
        self.widget.send_state("show_horizon")

    def set_show_origin(self, show_origin: bool) -> None:
        """Update origin visibility in the live viewer."""
        self.widget.show_origin = show_origin
        self.widget.send_state("show_origin")

    def set_show_stats(self, show_stats: bool) -> None:
        """Update stats visibility in the live viewer."""
        self.widget.show_stats = show_stats
        self.widget.send_state("show_stats")

    def set_origin(self, x: float, y: float, z: float) -> None:
        """Update the origin marker position in the live viewer."""
        self.widget.origin_x = x
        self.widget.origin_y = y
        self.widget.origin_z = z
        self.widget.send_state(["origin_x", "origin_y", "origin_z"])

    def set_keyboard_navigation(
        self,
        move_speed: float,
        sprint_multiplier: float,
    ) -> None:
        """Update keyboard navigation tuning in the live viewer."""
        self.widget.keyboard_move_speed = move_speed
        self.widget.keyboard_sprint_multiplier = sprint_multiplier
        self.widget.send_state(
            ["keyboard_move_speed", "keyboard_sprint_multiplier"]
        )

    def set_pointer_controls(
        self,
        orbit_invert_x: bool,
        orbit_invert_y: bool,
        pan_invert_x: bool,
        pan_invert_y: bool,
    ) -> None:
        """Update pointer interaction inversion controls in the live viewer."""
        self.widget.orbit_invert_x = orbit_invert_x
        self.widget.orbit_invert_y = orbit_invert_y
        self.widget.pan_invert_x = pan_invert_x
        self.widget.pan_invert_y = pan_invert_y
        self.widget.send_state(
            [
                "orbit_invert_x",
                "orbit_invert_y",
                "pan_invert_x",
                "pan_invert_y",
            ]
        )

    def rerender(self, *, interactive: bool = False) -> None:
        """Request a fresh render without changing the camera pose.

        When `interactive` is True the render uses interactive quality and
        scale, and a settled render is automatically scheduled afterward.
        """
        self.widget.error_text = ""
        if interactive:
            self.widget.interaction_active = True
        self.widget._camera_revision += 1
        self._wait_for_revision(self.widget._camera_revision)

    def _camera_state_with_max_side(
        self, camera_state: CameraState, max_side: int | None
    ) -> CameraState:
        """Return a camera state with its larger axis capped to ``max_side``."""
        if max_side is None:
            return camera_state
        larger_axis = max(camera_state.width, camera_state.height)
        if larger_axis <= max_side:
            return camera_state
        downscale = max_side / larger_axis
        scaled_width = max(1, round(camera_state.width * downscale))
        scaled_height = max(1, round(camera_state.height * downscale))
        if (
            scaled_width == camera_state.width
            and scaled_height == camera_state.height
        ):
            return camera_state
        return camera_state.with_size(scaled_width, scaled_height)

    def _render_camera_state(
        self, camera_state: CameraState, *, interaction_active: bool
    ) -> CameraState:
        """Return the effective render camera state for the current mode."""
        effective_camera_state = self._camera_state_with_max_side(
            camera_state,
            self._internal_render_max_side,
        )
        if interaction_active:
            effective_camera_state = self._camera_state_with_max_side(
                effective_camera_state,
                self._interactive_max_side,
            )
        return effective_camera_state

    def _on_camera_revision_change(self, change: dict[str, object]) -> None:
        del change
        camera_state = self._render_camera_state(
            CameraState.from_json(self.widget.camera_state_json),
            interaction_active=self.widget.interaction_active,
        )
        self._renderer.request(
            self.widget._camera_revision,
            camera_state,
            self.widget.interaction_active,
        )

    def _complete_revision(
        self, revision: int, error: Exception | None = None
    ) -> None:
        """Record the completion state for a render revision."""
        with self._render_completion_condition:
            self._completed_revisions[revision] = error
            self._render_completion_condition.notify_all()

    def _wait_for_revision(self, revision: int) -> None:
        """Block until the requested revision completes and re-raise errors."""
        if not self._raise_on_error:
            return
        with self._render_completion_condition:
            while revision not in self._completed_revisions:
                self._render_completion_condition.wait()
            error = self._completed_revisions.pop(revision)
        if error is not None:
            raise error

    def _publish_frame(
        self,
        revision: int,
        camera_state: CameraState,
        frame: np.ndarray,
        render_queue_time_ms: float,
        render_time_ms: float,
        interaction_active: bool,
    ) -> None:
        encode_started_at = time.perf_counter()
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if interaction_active:
            success, encoded = cv2.imencode(
                ".jpg",
                bgr_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._interactive_quality],
            )
            if not success:
                raise RuntimeError("Failed to encode rendered frame as JPEG.")
            mime_type = "image/jpeg"
        else:
            match self._settled_quality:
                case "png":
                    success, encoded = cv2.imencode(
                        ".png", bgr_frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
                    )
                    if not success:
                        raise RuntimeError(
                            "Failed to encode rendered frame as PNG."
                        )
                    mime_type = "image/png"
                case "jpeg_95":
                    success, encoded = cv2.imencode(
                        ".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    )
                    if not success:
                        raise RuntimeError(
                            "Failed to encode rendered frame as JPEG."
                        )
                    mime_type = "image/jpeg"
                case "jpeg_100":
                    success, encoded = cv2.imencode(
                        ".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                    )
                    if not success:
                        raise RuntimeError(
                            "Failed to encode rendered frame as JPEG."
                        )
                    mime_type = "image/jpeg"
        encode_time_ms = (time.perf_counter() - encode_started_at) * 1000.0
        encoded_bytes = encoded.tobytes()
        frame_width = int(frame.shape[1])
        frame_height = int(frame.shape[0])
        next_camera_state_json = None
        if not interaction_active and (
            camera_state.width != frame_width
            or camera_state.height != frame_height
        ):
            next_camera_state_json = camera_state.with_size(
                frame_width, frame_height
            ).to_json()

        self._latest_frame_array = frame.copy()
        now = time.perf_counter()
        self._render_frame_timestamps.append(now)
        cutoff = now - 1.0
        while (
            self._render_frame_timestamps
            and self._render_frame_timestamps[0] < cutoff
        ):
            self._render_frame_timestamps.pop(0)
        render_fps = float(len(self._render_frame_timestamps))
        packet = _pack_frame_packet(
            {
                "type": "frame",
                "mime_type": mime_type,
                "width": frame_width,
                "height": frame_height,
                "revision": revision,
                "render_time_ms": render_time_ms,
                "backend_frame_sent_perf_time_ms": (
                    time.perf_counter() * 1000.0
                ),
                "interaction_active": interaction_active,
            },
            encoded_bytes,
        )
        transport_mode = self.widget.transport_mode
        broadcast_future = None
        if transport_mode == "websocket":
            broadcast_future = self._stream_server.publish(
                self._stream_path.removeprefix("/streams/"),
                packet,
                scheduled_at=time.perf_counter(),
            )

        if broadcast_future is not None:

            def _on_broadcast_done(
                future: concurrent.futures.Future[dict[str, float]],
            ) -> None:
                try:
                    timings = future.result()
                except Exception:
                    return

                def _apply_stream_timing_update() -> None:
                    if not interaction_active:
                        return
                    smoothed_metrics = self._update_smoothed_debug_metrics(
                        render_queue_time_ms=render_queue_time_ms,
                        render_time_ms=render_time_ms,
                        encode_time_ms=encode_time_ms,
                        stream_queue_time_ms=timings["stream_queue_time_ms"],
                        stream_send_time_ms=timings["stream_send_time_ms"],
                    )
                    self.widget.render_queue_time_ms = smoothed_metrics[
                        "render_queue_time_ms"
                    ]
                    self.widget.render_time_ms = smoothed_metrics[
                        "render_time_ms"
                    ]
                    self.widget.encode_time_ms = smoothed_metrics[
                        "encode_time_ms"
                    ]
                    self.widget.stream_queue_time_ms = smoothed_metrics[
                        "stream_queue_time_ms"
                    ]
                    self.widget.stream_send_time_ms = smoothed_metrics[
                        "stream_send_time_ms"
                    ]
                    self.widget.render_fps = render_fps
                    self.widget.send_state(
                        [
                            "render_queue_time_ms",
                            "render_time_ms",
                            "encode_time_ms",
                            "stream_queue_time_ms",
                            "stream_send_time_ms",
                            "render_fps",
                        ]
                    )

                self._run_on_main_loop(_apply_stream_timing_update)

            broadcast_future.add_done_callback(_on_broadcast_done)

        def _apply_trait_updates() -> None:
            self.widget.error_text = ""
            self.widget.send_state("error_text")
            if transport_mode == "comm":
                self.widget.frame_packet = packet
                self.widget.send_state("frame_packet")
            self.widget.render_revision = revision
            self.widget.send_state("render_revision")
            if next_camera_state_json is not None:
                self.widget.camera_state_json = next_camera_state_json
                self.widget.send_state("camera_state_json")

        self._run_on_main_loop(_apply_trait_updates)
        self._complete_revision(revision)

    def _publish_error(
        self, revision: int, error: Exception, message: str
    ) -> None:
        def _apply_error_update() -> None:
            self.widget.error_text = message
            self.widget.send_state("error_text")

        self._run_on_main_loop(_apply_error_update)
        self._complete_revision(revision, error)

    def _set_rendering(self, value: bool) -> None:
        def _apply_rendering_update() -> None:
            self.widget.is_rendering = value
            self.widget.send_state("is_rendering")

        self._run_on_main_loop(_apply_rendering_update)


def marimo_viewer(
    render_fn: Callable[[CameraState], np.ndarray | torch.Tensor],
    *,
    aspect_ratio: float | None = None,
    interactive_quality: int | None = None,
    settled_quality: Literal["jpeg_95", "jpeg_100", "png"] | None = None,
    internal_render_max_side: int | None = None,
    interactive_max_side: int | None = None,
    transport_mode: ViewerTransportMode | None = None,
    camera_convention: CameraConvention | None = None,
    initial_view: CameraState | None = None,
    state: ViewerState | None = None,
    raise_on_error: bool | None = None,
) -> MarimoViewer:
    """Create a marimo-backed image-based 3D viewer.

    The render size comes from the measured notebook layout. `aspect_ratio`
    controls the initial widget height before the first render and resize
    measurement. `initial_view` sets the initial camera pose, FOV, convention,
    and nominal viewport size before the widget measures the live layout.
    `camera_convention` controls the default convention when no explicit camera
    state is provided via `initial_view` or `state`. Axis-gizmo visibility is
    stored in `ViewerState.show_axes`. Reuse the same `state` object
    across reruns to persist the camera state and last click even when
    `render_fn` is recreated. `internal_render_max_side` caps the larger image
    axis for all renders while preserving the live widget aspect ratio.
    `interactive_max_side` applies an additional motion-only cap on top of
    that; `None` disables motion downscaling. Interactive render rate is
    limited by the browser's pointer event frequency (typically 20-60 fps);
    use `rerender(interactive=True)` to drive rendering from Python instead.
    When `raise_on_error` is `True`, Python-triggered renders re-raise render
    exceptions instead of only surfacing them in widget state.
    `transport_mode="comm"` streams frames through marimo's existing widget
    channel and works through ordinary SSH port forwarding. Use
    `transport_mode="websocket"` for the older direct local websocket stream.

    The returned widget exposes:

    - `get_camera_state()` for the current synced view
    - `get_last_click()` for the last primary-button click
    - `get_snapshot()` for the latest rendered frame as a PIL image
    - `.value[...]` for direct access to synced anywidget traits

    Clicks are only registered for primary-button press/release interactions
    that stay below the drag threshold; orbiting and panning do not register as
    clicks.
    """
    if state is None:
        state = ViewerState(
            camera_state=initial_view,
            camera_convention=camera_convention or "opencv",
            aspect_ratio=aspect_ratio
            if aspect_ratio is not None
            else 16.3 / 9.0,
            interactive_quality=(
                interactive_quality if interactive_quality is not None else 50
            ),
            settled_quality=(
                settled_quality if settled_quality is not None else "jpeg_100"
            ),
            internal_render_max_side=(
                internal_render_max_side
                if internal_render_max_side is not None
                else 2560
            ),
            interactive_max_side=(
                interactive_max_side
                if interactive_max_side is not None
                else 1980
            ),
            raise_on_error=(
                raise_on_error if raise_on_error is not None else True
            ),
            transport_mode=(
                transport_mode if transport_mode is not None else "comm"
            ),
        )
    elif initial_view is not None:
        state.set_camera(initial_view)
    if transport_mode is not None:
        state.transport_mode = transport_mode

    existing_viewer = (
        None
        if state._active_marimo_viewer_ref is None
        else state._active_marimo_viewer_ref()
    )
    if existing_viewer is not None:
        existing_viewer.close()

    resolved_camera_state = state.camera_state
    stream_server = _get_frame_stream_server()
    stream_id, stream_token = stream_server.register_stream()
    anywidget_instance = _NativeViewerAnyWidget(
        camera_state=resolved_camera_state,
        aspect_ratio=state.aspect_ratio,
        show_axes=state.show_axes,
        show_horizon=state.show_horizon,
        show_origin=state.show_origin,
        show_stats=state.show_stats,
        viewer_rotation_x_degrees=state.viewer_rotation_x_degrees,
        viewer_rotation_y_degrees=state.viewer_rotation_y_degrees,
        viewer_rotation_z_degrees=state.viewer_rotation_z_degrees,
        origin_x=state.origin[0],
        origin_y=state.origin[1],
        origin_z=state.origin[2],
        keyboard_move_speed=state.keyboard_move_speed,
        keyboard_sprint_multiplier=state.keyboard_sprint_multiplier,
        orbit_invert_x=state.orbit_invert_x,
        orbit_invert_y=state.orbit_invert_y,
        pan_invert_x=state.pan_invert_x,
        pan_invert_y=state.pan_invert_y,
        stream_port=stream_server.port,
        stream_path=f"/streams/{stream_id}",
        stream_token=stream_token,
        transport_mode=state.transport_mode,
    )
    if state.last_click is not None:
        anywidget_instance.last_click_json = state.last_click.to_json()
    viewer = MarimoViewer(
        anywidget_instance,
        render_fn=render_fn,
        interactive_quality=state.interactive_quality,
        settled_quality=state.settled_quality,
        internal_render_max_side=state.internal_render_max_side,
        interactive_max_side=state.interactive_max_side,
        state=state,
        raise_on_error=state.raise_on_error,
    )
    state._active_marimo_viewer_ref = weakref.ref(viewer)
    return viewer


NativeViewerState = ViewerState
NativeViewerWidget = MarimoViewer
native_viewer = marimo_viewer
