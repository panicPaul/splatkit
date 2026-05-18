"""Viser-backed marimo viewer runtime built on Ember camera contracts."""

from __future__ import annotations

import contextlib
import math
import socket
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import marimo as mo
import numpy as np
import torch
from ember_core.core.contracts import CameraConvention, CameraState
from jaxtyping import Float, UInt8
from marimo_config_gui import create_config_gui
from pydantic import BaseModel, Field

RenderMode: TypeAlias = Literal["rendering", "training"]
ViewerStatus: TypeAlias = Literal[
    "preparing",
    "rendering",
    "training",
    "paused",
    "completed",
    "closed",
]
RenderAction: TypeAlias = Literal["move", "static", "rerender", "update"]


class ViserServerConfig(BaseModel):
    """Connection settings for a Viser server shown inside marimo."""

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8080, ge=0, le=65535)
    iframe_url: str | None = Field(default=None)
    public_url: str | None = Field(default=None)
    iframe_height: str = Field(default="640px")
    ssh_host: str | None = Field(default=None)
    ssh_user: str | None = Field(default=None)
    ssh_port: int = Field(default=22, ge=1, le=65535)
    local_forward_host: str = Field(default="127.0.0.1")
    local_forward_port: int | None = Field(default=None, ge=1, le=65535)
    verbose: bool = Field(default=False)


class ViserRenderConfig(BaseModel):
    """Render scheduling and image quality settings for the Viser runtime."""

    viewer_res: int = Field(default=1024, ge=64, le=8192)
    render_width: int = Field(default=1920, ge=1)
    render_height: int = Field(default=1080, ge=1)
    move_jpeg_quality: int = Field(default=40, ge=1, le=100)
    static_jpeg_quality: int = Field(default=70, ge=1, le=100)
    settle_seconds: float = Field(default=0.2, ge=0.0)
    move_pause_seconds: float = Field(default=0.1, ge=0.0)
    train_util: float = Field(default=0.9, ge=0.0, le=1.0)


class ViserControlsConfig(BaseModel):
    """Combined server and render controls for marimo-config-gui."""

    server: ViserServerConfig = Field(default_factory=ViserServerConfig)
    render: ViserRenderConfig = Field(default_factory=ViserRenderConfig)


@dataclass(frozen=True)
class ViserConnectionInfo:
    """URLs and forwarding command for a live or planned Viser server."""

    host: str
    port: int
    url: str
    iframe_url: str
    ssh_forward_command: str
    public_url: str | None = None


@dataclass
class ViserRenderState:
    """Mutable render state passed to user render functions."""

    viewer_res: int = 1024
    viewer_width: int = 1024
    viewer_height: int = 768
    render_width: int = 1920
    render_height: int = 1080
    preview_render: bool = False
    num_view_rays_per_sec: float | None = None
    num_train_rays_per_sec: float | None = None
    total_primitive_count: int | None = None
    rendered_primitive_count: int | None = None
    move_jpeg_quality: int = 40
    static_jpeg_quality: int = 70

    @classmethod
    def from_config(cls, config: ViserRenderConfig) -> ViserRenderState:
        """Create mutable render state from static notebook config."""
        return cls(
            viewer_res=config.viewer_res,
            viewer_width=config.viewer_res,
            viewer_height=round(config.viewer_res * 3 / 4),
            render_width=config.render_width,
            render_height=config.render_height,
            move_jpeg_quality=config.move_jpeg_quality,
            static_jpeg_quality=config.static_jpeg_quality,
        )


@dataclass
class ViserViewerState:
    """State that survives marimo cell reruns."""

    camera: CameraState | None = None
    camera_convention: CameraConvention = "opencv"
    server_port: int | None = None
    status: ViewerStatus = "preparing"
    step: int = 0
    training_active: bool = False
    interaction_active: bool = False
    latest_error: str | None = None


@dataclass(frozen=True)
class ViserControlsHandle:
    """Notebook-ready controls block for the Viser viewer."""

    config_model: type[ViserControlsConfig]
    default_config: ViserControlsConfig
    gui: Any
    config_bindings: Any | None = None

    @property
    def value(self) -> ViserControlsConfig:
        """Return the latest validated controls value."""
        if self.config_bindings is not None:
            if not self.config_bindings.is_valid():
                return self.default_config
            return self.config_bindings.validated_config()
        value = getattr(self.gui, "value", None)
        if value is None:
            return self.default_config
        return value


@dataclass(frozen=True)
class _StaticControls:
    value: Any


@dataclass
class _RenderTask:
    action: RenderAction
    camera: CameraState
    revision: int


def _find_free_port(start: int = 8080, attempts: int = 128) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"Could not find a free port in range {start}-{start + attempts}."
    )


def _url_host(host: str) -> str:
    if host in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def connection_info(
    config: ViserServerConfig,
    *,
    port: int,
) -> ViserConnectionInfo:
    """Build URLs and SSH-forwarding command for a Viser server."""
    display_host = _url_host(config.host)
    local_port = config.local_forward_port or port
    url = f"http://{display_host}:{port}"
    iframe_url = config.iframe_url or config.public_url or url
    target = config.ssh_host or "<remote-host>"
    if config.ssh_user:
        target = f"{config.ssh_user}@{target}"
    ssh_port_part = "" if config.ssh_port == 22 else f" -p {config.ssh_port}"
    ssh_forward_command = (
        "ssh -N"
        f"{ssh_port_part} -L "
        f"{config.local_forward_host}:{local_port}:{display_host}:{port} "
        f"{target}"
    )
    return ViserConnectionInfo(
        host=display_host,
        port=port,
        url=url,
        iframe_url=iframe_url,
        public_url=config.public_url,
        ssh_forward_command=ssh_forward_command,
    )


def _quaternion_wxyz_to_matrix(
    wxyz: Float[np.ndarray, " 4"],
) -> Float[np.ndarray, "4 4"]:
    w, x, y, z = np.asarray(wxyz, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        rotation = np.eye(3, dtype=np.float64)
    else:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        rotation = np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _matrix_to_quaternion_wxyz(
    rotation: Float[np.ndarray, "3 3"],
) -> Float[np.ndarray, " 4"]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )

    largest = int(np.argmax(np.diag(matrix)))
    if largest == 0:
        scale = (
            math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        )
        return np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    if largest == 1:
        scale = (
            math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        )
        return np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ],
            dtype=np.float64,
        )
    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return np.array(
        [
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
        ],
        dtype=np.float64,
    )


def _intrinsics_from_viser(
    *,
    width: int,
    height: int,
    vertical_fov_radians: float,
) -> tuple[float, Float[np.ndarray, "3 3"]]:
    focal = height / (2.0 * math.tan(vertical_fov_radians / 2.0))
    horizontal_fov = 2.0 * math.atan(width / (2.0 * focal))
    intrinsics = np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return math.degrees(horizontal_fov), intrinsics


def _camera_state_from_viser(
    camera: Any,
    *,
    width: int,
    height: int,
    camera_convention: CameraConvention = "opencv",
) -> CameraState:
    """Convert a Viser client camera handle into Ember camera state."""
    transform = _quaternion_wxyz_to_matrix(
        np.asarray(camera.wxyz, dtype=np.float64)
    )
    transform[:3, 3] = np.asarray(camera.position, dtype=np.float64)
    fov_degrees, intrinsics = _intrinsics_from_viser(
        width=width,
        height=height,
        vertical_fov_radians=float(camera.fov),
    )
    return CameraState(
        width=torch.tensor([width], dtype=torch.int64),
        height=torch.tensor([height], dtype=torch.int64),
        fov_degrees=torch.tensor([fov_degrees], dtype=torch.float32),
        cam_to_world=torch.from_numpy(transform.astype(np.float32))[None],
        intrinsics=torch.from_numpy(intrinsics)[None],
        camera_convention=camera_convention,
    )


def _core_camera_to_viser_pose(
    camera: CameraState,
) -> tuple[Float[np.ndarray, " 4"], Float[np.ndarray, " 3"], float]:
    """Return ``(wxyz, position, vertical_fov_radians)`` for a core camera."""
    selected = camera
    matrix = selected.cam_to_world[0].detach().cpu().numpy().astype(np.float64)
    intrinsics = selected.get_intrinsics()[0].detach().cpu().numpy()
    height = float(selected.height[0].item())
    focal_y = float(intrinsics[1, 1])
    vertical_fov = 2.0 * math.atan(height / (2.0 * focal_y))
    return (
        _matrix_to_quaternion_wxyz(matrix[:3, :3]),
        matrix[:3, 3].copy(),
        vertical_fov,
    )


def _img_wh_for_aspect(
    viewer_res: int,
    aspect: float,
) -> tuple[int, int]:
    safe_aspect = max(float(aspect), 1e-6)
    if safe_aspect >= 1.0:
        width = viewer_res
        height = max(1, round(viewer_res / safe_aspect))
    else:
        height = viewer_res
        width = max(1, round(viewer_res * safe_aspect))
    return width, height


def _numpy_image(value: Any) -> UInt8[np.ndarray, " height width 3"]:
    is_torch_tensor = isinstance(value, torch.Tensor)
    if is_torch_tensor:
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if (
        array.ndim == 3
        and array.shape[0] in {1, 3, 4}
        and (is_torch_tensor or array.shape[-1] not in {1, 3, 4})
    ):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(
            f"Rendered images must be 2D or 3D, got shape {array.shape}."
        )
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype.kind == "f":
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def _numpy_depth(value: Any) -> Float[np.ndarray, " height width"] | None:
    if value is None:
        return None
    array = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Depth images must be 2D, got shape {array.shape}.")
    return array.astype(np.float32, copy=False)


class _ViserRenderer(threading.Thread):
    """Per-client latest-only renderer for Viser background images."""

    def __init__(
        self,
        *,
        viewer: ViserViewer,
        client: Any,
        lock: threading.Lock,
    ) -> None:
        super().__init__(daemon=True)
        self.viewer = viewer
        self.client = client
        self.lock = lock
        self.running = True
        self._event = threading.Event()
        self._task: _RenderTask | None = None
        self._revision = 0
        self._last_static_at = 0.0

    def submit(self, action: RenderAction, camera: CameraState) -> None:
        """Replace the pending render task with the latest request."""
        self._revision += 1
        self._task = _RenderTask(
            action=action,
            camera=camera,
            revision=self._revision,
        )
        self._event.set()

    def run(self) -> None:
        """Render background images until the client disconnects."""
        while self.running:
            if not self._event.wait(
                timeout=self.viewer.render_config.settle_seconds
            ):
                if (
                    time.monotonic() - self._last_static_at
                    < self.viewer.render_config.settle_seconds
                ):
                    continue
                with contextlib.suppress(Exception):
                    self.submit(
                        "static", self.viewer.get_camera_state(self.client)
                    )
                continue
            self._event.clear()
            task = self._task
            if task is None:
                continue
            if task.action == "static":
                self._last_static_at = time.monotonic()
            self._render_task(task)

    def _render_task(self, task: _RenderTask) -> None:
        started_at = time.perf_counter()
        try:
            with self.lock:
                rendered = self.viewer.render_fn(
                    task.camera,
                    self.viewer.render_state,
                )
                self.viewer._after_render()
            if task.revision != self._revision:
                return
            image, depth = (
                rendered if isinstance(rendered, tuple) else (rendered, None)
            )
            image_array = _numpy_image(image)
            depth_array = _numpy_depth(depth)
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            width = int(task.camera.width[0].item())
            height = int(task.camera.height[0].item())
            self.viewer.render_state.num_view_rays_per_sec = (
                width * height
            ) / elapsed
            quality = (
                self.viewer.render_state.move_jpeg_quality
                if task.action == "move"
                else self.viewer.render_state.static_jpeg_quality
            )
            kwargs: dict[str, Any] = {
                "format": "jpeg",
                "jpeg_quality": quality,
            }
            if depth_array is not None:
                kwargs["depth"] = depth_array
            try:
                self.client.scene.set_background_image(image_array, **kwargs)
            except TypeError:
                fallback_kwargs: dict[str, Any] = {"format": "jpeg"}
                if depth_array is not None:
                    fallback_kwargs["depth"] = depth_array
                try:
                    self.client.scene.set_background_image(
                        image_array,
                        **fallback_kwargs,
                    )
                except TypeError:
                    self.client.scene.set_background_image(
                        image_array,
                        format="jpeg",
                    )
        except Exception:
            self.viewer.state.latest_error = traceback.format_exc().rstrip()


class ViserViewer:
    """Viser server-backed viewer with Ember camera render functions."""

    def __init__(
        self,
        render_fn: Callable[[CameraState, ViserRenderState], Any],
        *,
        state: ViserViewerState | None = None,
        server_config: ViserServerConfig | None = None,
        render_config: ViserRenderConfig | None = None,
        mode: RenderMode = "rendering",
        server: Any | None = None,
        output_dir: Path | str | None = None,
        title: str = "marimo-viser viewer",
    ) -> None:
        self.render_fn = render_fn
        self.state = state or ViserViewerState()
        self.server_config = server_config or ViserServerConfig()
        self.render_config = render_config or ViserRenderConfig()
        self.render_state = ViserRenderState.from_config(self.render_config)
        self.mode = mode
        self.output_dir = (
            Path(output_dir) if output_dir is not None else Path("results")
        )
        self.title = title
        self.lock = threading.Lock()
        self._renderers: dict[int, _ViserRenderer] = {}
        self._last_move_time = 0.0
        self._last_update_step = 0
        self.server = server if server is not None else self._create_server()
        self.state.status = "training" if mode == "training" else "rendering"
        self._configure_server()

    @property
    def interaction_active(self) -> bool:
        """Return whether any connected client is actively moving."""
        active = (
            time.perf_counter() - self._last_move_time
            < self.render_config.move_pause_seconds
        )
        self.state.interaction_active = active
        return active

    def anywidget(self) -> ViserViewer:
        """Expose a small compatibility shim for existing training hooks."""
        return self

    @property
    def connection_info(self) -> ViserConnectionInfo:
        """Return current URL and tunnel command details."""
        if self.state.server_port is None:
            raise RuntimeError("Viser server port has not been initialized.")
        return connection_info(self.server_config, port=self.state.server_port)

    def embed(self) -> Any:
        """Return a marimo iframe for the Viser server."""
        info = self.connection_info
        return mo.iframe(
            info.iframe_url,
            width="100%",
            height=self.server_config.iframe_height,
        )

    def connection_panel(self) -> Any:
        """Return a compact marimo panel with URLs and an SSH tunnel command."""
        info = self.connection_info
        public_line = (
            f"\n\nPublic URL: [{info.public_url}]({info.public_url})"
            if info.public_url is not None
            else ""
        )
        return mo.md(
            "\n".join(
                [
                    f"Viser URL: [{info.url}]({info.url})",
                    f"Iframe URL: [{info.iframe_url}]({info.iframe_url})",
                    public_line,
                    "",
                    "SSH tunnel:",
                    f"```bash\n{info.ssh_forward_command}\n```",
                ]
            )
        )

    def get_camera_state(self, client: Any) -> CameraState:
        """Return the current Ember camera for a Viser client."""
        width, height = _img_wh_for_aspect(
            self.render_state.viewer_res,
            float(client.camera.aspect),
        )
        self.render_state.viewer_width = width
        self.render_state.viewer_height = height
        camera = _camera_state_from_viser(
            client.camera,
            width=width,
            height=height,
            camera_convention=self.state.camera_convention,
        )
        self.state.camera = camera
        return camera

    def rerender(self, *, wait: bool = False) -> None:
        """Request a fresh render for all connected clients."""
        del wait
        for client_id, client in self.server.get_clients().items():
            if client_id not in self._renderers:
                continue
            self._renderers[client_id].submit(
                "rerender",
                self.get_camera_state(client),
            )

    def update(self, step: int, num_train_rays_per_step: int = 1) -> None:
        """Training-mode update following nerfview's train-util cadence."""
        if self.mode != "training":
            raise ValueError("`update` is only available in training mode.")
        if step < 5:
            return
        self.state.step = step
        while self.interaction_active:
            time.sleep(0.05)
        if not self._should_render_training_update(
            step,
            num_train_rays_per_step=num_train_rays_per_step,
        ):
            return
        self._last_update_step = step
        for client_id, client in self.server.get_clients().items():
            renderer = self._renderers.get(client_id)
            if renderer is None:
                continue
            renderer.submit("update", self.get_camera_state(client))

    def _should_render_training_update(
        self,
        step: int,
        *,
        num_train_rays_per_step: int,
    ) -> bool:
        """Return whether train-util scheduling allows this update render."""
        if not self._renderers:
            return False
        train_util = self.render_config.train_util
        if train_util >= 1.0:
            return False
        train_rays_per_sec = self.render_state.num_train_rays_per_sec
        view_rays_per_sec = self.render_state.num_view_rays_per_sec
        if (
            train_util <= 0.0
            or train_rays_per_sec is None
            or view_rays_per_sec is None
            or train_rays_per_sec <= 0.0
            or view_rays_per_sec <= 0.0
        ):
            return True
        train_rays = max(1, int(num_train_rays_per_step))
        view_rays = max(
            1,
            int(self.render_state.viewer_width)
            * int(self.render_state.viewer_height),
        )
        train_time = train_rays / train_rays_per_sec
        view_time = view_rays / view_rays_per_sec
        denominator = train_time * (1.0 - train_util)
        if denominator <= 0.0:
            return False
        update_every_steps = train_util * view_time / denominator
        return step > self._last_update_step + update_every_steps

    def complete(self) -> None:
        """Mark training as complete and disable training state."""
        self.state.status = "completed"
        self.state.training_active = False

    def close(self) -> None:
        """Stop render threads and close the Viser server if supported."""
        self.state.status = "closed"
        for renderer in list(self._renderers.values()):
            renderer.running = False
            renderer._event.set()
        self._renderers.clear()
        stop = getattr(self.server, "stop", None)
        if stop is not None:
            with contextlib.suppress(Exception):
                stop()

    def _create_server(self) -> Any:
        try:
            import viser
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Creating a Viser viewer requires the `viser` package. "
                "Install `marimo-viser` with its runtime dependencies."
            ) from error
        port = self.server_config.port or _find_free_port()
        self.state.server_port = port
        return viser.ViserServer(
            host=self.server_config.host,
            port=port,
            verbose=self.server_config.verbose,
        )

    def _configure_server(self) -> None:
        if self.state.server_port is None:
            self.state.server_port = int(
                getattr(self.server, "port", self.server_config.port)
            )
        scene = getattr(self.server, "scene", None)
        if scene is not None and hasattr(scene, "set_global_visibility"):
            scene.set_global_visibility(True)
        gui = getattr(self.server, "gui", None)
        if gui is not None:
            with contextlib.suppress(Exception):
                gui.set_panel_label(self.title)
            with contextlib.suppress(Exception):
                gui.configure_theme(
                    control_layout="collapsible",
                    dark_mode=True,
                )
        self.server.on_client_disconnect(self._disconnect_client)
        self.server.on_client_connect(self._connect_client)

    def _connect_client(self, client: Any) -> None:
        client_id = int(client.client_id)
        self._apply_initial_camera(client)
        renderer = _ViserRenderer(viewer=self, client=client, lock=self.lock)
        self._renderers[client_id] = renderer
        renderer.start()

        @client.camera.on_update
        def _(_: Any) -> None:
            self._last_move_time = time.perf_counter()
            self.state.interaction_active = True
            renderer.submit("move", self.get_camera_state(client))

        renderer.submit("rerender", self.get_camera_state(client))

    def _disconnect_client(self, client: Any) -> None:
        client_id = int(client.client_id)
        renderer = self._renderers.pop(client_id, None)
        if renderer is not None:
            renderer.running = False
            renderer._event.set()

    def _apply_initial_camera(self, client: Any) -> None:
        if self.state.camera is None:
            return
        wxyz, position, vertical_fov = _core_camera_to_viser_pose(
            self.state.camera
        )
        atomic = getattr(client, "atomic", None)
        context = atomic() if atomic is not None else contextlib.nullcontext()
        with context:
            client.camera.wxyz = wxyz
            client.camera.position = position
            client.camera.fov = vertical_fov

    def _after_render(self) -> None:
        """Hook called after every successful render_fn call."""


class NoopViserViewer:
    """Non-rendering placeholder used outside notebook viewer runtimes."""

    def __init__(
        self,
        *,
        state: ViserViewerState | None = None,
        server_config: ViserServerConfig | None = None,
    ) -> None:
        self.state = state or ViserViewerState()
        self.server_config = server_config or ViserServerConfig()
        if self.state.server_port is None:
            self.state.server_port = self.server_config.port or 8080

    @property
    def interaction_active(self) -> bool:
        """Return the stored interaction flag."""
        return self.state.interaction_active

    def anywidget(self) -> NoopViserViewer:
        """Expose a compatibility shim for training hooks."""
        return self

    @property
    def connection_info(self) -> ViserConnectionInfo:
        """Return deterministic connection details without launching Viser."""
        assert self.state.server_port is not None
        return connection_info(self.server_config, port=self.state.server_port)

    def embed(self) -> Any:
        """Return an empty marimo output outside live runtimes."""
        return mo.md("")

    def connection_panel(self) -> Any:
        """Return connection details without launching a server."""
        info = self.connection_info
        return mo.md(f"Viser server is not running. Planned URL: `{info.url}`")

    def rerender(self, *, wait: bool = False) -> None:
        """No-op render scheduling method."""
        del wait

    def update(self, step: int, num_train_rays_per_step: int = 1) -> None:
        """Record the latest training step without rendering."""
        del num_train_rays_per_step
        self.state.step = step

    def complete(self) -> None:
        """Mark the placeholder complete."""
        self.state.status = "completed"

    def close(self) -> None:
        """Mark the placeholder closed."""
        self.state.status = "closed"


def apply_viser_config(
    render_state: ViserRenderState,
    config: ViserRenderConfig,
) -> ViserRenderState:
    """Apply render controls to a live render state."""
    render_state.viewer_res = config.viewer_res
    render_state.render_width = config.render_width
    render_state.render_height = config.render_height
    render_state.move_jpeg_quality = config.move_jpeg_quality
    render_state.static_jpeg_quality = config.static_jpeg_quality
    return render_state


def viser_controls_handle(
    *,
    label: str = "",
    default_config: ViserControlsConfig | None = None,
) -> ViserControlsHandle:
    """Build marimo-config-gui controls for Viser server/render settings."""
    resolved_default = default_config or ViserControlsConfig()
    bindings = None
    if mo.running_in_notebook():
        bindings = create_config_gui(
            ViserControlsConfig,
            value=resolved_default,
            background=None,
            label=label,
        )
        gui = bindings.gui_panel()
    else:
        gui = _StaticControls(resolved_default)
    return ViserControlsHandle(
        config_model=ViserControlsConfig,
        default_config=resolved_default,
        gui=gui,
        config_bindings=bindings,
    )


def viser_controls_gui(
    *,
    label: str = "",
    default_config: ViserControlsConfig | None = None,
) -> ViserControlsHandle:
    """Build marimo-config-gui controls for Viser server/render settings."""
    return viser_controls_handle(label=label, default_config=default_config)


def viser_viewer(
    render_fn: Callable[[CameraState, ViserRenderState], Any],
    *,
    state: ViserViewerState | None = None,
    server_config: ViserServerConfig | None = None,
    render_config: ViserRenderConfig | None = None,
    mode: RenderMode = "rendering",
    title: str = "marimo-viser viewer",
    force: bool = False,
) -> ViserViewer | NoopViserViewer:
    """Create a Viser viewer in notebook mode or a no-op placeholder."""
    resolved_state = state or ViserViewerState()
    resolved_server_config = server_config or ViserServerConfig()
    if mo.running_in_notebook() or force:
        return ViserViewer(
            render_fn,
            state=resolved_state,
            server_config=resolved_server_config,
            render_config=render_config,
            mode=mode,
            title=title,
        )
    return NoopViserViewer(
        state=resolved_state,
        server_config=resolved_server_config,
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
    "apply_viser_config",
    "connection_info",
    "viser_controls_gui",
    "viser_controls_handle",
    "viser_viewer",
]
