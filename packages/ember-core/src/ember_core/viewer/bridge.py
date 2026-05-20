"""Camera-centric viewer bridge helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ember_core.core.contracts import CameraState
from ember_core.viewer.contracts import (
    Marimo3DVViewerConfig,
    ViewerBackend,
    ViewerMode,
    ViewerState,
)


@dataclass(frozen=True)
class ViewerCameraPayload:
    """Single-camera payload suitable for viewer runtimes."""

    width: int
    height: int
    fov_degrees: float
    cam_to_world: np.ndarray
    camera_convention: str


def select_viewer_camera(
    camera: CameraState,
    *,
    index: int = 0,
) -> CameraState:
    """Select one camera from a batched camera state."""
    num_cams = int(camera.width.shape[0])
    if not 0 <= index < num_cams:
        raise IndexError(
            f"Viewer camera index {index} is out of range for {num_cams} cameras."
        )
    return CameraState(
        width=camera.width[index : index + 1],
        height=camera.height[index : index + 1],
        fov_degrees=camera.fov_degrees[index : index + 1],
        cam_to_world=camera.cam_to_world[index : index + 1],
        intrinsics=(
            None
            if camera.intrinsics is None
            else camera.intrinsics[index : index + 1]
        ),
        camera_convention=camera.camera_convention,
        up_direction=camera.up_direction,
    )


def camera_to_viewer_payload(
    camera: CameraState,
    *,
    index: int = 0,
) -> ViewerCameraPayload:
    """Convert a batched ember-core camera to a single-camera viewer payload."""
    selected = select_viewer_camera(camera, index=index)
    return ViewerCameraPayload(
        width=int(selected.width[0].item()),
        height=int(selected.height[0].item()),
        fov_degrees=float(selected.fov_degrees[0].item()),
        cam_to_world=selected.cam_to_world[0].detach().cpu().numpy(),
        camera_convention=selected.camera_convention,
    )


def camera_from_viewer_payload(
    payload: ViewerCameraPayload,
) -> CameraState:
    """Convert a viewer payload back into a batched ember-core camera."""
    return CameraState(
        width=torch.tensor([payload.width], dtype=torch.int64),
        height=torch.tensor([payload.height], dtype=torch.int64),
        fov_degrees=torch.tensor([payload.fov_degrees], dtype=torch.float32),
        cam_to_world=torch.from_numpy(payload.cam_to_world).to(
            dtype=torch.float32
        )[None],
        camera_convention=payload.camera_convention,
    )


def resolve_viewer_mode(
    mode: ViewerMode,
    *,
    running_in_notebook: bool,
) -> bool:
    """Return whether a viewer should launch in the current runtime."""
    if mode == "force_on":
        return True
    if mode == "force_off":
        return False
    return running_in_notebook


def _to_native_camera(camera: CameraState) -> Any:
    """Build a marimo-3dv camera state lazily."""
    from marimo_3dv.viewer.widget import CameraState as NativeCameraState

    payload = camera_to_viewer_payload(camera)
    return NativeCameraState(
        width=payload.width,
        height=payload.height,
        fov_degrees=payload.fov_degrees,
        cam_to_world=payload.cam_to_world,
        camera_convention=payload.camera_convention,
    )


def _to_native_viewer_state(
    state: ViewerState,
    *,
    config: Marimo3DVViewerConfig | None = None,
) -> Any:
    """Build a marimo-3dv viewer state lazily."""
    from marimo_3dv.viewer.widget import ViewerState as NativeViewerState

    payload = camera_to_viewer_payload(state.camera)
    aspect_ratio = payload.width / payload.height
    kwargs: dict[str, Any] = {
        "camera_state": _to_native_camera(state.camera),
        "camera_convention": payload.camera_convention,
        "aspect_ratio": aspect_ratio,
    }
    if config is not None:
        if config.interactive_quality is not None:
            kwargs["interactive_quality"] = config.interactive_quality
        if config.settled_quality is not None:
            kwargs["settled_quality"] = config.settled_quality
        if config.internal_render_max_side is not None:
            kwargs["internal_render_max_side"] = (
                config.internal_render_max_side
            )
        if config.interactive_max_side is not None:
            kwargs["interactive_max_side"] = config.interactive_max_side
        if config.interactive_backpressure is not None:
            kwargs["interactive_backpressure"] = (
                config.interactive_backpressure
            )
        if config.interactive_max_fps is not None:
            kwargs["interactive_max_fps"] = config.interactive_max_fps
        if config.interactive_min_fps is not None:
            kwargs["interactive_min_fps"] = config.interactive_min_fps
        if config.interactive_latency_target_ms is not None:
            kwargs["interactive_latency_target_ms"] = (
                config.interactive_latency_target_ms
            )
        if config.interactive_probe_interval_s is not None:
            kwargs["interactive_probe_interval_s"] = (
                config.interactive_probe_interval_s
            )
        if config.interactive_reset_interval_s is not None:
            kwargs["interactive_reset_interval_s"] = (
                config.interactive_reset_interval_s
            )
        if config.transport_mode is not None:
            kwargs["transport_mode"] = config.transport_mode
    return NativeViewerState(**kwargs)


def launch_viewer(
    render_fn: Callable[[CameraState], Any],
    *,
    state: ViewerState,
    controls: Any | None = None,
    backend: ViewerBackend = "marimo_3dv",
    marimo_3dv_config: Marimo3DVViewerConfig | None = None,
    viser_server_config: Any | None = None,
    viser_render_config: Any | None = None,
) -> Any:
    """Launch the configured viewer backend using ember-core camera states."""
    import marimo as mo

    if not resolve_viewer_mode(
        state.viewer_mode,
        running_in_notebook=mo.running_in_notebook(),
    ):
        return None

    if backend == "viser":
        from marimo_viser import ViserViewer, ViserViewerState

        selected_camera = select_viewer_camera(state.camera)

        def viser_render_fn(
            camera: CameraState,
            _render_state: Any,
        ) -> Any:
            return render_fn(camera)

        return ViserViewer(
            viser_render_fn,
            state=ViserViewerState(
                camera=selected_camera,
                camera_convention=selected_camera.camera_convention,
                training_active=state.training_active,
                interaction_active=state.interaction_active,
            ),
            server_config=viser_server_config,
            render_config=viser_render_config,
            mode="rendering",
            title=state.title,
        )

    if backend != "marimo_3dv":
        raise ValueError(f"Unsupported viewer backend {backend!r}.")

    from marimo_3dv.viewer import Viewer as NativeViewer

    native_state = _to_native_viewer_state(
        state,
        config=marimo_3dv_config,
    )

    def native_render_fn(native_camera: Any) -> Any:
        payload = ViewerCameraPayload(
            width=int(native_camera.width),
            height=int(native_camera.height),
            fov_degrees=float(native_camera.fov_degrees),
            cam_to_world=np.asarray(
                native_camera.cam_to_world, dtype=np.float64
            ),
            camera_convention=native_camera.camera_convention,
        )
        core_camera = camera_from_viewer_payload(payload)
        return render_fn(core_camera)

    return NativeViewer(
        native_render_fn,
        state=native_state,
        title=state.title,
        controls=controls,
    )
