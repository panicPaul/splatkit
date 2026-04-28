"""Camera-centric viewer bridge helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ember_core.core.contracts import CameraState
from ember_core.viewer.contracts import ViewerMode, ViewerState


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


def _to_native_viewer_state(state: ViewerState) -> Any:
    """Build a marimo-3dv viewer state lazily."""
    from marimo_3dv.viewer.widget import ViewerState as NativeViewerState

    payload = camera_to_viewer_payload(state.camera)
    aspect_ratio = payload.width / payload.height
    return NativeViewerState(
        camera_state=_to_native_camera(state.camera),
        camera_convention=payload.camera_convention,
        aspect_ratio=aspect_ratio,
    )


def launch_viewer(
    render_fn: Callable[[CameraState], Any],
    *,
    state: ViewerState,
    controls: Any | None = None,
) -> Any:
    """Launch the configured viewer backend using ember-core camera states."""
    import marimo as mo
    from marimo_3dv.viewer import Viewer as NativeViewer

    if not resolve_viewer_mode(
        state.viewer_mode,
        running_in_notebook=mo.running_in_notebook(),
    ):
        return None

    native_state = _to_native_viewer_state(state)

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
