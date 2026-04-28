from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest
import torch
from marimo._runtime.virtual_file import InMemoryStorage, VirtualFileRegistry
from marimo_config_gui import PydanticGui
from pydantic import BaseModel

import marimo_3dv.viewer.defaults as viewer_defaults
from marimo_3dv import (
    CameraState,
    ViewerClick,
    ViewerControlsConfig,
    ViewerPipeline,
    ViewerState,
    apply_viewer_config,
    apply_viewer_pipeline_config,
    link_viewer_states,
    viewer_controls_config,
    viewer_controls_gui,
    viewer_controls_handle,
    viewer_pipeline_controls_gui,
    viewer_pipeline_controls_handle,
)
from marimo_3dv.viewer.controls import DesktopPydanticControls
from marimo_3dv.viewer.widget import (
    _cleanup_active_marimo_viewers,
    _convert_cam_to_world_between_conventions,
    _FrameStreamServer,
    _FrameStreamState,
    _LatestOnlyRenderer,
    _NativeViewerAnyWidget,
    _normalize_frame,
    _StableMarimoAnyWidget,
    marimo_viewer,
)


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 2.0,
    interval: float = 0.01,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("Timed out waiting for condition.")


def test_camera_state_json_round_trip() -> None:
    state = CameraState.default(
        width=320,
        height=240,
        fov_degrees=75.0,
    )

    restored = CameraState.from_json(state.to_json())

    assert restored.width == 320
    assert restored.height == 240
    assert restored.fov_degrees == 75.0
    assert restored.camera_convention == "opencv"
    assert np.allclose(restored.cam_to_world, state.cam_to_world)


def test_default_camera_state_uses_proper_rotation_matrix() -> None:
    rotation = CameraState.default().cam_to_world[:3, :3]

    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(rotation[:, 1], np.array([0.0, 1.0, 0.0]))


def test_viewer_state_defaults_show_axes_and_stats() -> None:
    state = ViewerState()

    assert state.show_axes is True
    assert state.show_horizon is False
    assert state.show_origin is False
    assert state.show_stats is True


def test_viewer_state_overlay_setters_are_fluent() -> None:
    state = ViewerState()

    chained_state = (
        state.set_show_axes(False)
        .set_show_origin(True)
        .set_show_horizon(True)
        .set_show_stats(True)
        .set_origin(1.0, 2.0, 3.0)
    )

    assert chained_state is state
    assert state.show_axes is False
    assert state.show_origin is True
    assert state.show_horizon is True
    assert state.show_stats is True
    assert state.origin == (1.0, 2.0, 3.0)


def test_viewer_controls_config_reflects_viewer_state() -> None:
    state = ViewerState(
        camera_state=CameraState.default(fov_degrees=75.0),
        interactive_quality=70,
        settled_quality="png",
        internal_render_max_side=2048,
        interactive_max_side=1024,
        show_axes=False,
        show_horizon=True,
        show_origin=True,
        show_stats=True,
    )
    state.set_pointer_controls(
        True,
        False,
        True,
        False,
    ).set_viewer_rotation(10.0, 20.0, 30.0).set_origin(1.0, 2.0, 3.0)

    config = viewer_controls_config(state)

    assert config.camera.fov_degrees == 75.0
    assert config.overlays.show_axes is False
    assert config.overlays.show_horizon is True
    assert config.overlays.show_origin is True
    assert config.overlays.show_stats is True
    assert config.render.interactive_quality == 70
    assert config.render.settled_quality == "png"
    assert config.render.interactive_max_side == 1024
    assert config.render.internal_render_max_side == 2048
    assert config.navigation.move_speed == 0.125
    assert config.navigation.sprint_multiplier == 4.0
    assert config.interaction.orbit_invert_x is True
    assert config.interaction.orbit_invert_y is False
    assert config.interaction.pan_invert_x is True
    assert config.interaction.pan_invert_y is False
    assert config.transform.rotation.x_degrees == 10.0
    assert config.transform.rotation.y_degrees == 20.0
    assert config.transform.rotation.z_degrees == 30.0
    assert config.transform.origin.x == 1.0
    assert config.transform.origin.y == 2.0
    assert config.transform.origin.z == 3.0


def test_apply_viewer_config_updates_viewer_state() -> None:
    state = ViewerState()
    config = ViewerControlsConfig(
        camera={"fov_degrees": 72.0},
        overlays={
            "show_axes": False,
            "show_horizon": True,
            "show_origin": True,
            "show_stats": True,
        },
        render={
            "interactive_quality": 80,
            "settled_quality": "png",
            "interactive_max_side": 900,
            "internal_render_max_side": 1800,
        },
        navigation={"move_speed": 0.3, "sprint_multiplier": 6.0},
        interaction={
            "orbit_invert_x": True,
            "orbit_invert_y": True,
            "pan_invert_x": False,
            "pan_invert_y": True,
        },
        transform={
            "rotation": {
                "x_degrees": 15.0,
                "y_degrees": -25.0,
                "z_degrees": 45.0,
            },
            "origin": {"x": 4.0, "y": 5.0, "z": 6.0},
        },
    )

    returned = apply_viewer_config(state, config)

    assert returned is state
    assert state.camera_state.fov_degrees == 72.0
    assert state.initial_camera_state.fov_degrees == 72.0
    assert state.show_axes is False
    assert state.show_horizon is True
    assert state.show_origin is True
    assert state.show_stats is True
    assert state.interactive_quality == 80
    assert state.settled_quality == "png"
    assert state.interactive_max_side == 900
    assert state.internal_render_max_side == 1800
    assert state.keyboard_move_speed == 0.3
    assert state.keyboard_sprint_multiplier == 6.0
    assert state.orbit_invert_x is True
    assert state.orbit_invert_y is True
    assert state.pan_invert_x is False
    assert state.pan_invert_y is True
    assert state.viewer_rotation_x_degrees == 15.0
    assert state.viewer_rotation_y_degrees == -25.0
    assert state.viewer_rotation_z_degrees == 45.0
    assert state.origin == (4.0, 5.0, 6.0)


def test_set_fov_degrees_can_skip_live_viewer_callback() -> None:
    state = ViewerState()
    callback_fov: list[float] = []

    def _capture(camera_state: CameraState) -> None:
        callback_fov.append(camera_state.fov_degrees)

    state._reset_camera_callback = _capture

    returned = state.set_fov_degrees(72.0, push_to_viewer=False)

    assert returned is state
    assert state.camera_state.fov_degrees == 72.0
    assert state.initial_camera_state.fov_degrees == 72.0
    assert callback_fov == []


def test_viewer_controls_gui_builds_live_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(viewer_defaults.mo, "running_in_notebook", lambda: True)

    handle = viewer_controls_gui(ViewerState())

    assert handle.config_model is ViewerControlsConfig
    assert isinstance(handle.gui, PydanticGui)


def test_viewer_controls_handle_uses_desktop_controls_outside_notebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        viewer_defaults.mo, "running_in_notebook", lambda: False
    )

    handle = viewer_controls_handle(ViewerState())

    assert handle.config_model is ViewerControlsConfig
    assert isinstance(handle.gui, DesktopPydanticControls)
    assert handle.value == handle.default_config


def test_combined_viewer_pipeline_controls_gui_builds_live_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(viewer_defaults.mo, "running_in_notebook", lambda: True)

    pipeline = ViewerPipeline(view_factory=lambda scene: scene)
    pipeline_result = pipeline.build(
        source_scene=None, viewer_state=ViewerState()
    )

    handle = viewer_pipeline_controls_gui(ViewerState(), pipeline_result)

    assert "viewer" in handle.config_model.model_fields
    assert "pipeline" in handle.config_model.model_fields
    assert isinstance(handle.gui, PydanticGui)


def test_combined_viewer_pipeline_controls_handle_uses_desktop_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        viewer_defaults.mo, "running_in_notebook", lambda: False
    )

    pipeline = ViewerPipeline(view_factory=lambda scene: scene)
    pipeline_result = pipeline.build(
        source_scene=None, viewer_state=ViewerState()
    )

    handle = viewer_pipeline_controls_handle(ViewerState(), pipeline_result)

    assert "viewer" in handle.config_model.model_fields
    assert "pipeline" in handle.config_model.model_fields
    assert isinstance(handle.gui, DesktopPydanticControls)
    assert handle.value == handle.default_config


def test_apply_viewer_pipeline_config_returns_pipeline_subtree() -> None:
    class _PipelineConfig(BaseModel):
        value: int = 3

    class _CombinedConfig(BaseModel):
        viewer: ViewerControlsConfig = ViewerControlsConfig()
        pipeline: _PipelineConfig = _PipelineConfig()

    state = ViewerState()
    config = _CombinedConfig(
        viewer=ViewerControlsConfig(
            camera={"fov_degrees": 68.0},
            overlays={"show_axes": False},
            navigation={"move_speed": 0.2, "sprint_multiplier": 3.0},
            transform={
                "rotation": {
                    "x_degrees": 5.0,
                    "y_degrees": 0.0,
                    "z_degrees": 0.0,
                }
            },
        ),
        pipeline=_PipelineConfig(value=9),
    )

    pipeline_config = apply_viewer_pipeline_config(state, config)

    assert pipeline_config.value == 9
    assert state.camera_state.fov_degrees == 68.0
    assert state.show_axes is False
    assert state.keyboard_move_speed == 0.2
    assert state.keyboard_sprint_multiplier == 3.0
    assert state.viewer_rotation_x_degrees == 5.0


def test_set_keyboard_navigation_updates_viewer_state() -> None:
    state = ViewerState()

    returned = state.set_keyboard_navigation(0.4, 5.0)

    assert returned is state
    assert state.keyboard_move_speed == 0.4
    assert state.keyboard_sprint_multiplier == 5.0


def test_set_pointer_controls_updates_viewer_state() -> None:
    state = ViewerState()

    returned = state.set_pointer_controls(True, False, True, True)

    assert returned is state
    assert state.orbit_invert_x is True
    assert state.orbit_invert_y is False
    assert state.pan_invert_x is True
    assert state.pan_invert_y is True


def test_link_viewer_states_initial_syncs_selected_fields() -> None:
    primary = ViewerState(
        camera_state=CameraState.default(width=64, height=48, fov_degrees=45.0),
        show_axes=False,
    )
    secondary = ViewerState(
        camera_state=CameraState.default(width=32, height=24, fov_degrees=70.0),
        show_axes=True,
    )

    link = link_viewer_states(
        primary,
        secondary,
        fields=("camera_state", "show_axes"),
    )

    assert secondary.camera_state.width == 64
    assert secondary.camera_state.height == 48
    assert secondary.camera_state.fov_degrees == 45.0
    assert secondary.show_axes is False

    link.close()


def test_link_viewer_states_bidirectionally_syncs_updates() -> None:
    primary = ViewerState(show_axes=True)
    secondary = ViewerState(show_axes=True)
    link = link_viewer_states(
        primary,
        secondary,
        fields=("camera_state", "show_axes"),
    )

    updated_camera = CameraState.default(width=40, height=30, fov_degrees=55.0)
    primary.set_camera(updated_camera)
    secondary.set_show_axes(False)

    assert secondary.camera_state.width == 40
    assert secondary.camera_state.height == 30
    assert secondary.camera_state.fov_degrees == 55.0
    assert primary.show_axes is False

    link.close()


def test_link_viewer_states_ignores_unselected_fields() -> None:
    primary = ViewerState(show_axes=True)
    secondary = ViewerState(show_axes=True)
    link = link_viewer_states(
        primary,
        secondary,
        fields=("camera_state",),
    )

    primary.set_show_axes(False)

    assert secondary.show_axes is True

    link.close()


def test_link_viewer_states_close_stops_sync() -> None:
    primary = ViewerState()
    secondary = ViewerState()
    link = link_viewer_states(primary, secondary)

    link.close()
    primary.set_camera(
        CameraState.default(width=44, height=22, fov_degrees=50.0)
    )

    assert secondary.camera_state.width != 44
    assert secondary.camera_state.height != 22


def test_link_viewer_states_follows_widget_camera_updates() -> None:
    primary_state = ViewerState(
        camera_state=CameraState.default(width=64, height=48)
    )
    secondary_state = ViewerState(
        camera_state=CameraState.default(width=32, height=24)
    )
    primary_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=primary_state,
    )
    secondary_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=secondary_state,
    )
    link = link_viewer_states(primary_state, secondary_state)
    updated_camera = CameraState.default(width=28, height=18, fov_degrees=47.0)

    primary_viewer.anywidget().camera_state_json = updated_camera.to_json()

    _wait_until(lambda: secondary_viewer.get_camera_state().width == 28)
    assert secondary_viewer.get_camera_state().height == 18
    assert secondary_viewer.get_camera_state().fov_degrees == 47.0

    link.close()
    primary_viewer.close()
    secondary_viewer.close()


def test_link_viewer_states_follows_widget_overlay_updates() -> None:
    primary_state = ViewerState(show_axes=True)
    secondary_state = ViewerState(show_axes=True)
    primary_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=primary_state,
    )
    secondary_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=secondary_state,
    )
    link = link_viewer_states(
        primary_state,
        secondary_state,
        fields=("show_axes",),
    )

    primary_viewer.anywidget().show_axes = False

    _wait_until(lambda: secondary_viewer.anywidget().show_axes is False)
    assert secondary_state.show_axes is False

    link.close()
    primary_viewer.close()
    secondary_viewer.close()


def test_camera_state_with_convention_round_trips() -> None:
    state = CameraState.default(width=48, height=32, camera_convention="opencv")

    converted = state.with_convention("opengl")
    restored = converted.with_convention("opencv")

    assert converted.camera_convention == "opengl"
    assert restored.camera_convention == "opencv"
    assert np.allclose(restored.cam_to_world, state.cam_to_world)


@pytest.mark.parametrize(
    "camera_convention",
    ["opencv", "opengl", "blender", "colmap"],
)
def test_camera_state_round_trips_supported_camera_conventions(
    camera_convention: str,
) -> None:
    state = CameraState(
        fov_degrees=60.0,
        width=32,
        height=24,
        cam_to_world=np.eye(4, dtype=np.float64),
        camera_convention=camera_convention,  # type: ignore[arg-type]
    )

    restored = CameraState.from_json(state.to_json())

    assert restored.camera_convention == camera_convention


@pytest.mark.parametrize(
    "camera_convention",
    ["opencv", "opengl", "blender", "colmap"],
)
def test_camera_convention_transform_round_trips(
    camera_convention: str,
) -> None:
    source = CameraState.default(
        width=48,
        height=32,
        camera_convention="opencv",
    ).cam_to_world

    converted = _convert_cam_to_world_between_conventions(
        source,
        source_convention="opencv",
        target_convention=camera_convention,  # type: ignore[arg-type]
    )
    round_tripped = _convert_cam_to_world_between_conventions(
        converted,
        source_convention=camera_convention,  # type: ignore[arg-type]
        target_convention="opencv",
    )

    assert np.allclose(round_tripped, source)


def test_camera_state_rejects_unknown_camera_convention() -> None:
    with pytest.raises(ValueError, match="camera_convention must be one of"):
        CameraState(
            fov_degrees=60.0,
            width=32,
            height=24,
            cam_to_world=np.eye(4, dtype=np.float64),
            camera_convention="unknown",  # type: ignore[arg-type]
        )


def test_viewer_click_json_round_trip() -> None:
    click = ViewerClick(
        x=10,
        y=12,
        width=32,
        height=24,
        camera_state=CameraState.default(width=32, height=24),
    )

    restored = ViewerClick.from_json(click.to_json())

    assert restored.x == 10
    assert restored.y == 12
    assert restored.width == 32
    assert restored.height == 24
    assert restored.camera_state.width == 32
    assert restored.camera_state.height == 24


def test_viewer_last_click_reads_synced_widget_state() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )
    click = ViewerClick(
        x=4,
        y=5,
        width=20,
        height=10,
        camera_state=CameraState.default(width=20, height=10),
    )

    viewer.anywidget().last_click_json = click.to_json()

    assert viewer.get_last_click() is not None
    assert viewer.get_last_click().x == click.x
    assert viewer.get_last_click().y == click.y
    assert viewer.get_last_click().width == click.width
    assert viewer.get_last_click().height == click.height
    assert np.allclose(
        viewer.get_last_click().camera_state.cam_to_world,
        click.camera_state.cam_to_world,
    )
    assert viewer.get_last_click().x == click.x


def test_frame_stream_server_ignores_send_after_close_race() -> None:
    class _FakeWebSocket:
        path_params = {"stream_id": "stream"}
        query_params = {"token": "token"}

        def __init__(self) -> None:
            self._receive_count = 0

        async def accept(self) -> None:
            return None

        async def receive(self) -> dict[str, object]:
            if self._receive_count == 0:
                self._receive_count += 1
                return {
                    "type": "websocket.receive",
                    "text": (
                        '{"type":"clock_sync_ping","ping_id":1,'
                        '"client_sent_at_ms":1.0}'
                    ),
                }
            return {"type": "websocket.disconnect"}

        async def send_json(self, payload: dict[str, object]) -> None:
            del payload
            raise RuntimeError(
                'Cannot call "send" once a close message has been sent.'
            )

        async def close(self, code: int) -> None:
            del code
            return None

    server = _FrameStreamServer.__new__(_FrameStreamServer)
    server._lock = threading.Lock()
    server._streams = {"stream": _FrameStreamState(token="token")}
    server._loop = None

    asyncio.run(server._websocket_endpoint(_FakeWebSocket()))


def test_stable_anywidget_uses_virtual_file_js_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeContext:
        def __init__(self) -> None:
            self.virtual_files_supported = True
            self.virtual_file_registry = VirtualFileRegistry(InMemoryStorage())

    monkeypatch.setattr(
        "marimo_3dv.viewer.widget.get_context",
        lambda: _FakeContext(),
    )

    widget = _NativeViewerAnyWidget(
        camera_state=CameraState.default(),
        aspect_ratio=1.0,
        show_axes=True,
        show_horizon=False,
        show_origin=False,
        show_stats=False,
        viewer_rotation_x_degrees=0.0,
        viewer_rotation_y_degrees=0.0,
        viewer_rotation_z_degrees=0.0,
        origin_x=0.0,
        origin_y=0.0,
        origin_z=0.0,
        orbit_invert_x=False,
        orbit_invert_y=False,
        pan_invert_x=False,
        pan_invert_y=False,
        stream_port=1,
        stream_path="/stream",
        stream_token="token",
        transport_mode="comm",
    )

    wrapped = _StableMarimoAnyWidget(widget)

    assert wrapped._args.args["js-url"].startswith("./@file/")


def test_stable_anywidget_retries_on_virtual_file_name_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRegistry:
        def __init__(self) -> None:
            self.calls = 0

        def add(self, virtual_file: object, context: object) -> None:
            del context
            self.calls += 1
            if self.calls == 1:
                raise FileExistsError("collision")

    class _FakeContext:
        def __init__(self) -> None:
            self.virtual_files_supported = True
            self.virtual_file_registry = _FakeRegistry()

    monkeypatch.setattr(
        "marimo_3dv.viewer.widget.get_context",
        lambda: _FakeContext(),
    )

    js_url = _StableMarimoAnyWidget._create_js_url(
        js="export default {};",
        js_filename="native_viewer_widget.js",
        js_hash="abc123",
    )

    assert js_url.startswith("./@file/")
    assert "abc123-" in js_url


def test_viewer_get_snapshot_decodes_latest_frame() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )
    viewer._latest_frame_array = np.full(
        (2, 3, 3),
        fill_value=np.array([12, 34, 56], dtype=np.uint8),
        dtype=np.uint8,
    )

    snapshot = viewer.get_snapshot()

    assert snapshot.size == (3, 2)
    assert snapshot.mode == "RGB"


def test_viewer_get_snapshot_requires_available_frame() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )
    viewer._latest_frame_array = None

    with pytest.raises(
        RuntimeError, match="No rendered frame is available yet"
    ):
        viewer.get_snapshot()


def test_normalize_frame_accepts_float_rgb() -> None:
    frame = np.array(
        [[[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]]],
        dtype=np.float32,
    )

    normalized = _normalize_frame(frame)

    assert normalized.dtype == np.uint8
    assert normalized.shape == (1, 2, 3)
    assert normalized.tolist() == [[[0, 127, 255], [255, 63, 0]]]


def test_normalize_frame_accepts_torch_tensor() -> None:
    frame = torch.tensor([[[0.0, 255.0, 32.0]]], dtype=torch.float32)

    normalized = _normalize_frame(frame)

    assert normalized.dtype == np.uint8
    assert normalized.tolist() == [[[0, 255, 32]]]


def test_normalize_frame_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Expected frame shape"):
        _normalize_frame(np.zeros((4, 4), dtype=np.uint8))


def test_latest_only_renderer_drops_stale_results() -> None:
    started_first = threading.Event()
    release_first = threading.Event()
    published: list[tuple[int, int]] = []

    def render_fn(camera_state: CameraState) -> np.ndarray:
        if camera_state.width == 10:
            started_first.set()
            assert release_first.wait(timeout=2.0)
        return np.full(
            (camera_state.height, camera_state.width, 3),
            fill_value=camera_state.width,
            dtype=np.uint8,
        )

    def publish_frame(
        revision: int,
        camera_state: CameraState,
        frame: np.ndarray,
        render_queue_time_ms: float,
        render_time_ms: float,
        interaction_active: bool,
    ) -> None:
        del frame
        assert render_queue_time_ms >= 0.0
        assert render_time_ms >= 0.0
        assert isinstance(interaction_active, bool)
        published.append((revision, camera_state.width))

    renderer = _LatestOnlyRenderer(
        render_fn=render_fn,
        publish_frame=publish_frame,
        publish_error=lambda revision, message: None,
        set_rendering=lambda value: None,
    )

    renderer.request(1, CameraState.default(width=10, height=4), True)
    assert started_first.wait(timeout=2.0)
    renderer.request(2, CameraState.default(width=20, height=4), False)
    release_first.set()

    _wait_until(lambda: published == [(2, 20)])


def test_marimo_viewer_set_camera_state_updates_widget_state() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        initial_view=CameraState.default(width=64, height=48),
    )
    updated = CameraState.default(width=32, height=24, fov_degrees=45.0)

    viewer.set_camera_state(updated)

    assert viewer.get_camera_state().width == 32
    assert viewer.get_camera_state().height == 24
    assert viewer.get_camera_state().fov_degrees == 45.0


def test_marimo_viewer_reuses_explicit_state_across_reruns() -> None:
    state = ViewerState(camera_state=CameraState.default(width=64, height=48))
    first_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
    )
    updated_camera_state = CameraState.default(
        width=32, height=24, fov_degrees=45.0
    )

    first_viewer.set_camera_state(updated_camera_state)

    second_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
    )

    assert second_viewer.get_camera_state().width == 32
    assert second_viewer.get_camera_state().height == 24
    assert second_viewer.get_camera_state().fov_degrees == 45.0
    assert first_viewer._closed is True


def test_process_cleanup_closes_active_viewers() -> None:
    first_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=ViewerState(),
    )
    second_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=ViewerState(),
    )

    _cleanup_active_marimo_viewers()

    assert first_viewer._closed is True
    assert second_viewer._closed is True


def test_viewer_state_can_reset_camera_to_initial_value() -> None:
    initial = CameraState.default(width=64, height=48, fov_degrees=60.0)
    state = ViewerState(camera_state=initial)
    viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
    )
    viewer.set_camera_state(
        CameraState.default(width=32, height=24, fov_degrees=45.0)
    )

    state.reset_camera()

    assert viewer.get_camera_state().width == 64
    assert viewer.get_camera_state().height == 48
    assert viewer.get_camera_state().fov_degrees == 60.0
    assert state.camera_state.width == 64
    assert state.camera_state.height == 48
    assert state.camera_state.fov_degrees == 60.0


def test_marimo_viewer_reuses_show_axes_from_explicit_state() -> None:
    state = ViewerState(show_axes=True)

    first_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
    )
    assert first_viewer.anywidget().show_axes is True

    state.show_axes = False
    second_viewer = marimo_viewer(
        lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        state=state,
    )

    assert second_viewer.anywidget().show_axes is False


def test_marimo_viewer_get_debug_info_reads_synced_metrics() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )
    widget = viewer.anywidget()
    widget.error_text = "boom"
    widget.latency_ms = 12.5
    widget.latency_sample_ms = 11.0
    widget.render_time_ms = 1.25
    widget.render_queue_time_ms = 3.5
    widget.encode_time_ms = 0.75
    widget.stream_queue_time_ms = 0.5
    widget.stream_send_time_ms = 1.0
    widget.backend_to_browser_time_ms = 3.0
    widget.packet_size_bytes = 12345
    widget.browser_receive_queue_ms = 4.0
    widget.browser_post_receive_ms = 14.0
    widget.browser_decode_time_ms = 2.0
    widget.browser_draw_time_ms = 0.25
    widget.browser_present_wait_ms = 8.5

    assert viewer.get_debug_info() == {
        "error_text": "boom",
        "latency_ms": 12.5,
        "latency_sample_ms": 11.0,
        "render_time_ms": 1.25,
        "render_queue_time_ms": 3.5,
        "encode_time_ms": 0.75,
        "stream_queue_time_ms": 0.5,
        "stream_send_time_ms": 1.0,
        "backend_to_browser_time_ms": 3.0,
        "packet_size_bytes": 12345,
        "browser_receive_queue_ms": 4.0,
        "browser_post_receive_ms": 14.0,
        "browser_decode_time_ms": 2.0,
        "browser_draw_time_ms": 0.25,
        "browser_present_wait_ms": 8.5,
        "accounted_leaf_latency_ms": 21.75,
        "unaccounted_leaf_latency_ms": -9.25,
        "unaccounted_leaf_latency_sample_ms": -10.75,
        "accounted_coarse_latency_ms": 25.0,
        "unaccounted_coarse_latency_ms": -12.5,
        "unaccounted_coarse_latency_sample_ms": -14.0,
    }


def test_marimo_viewer_exposes_configured_aspect_ratio() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        state=ViewerState(aspect_ratio=2.0),
    )

    assert viewer.anywidget().aspect_ratio == 2.0


def test_marimo_viewer_uses_comm_transport_by_default() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8)
    )

    assert viewer.anywidget().transport_mode == "comm"
    assert viewer.anywidget().frame_packet


def test_marimo_viewer_can_use_websocket_transport() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        transport_mode="websocket",
    )

    assert viewer.anywidget().transport_mode == "websocket"
    assert viewer.anywidget().frame_packet == b""


def test_marimo_viewer_uses_requested_default_camera_convention() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        state=ViewerState(camera_convention="opengl"),
    )

    assert viewer.get_camera_state().camera_convention == "opengl"


def test_marimo_viewer_caps_motion_render_larger_axis_only() -> None:
    rendered_sizes: list[tuple[int, int, bool]] = []
    viewer = marimo_viewer(
        lambda state: (
            rendered_sizes.append((state.width, state.height, True))
            or np.zeros((state.height, state.width, 3), dtype=np.uint8)
        ),
        initial_view=CameraState.default(width=100, height=80),
        state=ViewerState(interactive_max_side=50),
    )
    rendered_sizes.clear()
    viewer.anywidget().interaction_active = True

    viewer.rerender()
    _wait_until(lambda: len(rendered_sizes) >= 1)

    viewer.anywidget().interaction_active = False
    viewer.rerender()
    _wait_until(lambda: len(rendered_sizes) >= 2)

    assert rendered_sizes[0][:2] == (50, 40)
    assert rendered_sizes[1][:2] == (100, 80)
    assert viewer.get_camera_state().width == 100
    assert viewer.get_camera_state().height == 80


def test_marimo_viewer_caps_settled_render_larger_axis_with_internal_limit() -> (
    None
):
    rendered_sizes: list[tuple[int, int]] = []
    viewer = marimo_viewer(
        lambda state: (
            rendered_sizes.append((state.width, state.height))
            or np.zeros((state.height, state.width, 3), dtype=np.uint8)
        ),
        initial_view=CameraState.default(width=160, height=80),
        state=ViewerState(internal_render_max_side=100),
    )

    assert rendered_sizes[0] == (100, 50)
    assert viewer.get_camera_state().width == 160
    assert viewer.get_camera_state().height == 80


def test_marimo_viewer_render_errors_raise_by_default() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        marimo_viewer(
            lambda state: (_ for _ in ()).throw(RuntimeError("boom")),
            initial_view=CameraState.default(width=40, height=30),
        )


def test_marimo_viewer_can_surface_render_errors_in_widget_state() -> None:
    viewer = marimo_viewer(
        lambda state: (_ for _ in ()).throw(RuntimeError("boom")),
        initial_view=CameraState.default(width=40, height=30),
        state=ViewerState(raise_on_error=False),
    )

    viewer.rerender()

    _wait_until(lambda: "RuntimeError: boom" in viewer.anywidget().error_text)


def test_viewer_state_rejects_non_positive_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="aspect_ratio must be positive"):
        ViewerState(aspect_ratio=0.0)


def test_viewer_state_rejects_out_of_range_interactive_quality() -> None:
    with pytest.raises(ValueError, match="interactive_quality must be in"):
        ViewerState(interactive_quality=0)


def test_viewer_state_rejects_non_positive_interactive_max_side() -> None:
    with pytest.raises(
        ValueError, match="interactive_max_side must be None or a positive"
    ):
        ViewerState(interactive_max_side=0)


def test_viewer_state_rejects_non_positive_internal_render_max_side() -> None:
    with pytest.raises(
        ValueError, match="internal_render_max_side must be None or a positive"
    ):
        ViewerState(internal_render_max_side=0)


def test_viewer_state_accepts_none_interactive_max_side() -> None:
    viewer = marimo_viewer(
        lambda state: np.zeros((state.height, state.width, 3), dtype=np.uint8),
        state=ViewerState(interactive_max_side=None),
    )

    assert viewer is not None


def test_latest_only_renderer_close_stops_worker() -> None:
    renderer = _LatestOnlyRenderer(
        render_fn=lambda camera_state: np.zeros(
            (camera_state.height, camera_state.width, 3), dtype=np.uint8
        ),
        publish_frame=lambda *args: None,
        publish_error=lambda revision, error, message: None,
        set_rendering=lambda value: None,
    )

    renderer.close()

    assert not renderer._worker.is_alive()
