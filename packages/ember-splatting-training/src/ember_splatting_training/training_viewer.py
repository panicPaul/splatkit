"""Notebook training run and view-inspection support for Ember splatting."""

from __future__ import annotations

import atexit
import html
import threading
import time
import traceback
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Literal

import marimo as mo
import torch
from ember_core.core.contracts import CameraState, Scene
from ember_core.data import (
    MaterializationProgress,
    PreparedFrameDataset,
    PreparedFrameSample,
    materialization_progress_callback,
)
from ember_core.training import (
    TrainingConfig,
    TrainingHook,
    TrainingResult,
    TrainState,
    build_training_render_fn,
    run_training,
)
from ember_core.viewer import (
    Marimo3DVViewerConfig,
    ViewerState,
    camera_to_viewer_payload,
    launch_viewer,
)
from jaxtyping import Float, UInt8
from marimo._plugins.core.web_component import JSONType
from marimo._plugins.ui._core.ui_element import UIElement
from torch import Tensor

_ACTIVE_TRAINING_VIEWERS: dict[
    int, weakref.ReferenceType[TrainingViewerHandle]
] = {}
_CLEANUP_REGISTERED = False


def _register_process_cleanup() -> None:
    """Install best-effort cleanup for live notebook training threads."""
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return
    _CLEANUP_REGISTERED = True
    atexit.register(_close_active_training_viewers)


def _register_training_viewer(handle: TrainingViewerHandle) -> None:
    _register_process_cleanup()
    _ACTIVE_TRAINING_VIEWERS[id(handle)] = weakref.ref(handle)


def _unregister_training_viewer(handle: TrainingViewerHandle) -> None:
    _ACTIVE_TRAINING_VIEWERS.pop(id(handle), None)


def _active_training_viewer() -> TrainingViewerHandle | None:
    """Return the running notebook viewer handle, if one exists."""
    for handle_id, handle_ref in list(_ACTIVE_TRAINING_VIEWERS.items()):
        handle = handle_ref()
        if handle is None:
            _ACTIVE_TRAINING_VIEWERS.pop(handle_id, None)
            continue
        if handle.snapshot().status in {"running", "stopping"}:
            return handle
    return None


def _close_active_training_viewers(
    *,
    except_handle: TrainingViewerHandle | None = None,
) -> None:
    for handle_id, handle_ref in list(_ACTIVE_TRAINING_VIEWERS.items()):
        handle = handle_ref()
        if handle is None:
            _ACTIVE_TRAINING_VIEWERS.pop(handle_id, None)
            continue
        if handle is except_handle:
            continue
        handle.close(join_timeout=0.0)


def _bind_viewer_close_to_training(handle: TrainingViewerHandle) -> None:
    viewer = handle.viewer
    if viewer is None:
        return
    close = getattr(viewer, "close", None)
    if close is None:
        return

    def close_with_training_cancel() -> None:
        handle.request_stop()
        close()

    viewer.close = close_with_training_cancel


TrainingViewerBackend = Literal["marimo_3dv", "viser"]


@dataclass(frozen=True)
class TrainingViserViewerConfig:
    """Viser runtime controls for the optional training viewer backend."""

    host: str = "127.0.0.1"
    port: int = 8080
    iframe_url: str | None = None
    public_url: str | None = None
    iframe_height: str = "640px"
    ssh_host: str | None = None
    ssh_user: str | None = None
    ssh_port: int = 22
    local_forward_host: str = "127.0.0.1"
    local_forward_port: int | None = None
    verbose: bool = False
    viewer_res: int = 1024
    render_width: int = 1920
    render_height: int = 1080
    move_jpeg_quality: int = 40
    static_jpeg_quality: int = 70
    settle_seconds: float = 0.2
    move_pause_seconds: float = 0.1
    train_util: float = 0.9

    def __post_init__(self) -> None:
        """Validate Viser viewer controls."""
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be in range [0, 65535].")
        if self.ssh_port < 1 or self.ssh_port > 65535:
            raise ValueError("ssh_port must be in range [1, 65535].")
        if (
            self.local_forward_port is not None
            and not 1 <= self.local_forward_port <= 65535
        ):
            raise ValueError(
                "local_forward_port must be in range [1, 65535] or None."
            )
        if self.viewer_res < 64:
            raise ValueError("viewer_res must be at least 64.")
        if self.render_width < 1 or self.render_height < 1:
            raise ValueError("render dimensions must be positive.")
        if not 1 <= self.move_jpeg_quality <= 100:
            raise ValueError("move_jpeg_quality must be in range [1, 100].")
        if not 1 <= self.static_jpeg_quality <= 100:
            raise ValueError("static_jpeg_quality must be in range [1, 100].")
        if self.settle_seconds < 0.0:
            raise ValueError("settle_seconds must be non-negative.")
        if self.move_pause_seconds < 0.0:
            raise ValueError("move_pause_seconds must be non-negative.")
        if not 0.0 <= self.train_util <= 1.0:
            raise ValueError("train_util must be in range [0, 1].")


@dataclass(frozen=True)
class TrainingViewerConfig:
    """Runtime controls for a live training viewer."""

    enabled: bool = True
    viewer_backend: TrainingViewerBackend = "marimo_3dv"
    update_every_steps: int = 100
    min_update_seconds: float = 2.0
    pause_on_interaction: bool = True
    pause_poll_seconds: float = 0.05
    max_pause_seconds: float | None = None
    interaction_boost_seconds: float = 3.0
    boost_update_every_steps: int = 25
    boost_min_update_seconds: float = 0.25
    show_progress: bool = True
    progress_every_steps: int = 10
    wait_for_render: bool = False
    marimo_3dv: Marimo3DVViewerConfig = field(
        default_factory=lambda: Marimo3DVViewerConfig(
            interactive_quality=35,
            interactive_max_side=1024,
            interactive_backpressure=True,
            interactive_max_fps=6.0,
            interactive_min_fps=1.0,
        )
    )
    viser: TrainingViserViewerConfig = field(
        default_factory=TrainingViserViewerConfig
    )

    def __post_init__(self) -> None:
        """Validate runtime viewer controls."""
        if self.update_every_steps < 1:
            raise ValueError("update_every_steps must be at least 1.")
        if self.boost_update_every_steps < 1:
            raise ValueError("boost_update_every_steps must be at least 1.")
        if self.progress_every_steps < 1:
            raise ValueError("progress_every_steps must be at least 1.")
        if self.min_update_seconds < 0.0:
            raise ValueError("min_update_seconds must be non-negative.")
        if self.boost_min_update_seconds < 0.0:
            raise ValueError("boost_min_update_seconds must be non-negative.")
        if self.pause_poll_seconds < 0.0:
            raise ValueError("pause_poll_seconds must be non-negative.")
        if self.max_pause_seconds is not None and self.max_pause_seconds < 0.0:
            raise ValueError("max_pause_seconds must be non-negative or None.")
        if self.interaction_boost_seconds < 0.0:
            raise ValueError("interaction_boost_seconds must be non-negative.")
        if self.viewer_backend not in {"marimo_3dv", "viser"}:
            raise ValueError("viewer_backend must be 'marimo_3dv' or 'viser'.")


TrainingViewerStatus = Literal[
    "idle",
    "running",
    "stopping",
    "complete",
    "failed",
    "cancelled",
]
TrainingPreparationStatus = Literal[
    "idle",
    "loading_scene",
    "preparing_views",
    "ready",
    "failed",
    "cancelled",
]
TrainingPreparationPhase = Literal["loading_scene", "preparing_views"]

_INSPECTOR_VALIDATION_VIEW_KEY = "__validation_view__"
_INSPECTOR_SHOW_VALIDATION_KEY = "__show_validation__"
_INSPECTOR_TRAINING_VIEW_KEY = "__training_view__"
_INSPECTOR_SHOW_TRAINING_KEY = "__show_training__"
_INSPECTOR_L1_RANGE_KEY = "__l1_range__"


class TrainingViewerCancelled(RuntimeError):
    """Raised inside the training thread when viewer training is cancelled."""


@dataclass(frozen=True)
class TrainingViewerSnapshot:
    """Thread-safe public status for notebook polling cells."""

    status: TrainingViewerStatus = "idle"
    step: int = 0
    render_step: int | None = None
    max_steps: int | None = None
    latest_metrics: dict[str, float] = field(default_factory=dict)
    iterations_per_second: float | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    primitive_count: int | None = None
    result: TrainingResult | None = None
    error_text: str | None = None


@dataclass(frozen=True)
class TrainingPreparationSnapshot:
    """Thread-safe public status for background inspector preparation."""

    status: TrainingPreparationStatus = "idle"
    phase: TrainingPreparationPhase | None = None
    scene_record: Any | None = None
    frame_view_catalog: Any | None = None
    frame_dataset: PreparedFrameDataset | None = None
    elapsed_seconds: float | None = None
    error_text: str | None = None
    error_phase: TrainingPreparationPhase | None = None
    progress_label: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None


@dataclass
class TrainingPreparationHandle:
    """Background scene and frame-catalog preparation for notebooks."""

    load_scene: Callable[[], Any] = field(repr=False)
    prepare_frame_view_catalog: Callable[[Any], Any] = field(repr=False)
    title: str = "Preparing training inspector"
    _set_snapshot: Callable[[TrainingPreparationSnapshot], None] | None = field(
        default=None,
        repr=False,
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )
    _thread: threading.Thread | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _status: TrainingPreparationStatus = field(
        default="idle",
        init=False,
        repr=False,
    )
    _phase: TrainingPreparationPhase | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _scene_record: Any | None = field(default=None, init=False, repr=False)
    _frame_view_catalog: Any | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _frame_dataset: PreparedFrameDataset | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _started_at: float = field(default=0.0, init=False, repr=False)
    _elapsed_seconds: float | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _error_text: str | None = field(default=None, init=False, repr=False)
    _error_phase: TrainingPreparationPhase | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _progress_label: str | None = field(default=None, init=False, repr=False)
    _progress_current: int | None = field(default=None, init=False, repr=False)
    _progress_total: int | None = field(default=None, init=False, repr=False)

    def start(self, *, wait: bool = False) -> bool:
        """Start background preparation, or run inline when ``wait`` is true."""
        with self._lock:
            if self._status in {
                "loading_scene",
                "preparing_views",
                "ready",
            }:
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._status = "loading_scene"
            self._phase = "loading_scene"
            self._scene_record = None
            self._frame_view_catalog = None
            self._frame_dataset = None
            self._started_at = time.monotonic()
            self._elapsed_seconds = 0.0
            self._error_text = None
            self._error_phase = None
            self._progress_label = None
            self._progress_current = None
            self._progress_total = None
        self._publish_snapshot()
        if wait:
            self._run()
            return True
        thread_cls = mo.Thread if mo.running_in_notebook() else threading.Thread
        self._thread = thread_cls(
            target=self._run,
            daemon=True,
            name="ember-training-preparation",
        )
        self._thread.start()
        return True

    def snapshot(self) -> TrainingPreparationSnapshot:
        """Return a thread-safe copy of the current preparation state."""
        with self._lock:
            elapsed_seconds = self._elapsed_seconds
            if self._status in {"loading_scene", "preparing_views"}:
                elapsed_seconds = time.monotonic() - self._started_at
            return TrainingPreparationSnapshot(
                status=self._status,
                phase=self._phase,
                scene_record=self._scene_record,
                frame_view_catalog=self._frame_view_catalog,
                frame_dataset=self._frame_dataset,
                elapsed_seconds=elapsed_seconds,
                error_text=self._error_text,
                error_phase=self._error_phase,
                progress_label=self._progress_label,
                progress_current=self._progress_current,
                progress_total=self._progress_total,
            )

    def _run(self) -> None:
        try:
            scene_record = self.load_scene()
            if _current_marimo_thread_should_exit():
                self._cancel()
                return
            with self._lock:
                self._status = "preparing_views"
                self._phase = "preparing_views"
                self._scene_record = scene_record
                self._progress_label = None
                self._progress_current = None
                self._progress_total = None
            self._publish_snapshot()
            with materialization_progress_callback(
                self._update_materialization_progress
            ):
                frame_view_catalog = self.prepare_frame_view_catalog(
                    scene_record
                )
            if _current_marimo_thread_should_exit():
                self._cancel()
                return
            frame_dataset = frame_view_catalog.training_dataset
        except Exception:
            with self._lock:
                self._status = "failed"
                self._elapsed_seconds = time.monotonic() - self._started_at
                self._error_text = traceback.format_exc().rstrip()
                self._error_phase = self._phase
                self._phase = None
            self._publish_snapshot()
            return
        with self._lock:
            self._status = "ready"
            self._phase = None
            self._scene_record = scene_record
            self._frame_view_catalog = frame_view_catalog
            self._frame_dataset = frame_dataset
            self._elapsed_seconds = time.monotonic() - self._started_at
        self._publish_snapshot()

    def _update_materialization_progress(
        self,
        progress: MaterializationProgress,
    ) -> None:
        with self._lock:
            if self._status != "preparing_views":
                return
            self._progress_label = progress.label
            self._progress_current = progress.current
            self._progress_total = progress.total
            self._elapsed_seconds = time.monotonic() - self._started_at
        self._publish_snapshot()

    def _cancel(self) -> None:
        with self._lock:
            self._status = "cancelled"
            self._phase = None
            self._elapsed_seconds = time.monotonic() - self._started_at
        self._publish_snapshot()

    def _publish_snapshot(self) -> None:
        set_snapshot = self._set_snapshot
        if set_snapshot is not None:
            set_snapshot(self.snapshot())


@dataclass(frozen=True)
class TrainingViewerErrorMap:
    """Rendered per-pixel RGB reconstruction error for a dataset view."""

    image: UInt8[Tensor, " height width 3"]
    error: Float[Tensor, " height width"]
    max_error: float
    mean_error: float
    available: bool = True


@dataclass(frozen=True)
class TrainingViewInspectorConfig:
    """Notebook controls for fixed-view training inspection."""

    l1_range_bounds: tuple[float, float] = (0.0, 1.0)
    l1_range: tuple[float, float] = (0.0, 0.25)
    l1_range_step: float = 0.01

    def __post_init__(self) -> None:
        """Validate fixed-view inspector controls."""
        lower, upper = sorted(
            (float(self.l1_range_bounds[0]), float(self.l1_range_bounds[1]))
        )
        value_lower, value_upper = sorted(
            (float(self.l1_range[0]), float(self.l1_range[1]))
        )
        if upper <= lower:
            raise ValueError("l1_range_bounds must span a positive interval.")
        if self.l1_range_step <= 0.0:
            raise ValueError("l1_range_step must be positive.")
        if value_lower < lower or value_upper > upper:
            raise ValueError("l1_range must lie inside l1_range_bounds.")


@dataclass(frozen=True)
class TrainingViewMapContext:
    """Inputs available to custom fixed-view map functions."""

    sample: PreparedFrameSample
    target: Float[Tensor, " height width 3"]
    prediction: Float[Tensor, " height width 3"]
    l1_error: Float[Tensor, " height width"]
    render_output: Any
    snapshot: TrainingViewerSnapshot


TrainingViewMapFn = Callable[[TrainingViewMapContext], Tensor]
TrainingViewMapColor = Literal["viridis", "rgb"]


@dataclass(frozen=True)
class TrainingViewMapSpec:
    """Developer-defined map displayed by the fixed-view inspector."""

    key: str
    label: str
    fn: TrainingViewMapFn
    color: TrainingViewMapColor = "viridis"
    value_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate map identifiers and display mode."""
        if not self.key:
            raise ValueError("TrainingViewMapSpec.key must be non-empty.")
        if not self.label:
            raise ValueError("TrainingViewMapSpec.label must be non-empty.")
        if self.color not in {"viridis", "rgb"}:
            raise ValueError(
                "TrainingViewMapSpec.color must be 'viridis' or 'rgb'."
            )


@dataclass(frozen=True)
class TrainingViewMapResult:
    """Rendered image and scalar summary for one inspector map."""

    key: str
    label: str
    image: UInt8[Tensor, " height width 3"]
    values: Float[Tensor, " height width"] | None = None
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None


@dataclass(frozen=True)
class TrainingViewInspection:
    """Fixed-view render, target, L1 map, and custom map results."""

    target_image: UInt8[Tensor, " height width 3"]
    prediction_image: UInt8[Tensor, " height width 3"]
    l1_image: UInt8[Tensor, " height width 3"]
    l1_error: Float[Tensor, " height width"]
    l1_max: float
    l1_mean: float
    maps: tuple[TrainingViewMapResult, ...] = ()
    render_step: int | None = None
    available: bool = True
    error_text: str | None = None


@dataclass(frozen=True)
class TrainingViewInspectorControls:
    """Marimo widgets used by a fixed-view training inspector."""

    view: Any
    validation_view_selector: Any
    show_validation_view_button: Any
    training_view_selector: Any
    show_training_view_button: Any
    l1_range_slider: Any


class TrainingViewInspector(UIElement[dict[str, JSONType], dict[str, Any]]):
    """Reusable notebook panel for train/validation view inspection."""

    _name = "marimo-dict"

    def __init__(
        self,
        *,
        config: TrainingViewInspectorConfig,
        controls: TrainingViewInspectorControls,
        elements: dict[str, Any],
    ) -> None:
        self.config = config
        self.controls = controls
        self._elements = elements
        self._active_view_key: str | None = None
        self._processed_show_counts = (0, 0)
        initial_frontend_value = self._current_frontend_value()
        self._control_value = self._control_value_from_frontend(
            initial_frontend_value
        )
        super().__init__(
            component_name=self._name,
            initial_value=initial_frontend_value,
            label="",
            args={
                "element-ids": {
                    element._id: name for name, element in elements.items()
                }
            },
            slotted_html="",
            on_change=None,
        )
        for name, element in elements.items():
            element._register_as_view(parent=self, key=name)

    def selected_view_ref(self, frame_view_catalog: Any) -> Any | None:
        """Return the current selected train/validation view reference."""
        selected_key = self._active_view_key
        selected_view = _view_ref_from_key(frame_view_catalog, selected_key)
        if selected_view is not None:
            return selected_view
        selected_view = _view_ref_from_key(
            frame_view_catalog,
            self._control_value[_INSPECTOR_VALIDATION_VIEW_KEY],
        )
        if selected_view is not None:
            return selected_view
        return _view_ref_from_key(
            frame_view_catalog,
            self._control_value[_INSPECTOR_TRAINING_VIEW_KEY],
        )

    def l1_value_range(self) -> tuple[float, float]:
        """Return the selected L1 visualization range."""
        value = self._control_value[_INSPECTOR_L1_RANGE_KEY]
        if value is None:
            return self.config.l1_range
        lower, upper = value
        return float(lower), float(upper)

    def _current_frontend_value(self) -> dict[str, JSONType]:
        return {
            name: _current_element_frontend_value(element)
            for name, element in self._elements.items()
        }

    def _control_value_from_frontend(
        self,
        value: dict[str, JSONType],
    ) -> dict[str, Any]:
        next_value: dict[str, Any] = {}
        for name, element in self._elements.items():
            if name in value:
                frontend_value = value[name]
            else:
                frontend_value = _current_element_frontend_value(element)
            if name in {
                _INSPECTOR_SHOW_VALIDATION_KEY,
                _INSPECTOR_SHOW_TRAINING_KEY,
            }:
                next_value[name] = _button_count_from_frontend(frontend_value)
            else:
                next_value[name] = element._convert_value(frontend_value)
        return next_value

    def _convert_value(
        self,
        value: dict[str, JSONType],
    ) -> dict[str, Any]:
        if not getattr(self, "_initialized", False):
            return self._control_value
        next_value = self._control_value_from_frontend(
            {
                **self._current_frontend_value(),
                **value,
            }
        )
        validation_count = int(
            next_value.get(_INSPECTOR_SHOW_VALIDATION_KEY) or 0
        )
        training_count = int(next_value.get(_INSPECTOR_SHOW_TRAINING_KEY) or 0)
        processed_validation_count, processed_training_count = (
            self._processed_show_counts
        )
        if validation_count > processed_validation_count:
            self._active_view_key = str(
                next_value[_INSPECTOR_VALIDATION_VIEW_KEY]
            )
            processed_validation_count = validation_count
        if training_count > processed_training_count:
            self._active_view_key = str(
                next_value[_INSPECTOR_TRAINING_VIEW_KEY]
            )
            processed_training_count = training_count
        self._processed_show_counts = (
            processed_validation_count,
            processed_training_count,
        )
        self._control_value = next_value
        return next_value

    def panel(
        self,
        handle: TrainingViewerHandle | None,
        frame_view_catalog: Any,
        *,
        refresh: Any | None = None,
        map_specs: Sequence[TrainingViewMapSpec] = (),
    ) -> Any:
        """Return a marimo panel with controls and the selected view render."""
        _ = self.value
        if refresh is not None:
            _ = refresh.value
        view_ref = self.selected_view_ref(frame_view_catalog)
        view = render_training_view_inspector(
            handle,
            frame_view_catalog,
            view_ref,
            value_range=self.l1_value_range(),
            map_specs=map_specs,
        )
        return mo.vstack([self.controls.view, view], gap=0.75).style(
            max_height="none",
            overflow="visible",
        )


@dataclass
class TrainingViewerHandle:
    """Live viewer plus runtime hooks for a training run."""

    config: TrainingViewerConfig
    viewer: Any | None = None
    _training_config: TrainingConfig | None = field(default=None, repr=False)
    _state: TrainState | None = field(default=None, repr=False)
    _render_state: TrainState | None = field(
        default=None, init=False, repr=False
    )
    _render_fn: Any | None = field(default=None, init=False, repr=False)
    _progress_context: Any | None = field(default=None, init=False, repr=False)
    _progress_bar: Any | None = field(default=None, init=False, repr=False)
    _progress_step: int = field(default=0, init=False, repr=False)
    _stop_requested: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _running_in_notebook: bool = field(default=False, repr=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _thread: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _status: TrainingViewerStatus = field(
        default="idle", init=False, repr=False
    )
    _result: TrainingResult | None = field(default=None, init=False, repr=False)
    _error_text: str | None = field(default=None, init=False, repr=False)
    _latest_metrics: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _iterations_per_second: float | None = field(
        default=None, init=False, repr=False
    )
    _started_at: float = field(default=0.0, init=False, repr=False)
    _elapsed_seconds: float | None = field(default=None, init=False, repr=False)
    _eta_seconds: float | None = field(default=None, init=False, repr=False)
    _throughput_step: int = field(default=0, init=False, repr=False)
    _throughput_at: float = field(default=0.0, init=False, repr=False)
    _latest_step: int = field(default=0, init=False, repr=False)
    _max_steps: int | None = field(default=None, init=False, repr=False)
    _latest_primitive_count: int | None = field(
        default=None, init=False, repr=False
    )
    _render_step: int | None = field(default=None, init=False, repr=False)
    _last_render_step: int = field(default=-1, init=False, repr=False)
    _last_render_at: float = field(default=0.0, init=False, repr=False)
    _boost_until: float = field(default=0.0, init=False, repr=False)
    _inspection_cache_key: tuple[Any, ...] | None = field(
        default=None, init=False, repr=False
    )
    _inspection_cache: TrainingViewInspection | None = field(
        default=None, init=False, repr=False
    )

    def runtime_hooks(self) -> tuple[TrainingHook, ...]:
        """Return runtime-only hooks for ``ember.run_training``."""
        if not self._running_in_notebook:
            return ()
        has_render_hook = self.config.enabled
        has_progress_hook = (
            self._running_in_notebook and self.config.show_progress
        )
        if not has_render_hook and not has_progress_hook:
            return ()
        return (TrainingViewerHook(self),)

    @property
    def interaction_active(self) -> bool:
        """Return whether the browser viewer is currently being moved."""
        if self.viewer is None:
            return False
        interaction_active = getattr(self.viewer, "interaction_active", None)
        if interaction_active is not None:
            return bool(interaction_active)
        try:
            return bool(self.viewer.anywidget().interaction_active)
        except Exception:
            return False

    @property
    def stop_requested(self) -> bool:
        """Return whether the active background run should stop."""
        return self._stop_requested.is_set()

    def attach_state(self, state: TrainState) -> None:
        """Attach the current mutable training state for future renders."""
        should_render_initial = False
        with self._lock:
            had_state = self._state is not None
            self._state = state
            self._latest_step = state.step
            self._latest_primitive_count = _primitive_count(state)
            if self._training_config is not None:
                self._max_steps = self._training_config.runtime.max_steps
            if self._render_fn is None:
                if self._training_config is None:
                    raise RuntimeError(
                        "Training viewer has no training config."
                    )
                self._render_fn = build_training_render_fn(
                    self._training_config,
                    state,
                )
            should_render_initial = not had_state and self.config.enabled
        if should_render_initial:
            self.update_render_snapshot(state)
            self.rerender(force=True, step=state.step)

    def start_training(
        self,
        frame_dataset: PreparedFrameDataset,
        training_config: TrainingConfig | None = None,
    ) -> bool:
        """Start training in a notebook background thread.

        Returns ``False`` when a run is already active, or when called outside
        notebook mode where the viewer is intentionally zero-cost.
        """
        if not self._running_in_notebook:
            return False
        resolved_training_config = training_config or self._training_config
        if resolved_training_config is None:
            raise RuntimeError("Training viewer has no training config.")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._training_config = resolved_training_config
            self._status = "running"
            self._result = None
            self._error_text = None
            self._stop_requested.clear()
            self._state = None
            self._render_state = None
            self._render_fn = None
            self._latest_metrics = {}
            self._iterations_per_second = None
            self._started_at = time.monotonic()
            self._elapsed_seconds = 0.0
            self._eta_seconds = None
            self._throughput_step = 0
            self._throughput_at = 0.0
            self._progress_step = 0
            self._latest_step = 0
            self._max_steps = resolved_training_config.runtime.max_steps
            self._latest_primitive_count = None
            self._render_step = None
            self._last_render_step = -1
            self._last_render_at = 0.0
            self._boost_until = 0.0
            self._inspection_cache_key = None
            self._inspection_cache = None
            self._thread = threading.Thread(
                target=self._run_training_thread,
                args=(frame_dataset, resolved_training_config),
                daemon=True,
                name="ember-training-viewer",
            )
            self._thread.start()
            return True

    def request_stop(self) -> None:
        """Request cancellation of the active background training run."""
        self._stop_requested.set()
        with self._lock:
            if self._status == "running":
                self._status = "stopping"

    def raise_if_stop_requested(self) -> None:
        """Abort the training step before more work is scheduled."""
        if self.stop_requested:
            raise TrainingViewerCancelled("Training viewer run was cancelled.")

    def close(self, *, join_timeout: float | None = 1.0) -> None:
        """Cancel training and release viewer resources."""
        self.request_stop()
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=join_timeout)
        viewer = self.viewer
        if viewer is not None:
            self.viewer = None
            close = getattr(viewer, "close", None)
            if close is not None:
                close()
        self.close_progress()
        _unregister_training_viewer(self)

    def __del__(self) -> None:
        try:
            self.request_stop()
        except Exception:
            pass

    def _run_training_thread(
        self,
        frame_dataset: PreparedFrameDataset,
        training_config: TrainingConfig,
    ) -> None:
        try:
            result = run_training(
                frame_dataset,
                training_config,
                runtime_hooks=self.runtime_hooks(),
            )
        except TrainingViewerCancelled:
            with self._lock:
                self._status = "cancelled"
                self._error_text = None
            self.close_progress()
            return
        except Exception:
            with self._lock:
                self._status = "failed"
                self._error_text = traceback.format_exc().rstrip()
            self.close_progress()
            return
        with self._lock:
            self._status = "complete"
            self._result = result
            self._latest_step = result.state.step
            self._latest_primitive_count = _primitive_count(result.state)
        self.close_progress()
        self.update_render_snapshot(result.state)
        self.rerender(force=True, step=result.state.step)

    def snapshot(self) -> TrainingViewerSnapshot:
        """Return a thread-safe copy of the current training viewer state."""
        with self._lock:
            return TrainingViewerSnapshot(
                status=self._status,
                step=self._latest_step,
                render_step=self._render_step,
                max_steps=self._max_steps,
                latest_metrics=dict(self._latest_metrics),
                iterations_per_second=self._iterations_per_second,
                elapsed_seconds=self._elapsed_seconds,
                eta_seconds=self._eta_seconds,
                primitive_count=self._latest_primitive_count,
                result=self._result,
                error_text=self._error_text,
            )

    def rerender(self, *, force: bool = False, step: int | None = None) -> None:
        """Request a progress render if a live viewer exists."""
        if self.viewer is not None:
            if force:
                with self._lock:
                    self._last_render_at = time.monotonic()
                    if step is not None:
                        self._last_render_step = step
            self.viewer.rerender(wait=self.config.wait_for_render)

    def snap_to_camera(
        self,
        camera: CameraState,
        *,
        wait: bool | None = None,
    ) -> bool:
        """Snap the live viewer to a prepared camera."""
        viewer = self.viewer
        if viewer is None:
            return False
        set_camera_state = getattr(viewer, "set_camera_state", None)
        if set_camera_state is None:
            return False
        resolved_wait = self.config.wait_for_render if wait is None else wait
        viewer_camera = _viewer_camera_from_core(camera)
        try:
            set_camera_state(viewer_camera, wait=resolved_wait)
        except TypeError:
            set_camera_state(viewer_camera)
        self.start_interaction_boost()
        return True

    def snap_to_view(self, catalog: Any, view_ref: Any) -> bool:
        """Snap the live viewer to a view reference from a view catalog."""
        return self.snap_to_camera(catalog.camera(view_ref))

    def inspect_view(
        self,
        sample: PreparedFrameSample,
        *,
        value_range: tuple[float, float] | None = None,
        map_specs: Sequence[TrainingViewMapSpec] = (),
    ) -> TrainingViewInspection:
        """Render one fixed dataset view from the latest training snapshot."""
        with self._lock:
            state = self._render_state
            render_fn = self._render_fn
            render_step = self._render_step
        if state is None or render_fn is None:
            return _placeholder_inspection(sample)

        cache_key = _inspection_cache_key(
            sample,
            render_step=render_step,
            value_range=value_range,
            map_specs=map_specs,
        )
        with self._lock:
            if (
                self._inspection_cache_key == cache_key
                and self._inspection_cache is not None
            ):
                return self._inspection_cache

        camera = sample.camera.to(state.device)
        target = sample.image.to(state.device).to(torch.float32)
        with torch.no_grad():
            render_output, prediction = _render_prediction(
                render_fn,
                state,
                camera,
            )
        prediction = _display_image_as_float(prediction)
        if prediction.shape != target.shape:
            raise ValueError(
                "Training view inspection requires prediction and target "
                "images to have the same shape: "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        l1_error = (prediction - target).abs().mean(dim=-1)
        snapshot = self.snapshot()
        context = TrainingViewMapContext(
            sample=sample,
            target=target,
            prediction=prediction,
            l1_error=l1_error,
            render_output=render_output,
            snapshot=snapshot,
        )
        inspection = TrainingViewInspection(
            target_image=_display_image_to_uint8(target),
            prediction_image=_display_image_to_uint8(prediction),
            l1_image=viridis_error_map(
                l1_error,
                quantile=1.0,
                value_range=value_range,
            ),
            l1_error=l1_error.detach().cpu(),
            l1_max=float(l1_error.detach().amax().item()),
            l1_mean=float(l1_error.detach().mean().item()),
            maps=tuple(_render_map_spec(spec, context) for spec in map_specs),
            render_step=render_step,
            available=True,
        )
        with self._lock:
            self._inspection_cache_key = cache_key
            self._inspection_cache = inspection
        return inspection

    def render_view_error_map(
        self,
        sample: PreparedFrameSample,
        *,
        quantile: float = 0.98,
        value_range: tuple[float, float] | None = None,
    ) -> TrainingViewerErrorMap:
        """Render a viridis error map for one prepared frame sample."""
        with self._lock:
            state = self._render_state
            render_fn = self._render_fn
        if state is None or render_fn is None:
            return _placeholder_error_map(sample)

        camera = sample.camera.to(state.device)
        target = sample.image.to(state.device).to(torch.float32)
        with torch.no_grad():
            render_output = render_fn(state.model, camera)
        prediction = getattr(render_output, "render", render_output)
        if prediction.ndim == 4:
            prediction = prediction[0]
        prediction = _display_image_as_float(
            _sanitize_display_image(prediction.detach())
        )
        if prediction.shape != target.shape:
            raise ValueError(
                "Training viewer error map requires prediction and target "
                "images to have the same shape: "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        error = (prediction - target).abs().mean(dim=-1)
        return TrainingViewerErrorMap(
            image=viridis_error_map(
                error,
                quantile=quantile,
                value_range=value_range,
            ),
            error=error.detach().cpu(),
            max_error=float(error.detach().amax().item()),
            mean_error=float(error.detach().mean().item()),
            available=True,
        )

    def update_render_snapshot(self, state: TrainState) -> None:
        """Publish a stable model snapshot for async viewer renders."""
        with torch.no_grad():
            render_state = replace(
                state,
                model=_clone_model_for_render(state.model),
                diagnostics=dict(state.diagnostics),
            )
        with self._lock:
            self._render_state = render_state
            self._render_step = state.step
            self._inspection_cache_key = None
            self._inspection_cache = None

    def update_progress(
        self,
        state: TrainState,
        metrics: dict[str, float],
    ) -> None:
        """Update the live marimo progress bar when enabled."""
        with self._lock:
            self._latest_step = state.step
            self._latest_metrics = dict(metrics)
            now = time.monotonic()
            if self._started_at > 0.0:
                self._elapsed_seconds = now - self._started_at
            if self._throughput_at > 0.0 and state.step > self._throughput_step:
                elapsed = now - self._throughput_at
                if elapsed > 0.0:
                    self._iterations_per_second = (
                        state.step - self._throughput_step
                    ) / elapsed
                    if (
                        self._max_steps is not None
                        and self._iterations_per_second > 0.0
                    ):
                        remaining_steps = max(0, self._max_steps - state.step)
                        self._eta_seconds = (
                            remaining_steps / self._iterations_per_second
                        )
            self._throughput_step = state.step
            self._throughput_at = now
            self._latest_primitive_count = _primitive_count(state)
            if self._training_config is not None:
                self._max_steps = self._training_config.runtime.max_steps
        if not self._running_in_notebook or not self.config.show_progress:
            return
        if self._training_config is None:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        if self._progress_bar is None:
            self._progress_context = mo.status.progress_bar(
                total=self._training_config.runtime.max_steps,
                title="Training",
                completion_title="Training complete",
            )
            self._progress_bar = self._progress_context.__enter__()
        next_step = min(state.step, self._training_config.runtime.max_steps)
        is_final_step = next_step >= self._training_config.runtime.max_steps
        if (
            not is_final_step
            and next_step - self._progress_step
            < self.config.progress_every_steps
        ):
            return
        increment = next_step - self._progress_step
        if increment <= 0:
            return
        self._progress_step = next_step
        loss = metrics.get("loss")
        subtitle_parts = []
        if loss is not None:
            subtitle_parts.append(f"loss={loss:.6g}")
        primitive_count = _primitive_count(state)
        if primitive_count is not None:
            subtitle_parts.append(f"primitives={primitive_count:,}")
        subtitle = " | ".join(subtitle_parts) or None
        self._progress_bar.update(
            increment=increment,
            subtitle=subtitle,
        )
        if is_final_step:
            self.close_progress()

    def close_progress(self) -> None:
        """Close the live progress bar if it was opened."""
        if self._progress_context is None:
            return
        self._progress_context.__exit__(None, None, None)
        self._progress_context = None
        self._progress_bar = None

    def render(self, camera: Any) -> Tensor:
        """Render the latest training model, or a placeholder before training."""
        with self._lock:
            state = self._render_state
            render_fn = self._render_fn
        if state is None or render_fn is None:
            return _placeholder_image(camera)
        camera = camera.to(state.device)
        with torch.no_grad():
            render_output = render_fn(state.model, camera)
        image = getattr(render_output, "render", render_output)
        if image.ndim == 4:
            image = image[0]
        image = _sanitize_display_image(image.detach())
        if _needs_dc_display_fallback(image):
            fallback_model = _dc_only_model_for_render(state.model)
            if fallback_model is not state.model:
                with torch.no_grad():
                    render_output = render_fn(fallback_model, camera)
                fallback_image = getattr(render_output, "render", render_output)
                if fallback_image.ndim == 4:
                    fallback_image = fallback_image[0]
                image = _sanitize_display_image(fallback_image.detach())
        return image

    def pause_for_interaction(self) -> None:
        """Pause at a step boundary while the live viewer is being moved."""
        if self.viewer is None or not self.config.pause_on_interaction:
            return
        if not self.interaction_active:
            return
        started_at = time.monotonic()
        while self.interaction_active:
            self.raise_if_stop_requested()
            max_pause_seconds = self.config.max_pause_seconds
            if (
                max_pause_seconds is not None
                and time.monotonic() - started_at >= max_pause_seconds
            ):
                return
            time.sleep(self.config.pause_poll_seconds)
        self.start_interaction_boost()
        self.rerender(force=True)

    def start_interaction_boost(self) -> None:
        """Start the short high-frequency render window after interaction."""
        with self._lock:
            self._boost_until = (
                time.monotonic() + self.config.interaction_boost_seconds
            )

    def maybe_rerender_after_step(self, state: TrainState) -> None:
        """Request a render when the normal or boosted cadence is due."""
        if not self.config.enabled:
            return
        now = time.monotonic()
        with self._lock:
            in_boost = now < self._boost_until
            step_interval = (
                self.config.boost_update_every_steps
                if in_boost
                else self.config.update_every_steps
            )
            time_interval = (
                self.config.boost_min_update_seconds
                if in_boost
                else self.config.min_update_seconds
            )
            step_due = (
                state.step > 0
                and state.step % step_interval == 0
                and state.step != self._last_render_step
            )
            time_due = (
                self.viewer is None
                or now - self._last_render_at >= time_interval
            )
            if not step_due or not time_due:
                return
            self._last_render_step = state.step
            self._last_render_at = now
        self.update_render_snapshot(state)
        if self.viewer is None:
            return
        if self.config.viewer_backend == "viser":
            render_state = getattr(self.viewer, "render_state", None)
            if (
                render_state is not None
                and self._iterations_per_second is not None
            ):
                render_state.num_train_rays_per_sec = (
                    self._iterations_per_second
                )
            self.viewer.update(state.step, num_train_rays_per_step=1)
        else:
            self.viewer.rerender(wait=self.config.wait_for_render)


class TrainingViewerHook:
    """Runtime hook that drives progress renders and interaction pausing."""

    def __init__(self, handle: TrainingViewerHandle) -> None:
        self.handle = handle

    def before_step(self, state: TrainState) -> None:
        """Attach state and pause for active interaction at step boundaries."""
        self.handle.raise_if_stop_requested()
        self.handle.attach_state(state)
        self.handle.pause_for_interaction()

    def after_step(
        self,
        state: TrainState,
        metrics: dict[str, float],
    ) -> None:
        """Update notebook progress after a completed training step."""
        self.handle.update_progress(state, metrics)
        self.handle.maybe_rerender_after_step(state)


def create_training_run(
    frame_dataset: PreparedFrameDataset,
    training_config: TrainingConfig,
    *,
    config: TrainingViewerConfig | None = None,
    title: str = "Training inspector",
) -> TrainingViewerHandle:
    """Create a notebook training handle without launching a live viewer."""
    del frame_dataset, title
    viewer_config = config or TrainingViewerConfig()
    running_in_notebook = mo.running_in_notebook()
    if running_in_notebook:
        active_handle = _active_training_viewer()
        if active_handle is not None:
            return active_handle
        _close_active_training_viewers()
    handle = TrainingViewerHandle(
        config=viewer_config,
        _training_config=training_config,
        _running_in_notebook=running_in_notebook,
    )
    if running_in_notebook:
        _register_training_viewer(handle)
    return handle


def create_training_preparation(
    load_scene: Callable[[], Any],
    prepare_frame_view_catalog: Callable[[Any], Any],
    *,
    title: str = "Preparing training inspector",
) -> tuple[
    TrainingPreparationHandle,
    Callable[[], TrainingPreparationSnapshot],
]:
    """Create a reactive background preparation handle and snapshot getter."""
    snapshot_state, set_snapshot_state = mo.state(
        TrainingPreparationSnapshot(),
        allow_self_loops=True,
    )
    handle = TrainingPreparationHandle(
        load_scene=load_scene,
        prepare_frame_view_catalog=prepare_frame_view_catalog,
        title=title,
        _set_snapshot=set_snapshot_state,
    )
    return handle, snapshot_state


def create_training_viewer(
    frame_dataset: PreparedFrameDataset,
    training_config: TrainingConfig,
    *,
    config: TrainingViewerConfig | None = None,
    initial_camera: CameraState | None = None,
    title: str = "Training viewer",
) -> TrainingViewerHandle:
    """Create a live training viewer handle for notebook runtimes."""
    viewer_config = config or TrainingViewerConfig()
    running_in_notebook = mo.running_in_notebook()
    if running_in_notebook:
        active_handle = _active_training_viewer()
        if active_handle is not None:
            return active_handle
        _close_active_training_viewers()
    handle = TrainingViewerHandle(
        config=viewer_config,
        _training_config=training_config,
        _running_in_notebook=running_in_notebook,
    )
    if running_in_notebook:
        _register_training_viewer(handle)
    if not viewer_config.enabled or not running_in_notebook:
        return handle

    resolved_initial_camera = (
        initial_camera
        if initial_camera is not None
        else _initial_training_viewer_camera(frame_dataset)
    )
    if viewer_config.viewer_backend == "viser":
        handle.viewer = _launch_viser_training_viewer(
            handle,
            initial_camera=resolved_initial_camera,
            title=title,
        )
    else:
        viewer_state = ViewerState(
            camera=resolved_initial_camera,
            title=title,
        )
        handle.viewer = launch_viewer(
            handle.render,
            state=viewer_state,
            marimo_3dv_config=viewer_config.marimo_3dv,
        )
    _bind_viewer_close_to_training(handle)
    return handle


def create_training_view_inspector(
    frame_view_catalog: Any,
    *,
    config: TrainingViewInspectorConfig | None = None,
) -> TrainingViewInspector:
    """Create reusable train/validation fixed-view notebook controls."""
    inspector_config = config or TrainingViewInspectorConfig()
    validation_view_options = _view_key_options(frame_view_catalog, "val")
    training_view_options = _view_key_options(frame_view_catalog, "train")
    validation_view_selector = mo.ui.dropdown(
        validation_view_options,
        value=_first_option_label(validation_view_options),
        label="Validation view",
        searchable=True,
        full_width=True,
    )
    show_validation_view_button = mo.ui.button(
        on_click=lambda value: int(value or 0) + 1,
        value=0,
        label="Show",
        full_width=True,
    )
    training_view_selector = mo.ui.dropdown(
        training_view_options,
        value=_first_option_label(training_view_options),
        label="Training view",
        searchable=True,
        full_width=True,
    )
    show_training_view_button = mo.ui.button(
        on_click=lambda value: int(value or 0) + 1,
        value=0,
        label="Show",
        full_width=True,
    )
    l1_start, l1_stop = sorted(
        (
            float(inspector_config.l1_range_bounds[0]),
            float(inspector_config.l1_range_bounds[1]),
        )
    )
    l1_range_slider = mo.ui.range_slider(
        start=l1_start,
        stop=l1_stop,
        step=inspector_config.l1_range_step,
        value=tuple(sorted(inspector_config.l1_range)),
        label="L1 range",
        show_value=True,
        full_width=True,
    )
    controls = [
        mo.hstack(
            [validation_view_selector, show_validation_view_button],
            widths=[4.0, 1.0],
            align="end",
            gap=0.75,
        ),
        mo.hstack(
            [training_view_selector, show_training_view_button],
            widths=[4.0, 1.0],
            align="end",
            gap=0.75,
        ),
        l1_range_slider,
    ]
    controls_view = mo.vstack(
        controls,
        gap=0.75,
    )
    return TrainingViewInspector(
        config=inspector_config,
        controls=TrainingViewInspectorControls(
            view=controls_view,
            validation_view_selector=validation_view_selector,
            show_validation_view_button=show_validation_view_button,
            training_view_selector=training_view_selector,
            show_training_view_button=show_training_view_button,
            l1_range_slider=l1_range_slider,
        ),
        elements={
            _INSPECTOR_VALIDATION_VIEW_KEY: validation_view_selector,
            _INSPECTOR_SHOW_VALIDATION_KEY: show_validation_view_button,
            _INSPECTOR_TRAINING_VIEW_KEY: training_view_selector,
            _INSPECTOR_SHOW_TRAINING_KEY: show_training_view_button,
            _INSPECTOR_L1_RANGE_KEY: l1_range_slider,
        },
    )


def render_training_view_inspector(
    handle: TrainingViewerHandle | None,
    frame_view_catalog: Any,
    view_ref: Any | None,
    *,
    value_range: tuple[float, float] | None = None,
    map_specs: Sequence[TrainingViewMapSpec] = (),
) -> Any:
    """Render the selected fixed-view inspection panel as marimo output."""
    if frame_view_catalog is None or view_ref is None:
        return mo.md("Prepare the training inspector to select a dataset view.")
    if handle is None:
        return mo.md("Training inspector is not active in script mode.")
    sample = frame_view_catalog.sample(view_ref)
    try:
        inspection = handle.inspect_view(
            sample,
            value_range=value_range,
            map_specs=map_specs,
        )
    except Exception as error:
        return mo.callout(
            f"Training view inspection failed.\n\n```text\n{error}\n```",
            kind="danger",
        )
    status_text = (
        f"snapshot step {inspection.render_step}"
        if inspection.available and inspection.render_step is not None
        else "waiting"
    )
    images = [
        mo.image(
            inspection.target_image.numpy(),
            caption=f"{view_ref.label} | GT",
        ),
        mo.image(
            inspection.prediction_image.numpy(),
            caption=f"{view_ref.label} | rendered | {status_text}",
        ),
        mo.image(
            inspection.l1_image.numpy(),
            caption=(
                f"L1 mean {inspection.l1_mean:.5f} | "
                f"max {inspection.l1_max:.5f}"
            ),
        ),
    ]
    images.extend(
        mo.image(
            result.image.numpy(),
            caption=_map_caption(result),
        )
        for result in inspection.maps
    )
    return _two_column_image_grid(images)


def render_training_preparation_status(
    snapshot: TrainingPreparationSnapshot | None,
    *,
    title: str = "Preparing training inspector",
) -> Any:
    """Render the current background preparation state."""
    if snapshot is None or snapshot.status == "idle":
        return mo.md("")
    if snapshot.status in {"loading_scene", "preparing_views"}:
        subtitle = (
            "Loading scene..."
            if snapshot.status == "loading_scene"
            else "Preparing train and validation views..."
        )
        elapsed = _format_seconds(snapshot.elapsed_seconds)
        if elapsed is not None:
            subtitle = f"{subtitle} elapsed {elapsed}"
        if (
            snapshot.status == "preparing_views"
            and snapshot.progress_current is not None
            and snapshot.progress_total is not None
        ):
            progress_label = (
                snapshot.progress_label
                or "Preparing train and validation views"
            )
            progress_subtitle = (
                f"{progress_label}: "
                f"{snapshot.progress_current}/{snapshot.progress_total}"
            )
            if elapsed is not None:
                progress_subtitle = f"{progress_subtitle} | elapsed {elapsed}"
            return _training_preparation_progress_bar(
                title=title,
                subtitle=progress_subtitle,
                current=snapshot.progress_current,
                total=snapshot.progress_total,
            )
        return mo.status.spinner(title=title, subtitle=subtitle)
    if snapshot.status == "ready":
        elapsed = _format_seconds(snapshot.elapsed_seconds)
        suffix = f" in {elapsed}" if elapsed is not None else ""
        return mo.md(f"Training inspector ready{suffix}.").style(
            max_height="none",
            overflow="visible",
        )
    if snapshot.status == "cancelled":
        return mo.callout("Training inspector preparation was cancelled.")
    error_text = snapshot.error_text or "Training inspector preparation failed."
    return mo.callout(
        f"Training inspector preparation failed.\n\n```text\n{error_text}\n```",
        kind="danger",
    ).style(max_height="none", overflow="visible")


def training_preparation_outputs(
    snapshot: TrainingPreparationSnapshot | None,
) -> tuple[
    Any | None,
    Any | None,
    PreparedFrameDataset | None,
    Exception | None,
    Any | None,
]:
    """Return legacy notebook preparation values from a preparation snapshot."""
    if snapshot is None:
        return None, None, None, None, None
    if snapshot.status == "ready":
        return (
            None,
            snapshot.scene_record,
            snapshot.frame_dataset,
            None,
            snapshot.frame_view_catalog,
        )
    if snapshot.status == "failed":
        error = RuntimeError(
            snapshot.error_text or "Training inspector preparation failed."
        )
        if snapshot.error_phase == "loading_scene":
            return error, None, None, None, None
        return None, snapshot.scene_record, None, error, None
    return None, snapshot.scene_record, None, None, None


@contextmanager
def training_inspector_spinner(
    subtitle: str | None = None,
    *,
    title: str = "Preparing training inspector",
    enabled: bool = True,
) -> Iterator[None]:
    """Show a marimo spinner while preparing fixed-view inspection data."""
    if enabled and mo.running_in_notebook():
        mo.output.replace(_training_inspector_spinner_output(title, subtitle))
        try:
            with mo.status.spinner(title=title, subtitle=subtitle):
                yield
        finally:
            mo.output.clear()
    else:
        yield


def _two_column_image_grid(images: Sequence[Any]) -> Any:
    rows = []
    for row in _two_column_rows(images):
        if len(row) == 1:
            rows.append(row[0])
        else:
            rows.append(
                mo.hstack(
                    list(row),
                    align="start",
                    gap=0.75,
                    widths="equal",
                )
            )
    return mo.vstack(rows, gap=0.75).style(
        max_height="none",
        overflow="visible",
    )


def _two_column_rows(items: Sequence[Any]) -> list[tuple[Any, ...]]:
    return [
        tuple(items[index : index + 2]) for index in range(0, len(items), 2)
    ]


def _training_preparation_progress_bar(
    *,
    title: str,
    subtitle: str,
    current: int,
    total: int,
) -> Any:
    resolved_total = max(0, int(total))
    resolved_current = max(0, min(int(current), resolved_total))
    progress_context = mo.status.progress_bar(
        title=title,
        subtitle=subtitle,
        total=resolved_total,
        show_rate=False,
        show_eta=False,
        disabled=True,
    )
    if resolved_current > 0:
        progress_context.progress.update(increment=resolved_current)
    return progress_context.progress


def _current_element_frontend_value(element: Any) -> JSONType:
    if hasattr(element, "_value_frontend"):
        return element._value_frontend
    return element.value


def _button_count_from_frontend(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _current_marimo_thread_should_exit() -> bool:
    thread = threading.current_thread()
    return bool(getattr(thread, "should_exit", False))


def _format_seconds(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    total = max(0, int(seconds))
    minutes, remainder = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {remainder:02d}s"
    if minutes:
        return f"{minutes:d}m {remainder:02d}s"
    return f"{remainder:d}s"


def _training_inspector_spinner_output(
    title: str,
    subtitle: str | None,
) -> Any:
    escaped_title = html.escape(title)
    escaped_subtitle = html.escape(subtitle or "")
    subtitle_html = (
        f"<div style='color: #666; font-size: 0.875rem;'>{escaped_subtitle}</div>"
        if escaped_subtitle
        else ""
    )
    return mo.Html(
        f"""
        <style>
        @keyframes ember-training-inspector-spin {{
            to {{ transform: rotate(360deg); }}
        }}
        </style>
        <div style="
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 0;
            max-height: none;
            overflow: visible;
        ">
            <div style="
                width: 1.25rem;
                height: 1.25rem;
                border: 0.18rem solid #d1d5db;
                border-top-color: #2563eb;
                border-radius: 50%;
                animation: ember-training-inspector-spin 0.8s linear infinite;
                flex: 0 0 auto;
            "></div>
            <div>
                <div style="font-weight: 500;">{escaped_title}</div>
                {subtitle_html}
            </div>
        </div>
        """
    ).style(max_height="none", overflow="visible")


def _initial_training_viewer_camera(
    frame_dataset: PreparedFrameDataset,
) -> CameraState:
    """Return the first prepared camera used by the training viewer."""
    if len(frame_dataset) == 0:
        raise ValueError("Training viewer requires a non-empty frame dataset.")
    return frame_dataset.prepared_camera(0)


def _launch_viser_training_viewer(
    handle: TrainingViewerHandle,
    *,
    initial_camera: CameraState,
    title: str,
) -> Any:
    """Launch the optional Viser training viewer backend."""
    from marimo_viser import (
        ViserRenderConfig,
        ViserServerConfig,
        ViserViewer,
        ViserViewerState,
    )

    config = handle.config.viser

    def render(camera: CameraState, _render_state: Any) -> Tensor:
        return handle.render(camera)

    return ViserViewer(
        render,
        state=ViserViewerState(
            camera=initial_camera,
            camera_convention=initial_camera.camera_convention,
            training_active=True,
        ),
        server_config=ViserServerConfig(
            host=config.host,
            port=config.port,
            iframe_url=config.iframe_url,
            public_url=config.public_url,
            iframe_height=config.iframe_height,
            ssh_host=config.ssh_host,
            ssh_user=config.ssh_user,
            ssh_port=config.ssh_port,
            local_forward_host=config.local_forward_host,
            local_forward_port=config.local_forward_port,
            verbose=config.verbose,
        ),
        render_config=ViserRenderConfig(
            viewer_res=config.viewer_res,
            render_width=config.render_width,
            render_height=config.render_height,
            move_jpeg_quality=config.move_jpeg_quality,
            static_jpeg_quality=config.static_jpeg_quality,
            settle_seconds=config.settle_seconds,
            move_pause_seconds=config.move_pause_seconds,
            train_util=config.train_util,
        ),
        mode="training",
        title=title,
    )


def _placeholder_image(camera: Any) -> UInt8[Tensor, " height width 3"]:
    height = int(camera.height[0].item())
    width = int(camera.width[0].item())
    return torch.full((height, width, 3), 245, dtype=torch.uint8)


def _viewer_camera_from_core(camera: CameraState) -> Any:
    payload = camera_to_viewer_payload(camera)
    from marimo_3dv.viewer.widget import CameraState as NativeCameraState

    return NativeCameraState(
        width=payload.width,
        height=payload.height,
        fov_degrees=payload.fov_degrees,
        cam_to_world=payload.cam_to_world,
        camera_convention=payload.camera_convention,
    )


def _placeholder_error_map(
    sample: PreparedFrameSample,
) -> TrainingViewerErrorMap:
    height, width = sample.image.shape[:2]
    return TrainingViewerErrorMap(
        image=torch.full((height, width, 3), 245, dtype=torch.uint8),
        error=torch.zeros((height, width), dtype=torch.float32),
        max_error=0.0,
        mean_error=0.0,
        available=False,
    )


def _placeholder_inspection(
    sample: PreparedFrameSample,
) -> TrainingViewInspection:
    height, width = sample.image.shape[:2]
    placeholder = torch.full((height, width, 3), 245, dtype=torch.uint8)
    return TrainingViewInspection(
        target_image=_display_image_to_uint8(sample.image),
        prediction_image=placeholder,
        l1_image=placeholder,
        l1_error=torch.zeros((height, width), dtype=torch.float32),
        l1_max=0.0,
        l1_mean=0.0,
        render_step=None,
        available=False,
    )


def _render_prediction(
    render_fn: Any,
    state: TrainState,
    camera: Any,
) -> tuple[Any, Tensor]:
    render_output = render_fn(state.model, camera)
    image = getattr(render_output, "render", render_output)
    if image.ndim == 4:
        image = image[0]
    image = _sanitize_display_image(image.detach())
    if _needs_dc_display_fallback(image):
        fallback_model = _dc_only_model_for_render(state.model)
        if fallback_model is not state.model:
            render_output = render_fn(fallback_model, camera)
            image = getattr(render_output, "render", render_output)
            if image.ndim == 4:
                image = image[0]
            image = _sanitize_display_image(image.detach())
    return render_output, image


def _render_map_spec(
    spec: TrainingViewMapSpec,
    context: TrainingViewMapContext,
) -> TrainingViewMapResult:
    values = spec.fn(context)
    if not isinstance(values, Tensor):
        raise TypeError(
            f"Training view map {spec.key!r} returned "
            f"{type(values).__name__}; expected torch.Tensor."
        )
    values = values.detach()
    if spec.color == "rgb":
        return TrainingViewMapResult(
            key=spec.key,
            label=spec.label,
            image=_display_image_to_uint8(values),
        )
    scalar = values.to(torch.float32)
    if scalar.ndim == 3 and scalar.shape[-1] == 1:
        scalar = scalar[..., 0]
    if scalar.ndim != 2:
        raise ValueError(
            f"Training view map {spec.key!r} must return a 2D scalar map "
            "for viridis display."
        )
    return TrainingViewMapResult(
        key=spec.key,
        label=spec.label,
        image=viridis_error_map(
            scalar, quantile=1.0, value_range=spec.value_range
        ),
        values=scalar.detach().cpu(),
        min_value=float(scalar.detach().amin().item()),
        max_value=float(scalar.detach().amax().item()),
        mean_value=float(scalar.detach().mean().item()),
    )


def _inspection_cache_key(
    sample: PreparedFrameSample,
    *,
    render_step: int | None,
    value_range: tuple[float, float] | None,
    map_specs: Sequence[TrainingViewMapSpec],
) -> tuple[Any, ...]:
    frame = sample.frame
    map_key = tuple(
        (
            spec.key,
            id(spec.fn),
            spec.color,
            spec.value_range,
        )
        for spec in map_specs
    )
    return (
        getattr(frame, "frame_id", None),
        getattr(frame, "camera_index", None),
        tuple(sample.image.shape),
        render_step,
        value_range,
        map_key,
    )


def _view_key_options(frame_view_catalog: Any, split: str) -> dict[str, str]:
    if frame_view_catalog is None:
        return {f"No {split} views": ""}
    options = frame_view_catalog.view_key_options(split)
    if not options:
        return {f"No {split} views": ""}
    return options


def _first_option_label(options: dict[str, str]) -> str:
    return next(iter(options), "")


def _view_ref_from_key(frame_view_catalog: Any, key: str | None) -> Any | None:
    if frame_view_catalog is None:
        return None
    return frame_view_catalog.view_ref_by_key(key)


def _map_caption(result: TrainingViewMapResult) -> str:
    if result.mean_value is None or result.max_value is None:
        return result.label
    return (
        f"{result.label} | mean {result.mean_value:.5f} | "
        f"max {result.max_value:.5f}"
    )


def _primitive_count(state: TrainState) -> int | None:
    scene = getattr(state.model, "scene", None)
    num_primitives = getattr(scene, "num_primitives", None)
    if isinstance(num_primitives, int):
        return num_primitives
    center_position = getattr(scene, "center_position", None)
    if isinstance(center_position, Tensor) and center_position.ndim > 0:
        return int(center_position.shape[0])
    return None


def _clone_model_for_render(model: Any) -> Any:
    if model is None:
        return None
    if not is_dataclass(model):
        return model
    return replace(
        model,
        scene=_clone_scene_for_render(model.scene),
        parameters=_clone_tensor_mapping(getattr(model, "parameters", {})),
        buffers=_clone_tensor_mapping(getattr(model, "buffers", {})),
        metadata=dict(getattr(model, "metadata", {})),
    )


def _clone_scene_for_render(value: Any) -> Any:
    if isinstance(value, Scene):
        return value.detached_copy()
    return _clone_dataclass_tensors(value)


def _clone_dataclass_tensors(value: Any) -> Any:
    if not is_dataclass(value):
        return value
    updates = {}
    for field_def in value.__dataclass_fields__.values():
        field_value = getattr(value, field_def.name)
        updates[field_def.name] = (
            _clone_tensor(field_value)
            if isinstance(field_value, Tensor)
            else field_value
        )
    return replace(value, **updates)


def _clone_tensor_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: _clone_tensor(value) if isinstance(value, Tensor) else value
        for name, value in values.items()
    }


def _clone_tensor(value: Tensor) -> Tensor:
    if isinstance(value, torch.nn.Parameter):
        return torch.nn.Parameter(
            value.detach().clone(),
            requires_grad=False,
        )
    return value.detach().clone()


def _sanitize_display_image(image: Tensor) -> Tensor:
    if not image.dtype.is_floating_point:
        return image
    return torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(
        0.0,
        1.0,
    )


def _display_image_as_float(image: Tensor) -> Tensor:
    if image.dtype == torch.uint8:
        return image.to(torch.float32) / 255.0
    return image.to(torch.float32)


def _display_image_to_uint8(
    image: Tensor,
) -> UInt8[Tensor, " height width 3"]:
    image = _sanitize_display_image(image.detach()).cpu()
    if image.dtype == torch.uint8:
        return image
    return (
        (image.to(torch.float32) * 255.0).round().clamp(0, 255).to(torch.uint8)
    )


def viridis_error_map(
    error: Float[Tensor, " height width"],
    *,
    quantile: float = 0.98,
    value_range: tuple[float, float] | None = None,
) -> UInt8[Tensor, " height width 3"]:
    """Map a scalar error image to RGB with a viridis-like color ramp."""
    error_cpu = torch.nan_to_num(
        error.detach().to(torch.float32).cpu(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if error_cpu.numel() == 0:
        normalized = error_cpu
    elif value_range is not None:
        lower, upper = sorted((float(value_range[0]), float(value_range[1])))
        scale = max(upper - lower, 1e-8)
        normalized = torch.clamp((error_cpu - lower) / scale, 0.0, 1.0)
    else:
        finite = error_cpu[torch.isfinite(error_cpu)]
        if finite.numel() == 0:
            scale = torch.tensor(1.0)
        elif 0.0 < quantile < 1.0:
            scale = torch.quantile(finite, quantile)
        else:
            scale = finite.amax()
        scale = torch.clamp(scale, min=1e-8)
        normalized = torch.clamp(error_cpu / scale, 0.0, 1.0)
    return _apply_viridis(normalized)


def _apply_viridis(
    values: Float[Tensor, " height width"],
) -> UInt8[Tensor, " height width 3"]:
    anchors = torch.tensor(
        [
            [0.267004, 0.004874, 0.329415],
            [0.282327, 0.094955, 0.417331],
            [0.253935, 0.265254, 0.529983],
            [0.163625, 0.471133, 0.558148],
            [0.134692, 0.658636, 0.517649],
            [0.477504, 0.821444, 0.318195],
            [0.993248, 0.906157, 0.143936],
        ],
        dtype=torch.float32,
    )
    scaled = torch.clamp(values, 0.0, 1.0) * (anchors.shape[0] - 1)
    lower = torch.floor(scaled).to(torch.int64)
    upper = torch.clamp(lower + 1, max=anchors.shape[0] - 1)
    blend = (scaled - lower.to(torch.float32)).unsqueeze(-1)
    colors = anchors[lower] * (1.0 - blend) + anchors[upper] * blend
    return (colors * 255.0).round().clamp(0, 255).to(torch.uint8)


def _needs_dc_display_fallback(image: Tensor) -> bool:
    if not image.dtype.is_floating_point or image.numel() == 0:
        return False
    return bool(image.amax().item() <= 1e-5)


def _dc_only_model_for_render(model: Any) -> Any:
    scene = getattr(model, "scene", None)
    feature = getattr(scene, "feature", None)
    if not (
        is_dataclass(model)
        and isinstance(scene, Scene)
        and isinstance(feature, Tensor)
        and feature.ndim == 3
        and feature.shape[1] > 1
    ):
        return model
    display_feature = feature.clone()
    display_feature[:, 1:, :] = 0.0
    return replace(
        model,
        scene=scene.with_fields(feature=display_feature),
    )


__all__ = [
    "TrainingPreparationHandle",
    "TrainingPreparationSnapshot",
    "TrainingViewInspection",
    "TrainingViewInspector",
    "TrainingViewInspectorConfig",
    "TrainingViewInspectorControls",
    "TrainingViewMapContext",
    "TrainingViewMapResult",
    "TrainingViewMapSpec",
    "TrainingViewerConfig",
    "TrainingViewerErrorMap",
    "TrainingViewerHandle",
    "TrainingViewerHook",
    "TrainingViewerSnapshot",
    "TrainingViserViewerConfig",
    "create_training_preparation",
    "create_training_run",
    "create_training_view_inspector",
    "create_training_viewer",
    "render_training_preparation_status",
    "render_training_view_inspector",
    "training_inspector_spinner",
    "training_preparation_outputs",
    "viridis_error_map",
]
