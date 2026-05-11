"""Live marimo viewer support for Ember splatting training."""

from __future__ import annotations

import atexit
import threading
import time
import traceback
import weakref
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Literal

import marimo as mo
import torch
from ember_core.core.contracts import Scene
from ember_core.data import PreparedFrameDataset
from ember_core.training import (
    TrainingConfig,
    TrainingHook,
    TrainingResult,
    TrainState,
    build_training_render_fn,
    run_training,
)
from ember_core.viewer import ViewerState, launch_viewer, select_viewer_camera
from jaxtyping import UInt8
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


@dataclass(frozen=True)
class TrainingViewerConfig:
    """Runtime controls for a live training viewer."""

    enabled: bool = True
    update_every_steps: int = 500
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


TrainingViewerStatus = Literal[
    "idle",
    "running",
    "stopping",
    "complete",
    "failed",
    "cancelled",
]


class TrainingViewerCancelled(RuntimeError):
    """Raised inside the training thread when viewer training is cancelled."""


@dataclass(frozen=True)
class TrainingViewerSnapshot:
    """Thread-safe public status for notebook polling cells."""

    status: TrainingViewerStatus = "idle"
    step: int = 0
    max_steps: int | None = None
    latest_metrics: dict[str, float] = field(default_factory=dict)
    iterations_per_second: float | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    primitive_count: int | None = None
    result: TrainingResult | None = None
    error_text: str | None = None


@dataclass
class TrainingViewerHandle:
    """Live viewer plus runtime hooks for a training run."""

    config: TrainingViewerConfig
    viewer: Any | None = None
    _training_config: TrainingConfig | None = field(default=None, repr=False)
    _state: TrainState | None = field(default=None, repr=False)
    _render_state: TrainState | None = field(default=None, init=False, repr=False)
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
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _status: TrainingViewerStatus = field(default="idle", init=False, repr=False)
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
    _last_render_step: int = field(default=-1, init=False, repr=False)
    _last_render_at: float = field(default=0.0, init=False, repr=False)
    _boost_until: float = field(default=0.0, init=False, repr=False)

    def runtime_hooks(self) -> tuple[TrainingHook, ...]:
        """Return runtime-only hooks for ``ember.run_training``."""
        if not self._running_in_notebook:
            return ()
        has_viewer_hook = self.viewer is not None and self.config.enabled
        has_progress_hook = self._running_in_notebook and self.config.show_progress
        if not has_viewer_hook and not has_progress_hook:
            return ()
        return (TrainingViewerHook(self),)

    @property
    def interaction_active(self) -> bool:
        """Return whether the browser viewer is currently being moved."""
        if self.viewer is None:
            return False
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
                    raise RuntimeError("Training viewer has no training config.")
                self._render_fn = build_training_render_fn(
                    self._training_config,
                    state,
                )
            should_render_initial = (
                self.viewer is not None
                and not had_state
                and self.config.enabled
            )
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
            self._last_render_step = -1
            self._last_render_at = 0.0
            self._boost_until = 0.0
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
        self.rerender(force=True, step=result.state.step)

    def snapshot(self) -> TrainingViewerSnapshot:
        """Return a thread-safe copy of the current training viewer state."""
        with self._lock:
            return TrainingViewerSnapshot(
                status=self._status,
                step=self._latest_step,
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
            and next_step - self._progress_step < self.config.progress_every_steps
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
        if self.viewer is None:
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
            time_due = now - self._last_render_at >= time_interval
            if not step_due or not time_due:
                return
            self._last_render_step = state.step
            self._last_render_at = now
        self.update_render_snapshot(state)
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


def create_training_viewer(
    frame_dataset: PreparedFrameDataset,
    training_config: TrainingConfig,
    *,
    config: TrainingViewerConfig | None = None,
    title: str = "Training viewer",
) -> TrainingViewerHandle:
    """Create a live training viewer handle for notebook runtimes."""
    viewer_config = config or TrainingViewerConfig()
    running_in_notebook = mo.running_in_notebook()
    if running_in_notebook:
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

    first_index = frame_dataset.indices[0]
    first_frame = frame_dataset.camera_stream.frames[first_index]
    initial_camera = select_viewer_camera(
        frame_dataset.camera_stream.camera,
        index=first_frame.camera_index,
    )
    viewer_state = ViewerState(camera=initial_camera, title=title)
    handle.viewer = launch_viewer(
        handle.render,
        state=viewer_state,
    )
    _bind_viewer_close_to_training(handle)
    return handle


def _placeholder_image(camera: Any) -> UInt8[Tensor, " height width 3"]:
    height = int(camera.height[0].item())
    width = int(camera.width[0].item())
    return torch.full((height, width, 3), 245, dtype=torch.uint8)


def _primitive_count(state: TrainState) -> int | None:
    scene = getattr(state.model, "scene", None)
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
    "TrainingViewerConfig",
    "TrainingViewerHandle",
    "TrainingViewerHook",
    "TrainingViewerSnapshot",
    "create_training_viewer",
]
