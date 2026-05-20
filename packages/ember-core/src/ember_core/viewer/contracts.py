"""Viewer-facing state built on top of ember-core camera contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ember_core.core.contracts import CameraState

ViewerMode = Literal["auto", "force_on", "force_off"]
ViewerBackend = Literal["marimo_3dv", "viser"]
Marimo3DVTransportMode = Literal["widget", "websocket"]
Marimo3DVSettledQuality = Literal["jpeg_95", "jpeg_100", "png"]


@dataclass(frozen=True)
class Marimo3DVViewerConfig:
    """Typed marimo-3dv viewer tuning exposed through Ember."""

    interactive_quality: int | None = None
    settled_quality: Marimo3DVSettledQuality | None = None
    internal_render_max_side: int | None = None
    interactive_max_side: int | None = None
    interactive_backpressure: bool | None = None
    interactive_max_fps: float | None = None
    interactive_min_fps: float | None = None
    interactive_latency_target_ms: float | None = None
    interactive_probe_interval_s: float | None = None
    interactive_reset_interval_s: float | None = None
    transport_mode: Marimo3DVTransportMode | None = None

    def __post_init__(self) -> None:
        if (
            self.interactive_quality is not None
            and not 1 <= self.interactive_quality <= 100
        ):
            raise ValueError("interactive_quality must be in [1, 100].")
        for name in (
            "internal_render_max_side",
            "interactive_max_side",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set.")
        for name in (
            "interactive_max_fps",
            "interactive_min_fps",
            "interactive_latency_target_ms",
            "interactive_probe_interval_s",
            "interactive_reset_interval_s",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when set.")
        if (
            self.interactive_min_fps is not None
            and self.interactive_max_fps is not None
            and self.interactive_min_fps > self.interactive_max_fps
        ):
            raise ValueError(
                "interactive_min_fps must be less than or equal to "
                "interactive_max_fps."
            )


@dataclass
class ViewerState:
    """Persistent viewer state that stays in ember-core camera space."""

    camera: CameraState
    viewer_mode: ViewerMode = "auto"
    training_active: bool = False
    interaction_active: bool = False
    title: str = "ember-core viewer"

    def set_training_active(self, active: bool) -> ViewerState:
        """Update the training-active flag in place."""
        self.training_active = active
        return self

    def set_interaction_active(self, active: bool) -> ViewerState:
        """Update the interaction flag in place."""
        self.interaction_active = active
        return self
