"""Viewer-facing state built on top of ember-core camera contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ember_core.core.contracts import CameraState

ViewerMode = Literal["auto", "force_on", "force_off"]


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
