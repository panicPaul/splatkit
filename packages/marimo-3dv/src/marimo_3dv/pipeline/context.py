"""ViewerContext: renderer-agnostic event and state access for GUI ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marimo_3dv.viewer.widget import ViewerClick, ViewerState


@dataclass
class ViewerContext:
    """Renderer-agnostic context passed to stateful GUI ops each frame.

    Ops should read viewer state and click events through this object rather
    than accessing notebook globals directly.
    """

    viewer_state: ViewerState
    last_click: ViewerClick | None
