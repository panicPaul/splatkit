"""Helpers for linking multiple viewer states together."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from marimo_3dv.viewer.widget import LinkedViewerStateField, ViewerState


@dataclass
class ViewerStateLink:
    """Live link handle between two viewer states."""

    _cleanup_callbacks: list[Callable[[], None]]
    _closed: bool = False

    def close(self) -> None:
        """Remove all registered state listeners."""
        if self._closed:
            return
        self._closed = True
        for cleanup_callback in self._cleanup_callbacks:
            cleanup_callback()
        self._cleanup_callbacks.clear()


def _normalized_fields(
    fields: Sequence[LinkedViewerStateField],
) -> tuple[LinkedViewerStateField, ...]:
    """Return a deterministic field list without duplicates."""
    return tuple(dict.fromkeys(fields))


def _copy_field(
    source: ViewerState,
    target: ViewerState,
    field: LinkedViewerStateField,
) -> None:
    """Copy one linked field from source to target."""
    if field == "camera_state":
        target.set_camera(source.camera_state)
        return
    if field == "show_axes":
        target.set_show_axes(source.show_axes)
        return
    if field == "show_horizon":
        target.set_show_horizon(source.show_horizon)
        return
    if field == "show_origin":
        target.set_show_origin(source.show_origin)
        return
    if field == "show_stats":
        target.set_show_stats(source.show_stats)
        return
    raise ValueError(f"Unsupported linked viewer state field {field!r}.")


def link_viewer_states(
    primary: ViewerState,
    secondary: ViewerState,
    *,
    fields: Sequence[LinkedViewerStateField] = ("camera_state",),
    bidirectional: bool = True,
) -> ViewerStateLink:
    """Link selected fields across two viewer states."""
    resolved_fields = _normalized_fields(fields)
    cleanup_callbacks: list[Callable[[], None]] = []
    active_propagations = 0

    def propagate(
        source: ViewerState, target: ViewerState
    ) -> Callable[[LinkedViewerStateField], None]:
        def _listener(field: LinkedViewerStateField) -> None:
            nonlocal active_propagations
            if field not in resolved_fields or active_propagations > 0:
                return
            active_propagations += 1
            try:
                _copy_field(source, target, field)
            finally:
                active_propagations -= 1

        return _listener

    primary_listener = propagate(primary, secondary)
    primary._register_field_listener(primary_listener)
    cleanup_callbacks.append(
        lambda: primary._unregister_field_listener(primary_listener)
    )

    if bidirectional:
        secondary_listener = propagate(secondary, primary)
        secondary._register_field_listener(secondary_listener)
        cleanup_callbacks.append(
            lambda: secondary._unregister_field_listener(secondary_listener)
        )

    active_propagations += 1
    try:
        for field in resolved_fields:
            _copy_field(primary, secondary, field)
    finally:
        active_propagations -= 1

    return ViewerStateLink(cleanup_callbacks)


__all__ = [
    "LinkedViewerStateField",
    "ViewerStateLink",
    "link_viewer_states",
]
