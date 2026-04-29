"""Bounded statistics payloads for viewer-side plots."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from jaxtyping import Float
from numpy import ndarray
from torch import Tensor

ViewerStatsPlotKind = Literal["histogram", "sorted_rank"]
ViewerStatsKeepMode = Literal["higher", "lower"]


@dataclass(frozen=True)
class ViewerStatsSummary:
    """Summary statistics for a viewer stats series."""

    total_count: int
    finite_count: int
    selected_count: int
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    threshold: float | None = None


@dataclass(frozen=True)
class ViewerStatsSeries:
    """Small, UI-ready stats payload derived from a larger source array."""

    name: str
    plot_kind: ViewerStatsPlotKind
    rows: tuple[dict[str, float], ...]
    summary: ViewerStatsSummary
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewerStatsUpdateGate:
    """Throttle live stats updates independently from viewer render updates."""

    min_interval_seconds: float = 0.75
    update_while_active: bool = False
    _last_update_at: float = field(default=0.0, init=False)
    _last_revision: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate update interval."""
        if self.min_interval_seconds < 0.0:
            raise ValueError("min_interval_seconds must be non-negative.")

    def should_update(
        self,
        revision: int,
        *,
        active: bool = False,
        now: float | None = None,
    ) -> bool:
        """Return whether a live stats plot should update for ``revision``."""
        if active and not self.update_while_active:
            return False
        if self._last_revision == revision:
            return False
        current_time = time.monotonic() if now is None else float(now)
        if (
            self._last_revision is not None
            and current_time - self._last_update_at < self.min_interval_seconds
        ):
            return False
        self._last_revision = revision
        self._last_update_at = current_time
        return True

    def reset(self) -> None:
        """Allow the next new revision to update immediately."""
        self._last_update_at = 0.0
        self._last_revision = None


def _as_numpy_1d(
    values: Float[Tensor, " ..."] | Float[ndarray, " ..."] | Any,
) -> ndarray:
    """Return a detached 1D NumPy view/copy for stats preparation."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().reshape(-1)
    return np.asarray(values).reshape(-1)


def prepare_viewer_stats_series(
    values: Float[Tensor, " ..."] | Float[ndarray, " ..."] | Any,
    *,
    name: str = "value",
    plot_kind: ViewerStatsPlotKind = "histogram",
    max_points: int = 256,
    positive_only: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
    keep_mode: ViewerStatsKeepMode = "higher",
    top_count: int | None = None,
) -> ViewerStatsSeries:
    """Prepare a bounded stats series without mutating the source values.

    The returned rows are intentionally small enough to hand to UI layers such
    as Altair without pushing full per-primitive arrays through the notebook.
    """
    flat_values = _as_numpy_1d(values).astype(np.float64, copy=False)
    finite_mask = np.isfinite(flat_values)
    selected_mask = finite_mask.copy()
    if positive_only:
        selected_mask &= flat_values > 0.0
    threshold = None
    if min_value is not None:
        threshold = float(min_value)
        selected_mask &= flat_values >= threshold
    if max_value is not None:
        selected_mask &= flat_values <= float(max_value)
    selected_values = flat_values[selected_mask]
    selected_top_count = None
    if top_count is not None:
        selected_top_count = max(0, int(top_count))
        if selected_top_count <= 0:
            selected_values = selected_values[:0]
        elif selected_top_count < selected_values.size:
            if keep_mode == "higher":
                selected_indices = np.argpartition(
                    selected_values,
                    -selected_top_count,
                )[-selected_top_count:]
            else:
                selected_indices = np.argpartition(
                    selected_values,
                    selected_top_count - 1,
                )[:selected_top_count]
            selected_values = selected_values[selected_indices]

    summary = ViewerStatsSummary(
        total_count=int(flat_values.size),
        finite_count=int(finite_mask.sum()),
        selected_count=int(selected_values.size),
        min_value=(
            None if selected_values.size == 0 else float(selected_values.min())
        ),
        max_value=(
            None if selected_values.size == 0 else float(selected_values.max())
        ),
        mean_value=(
            None if selected_values.size == 0 else float(selected_values.mean())
        ),
        threshold=threshold,
    )
    if selected_values.size == 0:
        return ViewerStatsSeries(
            name=name,
            plot_kind=plot_kind,
            rows=(),
            summary=summary,
        )

    capped_points = max(1, int(max_points))
    if plot_kind == "histogram":
        bin_count = min(capped_points, int(selected_values.size))
        counts, edges = np.histogram(selected_values, bins=bin_count)
        rows = tuple(
            {
                "bin_start": float(start),
                "bin_end": float(end),
                "bin_center": float((start + end) * 0.5),
                "count": float(count),
            }
            for start, end, count in zip(
                edges[:-1],
                edges[1:],
                counts,
                strict=True,
            )
            if count > 0
        )
        return ViewerStatsSeries(
            name=name,
            plot_kind=plot_kind,
            rows=rows,
            summary=summary,
            metadata={
                "bin_count": float(bin_count),
                "top_count": selected_top_count,
            },
        )

    quantiles = np.linspace(0.0, 1.0, min(capped_points, selected_values.size))
    sampled_values = np.quantile(selected_values, quantiles)
    if keep_mode == "higher":
        sampled_values = sampled_values[::-1]
        ranks = (1.0 - quantiles[::-1]) * max(selected_values.size - 1, 1) + 1
    else:
        ranks = quantiles * max(selected_values.size - 1, 1) + 1
    rows = tuple(
        {
            "rank": float(rank),
            "value": float(value),
        }
        for rank, value in zip(ranks, sampled_values, strict=True)
    )
    return ViewerStatsSeries(
        name=name,
        plot_kind=plot_kind,
        rows=rows,
        summary=summary,
        metadata={"keep_mode": keep_mode, "top_count": selected_top_count},
    )


__all__ = [
    "ViewerStatsKeepMode",
    "ViewerStatsPlotKind",
    "ViewerStatsSeries",
    "ViewerStatsSummary",
    "ViewerStatsUpdateGate",
    "prepare_viewer_stats_series",
]
