"""TensorBoard checkpoint analysis helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import polars as pl


@dataclass(frozen=True)
class TensorBoardScalarRecord:
    """One scalar event read from TensorBoard logs."""

    run: str
    tag: str
    step: int
    wall_time: float
    value: float


def checkpoint_logs_dir(checkpoint_dir: str | Path) -> Path:
    """Return the canonical logs directory for one checkpoint."""
    return Path(checkpoint_dir).expanduser() / "logs"


def find_event_files(path: str | Path) -> tuple[Path, ...]:
    """Find TensorBoard event files under a checkpoint or logs directory."""
    root = Path(path).expanduser()
    search_root = checkpoint_logs_dir(root) if (root / "logs").exists() else root
    return tuple(sorted(search_root.rglob("events.out.tfevents.*")))


def read_scalar_records(path: str | Path) -> list[TensorBoardScalarRecord]:
    """Read TensorBoard scalar events from a checkpoint or logs directory."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    records: list[TensorBoardScalarRecord] = []
    event_dirs = sorted({event_file.parent for event_file in find_event_files(path)})
    for event_dir in event_dirs:
        run = _run_name_for_event_file(path, event_dir)
        accumulator = EventAccumulator(str(event_dir))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", ()):
            for event in accumulator.Scalars(tag):
                records.append(
                    TensorBoardScalarRecord(
                        run=run,
                        tag=tag,
                        step=int(event.step),
                        wall_time=float(event.wall_time),
                        value=float(event.value),
                    )
                )
    return records


def empty_scalar_frame() -> pl.DataFrame:
    """Return an empty scalar dataframe with the canonical schema."""
    return pl.DataFrame(
        schema={
            "run": pl.String,
            "tag": pl.String,
            "step": pl.Int64,
            "wall_time": pl.Float64,
            "value": pl.Float64,
        }
    )


def read_scalars(path: str | Path) -> pl.DataFrame:
    """Read TensorBoard scalar events into a Polars dataframe."""
    records = read_scalar_records(path)
    if len(records) == 0:
        return empty_scalar_frame()
    return pl.DataFrame(
        [record.__dict__ for record in records],
        schema={
            "run": pl.String,
            "tag": pl.String,
            "step": pl.Int64,
            "wall_time": pl.Float64,
            "value": pl.Float64,
        },
    )


def scalar_tags(frame: pl.DataFrame) -> tuple[str, ...]:
    """Return sorted scalar tags from a scalar dataframe."""
    if frame.is_empty():
        return ()
    return tuple(frame.select("tag").unique().sort("tag")["tag"].to_list())


def filter_scalars(
    frame: pl.DataFrame,
    tags: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Filter scalar events by tag."""
    if tags is None or len(tags) == 0:
        return frame
    return frame.filter(pl.col("tag").is_in(list(tags)))


def scalar_line_chart(
    frame: pl.DataFrame,
    *,
    tags: Sequence[str] | None = None,
    title: str = "TensorBoard scalars",
) -> alt.Chart:
    """Build an Altair line chart for scalar events."""
    selected = filter_scalars(frame, tags)
    if selected.is_empty():
        selected = pl.DataFrame(
            [{"run": "", "tag": "", "step": 0, "wall_time": 0.0, "value": 0.0}]
        )
    return (
        alt.Chart(selected)
        .mark_line()
        .encode(
            x=alt.X("step:Q", title="step"),
            y=alt.Y("value:Q", title="value"),
            color=alt.Color("tag:N", title="metric"),
            tooltip=["run:N", "tag:N", "step:Q", "value:Q"],
        )
        .properties(title=title, height=280)
        .interactive()
    )


def load_checkpoint(path: str | Path) -> Any:
    """Load an Ember training checkpoint directory."""
    from ember_core.training import load_checkpoint_dir

    return load_checkpoint_dir(path)


def _run_name_for_event_file(root: str | Path, event_path: Path) -> str:
    root_path = Path(root).expanduser()
    logs_root = (
        checkpoint_logs_dir(root_path) if (root_path / "logs").exists() else root_path
    )
    try:
        relative_parent = event_path.relative_to(logs_root)
    except ValueError:
        relative_parent = event_path
    if str(relative_parent) == ".":
        return logs_root.parent.name
    return str(relative_parent)


__all__ = [
    "TensorBoardScalarRecord",
    "checkpoint_logs_dir",
    "empty_scalar_frame",
    "filter_scalars",
    "find_event_files",
    "load_checkpoint",
    "read_scalar_records",
    "read_scalars",
    "scalar_line_chart",
    "scalar_tags",
]
