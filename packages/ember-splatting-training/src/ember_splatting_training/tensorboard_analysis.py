"""TensorBoard checkpoint analysis helpers and marimo notebook."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import marimo
import polars as pl

__generated_with = "0.23.2"
app = marimo.App(width="columns")


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


def read_scalars(path: str | Path) -> pl.DataFrame:
    """Read TensorBoard scalar events into a Polars dataframe."""
    records = read_scalar_records(path)
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


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md("# TensorBoard checkpoint analysis")
    return (mo,)


@app.cell
def _(mo):
    checkpoint_path = mo.ui.text(
        value="checkpoints/papers/fastergs/garden_baseline/faster_gs.core",
        label="Checkpoint directory",
        full_width=True,
    )
    checkpoint_path
    return (checkpoint_path,)


@app.cell
def _(mo, tags):
    metric_selector = mo.ui.multiselect(
        options=list(tags),
        value=list(tags[: min(4, len(tags))]),
        label="Metrics",
        full_width=True,
    )
    metric_selector
    return (metric_selector,)


@app.cell(hide_code=True)
def _(metric_selector, scalar_frame):
    from ember_splatting_training.tensorboard_analysis import scalar_line_chart

    scalar_line_chart(
        scalar_frame,
        tags=metric_selector.value,
        title="Selected scalar metrics",
    )
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md("## Scalar Data")
    return


@app.cell(column=1)
def _(checkpoint_path):
    from pathlib import Path

    import polars as _pl

    from ember_splatting_training.tensorboard_analysis import (
        find_event_files,
        read_scalars,
    )

    checkpoint_dir = Path(checkpoint_path.value).expanduser()
    event_files = find_event_files(checkpoint_dir)
    scalar_frame = read_scalars(checkpoint_dir) if event_files else _pl.DataFrame(
        schema={
            "run": _pl.String,
            "tag": _pl.String,
            "step": _pl.Int64,
            "wall_time": _pl.Float64,
            "value": _pl.Float64,
        }
    )
    return checkpoint_dir, event_files, scalar_frame


@app.cell(column=1)
def _(scalar_frame):
    from ember_splatting_training.tensorboard_analysis import scalar_tags

    tags = scalar_tags(scalar_frame)
    return (tags,)


@app.cell(column=1)
def _(checkpoint_dir, event_files, mo, scalar_frame):
    summary = mo.md(
        "\n".join(
            [
                f"Checkpoint: `{checkpoint_dir}`",
                f"Event files: `{len(event_files)}`",
                f"Scalar rows: `{scalar_frame.height}`",
            ]
        )
    )
    summary
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md("## Checkpoint")
    return


@app.cell(column=2)
def _(checkpoint_dir):
    from ember_splatting_training.tensorboard_analysis import load_checkpoint

    checkpoint_files = (
        checkpoint_dir / "config.json",
        checkpoint_dir / "metadata.json",
        checkpoint_dir / "model.ckpt",
    )
    checkpoint_error = ""
    if all(checkpoint_file.exists() for checkpoint_file in checkpoint_files):
        try:
            checkpoint = load_checkpoint(checkpoint_dir)
        except Exception as error:
            checkpoint = None
            checkpoint_error = str(error)
    else:
        checkpoint = None
    return checkpoint, checkpoint_error


@app.cell(column=2)
def _(checkpoint, checkpoint_error, mo):
    if checkpoint is None:
        checkpoint_view = mo.md(
            f"Checkpoint not loaded: `{checkpoint_error}`"
            if checkpoint_error
            else "No checkpoint loaded."
        )
    else:
        checkpoint_view = mo.md(
            "\n".join(
                [
                    f"Backend: `{checkpoint.config.render.backend}`",
                    f"Scene: `{type(checkpoint.model.scene).__name__}`",
                    f"Step: `{checkpoint.model.metadata.get('checkpoint_step', 0)}`",
                ]
            )
        )
    checkpoint_view
    return


if __name__ == "__main__":
    app.run()
