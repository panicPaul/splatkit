"""Summarize Ember training profiler JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProfileRecord:
    """One profiler JSONL record."""

    source: Path
    step: int
    metrics: dict[str, float]
    refinement: dict[str, float]

    @property
    def is_refinement(self) -> bool:
        """Return whether this step emitted refinement diagnostics."""
        return bool(self.refinement)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Summarize phase timings from training profiler JSONL."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Profiler JSONL files or directories containing *.jsonl files.",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=1,
        help="Skip this many earliest records per file before summarizing.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of Markdown tables.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=14,
        help="Maximum timing metrics to show per section.",
    )
    return parser.parse_args()


def expand_paths(paths: list[Path]) -> list[Path]:
    """Expand input files/directories into sorted JSONL files."""
    expanded: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved.is_dir():
            expanded.extend(sorted(resolved.glob("*.jsonl")))
        else:
            expanded.append(resolved)
    return expanded


def load_records(path: Path, *, skip_first: int) -> list[ProfileRecord]:
    """Load profiler records from one JSONL file."""
    records: list[ProfileRecord] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            metrics = _float_dict(raw.get("metrics", {}))
            refinement = _float_dict(raw.get("refinement", {}))
            records.append(
                ProfileRecord(
                    source=path,
                    step=int(raw["step"]),
                    metrics=metrics,
                    refinement=refinement,
                )
            )
    return records[skip_first:]


def _float_dict(raw: Any) -> dict[str, float]:
    """Convert numeric mapping values to floats."""
    if not isinstance(raw, dict):
        return {}
    return {
        str(name): float(value)
        for name, value in raw.items()
        if isinstance(value, int | float)
    }


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize_values(values: list[float]) -> dict[str, float]:
    """Summarize a list of numeric values."""
    if not values:
        return {}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def summarize_records(records: list[ProfileRecord]) -> dict[str, Any]:
    """Summarize one record collection."""
    metric_names = sorted(
        {
            name
            for record in records
            for name in record.metrics
            if name.startswith("time_")
        }
    )
    timings = {
        name: summarize_values(
            [record.metrics[name] for record in records if name in record.metrics]
        )
        for name in metric_names
    }
    primitive_values = [
        record.metrics["primitives"]
        for record in records
        if "primitives" in record.metrics
    ]
    return {
        "count": len(records),
        "step_min": min((record.step for record in records), default=None),
        "step_max": max((record.step for record in records), default=None),
        "refinement_count": sum(record.is_refinement for record in records),
        "primitives": summarize_values(primitive_values),
        "timings": timings,
    }


def summarize_file(records: list[ProfileRecord]) -> dict[str, Any]:
    """Summarize all/refinement/non-refinement records for one file."""
    return {
        "all": summarize_records(records),
        "non_refinement": summarize_records(
            [record for record in records if not record.is_refinement]
        ),
        "refinement": summarize_records(
            [record for record in records if record.is_refinement]
        ),
    }


def top_timing_rows(
    summary: dict[str, Any],
    *,
    limit: int,
) -> list[tuple[str, dict[str, float]]]:
    """Return timing rows ordered by mean descending."""
    timings = summary.get("timings", {})
    rows = [
        (name, values)
        for name, values in timings.items()
        if isinstance(values, dict) and values
    ]
    rows.sort(key=lambda item: item[1].get("mean", 0.0), reverse=True)
    return rows[:limit]


def print_markdown_report(
    summaries: dict[str, dict[str, Any]],
    *,
    top: int,
) -> None:
    """Print a compact Markdown summary."""
    for source, sections in summaries.items():
        print(f"## {source}")
        for section_name in ("all", "non_refinement", "refinement"):
            section = sections[section_name]
            print(
                f"\n### {section_name} "
                f"(n={section['count']}, refinements={section['refinement_count']})"
            )
            if section["count"] == 0:
                continue
            primitive_mean = section["primitives"].get("mean")
            if primitive_mean is not None:
                print(f"primitives mean: {primitive_mean:.0f}")
            print("| metric | mean ms | p50 ms | p90 ms | max ms |")
            print("|---|---:|---:|---:|---:|")
            for name, values in top_timing_rows(section, limit=top):
                print(
                    f"| `{name}` | {values['mean']:.3f} | "
                    f"{values['p50']:.3f} | {values['p90']:.3f} | "
                    f"{values['max']:.3f} |"
                )
        print()


def main() -> None:
    """Summarize requested profile files."""
    args = parse_args()
    summaries: dict[str, dict[str, Any]] = {}
    for path in expand_paths(args.paths):
        records = load_records(path, skip_first=args.skip_first)
        summaries[str(path)] = summarize_file(records)
    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return
    print_markdown_report(summaries, top=args.top)


if __name__ == "__main__":
    main()
