"""Lightweight training profiling utilities."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ember_core.training.config import TrainingProfilerConfig
from ember_core.training.protocols import TrainState


@dataclass
class TrainingStepProfile:
    """Per-step mutable timing and memory record."""

    step: int
    device: torch.device
    sync_timing: bool
    cuda_memory: bool
    phases_ms: dict[str, float] = field(default_factory=dict)
    start_allocated_bytes: int | None = None
    start_reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.cuda_memory and self.device.type == "cuda":
            self.start_allocated_bytes = int(torch.cuda.memory_allocated(self.device))
            self.start_reserved_bytes = int(torch.cuda.memory_reserved(self.device))

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Measure one named phase."""
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.phases_ms[name] = self.phases_ms.get(name, 0.0) + elapsed_ms

    def metrics(
        self,
        state: TrainState,
        *,
        step_seconds: float | None = None,
    ) -> dict[str, float]:
        """Return flat metric values for this profile."""
        values = {
            f"time_{name}_ms": elapsed
            for name, elapsed in self.phases_ms.items()
        }
        profiled_total_ms = float(sum(self.phases_ms.values()))
        values["time_profiled_phases_total_ms"] = profiled_total_ms
        if step_seconds is not None:
            step_wall_ms = float(step_seconds) * 1000.0
            values["time_step_wall_ms"] = step_wall_ms
            values["time_profiled_unaccounted_ms"] = (
                step_wall_ms - profiled_total_ms
            )
        primitive_count = _primitive_count(state)
        if primitive_count is not None:
            values["primitives"] = float(primitive_count)
        if self.cuda_memory and self.device.type == "cuda":
            allocated = int(torch.cuda.memory_allocated(self.device))
            reserved = int(torch.cuda.memory_reserved(self.device))
            max_allocated = int(torch.cuda.max_memory_allocated(self.device))
            values.update(
                {
                    "cuda_allocated_bytes": float(allocated),
                    "cuda_reserved_bytes": float(reserved),
                    "cuda_max_allocated_bytes": float(max_allocated),
                }
            )
            if self.start_allocated_bytes is not None:
                values["cuda_allocated_delta_bytes"] = float(
                    allocated - self.start_allocated_bytes
                )
            if self.start_reserved_bytes is not None:
                values["cuda_reserved_delta_bytes"] = float(
                    reserved - self.start_reserved_bytes
                )
        return values

    def _sync(self) -> None:
        if self.sync_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class TrainingProfiler:
    """Disabled-by-default script-mode training profiler."""

    def __init__(self, config: TrainingProfilerConfig) -> None:
        self.config = config
        self._output_path = config.output_path
        if self._output_path is not None:
            self._output_path.expanduser().parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    @property
    def enabled(self) -> bool:
        """Return whether profiling is active."""
        return bool(self.config.enabled)

    def start_step(self, state: TrainState) -> TrainingStepProfile | None:
        """Start a new per-step profile if profiling is enabled."""
        if not self.enabled:
            return None
        return TrainingStepProfile(
            step=state.step,
            device=state.device,
            sync_timing=self.config.sync_timing,
            cuda_memory=self.config.cuda_memory,
        )

    def finish_step(
        self,
        state: TrainState,
        metrics: dict[str, float],
        profile: TrainingStepProfile | None,
    ) -> None:
        """Merge and optionally emit one profile record."""
        if profile is None:
            return
        step_seconds = metrics.get("step_seconds")
        profile_metrics = profile.metrics(
            state,
            step_seconds=(
                float(step_seconds)
                if isinstance(step_seconds, int | float)
                else None
            ),
        )
        metrics.update(profile_metrics)
        if (state.step % self.config.log_every) != 0 and not _has_refinement(
            metrics
        ):
            return
        record = {
            "step": state.step,
            "metrics": profile_metrics,
        }
        refinement = {
            name: value
            for name, value in metrics.items()
            if name.startswith("refinement_")
        }
        if refinement:
            record["refinement"] = refinement
        text = json.dumps(record, sort_keys=True)
        print(text, flush=True)
        if self._output_path is not None:
            with self._output_path.expanduser().open("a", encoding="utf-8") as f:
                f.write(text)
                f.write("\n")


def build_training_profiler(
    config: TrainingProfilerConfig,
) -> TrainingProfiler | None:
    """Build the profiler when explicitly enabled."""
    if not config.enabled:
        return None
    return TrainingProfiler(config)


def _primitive_count(state: TrainState) -> int | None:
    scene = getattr(state.model, "scene", None)
    center_position = getattr(scene, "center_position", None)
    if isinstance(center_position, Tensor) and center_position.ndim > 0:
        return int(center_position.shape[0])
    return None


def _has_refinement(metrics: dict[str, Any]) -> bool:
    return any(name.startswith("refinement_") for name in metrics)


__all__ = [
    "TrainingProfiler",
    "TrainingStepProfile",
    "build_training_profiler",
]
