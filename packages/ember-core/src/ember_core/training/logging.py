"""Checkpoint-local TensorBoard scalar logging for training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ember_core.training.config import TrainingLoggingConfig

_ALWAYS_LOGGED_METRICS = (
    "elapsed_seconds",
    "iterations_per_second",
    "step_seconds",
)


def checkpoint_log_dir(checkpoint_dir: str | Path) -> Path:
    """Return the canonical TensorBoard log directory for a checkpoint."""
    return Path(checkpoint_dir).expanduser() / "logs"


class TensorBoardTrainingLogger:
    """Low-overhead scalar writer for one checkpoint directory."""

    def __init__(
        self,
        config: TrainingLoggingConfig,
        *,
        checkpoint_dir: str | Path,
    ) -> None:
        self.config = config
        self.log_dir = checkpoint_log_dir(checkpoint_dir)
        self._writer: Any | None = None
        if config.enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self.log_dir))

    @property
    def enabled(self) -> bool:
        """Return whether this logger writes events."""
        return self._writer is not None

    def write_step(self, step: int, metrics: dict[str, float]) -> None:
        """Write scalar metrics for one completed optimization step."""
        if self._writer is None:
            return
        if step % self.config.log_every != 0 and not _has_refinement(metrics):
            for name in _ALWAYS_LOGGED_METRICS:
                _write_named_scalar(self._writer, step, name, metrics.get(name))
            return
        for name, value in sorted(metrics.items()):
            _write_named_scalar(self._writer, step, name, value)

    def close(self) -> None:
        """Flush and close the underlying TensorBoard writer."""
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None


def build_training_logger(
    config: TrainingLoggingConfig,
    *,
    checkpoint_dir: str | Path,
) -> TensorBoardTrainingLogger | None:
    """Build the checkpoint-local training logger when enabled."""
    if not config.enabled:
        return None
    return TensorBoardTrainingLogger(config, checkpoint_dir=checkpoint_dir)


def scalar_tag_for_metric(name: str) -> str:
    """Map internal metric names to stable TensorBoard scalar tags."""
    match name:
        case "loss":
            return "train/loss"
        case "iterations_per_second":
            return "train/iterations_per_second"
        case "primitives":
            return "train/primitives"
        case "elapsed_seconds" | "step_seconds":
            return f"time/{name}"
        case "l1" | "dssim":
            return f"loss/{name}"
        case _ if name.endswith("_regularization"):
            return f"loss/{name}"
        case _ if name.startswith("time_") and name.endswith("_ms"):
            phase = name.removeprefix("time_").removesuffix("_ms")
            return f"time/{phase}_ms"
        case _ if name.startswith("cuda_"):
            return f"cuda/{name.removeprefix('cuda_')}"
        case _ if name.startswith("refinement_"):
            return f"densification/{name.removeprefix('refinement_')}"
        case _:
            return f"metrics/{name}"


def _is_finite_scalar(value: object) -> bool:
    if not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))


def _has_refinement(metrics: dict[str, float]) -> bool:
    return any(name.startswith("refinement_") for name in metrics)


def _write_named_scalar(
    writer: Any,
    step: int,
    name: str,
    value: object,
) -> None:
    if name == "step" or not _is_finite_scalar(value):
        return
    writer.add_scalar(scalar_tag_for_metric(name), float(value), global_step=step)


__all__ = [
    "TensorBoardTrainingLogger",
    "build_training_logger",
    "checkpoint_log_dir",
    "scalar_tag_for_metric",
]
