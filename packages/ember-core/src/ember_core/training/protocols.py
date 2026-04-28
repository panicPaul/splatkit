"""Runtime dataclasses and protocols for declarative training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor

from ember_core.core.contracts import CameraState
from ember_core.initialization import InitializedModel
from ember_core.training.config import CheckpointMetadata, TrainingConfig


@dataclass
class TrainState:
    """In-memory training state."""

    model: InitializedModel
    step: int
    seed: int
    device: torch.device


@dataclass(frozen=True)
class LossResult:
    """Normalized loss return value."""

    loss: Tensor
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    """Result of a training run."""

    state: TrainState
    history: list[dict[str, float]]
    checkpoint_dir: str


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Loaded inference-ready training artifact."""

    model: InitializedModel
    render_fn: Callable[[InitializedModel, CameraState], Any]
    config: TrainingConfig
    metadata: CheckpointMetadata


class TrainingHook(Protocol):
    """Optional training loop hook."""

    def pre_backward(
        self,
        state: TrainState,
        batch: Any,
        render_output: Any,
        loss_result: LossResult,
    ) -> None:
        """Run after forward/loss and before backward."""

    def post_backward(
        self,
        state: TrainState,
        batch: Any,
        render_output: Any,
        loss_result: LossResult,
    ) -> None:
        """Run after backward and before optimizer steps."""

    def after_step(
        self,
        state: TrainState,
        metrics: dict[str, float],
    ) -> None:
        """Run after optimizer steps."""

    def post_optimizer_step(
        self,
        state: TrainState,
        batch: Any,
        render_output: Any,
        loss_result: LossResult,
    ) -> None:
        """Run after optimizer steps and before metric hooks."""


RenderFn = Callable[[InitializedModel, CameraState], Any]
LossFn = Callable[[TrainState, Any, Any], LossResult]


__all__ = [
    "LoadedCheckpoint",
    "LossFn",
    "LossResult",
    "RenderFn",
    "TrainState",
    "TrainingHook",
    "TrainingResult",
]
