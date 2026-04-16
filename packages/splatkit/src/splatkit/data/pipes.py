"""Typed dataset pipe specs and registries.

This module intentionally keeps the common path explicit:

- dataset configs own concrete ordered tuples of pipe specs
- each pipe spec serializes with an explicit ``kind`` field
- runtime dispatch is implicit by spec class via registration metadata

Researchers extending dataset behavior should typically:

1. subclass an existing dataset config
2. register new pipe spec classes plus implementations
3. override the phase tuples on the subclass
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import torch
from pydantic import BaseModel, Field

from splatkit.data.contracts import HorizonAdjustmentSpec, SceneDataset

SourcePipeKind = Literal["horizon_align"]
CachePipeKind = Literal["resize"]
PreparePipeKind = Literal["normalize"]

SourcePipeT = TypeVar("SourcePipeT", bound="SourcePipeConfig")
CachePipeT = TypeVar("CachePipeT", bound="CachePipeConfig")
PreparePipeT = TypeVar("PreparePipeT", bound="PreparePipeConfig")


class PipeConfigBase(BaseModel):
    """Base class for serializable dataset pipe specs."""

    model_config = {
        "extra": "forbid",
    }


class SourcePipeConfig(PipeConfigBase):
    """Serializable source-phase pipe spec."""


class CachePipeConfig(PipeConfigBase):
    """Serializable cache-phase pipe spec."""


class PreparePipeConfig(PipeConfigBase):
    """Serializable prepare-phase pipe spec."""


class HorizonAlignPipeConfig(SourcePipeConfig):
    """Rotate and center a scene dataset into a canonical up frame."""

    kind: SourcePipeKind = "horizon_align"
    enabled: bool = True
    target_up: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def to_spec(self) -> HorizonAdjustmentSpec:
        """Convert this config into an execution-time adjustment spec."""
        return HorizonAdjustmentSpec(
            enabled=self.enabled,
            target_up=torch.tensor(self.target_up, dtype=torch.float32),
        )


class ResizePipeConfig(CachePipeConfig):
    """Resize images before prepared samples are cached or produced."""

    kind: CachePipeKind = "resize"
    width_scale: float | None = Field(default=None, gt=0.0)
    width_target: int | None = Field(default=None, ge=1)
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"


class NormalizePipeConfig(PreparePipeConfig):
    """Scale RGB images into the [0, 1] range."""

    kind: PreparePipeKind = "normalize"
    enabled: bool = True


@dataclass(frozen=True)
class RegisteredSourcePipe(Generic[SourcePipeT]):
    """Registered mapping from source-phase spec class to apply function."""

    kind: str
    spec_cls: type[SourcePipeT]
    apply_fn: Any


@dataclass(frozen=True)
class RegisteredCachePipe(Generic[CachePipeT]):
    """Registered mapping from cache-phase spec class."""

    kind: str
    spec_cls: type[CachePipeT]


@dataclass(frozen=True)
class RegisteredPreparePipe(Generic[PreparePipeT]):
    """Registered mapping from prepare-phase spec class."""

    kind: str
    spec_cls: type[PreparePipeT]


SOURCE_PIPE_REGISTRY: dict[
    type[SourcePipeConfig], RegisteredSourcePipe[Any]
] = {}
CACHE_PIPE_REGISTRY: dict[type[CachePipeConfig], RegisteredCachePipe[Any]] = {}
PREPARE_PIPE_REGISTRY: dict[
    type[PreparePipeConfig], RegisteredPreparePipe[Any]
] = {}


def register_source_pipe(
    *,
    kind: str,
    spec_cls: type[SourcePipeT],
) -> Any:
    """Register a source-phase pipe implementation for a spec class."""

    def decorator(apply_fn: Any) -> Any:
        SOURCE_PIPE_REGISTRY[spec_cls] = RegisteredSourcePipe(
            kind=kind,
            spec_cls=spec_cls,
            apply_fn=apply_fn,
        )
        return apply_fn

    return decorator


def register_cache_pipe(
    *,
    kind: str,
    spec_cls: type[CachePipeT],
) -> None:
    """Register a cache-phase pipe spec class."""
    CACHE_PIPE_REGISTRY[spec_cls] = RegisteredCachePipe(
        kind=kind,
        spec_cls=spec_cls,
    )


def register_prepare_pipe(
    *,
    kind: str,
    spec_cls: type[PreparePipeT],
) -> None:
    """Register a prepare-phase pipe spec class."""
    PREPARE_PIPE_REGISTRY[spec_cls] = RegisteredPreparePipe(
        kind=kind,
        spec_cls=spec_cls,
    )


def apply_source_pipe(
    dataset: SceneDataset,
    pipe: SourcePipeConfig,
) -> SceneDataset:
    """Apply one registered source-phase pipe."""
    registered = SOURCE_PIPE_REGISTRY.get(type(pipe))
    if registered is None:
        raise ValueError(f"Unregistered source pipe spec {type(pipe)!r}.")
    return registered.apply_fn(dataset, pipe)


register_cache_pipe(kind="resize", spec_cls=ResizePipeConfig)
register_prepare_pipe(kind="normalize", spec_cls=NormalizePipeConfig)
