"""Typed buffer references and buffer specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from ember_core.core.enums import BufferLifetime, DeviceKind, DTypeName
from ember_core.core.keys import BufferKey

T = TypeVar("T")
ShapeExpr = tuple[int | str, ...]


@dataclass(frozen=True, slots=True)
class BufferRef(Generic[T]):
    """Typed logical buffer reference."""

    key: BufferKey
    doc: str = ""

    @property
    def serialized(self) -> str:
        """Return the stable buffer ID."""
        return self.key.serialized


@dataclass(frozen=True, slots=True)
class BufferSpec(Generic[T]):
    """Logical buffer allocation requirement."""

    ref: BufferRef[T]
    dtype: DTypeName
    shape: ShapeExpr
    device: DeviceKind = DeviceKind.CUDA
    lifetime: BufferLifetime = BufferLifetime.SCRATCH
    differentiable: bool = False
    persistent: bool = False
    alias_of: BufferRef[object] | None = None
