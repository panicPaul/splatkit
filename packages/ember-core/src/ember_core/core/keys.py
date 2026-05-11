"""Typed extensible keys for Ember composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

Namespace = NewType("Namespace", str)
SymbolName = NewType("SymbolName", str)


@dataclass(frozen=True, slots=True, order=True)
class Symbol:
    """A stable, serializable, user-extensible symbol."""

    namespace: str
    name: str

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("Symbol namespace must not be empty.")
        if not self.name:
            raise ValueError("Symbol name must not be empty.")
        if "." in self.namespace:
            raise ValueError("Symbol namespace must not contain '.'.")
        if "." in self.name:
            raise ValueError("Symbol name must not contain '.'.")

    @property
    def serialized(self) -> str:
        """Return the stable serialized symbol ID."""
        return f"{self.namespace}.{self.name}"

    def __str__(self) -> str:
        return self.serialized


@dataclass(frozen=True, slots=True, order=True)
class ProductKey(Symbol):
    """Render/training product key."""


@dataclass(frozen=True, slots=True, order=True)
class StageKey(Symbol):
    """Execution-stage key."""


@dataclass(frozen=True, slots=True, order=True)
class BufferKey(Symbol):
    """Logical buffer key."""


@dataclass(frozen=True, slots=True, order=True)
class ProviderKey(Symbol):
    """Provider key."""


@dataclass(frozen=True, slots=True, order=True)
class TraitKey(Symbol):
    """Trait key."""


@dataclass(frozen=True, slots=True, order=True)
class SceneFamilyKey(Symbol):
    """Scene-family key."""


@dataclass(frozen=True, slots=True, order=True)
class BackendId(Symbol):
    """Backend ID used for registry and serialization."""


@dataclass(frozen=True, slots=True, order=True)
class OptimizerRef(Symbol):
    """Optimizer ID used for authoring and serialization."""


def serialized_id(value: str | Symbol) -> str:
    """Return the stable serialized ID for a string or typed symbol."""
    return value.serialized if isinstance(value, Symbol) else value
