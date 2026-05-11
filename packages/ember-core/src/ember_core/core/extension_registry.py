"""Registries for typed Ember symbols and extension discovery."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Generic, TypeVar

from ember_core.core.keys import (
    BackendId,
    BufferKey,
    ProductKey,
    ProviderKey,
    SceneFamilyKey,
    StageKey,
    TraitKey,
)

T = TypeVar("T")


class SymbolRegistry(Generic[T]):
    """Small stable registry for typed symbols."""

    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, item: T, *, serialized: str) -> T:
        """Register an item by stable serialized ID."""
        existing = self._items.get(serialized)
        if existing is not None and existing != item:
            raise ValueError(f"Conflicting registration for {serialized!r}.")
        self._items[serialized] = item
        return item

    def get(self, serialized: str) -> T:
        """Return a registered item by stable serialized ID."""
        try:
            return self._items[serialized]
        except KeyError as exc:
            raise KeyError(f"Unknown Ember symbol {serialized!r}.") from exc

    def values(self) -> tuple[T, ...]:
        """Return registered values in insertion order."""
        return tuple(self._items.values())


PRODUCTS = SymbolRegistry[ProductKey]()
STAGES = SymbolRegistry[StageKey]()
BUFFERS = SymbolRegistry[BufferKey]()
PROVIDERS = SymbolRegistry[ProviderKey]()
TRAITS = SymbolRegistry[TraitKey]()
SCENE_FAMILIES = SymbolRegistry[SceneFamilyKey]()
BACKEND_IDS = SymbolRegistry[BackendId]()

_EXTENSIONS_LOADED = False


def load_ember_extensions(*, force: bool = False) -> None:
    """Load third-party Ember extension entry points."""
    global _EXTENSIONS_LOADED
    if _EXTENSIONS_LOADED and not force:
        return
    for entry_point in entry_points(group="ember.extensions"):
        register = entry_point.load()
        if not callable(register):
            raise TypeError(
                f"Ember extension entry point {entry_point.name!r} is not callable."
            )
        register()
    _EXTENSIONS_LOADED = True
