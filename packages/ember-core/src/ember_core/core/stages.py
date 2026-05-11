"""Typed stage specs for backend/provider composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ember_core.core.buffers import BufferRef, BufferSpec
from ember_core.core.keys import ProductKey, ProviderKey, StageKey, TraitKey


@dataclass(frozen=True, slots=True)
class StageSpec:
    """A declarative execution stage."""

    key: StageKey
    provider: ProviderKey
    reads: tuple[BufferRef[Any], ...] = ()
    writes: tuple[BufferRef[Any], ...] = ()
    mutates: tuple[BufferRef[Any], ...] = ()
    allocates: tuple[BufferSpec[Any], ...] = ()
    products: frozenset[ProductKey] = frozenset()
    requires_products: frozenset[ProductKey] = frozenset()
    requires_traits: tuple[TraitKey, ...] = ()
