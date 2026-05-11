"""Typed backend references for IDE-friendly authoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ember_core.core.keys import BackendId

SceneT = TypeVar("SceneT")
OptionsT = TypeVar("OptionsT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class BackendRef(Generic[SceneT, OptionsT, OutputT]):
    """Typed backend handle."""

    id: BackendId
    scene_type: type[SceneT]
    options_type: type[OptionsT]
    output_type: type[OutputT]

    @property
    def serialized(self) -> str:
        """Return the stable backend ID."""
        return self.id.serialized

    def options(self, **kwargs: Any) -> OptionsT:
        """Build typed options for this backend."""
        return self.options_type(**kwargs)
