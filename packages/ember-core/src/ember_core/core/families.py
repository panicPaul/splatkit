"""Core scene-family keys."""

from __future__ import annotations

from typing import Final

from ember_core.core.keys import SceneFamilyKey, serialized_id

GAUSSIAN: Final = SceneFamilyKey("core", "gaussian")
SPARSE_VOXEL: Final = SceneFamilyKey("core", "sparse_voxel")


def scene_family_id(value: str | SceneFamilyKey) -> str:
    """Return the runtime scene-family string."""
    serialized = serialized_id(value)
    if serialized == GAUSSIAN.serialized:
        return "gaussian"
    if serialized == SPARSE_VOXEL.serialized:
        return "sparse_voxel"
    return serialized
