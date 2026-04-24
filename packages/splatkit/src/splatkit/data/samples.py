"""Helpers for bundled sample datasets."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import Literal

SampleSceneName = Literal["bicycle_smoke"]

_DEFAULT_SAMPLE_SCENE: SampleSceneName = "bicycle_smoke"
_COLMAP_ENV_VAR = "SPLATKIT_COLMAP_ROOT"


def get_sample_scene_path(
    name: SampleSceneName = _DEFAULT_SAMPLE_SCENE,
) -> Path:
    """Return the filesystem path for a bundled sample scene."""
    scene_root = Path(
        str(files("splatkit").joinpath("assets", "scenes", name))
    )
    if not scene_root.exists():
        raise FileNotFoundError(
            f"Bundled sample scene {name!r} was not found at {scene_root}."
        )
    return scene_root


def resolve_colmap_scene_path(
    path: str | Path | None = None,
    *,
    env_var: str = _COLMAP_ENV_VAR,
    default_sample: SampleSceneName = _DEFAULT_SAMPLE_SCENE,
) -> Path:
    """Resolve an explicit path, env override, or bundled sample scene."""
    if path is not None:
        return Path(path).expanduser()
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    return get_sample_scene_path(default_sample)


__all__ = [
    "SampleSceneName",
    "get_sample_scene_path",
    "resolve_colmap_scene_path",
]
