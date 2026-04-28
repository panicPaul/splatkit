"""Generic scene loader/saver dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ember_core.core.contracts import GaussianScene3D, Scene, SparseVoxelScene
from ember_core.io.gaussian_ply import load_gaussian_ply, save_gaussian_ply
from ember_core.io.svraster import (
    load_svraster_checkpoint,
    save_svraster_checkpoint,
)

SceneFormat = Literal["gaussian_ply", "svraster_checkpoint"]


def _infer_load_format(path: Path) -> SceneFormat:
    if path.suffix == ".ply":
        return "gaussian_ply"
    if path.is_dir():
        return "svraster_checkpoint"
    if path.suffix == ".pt":
        return "svraster_checkpoint"
    raise ValueError(f"Could not infer scene format from path {path}.")


def load_scene(
    path: str | Path,
    *,
    format: SceneFormat | None = None,
    iteration: int | None = None,
) -> Scene:
    """Load a scene-only artifact and return the matching scene object."""
    resolved_path = Path(path)
    resolved_format = format or _infer_load_format(resolved_path)
    if resolved_format == "gaussian_ply":
        return load_gaussian_ply(resolved_path)
    if resolved_format == "svraster_checkpoint":
        return load_svraster_checkpoint(resolved_path, iteration=iteration)
    raise ValueError(f"Unsupported scene format {resolved_format!r}.")


def save_scene(
    scene: Scene,
    path: str | Path,
    *,
    format: SceneFormat | None = None,
    iteration: int | None = None,
) -> None:
    """Save a scene-only artifact in the requested format."""
    resolved_path = Path(path)
    resolved_format = format
    if resolved_format is None:
        if isinstance(scene, GaussianScene3D):
            resolved_format = "gaussian_ply"
        elif isinstance(scene, SparseVoxelScene):
            resolved_format = "svraster_checkpoint"
        else:
            raise ValueError(
                f"Could not infer save format for scene type {type(scene).__name__}."
            )

    if resolved_format == "gaussian_ply":
        if not isinstance(scene, GaussianScene3D):
            raise ValueError(
                "gaussian_ply export expects GaussianScene3D, got "
                f"{type(scene).__name__}."
            )
        save_gaussian_ply(scene, resolved_path)
        return
    if resolved_format == "svraster_checkpoint":
        if not isinstance(scene, SparseVoxelScene):
            raise ValueError(
                "svraster_checkpoint export expects SparseVoxelScene, got "
                f"{type(scene).__name__}."
            )
        save_svraster_checkpoint(scene, resolved_path, iteration=iteration)
        return
    raise ValueError(f"Unsupported scene format {resolved_format!r}.")
