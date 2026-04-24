"""Dataset source loaders."""

from splatkit.data.loaders.colmap import load_colmap_dataset
from splatkit.data.loaders.must3r import (
    Must3rCheckpointPaths,
    Must3rRuntime,
    SubprocessMust3rSlamRuntime,
    load_must3r_dataset,
    resolve_must3r_checkpoints,
    run_must3r_dataset,
)
from splatkit.data.loaders.ncore import load_ncore_dataset

load_colmap_scene_record = load_colmap_dataset
load_must3r_scene_record = load_must3r_dataset
run_must3r_scene_record = run_must3r_dataset
load_ncore_scene_record = load_ncore_dataset

__all__ = [
    "Must3rCheckpointPaths",
    "Must3rRuntime",
    "SubprocessMust3rSlamRuntime",
    "load_colmap_scene_record",
    "load_must3r_scene_record",
    "load_ncore_scene_record",
    "resolve_must3r_checkpoints",
    "run_must3r_scene_record",
]
