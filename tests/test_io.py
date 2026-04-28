from __future__ import annotations

from pathlib import Path

import torch
from ember_core import (
    GaussianScene3D,
    SparseVoxelScene,
    load_gaussian_ply,
    load_scene,
    load_svraster_checkpoint,
    save_gaussian_ply,
    save_scene,
    save_svraster_checkpoint,
)


def test_gaussian_ply_roundtrip(
    tmp_path: Path, cpu_scene: GaussianScene3D
) -> None:
    path = tmp_path / "scene.ply"
    save_gaussian_ply(cpu_scene, path)
    loaded = load_gaussian_ply(path)
    assert loaded.center_position.shape == cpu_scene.center_position.shape
    assert loaded.log_scales.shape == cpu_scene.log_scales.shape
    assert (
        loaded.quaternion_orientation.shape
        == cpu_scene.quaternion_orientation.shape
    )
    assert loaded.logit_opacity.shape == cpu_scene.logit_opacity.shape
    assert loaded.feature.shape == cpu_scene.feature.shape


def test_svraster_checkpoint_roundtrip(
    tmp_path: Path,
    cpu_sparse_voxel_scene: SparseVoxelScene,
) -> None:
    run_dir = tmp_path / "scene"
    save_svraster_checkpoint(cpu_sparse_voxel_scene, run_dir, iteration=7)
    loaded = load_svraster_checkpoint(run_dir, iteration=7)
    assert torch.equal(loaded.octpath, cpu_sparse_voxel_scene.octpath)
    assert torch.equal(loaded.octlevel, cpu_sparse_voxel_scene.octlevel)
    assert torch.equal(loaded.geo_grid_pts, cpu_sparse_voxel_scene.geo_grid_pts)
    assert torch.equal(loaded.sh0, cpu_sparse_voxel_scene.sh0)
    assert torch.equal(loaded.shs, cpu_sparse_voxel_scene.shs)


def test_generic_scene_load_and_save(
    tmp_path: Path,
    cpu_scene: GaussianScene3D,
    cpu_sparse_voxel_scene: SparseVoxelScene,
) -> None:
    ply_path = tmp_path / "scene.ply"
    save_scene(cpu_scene, ply_path)
    loaded_gaussian = load_scene(ply_path)
    assert isinstance(loaded_gaussian, GaussianScene3D)

    checkpoint_dir = tmp_path / "sv"
    save_scene(
        cpu_sparse_voxel_scene,
        checkpoint_dir,
        iteration=3,
    )
    loaded_sparse = load_scene(
        checkpoint_dir,
        format="svraster_checkpoint",
        iteration=3,
    )
    assert isinstance(loaded_sparse, SparseVoxelScene)
