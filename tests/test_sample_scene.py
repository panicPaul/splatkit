from __future__ import annotations

from pathlib import Path

from ember_core import (
    get_sample_scene_path,
    initialize_gaussian_scene_from_scene_record,
    load_colmap_scene_record,
)


def test_get_sample_scene_path_resolves_packaged_colmap_root() -> None:
    scene_root = get_sample_scene_path()
    assert isinstance(scene_root, Path)
    assert scene_root.exists()
    assert (scene_root / "images").exists()
    assert (scene_root / "sparse" / "0" / "cameras.txt").exists()


def test_bundled_sample_scene_loads_and_initializes() -> None:
    scene_record = load_colmap_scene_record(get_sample_scene_path())
    assert scene_record.num_frames == 6
    assert scene_record.point_cloud is not None
    assert scene_record.point_cloud.points.shape == (8332, 3)

    scene = initialize_gaussian_scene_from_scene_record(scene_record)
    assert scene.center_position.shape == (8332, 3)
    assert scene.feature.shape == (8332, 1, 3)
