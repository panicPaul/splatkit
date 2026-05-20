from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import ember_core as ember
import pytest
import torch
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "radfoam" / "notebook.py"
RADFOAM_PACKAGE_SRC = REPO_ROOT / "packages" / "ember-native-radfoam" / "src"
sys.path.insert(0, str(RADFOAM_PACKAGE_SRC))


def make_radfoam_scene(num_points: int = 32) -> ember.RadFoamScene:
    return ember.RadFoamScene(
        primal_points=torch.zeros((num_points, 3), dtype=torch.float32),
        density=torch.zeros((num_points, 1), dtype=torch.float32),
        att_dc=torch.zeros((num_points, 3), dtype=torch.float32),
        att_sh=torch.zeros((num_points, 24), dtype=torch.float32),
        point_adjacency=torch.zeros((0,), dtype=torch.uint32),
        point_adjacency_offsets=torch.zeros((num_points + 1,), dtype=torch.uint32),
        sh_degree=2,
    )


def load_radfoam_preset(radfoam_config_module, name: str):
    return load_preset_config(
        radfoam_config_module.radfoam_preset_catalog(),
        name,
    )


@pytest.fixture
def radfoam_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.radfoam.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_radfoam_scene_contract_exports_core_family() -> None:
    scene = make_radfoam_scene()

    assert scene.scene_family == "radfoam"
    assert ember.RADFOAM.serialized == "core.radfoam"
    assert ember.scene_family_id(ember.RADFOAM.serialized) == "foam"


def test_radfoam_scene_validates_sh_layout() -> None:
    with pytest.raises(ValueError, match="att_sh"):
        ember.RadFoamScene(
            primal_points=torch.zeros((32, 3), dtype=torch.float32),
            density=torch.zeros((32, 1), dtype=torch.float32),
            att_dc=torch.zeros((32, 3), dtype=torch.float32),
            att_sh=torch.zeros((32, 1), dtype=torch.float32),
            point_adjacency=torch.zeros((0,), dtype=torch.uint32),
            point_adjacency_offsets=torch.zeros((33,), dtype=torch.uint32),
            sh_degree=2,
        )


def test_radfoam_scene_validates_minimum_topology_size() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        make_radfoam_scene(num_points=31)


def test_radfoam_package_import_and_register() -> None:
    import ember_native_radfoam as radfoam
    from ember_core.core.registry import BACKEND_REGISTRY

    radfoam.register()

    assert "radfoam.core" in BACKEND_REGISTRY


def test_radfoam_default_preset_is_base(radfoam_config_module) -> None:
    catalog = radfoam_config_module.radfoam_preset_catalog()
    config = load_preset_config(catalog)

    assert catalog.default == "garden_base"
    assert config.preset == "garden_base"


def test_radfoam_notebook_resolves_training_config(
    radfoam_config_module,
) -> None:
    config = load_radfoam_preset(radfoam_config_module, "garden_debug")
    training_config = radfoam_config_module.resolve_training_config(config)

    assert training_config.runtime.device == "cuda"
    assert training_config.runtime.max_steps == 100
    assert training_config.render.backend == "radfoam.core"
    assert training_config.render.backend_options == {
        "weight_threshold": 0.001,
        "max_intersections": 512,
        "density_beta": 10.0,
        "background_color": [0.0, 0.0, 0.0],
        "clamp_output": False,
    }
    assert (
        training_config.initialization.initializer.target
        == "ember_native_radfoam.initialize_radfoam_model_from_scene_record"
    )
    assert [
        group.target.name
        for group in training_config.optimization.parameter_groups
    ] == [
        "primal_points",
        "density",
        "att_dc",
        "att_sh",
    ]
    assert (
        training_config.loss.target.target
        == "ember_native_radfoam.radfoam_rgb_loss"
    )
    assert training_config.densification is not None
    assert (
        training_config.densification.builders[0].target
        == "ember_native_radfoam.RadFoamTopologyRefresh"
    )


def test_radfoam_notebook_builds_scene_and_dataset_configs(
    radfoam_config_module,
) -> None:
    config = load_radfoam_preset(radfoam_config_module, "garden_debug")
    config.data.cache_resized_images = False

    scene_config = radfoam_config_module.build_scene_load_config(config)
    dataset_config = (
        radfoam_config_module.build_prepared_frame_dataset_config(config)
    )

    assert scene_config.path == config.scene.path
    assert scene_config.image_root is None
    assert len(scene_config.source_pipes) == 1
    assert dataset_config.camera_sensor_id is None
    assert dataset_config.split is not None
    assert dataset_config.split.target == "train"
    assert dataset_config.split.every_n == 8
    assert dataset_config.materialization is not None
    assert dataset_config.materialization.stage == "prepared"
    assert dataset_config.materialization.mode == "eager"
    assert dataset_config.image_preparation is not None
    assert dataset_config.image_preparation.resize_width_scale == 0.25
    assert dataset_config.image_preparation.interpolation == "bicubic"
    assert config.training.viewer.enabled is True
