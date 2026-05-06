from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import ember_core as ember
import pytest
import torch
from marimo_config_gui.api import load_script_config
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "scaffold_gs" / "notebook.py"


def load_scaffold_gs_preset(scaffold_gs_config_module, name: str):
    return load_preset_config(
        scaffold_gs_config_module.scaffold_gs_preset_catalog(), name
    )


def load_scaffold_gs_script_config(
    scaffold_gs_config_module,
    args: list[str],
):
    return load_script_config(
        scaffold_gs_config_module.ScaffoldGSExperimentConfig,
        args=args,
        presets=scaffold_gs_config_module.scaffold_gs_preset_catalog(),
    )


@pytest.fixture
def scaffold_gs_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.scaffold_gs.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scaffold_gs_resolved_training_config_uses_direct_rgb_backend(
    scaffold_gs_config_module,
) -> None:
    experiment_config = load_scaffold_gs_preset(
        scaffold_gs_config_module, "garden_scaffold_gs"
    )
    training_config = scaffold_gs_config_module.resolve_training_config(
        experiment_config
    )

    assert training_config.render.backend == "faster_gs.core"
    assert training_config.render.backend_options == {
        "near_plane": 0.01,
        "far_plane": 1000.0,
        "color_source": "direct_rgb",
        "mip_splatting_screen_filter": True,
        "clamp_output": True,
        "background_color": [0.0, 0.0, 0.0],
    }
    assert (
        training_config.render.feature_fn.target
        == "papers.scaffold_gs.notebook.scaffold_gs_render_scene"
    )
    assert (
        training_config.initialization.initializer.target
        == (
            "papers.scaffold_gs.notebook."
            "initialize_scaffold_gs_model_from_scene_record"
        )
    )
    assert training_config.initialization.initializer.context_kwargs == {
        "device": "device"
    }
    assert [
        group.target.name
        for group in training_config.optimization.parameter_groups
    ] == [
        "center_position",
        "feature",
        "logit_opacity",
        "log_scales",
        "quaternion_orientation",
        "anchor_offsets",
        "opacity_mlp",
        "covariance_mlp",
        "color_mlp",
        "feature_bank_mlp",
        "appearance_embedding",
    ]
    assert (
        training_config.loss.target.target
        == "papers.scaffold_gs.notebook.scaffold_gs_rgb_loss"
    )
    assert training_config.checkpoint.output_dir == (
        REPO_ROOT
        / "checkpoints"
        / "papers"
        / "scaffold_gs"
        / "garden_scaffold_gs"
        / "faster_gs.core"
    )


def test_scaffold_gs_default_preset_is_paper_training_not_debug(
    scaffold_gs_config_module,
) -> None:
    catalog = scaffold_gs_config_module.scaffold_gs_preset_catalog()
    default_config = load_preset_config(catalog, catalog.default)
    debug_config = load_scaffold_gs_preset(
        scaffold_gs_config_module, "garden_debug_val"
    )

    assert catalog.default == "garden_scaffold_gs"
    assert default_config.training.optimization.iterations == 30000
    assert default_config.data.split_target == "train"
    assert default_config.data.image_scale_factor == 1.0
    assert debug_config.training.optimization.iterations == 3000
    assert debug_config.data.split_target == "val"
    assert debug_config.data.image_scale_factor == 0.1


def test_scaffold_gs_script_loader_applies_preset_then_cli_overrides(
    scaffold_gs_config_module,
) -> None:
    loaded = load_scaffold_gs_script_config(
        scaffold_gs_config_module,
        args=[
            "--preset",
            "garden_scaffold_gs",
            "--training.optimization.iterations",
            "5",
            "--training.profiler.enabled",
            "True",
            "--training.profiler.log-every",
            "7",
        ],
    )

    assert isinstance(
        loaded,
        scaffold_gs_config_module.ScaffoldGSExperimentConfig,
    )
    assert loaded.preset == "garden_scaffold_gs"
    assert loaded.training.optimization.iterations == 5
    assert loaded.training.profiler.enabled is True
    assert loaded.training.profiler.log_every == 7

    training_config = scaffold_gs_config_module.resolve_training_config(loaded)
    assert training_config.runtime.max_steps == 5
    assert training_config.profiler.enabled is True
    assert training_config.profiler.log_every == 7


def test_scaffold_gs_user_config_does_not_expose_runtime_kwargs(
    scaffold_gs_config_module,
) -> None:
    config = load_scaffold_gs_preset(
        scaffold_gs_config_module, "garden_scaffold_gs"
    )
    serialized = json.dumps(config.model_dump(mode="json"))

    assert '"kwargs"' not in serialized
    assert '"context_kwargs"' not in serialized


def test_scaffold_gs_script_loader_replays_json_config(
    scaffold_gs_config_module,
    tmp_path: Path,
) -> None:
    config = load_scaffold_gs_preset(
        scaffold_gs_config_module, "garden_scaffold_gs"
    )
    json_path = tmp_path / "scaffold_gs_config.json"
    json_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))

    loaded = load_scaffold_gs_script_config(
        scaffold_gs_config_module,
        args=[str(json_path)],
    )

    assert isinstance(
        loaded,
        scaffold_gs_config_module.ScaffoldGSExperimentConfig,
    )
    assert loaded == config


def test_scaffold_gs_initializer_builds_anchor_scene_and_modules(
    scaffold_gs_config_module,
) -> None:
    point_cloud = ember.PointCloudState(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        colors=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )
    scene_record = ember.SceneRecord(
        sensors=(),
        source_format="colmap",
        point_cloud=point_cloud,
    )

    initialized = (
        scaffold_gs_config_module.initialize_scaffold_gs_model_from_scene_record(
            scene_record,
            anchor_feature_dimension=8,
            neural_offsets_per_anchor=4,
            appearance_embedding_dimension=2,
            device=torch.device("cpu"),
        )
    )

    assert isinstance(initialized.scene, ember.GaussianScene3D)
    assert initialized.scene.sh_degree == 0
    assert initialized.scene.feature.shape == (3, 8)
    torch.testing.assert_close(
        initialized.scene.feature[:, :3],
        point_cloud.colors,
    )
    assert initialized.parameters["anchor_offsets"].shape == (3, 4, 3)
    assert set(initialized.modules) == {
        "opacity_mlp",
        "covariance_mlp",
        "color_mlp",
        "feature_bank_mlp",
        "appearance_embedding",
    }
