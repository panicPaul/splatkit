from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from marimo_config_gui.api import load_script_config
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "fastgs" / "notebook.py"


def load_fastgs_preset(fastgs_config_module, name: str):
    return load_preset_config(
        fastgs_config_module.fastgs_preset_catalog(), name
    )


def load_fastgs_script_config(
    fastgs_config_module,
    args: list[str],
):
    return load_script_config(
        fastgs_config_module.FastGSExperimentConfig,
        args=args,
        presets=fastgs_config_module.fastgs_preset_catalog(),
    )


def load_fastgs_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.fastgs.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fastgs_resolved_training_config_defaults_to_native_backend() -> None:
    module = load_fastgs_config_module()
    experiment_config = load_fastgs_preset(module, "garden_base")

    training_config = module.resolve_training_config(experiment_config)

    assert experiment_config.training.render.backend == "faster_gs.fastgs"
    assert training_config.render.backend == "faster_gs.fastgs"
    assert training_config.render.backend_options == {
        "near_plane": 0.2,
        "far_plane": 10_000.0,
        "mip_splatting_screen_filter": False,
        "compact_box_scale": 0.5,
        "background_color": [0.0, 0.0, 0.0],
    }
    assert (
        training_config.initialization.initializer.target
        == "papers.fastgs.notebook.initialize_fastgs_model_from_scene_record"
    )
    assert (
        training_config.densification.builders[0].target
        == "papers.fastgs.notebook.FastGSVanillaDensification"
    )
    assert (
        training_config.densification.builders[-1].target
        == "papers.fastgs.notebook.FastGSFinalCleanup"
    )
    assert (
        training_config.checkpoint.output_dir
        == REPO_ROOT / "checkpoints/papers/fastgs/garden_base/faster_gs.fastgs"
    )


def test_fastgs_script_loader_applies_preset_then_cli_overrides() -> None:
    module = load_fastgs_config_module()

    loaded = load_fastgs_script_config(
        module,
        args=[
            "--preset",
            "garden_base",
            "--training.render.backend",
            "adapter.fastgs",
            "--training.runtime.max-steps",
            "5",
        ],
    )

    assert isinstance(loaded, module.FastGSExperimentConfig)
    assert loaded.preset == "garden_base"
    assert loaded.training.render.backend == "adapter.fastgs"
    assert loaded.training.runtime.max_steps == 5

    training_config = module.resolve_training_config(loaded)
    assert training_config.render.backend == "adapter.fastgs"
    assert training_config.render.backend_options == {
        "mult": 0.5,
        "background_color": [0.0, 0.0, 0.0],
    }


def test_fastgs_user_config_does_not_expose_runtime_kwargs() -> None:
    module = load_fastgs_config_module()
    config = load_fastgs_preset(module, "garden_base")
    serialized = json.dumps(config.model_dump(mode="json"))

    assert '"kwargs"' not in serialized
    assert '"context_kwargs"' not in serialized
    assert "proper_antialiasing" not in serialized


def test_fastgs_script_loader_replays_json_config(tmp_path: Path) -> None:
    module = load_fastgs_config_module()
    config = load_fastgs_preset(module, "garden_base")
    json_path = tmp_path / "fastgs_config.json"
    json_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))

    loaded = load_fastgs_script_config(
        module,
        args=[str(json_path)],
    )

    assert isinstance(loaded, module.FastGSExperimentConfig)
    assert loaded == config


def test_fastgs_scene_loader_uses_prebuilt_scaled_images(
    tmp_path: Path,
) -> None:
    module = load_fastgs_config_module()
    scene_path = tmp_path / "garden"
    image_root = scene_path / "images_4"
    image_root.mkdir(parents=True)
    config = load_fastgs_preset(module, "garden_base").model_copy(
        update={
            "scene": module.FastGSSceneConfig(path=scene_path),
        }
    )

    scene_config = module.build_scene_load_config(config)

    assert scene_config.image_root == image_root


def test_fastgs_densification_accumulators_follow_clone_and_split() -> None:
    module = load_fastgs_config_module()
    densification = module.FastGSVanillaDensification()
    densification.clone_grad_sum = torch.arange(4, dtype=torch.float32)
    densification.split_grad_sum = torch.arange(10, 14, dtype=torch.float32)
    densification.visible_count = torch.ones(4)
    densification.max_screen_radii = torch.arange(20, 24, dtype=torch.float32)

    densification._append_zero_accumulator_values(2)
    split_mask = torch.tensor([False, True, False, False, False, True])
    densification._split_accumulator_values(split_mask, num_children=2)

    assert densification.max_screen_radii is not None
    assert densification.max_screen_radii.shape == (8,)
    torch.testing.assert_close(
        densification.max_screen_radii,
        torch.tensor([20.0, 22.0, 23.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
