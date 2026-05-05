from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from marimo_config_gui.api import load_script_config
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "svraster" / "notebook.py"
SVRASTER_TRAINING_SOURCE = (
    REPO_ROOT / "packages" / "ember-svraster-training" / "src"
)
NATIVE_SVRASTER_SOURCE = (
    REPO_ROOT / "packages" / "ember-native-svraster" / "src"
)

for source_path in (SVRASTER_TRAINING_SOURCE, NATIVE_SVRASTER_SOURCE):
    source_path_text = str(source_path)
    if source_path_text not in sys.path:
        sys.path.insert(0, source_path_text)


def load_svraster_preset(svraster_config_module, name: str):
    return load_preset_config(
        svraster_config_module.svraster_preset_catalog(), name
    )


def load_svraster_script_config(
    svraster_config_module,
    args: list[str],
):
    return load_script_config(
        svraster_config_module.SVRasterExperimentConfig,
        args=args,
        presets=svraster_config_module.svraster_preset_catalog(),
    )


def load_svraster_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.svraster.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_svraster_training_package_recipe_targets_native_sparse_adam() -> None:
    import ember_svraster_training

    parameter_groups = ember_svraster_training.svraster_parameter_groups()

    assert [group.target.name for group in parameter_groups] == [
        "geo_grid_pts",
        "sh0",
        "shs",
    ]
    assert all(
        group.optimizer == "ember_svraster_training.SVRasterSparseAdam"
        for group in parameter_groups
    )
    assert [group.lr for group in parameter_groups] == [0.025, 0.010, 0.00025]
    assert parameter_groups[0].optimizer_kwargs == {
        "betas": (0.1, 0.99),
        "eps": 1e-15,
        "sparse": False,
        "biased": False,
    }
    assert parameter_groups[0].scheduler is not None
    assert (
        parameter_groups[0].scheduler.target
        == "torch.optim.lr_scheduler.MultiStepLR"
    )
    assert parameter_groups[0].scheduler.kwargs == {
        "milestones": [19000],
        "gamma": 0.1,
    }


def test_svraster_resolved_training_config_uses_native_training_package() -> (
    None
):
    module = load_svraster_config_module()
    experiment_config = load_svraster_preset(module, "garden_svraster")

    training_config = module.resolve_training_config(experiment_config)

    assert training_config.render.backend == "svraster.core"
    assert training_config.render.backend_options == {
        "near_plane": 0.02,
        "background_color": [0.0, 0.0, 0.0],
    }
    assert training_config.initialization.initializer.target == (
        "papers.svraster.notebook.initialize_svraster_model_from_scene_record"
    )
    assert training_config.optimization.builder is not None
    assert (
        training_config.optimization.builder.target
        == "ember_svraster_training.svraster_optimization_config"
    )
    recipe = training_config.optimization.builder.kwargs["recipe"]
    assert recipe["geo_lr"] == 0.025
    assert recipe["betas"] == (0.1, 0.99)
    assert (
        training_config.loss.target.target
        == "papers.svraster.notebook.svraster_rgb_loss"
    )
    assert (
        training_config.hooks.builders[0].target
        == "ember_svraster_training.SVRasterTVDensityHook"
    )
    assert (
        training_config.checkpoint.output_dir
        == REPO_ROOT
        / "checkpoints/papers/svraster/garden_svraster/svraster.core"
    )


def test_svraster_script_loader_applies_preset_then_cli_overrides() -> None:
    module = load_svraster_config_module()

    loaded = load_svraster_script_config(
        module,
        args=[
            "--preset",
            "garden_svraster",
            "--training.runtime.max-steps",
            "5",
            "--training.optimization.geo-lr",
            "0.03",
        ],
    )

    assert isinstance(loaded, module.SVRasterExperimentConfig)
    assert loaded.preset == "garden_svraster"
    assert loaded.training.runtime.max_steps == 5
    assert loaded.training.optimization.geo_lr == 0.03

    training_config = module.resolve_training_config(loaded)
    assert training_config.runtime.max_steps == 5
    assert training_config.optimization.builder is not None
    assert (
        training_config.optimization.builder.kwargs["recipe"]["geo_lr"] == 0.03
    )


def test_svraster_user_config_does_not_expose_runtime_kwargs() -> None:
    module = load_svraster_config_module()
    config = load_svraster_preset(module, "garden_svraster")
    serialized = json.dumps(config.model_dump(mode="json"))

    assert '"kwargs"' not in serialized
    assert '"context_kwargs"' not in serialized


def test_svraster_script_loader_replays_json_config(tmp_path: Path) -> None:
    module = load_svraster_config_module()
    config = load_svraster_preset(module, "garden_svraster")
    json_path = tmp_path / "svraster_config.json"
    json_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))

    loaded = load_svraster_script_config(
        module,
        args=[str(json_path)],
    )

    assert isinstance(loaded, module.SVRasterExperimentConfig)
    assert loaded == config


def test_svraster_adaptive_schedules_match_upstream_shapes() -> None:
    module = load_svraster_config_module()

    assert module.svraster_step_includes(500, 500, 15000, 100) is True
    assert module.svraster_step_includes(550, 500, 15000, 100) is False
    assert module.svraster_step_includes(15100, 500, 15000, 100) is False
    assert (
        module.svraster_prune_threshold(
            500,
            adapt_from=500,
            prune_until=15000,
            initial=0.0001,
            final=0.01,
        )
        == 0.0001
    )
    assert (
        module.svraster_prune_threshold(
            15000,
            adapt_from=500,
            prune_until=15000,
            initial=0.0001,
            final=0.01,
        )
        == 0.01
    )
    assert module.svraster_max_subdivide_count(100, 170) == 10
