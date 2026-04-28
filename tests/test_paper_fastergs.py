from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "papers" / "fastergs" / "config.py"


@pytest.fixture
def fastergs_config_module():
    spec = importlib.util.spec_from_file_location(
        "paper_fastergs_config",
        CONFIG_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fastergs_build_training_config_supports_both_backends(
    fastergs_config_module,
) -> None:
    for backend in ("adapter.fastergs", "faster_gs.core"):
        experiment_config = fastergs_config_module.load_default_experiment_config(
            "garden_baseline"
        ).model_copy(update={"backend": backend})
        training_config = fastergs_config_module.build_training_config(
            experiment_config
        )

        assert training_config.render.backend == backend
        assert training_config.render.backend_options == {
            "near_plane": 0.2,
            "far_plane": 10_000.0,
            "proper_antialiasing": False,
            "background_color": [0.0, 0.0, 0.0],
        }
        assert [group.target.name for group in training_config.optimization.parameter_groups] == [
            "center_position",
            "feature",
            "feature",
            "logit_opacity",
            "log_scales",
            "quaternion_orientation",
        ]
        assert (
            training_config.optimization.parameter_groups[0].scheduler.target
            == "ember_core.training.exponential_decay_to"
        )
        assert training_config.model_dump(mode="json")["render"][
            "backend_options"
        ]["background_color"] == [0.0, 0.0, 0.0]


def test_fastergs_script_loader_applies_preset_then_cli_overrides(
    fastergs_config_module,
) -> None:
    loaded = fastergs_config_module.load_experiment_script_config(
        fastergs_config_module.FasterGSExperimentConfig,
        args=[
            "cli",
            "--preset",
            "garden_mcmc",
            "--backend",
            "faster_gs.core",
            "--execution.max-steps",
            "5",
            "--loss.lambda-l1",
            "1.0",
        ],
    )

    assert isinstance(
        loaded,
        fastergs_config_module.FasterGSExperimentConfig,
    )
    assert loaded.preset == "garden_mcmc"
    assert loaded.backend == "faster_gs.core"
    assert loaded.execution.max_steps == 5
    assert loaded.loss.lambda_l1 == 1.0
    assert loaded.loss.lambda_opacity_regularization == 0.01
    assert loaded.loss.lambda_scale_regularization == 0.01
    assert loaded.densification.use_mcmc is True


def test_fastergs_script_loader_replays_json_config(
    fastergs_config_module,
    tmp_path: Path,
) -> None:
    config = fastergs_config_module.load_default_experiment_config(
        "garden_baseline"
    )
    json_path = tmp_path / "fastergs_config.json"
    json_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))

    loaded = fastergs_config_module.load_experiment_script_config(
        fastergs_config_module.FasterGSExperimentConfig,
        args=["json", str(json_path)],
    )

    assert isinstance(
        loaded,
        fastergs_config_module.FasterGSExperimentConfig,
    )
    assert loaded == config


def test_fastergs_script_loader_resolves_relative_paths_from_json_file(
    fastergs_config_module,
) -> None:
    default_json_path = (
        REPO_ROOT
        / "papers"
        / "fastergs"
        / "defaults"
        / "garden_baseline.json"
    )

    loaded = fastergs_config_module.load_experiment_script_config(
        fastergs_config_module.FasterGSExperimentConfig,
        args=["json", str(default_json_path)],
    )

    assert loaded.scene.path == (REPO_ROOT / "dataset/mipnerf360/garden")
    assert loaded.checkpoint.output_dir == (
        REPO_ROOT
        / "checkpoints"
        / "papers"
        / "fastergs"
        / "garden_baseline"
        / "adapter.fastergs"
    )


def test_fastergs_default_checkpoint_layout_is_mirrored_by_paper_and_backend(
    fastergs_config_module,
) -> None:
    baseline = fastergs_config_module.load_default_experiment_config(
        "garden_baseline"
    )
    mcmc = fastergs_config_module.load_default_experiment_config("garden_mcmc")

    assert baseline.checkpoint.output_dir == (
        REPO_ROOT
        / "checkpoints"
        / "papers"
        / "fastergs"
        / "garden_baseline"
        / "adapter.fastergs"
    )
    assert mcmc.checkpoint.output_dir == (
        REPO_ROOT
        / "checkpoints"
        / "papers"
        / "fastergs"
        / "garden_mcmc"
        / "adapter.fastergs"
    )


def test_fastergs_training_config_retargets_default_checkpoint_dir_with_backend(
    fastergs_config_module,
) -> None:
    experiment_config = fastergs_config_module.load_default_experiment_config(
        "garden_baseline"
    ).model_copy(update={"backend": "faster_gs.core"})

    training_config = fastergs_config_module.build_training_config(
        experiment_config
    )

    assert training_config.checkpoint.output_dir == (
        REPO_ROOT
        / "checkpoints"
        / "papers"
        / "fastergs"
        / "garden_baseline"
        / "faster_gs.core"
    )
