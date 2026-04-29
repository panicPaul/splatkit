from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from marimo_config_gui import load_preset_config, load_script_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "fastergs" / "notebook.py"


def load_fastergs_preset(fastergs_config_module, name: str):
    return load_preset_config(
        fastergs_config_module.fastergs_preset_catalog(), name
    )


def load_fastergs_script_config(
    fastergs_config_module,
    args: list[str],
):
    return load_script_config(
        fastergs_config_module.FasterGSExperimentConfig,
        args=args,
        presets=fastergs_config_module.fastergs_preset_catalog(),
    )


@pytest.fixture
def fastergs_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.fastergs.notebook",
        NOTEBOOK_PATH,
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
        experiment_config = load_fastergs_preset(
            fastergs_config_module, "garden_baseline"
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
        assert [
            group.target.name
            for group in training_config.optimization.parameter_groups
        ] == [
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
    loaded = load_fastergs_script_config(
        fastergs_config_module,
        args=[
            "preset",
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
    config = load_fastergs_preset(fastergs_config_module, "garden_baseline")
    json_path = tmp_path / "fastergs_config.json"
    json_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))

    loaded = load_fastergs_script_config(
        fastergs_config_module,
        args=["json", str(json_path)],
    )

    assert isinstance(
        loaded,
        fastergs_config_module.FasterGSExperimentConfig,
    )
    assert loaded == config


def test_fastergs_script_loader_resolves_relative_paths_from_json_file(
    fastergs_config_module,
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "config.json"
    json_path.write_text(
        json.dumps(
            {
                "preset": "garden_baseline",
                "backend": "adapter.fastergs",
                "scene": {
                    "path": "dataset",
                    "image_root": None,
                    "undistort_output_dir": None,
                    "align_horizon": True,
                },
                "data": {
                    "camera_sensor_id": None,
                    "image_scale_factor": 0.25,
                    "split_target": "train",
                    "split_every_n": 8,
                    "materialization_stage": "decoded",
                    "materialization_mode": "eager",
                    "materialization_num_workers": 0,
                    "normalize_images": True,
                    "interpolation": "bicubic",
                },
                "model": {
                    "sh_degree": 3,
                    "initial_scale": 0.01,
                    "initial_opacity": 0.1,
                    "default_color": [0.5, 0.5, 0.5],
                },
                "render": {
                    "proper_antialiasing": False,
                    "near_plane": 0.2,
                    "far_plane": 10_000.0,
                    "background_color": [0.0, 0.0, 0.0],
                },
                "optimization": {
                    "optimizer": "ember_splatting_training.FusedAdam",
                    "means_lr_init": 0.00016,
                    "means_lr_final": 0.0000016,
                    "means_lr_max_steps": 30_000,
                    "sh_dc_lr": 0.0025,
                    "sh_rest_lr": 0.000125,
                    "opacity_lr": 0.025,
                    "scale_lr": 0.005,
                    "rotation_lr": 0.001,
                },
                "loss": {
                    "lambda_l1": 0.8,
                    "lambda_dssim": 0.2,
                    "lambda_opacity_regularization": 0.0,
                    "lambda_scale_regularization": 0.0,
                },
                "densification": {
                    "use_mcmc": False,
                    "refine_every": 100,
                    "start_iter": 600,
                    "stop_iter": 14_900,
                    "grad_threshold": 0.0002,
                    "dense_fraction": 0.01,
                    "prune_opacity_threshold": 0.005,
                    "opacity_reset_every": 3000,
                    "max_reset_opacity": 0.01,
                    "min_opacity": 0.005,
                    "max_primitives": 1_000_000,
                    "noise_lr_scale": 500_000.0,
                },
                "checkpoint": {
                    "output_dir": "checkpoints/run",
                    "export_ply": True,
                    "overwrite": False,
                },
                "execution": {
                    "device": "cuda",
                    "seed": 0,
                    "max_steps": 30_000,
                    "batch_size": 1,
                    "shuffle": True,
                },
            }
        )
    )

    loaded = load_fastergs_script_config(
        fastergs_config_module,
        args=["json", str(json_path)],
    )

    assert loaded.scene.path == (tmp_path / "dataset")
    assert loaded.checkpoint.output_dir == (tmp_path / "checkpoints/run")


def test_fastergs_default_checkpoint_layout_is_mirrored_by_paper_and_backend(
    fastergs_config_module,
) -> None:
    baseline = load_fastergs_preset(fastergs_config_module, "garden_baseline")
    mcmc = load_fastergs_preset(fastergs_config_module, "garden_mcmc")

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
    experiment_config = load_fastergs_preset(
        fastergs_config_module, "garden_baseline"
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


def test_fastergs_checkpoint_overwrite_guard(
    fastergs_config_module,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "checkpoint"
    output_dir.mkdir()
    (output_dir / "model.ckpt").write_text("existing")

    with pytest.raises(FileExistsError, match=r"checkpoint\.overwrite=true"):
        fastergs_config_module.ensure_checkpoint_output_writable(
            output_dir,
            overwrite=False,
        )

    fastergs_config_module.ensure_checkpoint_output_writable(
        output_dir,
        overwrite=True,
    )
