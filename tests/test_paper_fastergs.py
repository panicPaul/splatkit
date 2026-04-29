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


def test_fastergs_resolved_training_config_supports_both_backends(
    fastergs_config_module,
) -> None:
    for backend in ("adapter.fastergs", "faster_gs.core"):
        experiment_config = load_fastergs_preset(
            fastergs_config_module, "garden_baseline"
        )
        experiment_config.training.render.backend = backend
        training_config = fastergs_config_module.resolve_training_config(
            experiment_config
        )

        assert training_config.render.backend == backend
        assert training_config.render.backend_options == {
            "near_plane": 0.2,
            "far_plane": 10_000.0,
            "proper_antialiasing": True,
            "background_color": [0.0, 0.0, 0.0],
        }
        assert (
            training_config.initialization.initializer.target
            == "papers.fastergs.notebook.initialize_fastergs_model_from_scene_record"
        )
        assert training_config.initialization.initializer.context_kwargs == {
            "device": "device"
        }
        assert (
            training_config.optimization.builder.target
            == "ember_splatting_training.recipes.gaussian_3dgs_optimization_config"
        )
        assert (
            training_config.loss.target.target
            == "ember_splatting_training.losses.rgb_l1_dssim_loss"
        )
        assert (
            training_config.densification.builders[0].target
            == "papers.fastergs.notebook.FasterGSVanillaDensification"
        )
        assert (
            training_config.render.training_backend_options_builder.target
            == "ember_splatting_training.fastergs_training_backend_options"
        )
        assert (
            training_config.densification.builders[-1].target
            == "papers.fastergs.notebook.FasterGSFinalCleanup"
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
            "--training.render.backend",
            "faster_gs.core",
            "--training.runtime.max-steps",
            "5",
        ],
    )

    assert isinstance(
        loaded,
        fastergs_config_module.FasterGSExperimentConfig,
    )
    assert loaded.preset == "garden_mcmc"
    assert loaded.training.render.backend == "faster_gs.core"
    assert loaded.training.runtime.max_steps == 5
    assert loaded.training.loss.target.kwargs[
        "lambda_opacity_regularization"
    ] == 0.01
    assert (
        loaded.training.densification.builders[0].target
        == "papers.fastergs.notebook.build_fastergs_mcmc_densification"
    )
    assert (
        loaded.training.optimization.builder.kwargs["recipe"][
            "logit_opacity_lr"
        ]
        == 0.05
    )
    assert loaded.training.initialization.initializer.kwargs["use_mcmc"] is True


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
                    "training": {
                        **load_fastergs_preset(
                            fastergs_config_module,
                            "garden_baseline",
                        ).training.model_dump(mode="json"),
                        "checkpoint": {
                            "output_dir": "checkpoints/run",
                            "export_ply": True,
                            "overwrite": False,
                        },
                    },
                }
            )
    )

    loaded = load_fastergs_script_config(
        fastergs_config_module,
        args=["json", str(json_path)],
    )

    assert loaded.scene.path == (tmp_path / "dataset")
    assert loaded.training.checkpoint.output_dir == (
        tmp_path / "checkpoints/run"
    )


def test_fastergs_default_checkpoint_layout_is_mirrored_by_paper_and_backend(
    fastergs_config_module,
) -> None:
    baseline = load_fastergs_preset(fastergs_config_module, "garden_baseline")
    mcmc = load_fastergs_preset(fastergs_config_module, "garden_mcmc")

    assert baseline.training.checkpoint.output_dir == (
        REPO_ROOT
        / "checkpoints"
        / "papers"
        / "fastergs"
        / "garden_baseline"
        / "adapter.fastergs"
    )
    assert mcmc.training.checkpoint.output_dir == (
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
    )
    experiment_config.training.render.backend = "faster_gs.core"

    training_config = fastergs_config_module.resolve_training_config(
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
        ember.ensure_checkpoint_output_writable(
            output_dir,
            overwrite=False,
        )

    ember.ensure_checkpoint_output_writable(
        output_dir,
        overwrite=True,
    )


def test_fastergs_initializer_matches_upstream_parameterization(
    fastergs_config_module,
) -> None:
    point_cloud = ember.PointCloudState(
        points=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        colors=torch.tensor(
            [
                [0.5, 0.25, 0.75],
                [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.5],
                [0.25, 0.25, 0.25],
            ],
            dtype=torch.float32,
        ),
    )
    scene_record = ember.SceneRecord(
        sensors=(),
        source_format="colmap",
        point_cloud=point_cloud,
    )

    baseline = fastergs_config_module.initialize_fastergs_model_from_scene_record(
        scene_record,
        sh_degree=3,
        use_mcmc=False,
        device=torch.device("cpu"),
    )
    mcmc = fastergs_config_module.initialize_fastergs_model_from_scene_record(
        scene_record,
        sh_degree=3,
        use_mcmc=True,
        device=torch.device("cpu"),
    )

    expected_dc = (point_cloud.colors - 0.5) / 0.28209479177387814
    torch.testing.assert_close(baseline.scene.feature[:, 0, :], expected_dc)
    torch.testing.assert_close(
        baseline.scene.feature[:, 1:, :],
        torch.zeros_like(baseline.scene.feature[:, 1:, :]),
    )
    torch.testing.assert_close(
        torch.sigmoid(baseline.scene.logit_opacity),
        torch.full((4,), 0.1),
    )
    torch.testing.assert_close(
        torch.sigmoid(mcmc.scene.logit_opacity),
        torch.full((4,), 0.5),
    )
    torch.testing.assert_close(
        torch.exp(mcmc.scene.log_scales),
        torch.exp(baseline.scene.log_scales) * 0.1,
    )
