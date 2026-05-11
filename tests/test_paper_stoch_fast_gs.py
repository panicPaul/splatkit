from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import ember_splatting_training as ember_splatting
from marimo_config_gui.api import load_script_config
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "stoch_fast_gs" / "notebook.py"


def load_stoch_fast_gs_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.stoch_fast_gs.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_stoch_fast_gs_preset(stoch_fast_gs_config_module, name: str):
    return load_preset_config(
        stoch_fast_gs_config_module.stoch_fast_gs_preset_catalog(), name
    )


def test_stoch_fast_gs_resolved_training_config_uses_hybrid_defaults() -> None:
    module = load_stoch_fast_gs_config_module()
    experiment_config = load_stoch_fast_gs_preset(
        module,
        "garden_stoch_fast_gs",
    )

    training_config = module.resolve_training_config(experiment_config)

    assert experiment_config.training.render.backend == "3dgrt.stoch_fast_gs"
    assert training_config.render.backend == "3dgrt.stoch_fast_gs"
    assert training_config.render.return_alpha is True
    assert training_config.render.return_depth is True
    assert training_config.render.backend_options["min_transmittance"] == 0.001
    assert (
        training_config.densification.builders[0].target
        == "ember_splatting_training.GaussianFastGS"
    )
    assert (
        training_config.densification.builders[0].kwargs["probe_view_count"]
        == 10
    )
    assert training_config.densification.builders[0].kwargs[
        "refine_every"
    ] == 500
    assert training_config.densification.builders[0].kwargs[
        "start_iter"
    ] == 500
    assert training_config.densification.builders[0].kwargs[
        "stop_iter"
    ] == 15000
    assert training_config.densification.builders[0].kwargs[
        "loss_thresh"
    ] == 0.06
    assert training_config.densification.builders[0].kwargs[
        "grad_abs_threshold"
    ] == 0.0008
    assert training_config.densification.builders[0].kwargs[
        "dense_fraction"
    ] == 0.001
    assert (
        training_config.densification.builders[0].kwargs[
            "extra_opacity_reset_iter"
        ]
        is None
    )
    assert (
        training_config.densification.builders[0].kwargs["max_reset_opacity"]
        == 0.8
    )
    assert (
        training_config.densification.builders[0].kwargs[
            "scheduled_reset_opacity"
        ]
        == 0.01
    )
    assert (
        training_config.densification.builders[-1].target
        == "papers.stoch3dgs.notebook.Stoch3DGSFinalCleanup"
    )
    assert (
        training_config.render.feature_fn.target
        == "papers.stoch3dgs.notebook.stoch3dgs_active_sh_scene"
    )
    assert (
        training_config.checkpoint.output_dir
        == REPO_ROOT
        / "checkpoints/papers/stoch_fast_gs/garden_stoch_fast_gs/3dgrt.stoch_fast_gs"
    )


def test_stoch_fast_gs_densification_target_is_importable() -> None:
    assert ember_splatting.GaussianFastGS.__name__ == "GaussianFastGS"


def test_stoch_fast_gs_script_loader_applies_preset_then_cli_overrides() -> None:
    module = load_stoch_fast_gs_config_module()

    loaded = load_script_config(
        module.StochFastGSExperimentConfig,
        args=[
            "--preset",
            "garden_stoch_fast_gs",
            "--training.runtime.max-steps",
            "5",
            "--training.densification.probe-view-count",
            "3",
        ],
        presets=module.stoch_fast_gs_preset_catalog(),
    )

    assert isinstance(loaded, module.StochFastGSExperimentConfig)
    assert loaded.preset == "garden_stoch_fast_gs"
    assert loaded.training.runtime.max_steps == 5
    assert loaded.training.densification.probe_view_count == 3
    training_config = module.resolve_training_config(loaded)
    assert (
        training_config.densification.builders[0].kwargs["probe_view_count"]
        == 3
    )


def test_stoch_fast_gs_user_config_does_not_expose_runtime_kwargs() -> None:
    module = load_stoch_fast_gs_config_module()
    config = load_stoch_fast_gs_preset(module, "garden_stoch_fast_gs")
    serialized = json.dumps(config.model_dump(mode="json"))

    assert '"kwargs"' not in serialized
    assert '"context_kwargs"' not in serialized
