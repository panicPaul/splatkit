from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from marimo_config_gui.api import load_script_config
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "nht_fast_gs" / "notebook.py"


def load_nht_fast_gs_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.nht_fast_gs.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_nht_fast_gs_preset(nht_fast_gs_config_module, name: str):
    return load_preset_config(
        nht_fast_gs_config_module.nht_fast_gs_preset_catalog(),
        name,
    )


def test_nht_fast_gs_resolved_training_config_uses_hybrid_backend() -> None:
    module = load_nht_fast_gs_config_module()
    experiment_config = load_nht_fast_gs_preset(
        module,
        "garden_nht_fast_gs",
    )

    training_config = module.resolve_training_config(experiment_config)

    assert training_config.render.backend == "nht.3dgut_fast_gs"
    assert training_config.render.return_alpha is True
    assert training_config.render.return_depth is True
    assert (
        training_config.render.feature_fn.target
        == "papers.nht.notebook.nht_feature_scene"
    )
    assert (
        training_config.render.postprocess_fn.target
        == "papers.nht.notebook.nht_decode_render"
    )
    assert (
        training_config.densification.builders[0].target
        == "papers.nht_fast_gs.notebook.NHTFastGSDensification"
    )
    assert (
        training_config.densification.builders[0].kwargs["probe_view_count"]
        == 10
    )
    assert (
        training_config.densification.builders[0].kwargs["refine_every"] == 500
    )
    assert training_config.densification.builders[0].kwargs["start_iter"] == 500
    assert (
        training_config.densification.builders[0].kwargs["stop_iter"] == 15000
    )
    assert (
        training_config.densification.builders[0].kwargs["loss_thresh"] == 0.06
    )
    assert (
        training_config.densification.builders[0].kwargs["grad_abs_threshold"]
        == 0.0008
    )
    assert (
        training_config.densification.builders[0].kwargs["dense_fraction"]
        == 0.001
    )
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
        training_config.densification.builders[0].kwargs["final_prune_mode"]
        == "disabled"
    )
    assert (
        training_config.checkpoint.output_dir
        == REPO_ROOT
        / "checkpoints/papers/nht_fast_gs/garden_nht_fast_gs/nht.3dgut_fast_gs"
    )


def test_nht_fast_gs_script_loader_applies_preset_then_cli_overrides() -> None:
    module = load_nht_fast_gs_config_module()

    loaded = load_script_config(
        module.NHTFastGSExperimentConfig,
        args=[
            "--preset",
            "garden_nht_fast_gs",
            "--training.runtime.max-steps",
            "5",
            "--training.densification.probe-view-count",
            "3",
        ],
        presets=module.nht_fast_gs_preset_catalog(),
    )

    assert isinstance(loaded, module.NHTFastGSExperimentConfig)
    assert loaded.preset == "garden_nht_fast_gs"
    assert loaded.training.runtime.max_steps == 5
    assert loaded.training.densification.probe_view_count == 3
    training_config = module.resolve_training_config(loaded)
    assert (
        training_config.densification.builders[0].kwargs["probe_view_count"]
        == 3
    )


def test_nht_fast_gs_big_preset_uses_fastgs_big_densification() -> None:
    module = load_nht_fast_gs_config_module()
    base_config = load_nht_fast_gs_preset(module, "garden_nht_fast_gs")
    big_config = load_nht_fast_gs_preset(module, "garden_big")

    base_training_config = module.resolve_training_config(base_config)
    big_training_config = module.resolve_training_config(big_config)
    base_densifier = base_training_config.densification.builders[0]
    big_densifier = big_training_config.densification.builders[0]

    assert (
        base_densifier.target
        == "papers.nht_fast_gs.notebook.NHTFastGSDensification"
    )
    assert (
        big_densifier.target
        == "papers.nht_fast_gs.notebook.NHTFastGSDensification"
    )
    assert base_densifier.kwargs["refine_every"] == 500
    assert base_densifier.kwargs["grad_abs_threshold"] == 0.0008
    assert big_densifier.kwargs["refine_every"] == 100
    assert big_densifier.kwargs["grad_abs_threshold"] == 0.0003
    assert big_training_config.render.backend == "nht.3dgut_fast_gs"
    assert (
        big_training_config.checkpoint.output_dir
        == REPO_ROOT
        / "checkpoints/papers/nht_fast_gs/garden_big/nht.3dgut_fast_gs"
    )


def test_nht_fast_gs_jit_notice_matches_initial_running_snapshot() -> None:
    module = load_nht_fast_gs_config_module()
    config = load_nht_fast_gs_preset(module, "garden_nht_fast_gs")
    snapshot = SimpleNamespace(
        status="running",
        step=0,
        latest_metrics={},
    )

    assert module.nht_fast_gs_should_show_jit_compile_notice(
        config,
        snapshot,
        is_script_mode=False,
    )
    assert not module.nht_fast_gs_should_show_jit_compile_notice(
        config,
        snapshot,
        is_script_mode=True,
    )
    snapshot.latest_metrics = {"loss": 1.0}
    assert not module.nht_fast_gs_should_show_jit_compile_notice(
        config,
        snapshot,
        is_script_mode=False,
    )


def test_nht_fast_gs_user_config_does_not_expose_runtime_kwargs() -> None:
    module = load_nht_fast_gs_config_module()
    config = load_nht_fast_gs_preset(module, "garden_nht_fast_gs")
    serialized = json.dumps(config.model_dump(mode="json"))

    assert '"kwargs"' not in serialized
    assert '"context_kwargs"' not in serialized
