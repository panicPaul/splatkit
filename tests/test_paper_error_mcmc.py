from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import ember_core as ember
import pytest
import torch
from ember_core.densification import DensificationContext, GaussianFamilyOps
from ember_core.training import TrainState
from marimo_config_gui.api import load_script_config
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "error_mcmc" / "notebook.py"


def load_error_mcmc_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.error_mcmc.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_error_mcmc_preset(error_mcmc_config_module, name: str):
    return load_preset_config(
        error_mcmc_config_module.error_mcmc_preset_catalog(), name
    )


def load_error_mcmc_script_config(
    error_mcmc_config_module,
    args: list[str],
):
    return load_script_config(
        error_mcmc_config_module.ErrorMCMCExperimentConfig,
        args=args,
        presets=error_mcmc_config_module.error_mcmc_preset_catalog(),
    )


def test_error_mcmc_training_config_uses_native_error_guided_densification() -> (
    None
):
    module = load_error_mcmc_config_module()
    experiment_config = load_error_mcmc_preset(module, "garden_error_mcmc")

    training_config = module.resolve_training_config(experiment_config)

    assert experiment_config.training.render.backend == "faster_gs.fastgs"
    assert experiment_config.training.initialization.use_mcmc is True
    assert experiment_config.data.materialization_stage == "prepared"
    assert experiment_config.data.materialization_mode == "eager"
    assert experiment_config.data.materialization_num_workers == 8
    assert experiment_config.training.mip_splatting.enabled is True
    assert (
        experiment_config.training.mip_splatting.screen_filter_enabled is True
    )
    assert training_config.render.backend == "faster_gs.fastgs"
    assert (
        training_config.render.backend_options["mip_splatting_screen_filter"]
        is True
    )
    assert (
        training_config.initialization.initializer.target
        == "papers.error_mcmc.notebook."
        "initialize_error_mcmc_model_from_scene_record"
    )
    assert (
        training_config.densification.builders[0].target
        == "papers.error_mcmc.notebook.ErrorMCMCDensification"
    )
    assert (
        training_config.densification.builders[0].kwargs["score_aggregation"]
        == "topk_mean"
    )
    assert training_config.densification.builders[0].kwargs["score_top_k"] == 3
    assert training_config.densification.builders[0].kwargs[
        "opacity_floor"
    ] == pytest.approx(0.05)
    assert (
        training_config.densification.builders[0]
        .kwargs["schedule"]
        .start_iteration
        == 600
    )
    assert any(
        builder.target
        == "ember_splatting_training.GaussianMipSplatting3DFilter"
        for builder in training_config.densification.builders
    )
    assert (
        training_config.densification.builders[-1].target
        == "papers.error_mcmc.notebook.ErrorMCMCFinalCleanup"
    )
    assert (
        training_config.checkpoint.output_dir
        == REPO_ROOT
        / "checkpoints/papers/error_mcmc/garden_error_mcmc/faster_gs.fastgs"
    )
    assert training_config.model_dump(mode="json")


def test_error_mcmc_script_loader_applies_preset_then_cli_overrides() -> None:
    module = load_error_mcmc_config_module()

    loaded = load_error_mcmc_script_config(
        module,
        args=[
            "--preset",
            "garden_debug_val",
            "--training.runtime.max-steps",
            "5",
            "--training.densification.error-mcmc.score-aggregation",
            "max",
        ],
    )

    assert isinstance(loaded, module.ErrorMCMCExperimentConfig)
    assert loaded.preset == "garden_debug_val"
    assert loaded.training.runtime.max_steps == 5
    assert loaded.training.densification.error_mcmc.score_aggregation == "max"


def test_error_mcmc_script_loader_replays_json_config(tmp_path: Path) -> None:
    module = load_error_mcmc_config_module()
    config = load_error_mcmc_preset(module, "garden_error_mcmc")
    json_path = tmp_path / "error_mcmc_config.json"
    json_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))

    loaded = load_error_mcmc_script_config(
        module,
        args=[str(json_path)],
    )

    assert isinstance(loaded, module.ErrorMCMCExperimentConfig)
    assert loaded == config


def test_error_mcmc_user_config_does_not_expose_runtime_kwargs() -> None:
    module = load_error_mcmc_config_module()
    config = load_error_mcmc_preset(module, "garden_error_mcmc")
    serialized = json.dumps(config.model_dump(mode="json"))

    assert '"kwargs"' not in serialized
    assert '"context_kwargs"' not in serialized


def test_error_mcmc_topk_score_keeps_rare_view_background_signal() -> None:
    module = load_error_mcmc_config_module()
    method = module.ErrorMCMCDensification(
        score_aggregation="topk_mean",
        score_top_k=2,
    )
    per_view_scores = torch.tensor(
        [
            [12.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    topk_score = method.aggregate_error_scores(per_view_scores)
    mean_method = module.ErrorMCMCDensification(score_aggregation="mean")
    mean_score = mean_method.aggregate_error_scores(per_view_scores)

    torch.testing.assert_close(topk_score, torch.tensor([6.0, 1.0]))
    assert topk_score[0] > mean_score[0]
    assert topk_score[0] > topk_score[1]


def test_error_mcmc_soft_opacity_floor_keeps_low_opacity_sources_sampleable() -> (
    None
):
    module = load_error_mcmc_config_module()
    method = module.ErrorMCMCDensification(
        min_opacity=0.005,
        opacity_floor=0.05,
        opacity_power=0.5,
    )
    scene = ember.GaussianScene3D(
        center_position=torch.zeros((2, 3), dtype=torch.float32),
        log_scales=torch.zeros((2, 3), dtype=torch.float32),
        quaternion_orientation=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        logit_opacity=torch.tensor([0.006, 0.9], dtype=torch.float32).logit(),
        feature=torch.zeros((2, 1, 3), dtype=torch.float32),
        sh_degree=0,
    )
    error_score = torch.tensor([100.0, 1.0], dtype=torch.float32)

    weights = method._mask_invalid_sources(
        scene,
        method._opacity_weighted_error_score(scene, error_score),
    )

    assert weights[0] > 0
    assert weights[0] > weights[1]


def test_error_mcmc_relocates_and_grows_from_high_error_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_error_mcmc_config_module()
    from ember_splatting_training.densification import mcmc as mcmc_module

    scene = ember.GaussianScene3D(
        center_position=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ).requires_grad_(True),
        log_scales=torch.zeros((3, 3), dtype=torch.float32).requires_grad_(
            True
        ),
        quaternion_orientation=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ).requires_grad_(True),
        logit_opacity=torch.tensor(
            [-20.0, 0.8, 0.7],
            dtype=torch.float32,
        ).requires_grad_(True),
        feature=torch.zeros((3, 1, 3), dtype=torch.float32).requires_grad_(
            True
        ),
        sh_degree=0,
    )
    state = TrainState(
        model=ember.InitializedModel(
            scene=scene,
            modules={},
            parameters={},
        ),
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )
    method = module.ErrorMCMCDensification(
        schedule=module.Schedule(start_iteration=1, frequency=1),
        min_opacity=0.005,
        cap_growth_factor=2.0,
        cap_max=6,
        inject_position_noise=False,
    )
    family_ops = GaussianFamilyOps(state, [])
    method.bind(state, [], family_ops)

    def fake_relocation_adjustment(
        old_opacities: torch.Tensor,
        old_scales: torch.Tensor,
        n_samples_per_primitive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del n_samples_per_primitive
        return old_opacities[:, None], old_scales

    monkeypatch.setattr(
        mcmc_module,
        "relocation_adjustment",
        fake_relocation_adjustment,
    )
    monkeypatch.setattr(
        method,
        "compute_error_score",
        lambda _context, _scene: torch.tensor(
            [0.0, 10.0, 0.0],
            dtype=torch.float32,
        ),
    )

    method.post_optimizer_step(
        DensificationContext(
            state=state,
            batch=None,
            render_output=None,
            loss_result=None,
            step=0,
            optimizers=[],
        )
    )

    assert state.model.scene.center_position.shape == (6, 3)
    torch.testing.assert_close(
        state.model.scene.center_position[0],
        torch.tensor([10.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        state.model.scene.center_position[3:],
        torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
