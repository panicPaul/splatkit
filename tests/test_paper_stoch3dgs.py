from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import ember_core as ember
import torch
from marimo_config_gui.api import load_script_config
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "stoch3dgs" / "notebook.py"


def load_stoch3dgs_preset(stoch3dgs_config_module, name: str):
    return load_preset_config(
        stoch3dgs_config_module.stoch3dgs_preset_catalog(), name
    )


def load_stoch3dgs_script_config(
    stoch3dgs_config_module,
    args: list[str],
):
    return load_script_config(
        stoch3dgs_config_module.Stoch3DGSExperimentConfig,
        args=args,
        presets=stoch3dgs_config_module.stoch3dgs_preset_catalog(),
    )


def load_stoch3dgs_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.stoch3dgs.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stoch3dgs_resolved_training_config_uses_native_defaults() -> None:
    module = load_stoch3dgs_config_module()
    experiment_config = load_stoch3dgs_preset(module, "garden_stoch")

    training_config = module.resolve_training_config(experiment_config)

    assert experiment_config.training.render.backend == "3dgrt.stoch3dgs"
    assert training_config.render.backend == "3dgrt.stoch3dgs"
    assert training_config.render.return_alpha is True
    assert training_config.render.return_depth is True
    assert training_config.render.backend_options == {
        "particle_kernel_degree": 4,
        "particle_kernel_density_clamping": True,
        "particle_kernel_min_response": 0.0113,
        "particle_kernel_min_alpha": 1.0 / 255.0,
        "particle_kernel_max_alpha": 0.99,
        "primitive_type": "instances",
        "min_transmittance": 0.001,
        "enable_normals": False,
        "enable_hitcounts": True,
        "max_consecutive_bvh_update": 15,
        "ray_principal_point_mode": "image_center",
        "background_color": [0.0, 0.0, 0.0],
    }
    assert (
        training_config.initialization.initializer.target
        == "papers.stoch3dgs.notebook.initialize_stoch3dgs_model_from_scene_record"
    )
    assert (
        training_config.render.feature_fn.target
        == "papers.stoch3dgs.notebook.stoch3dgs_active_sh_scene"
    )
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
    assert training_config.optimization.parameter_groups[3].lr == 0.04
    assert (
        training_config.loss.target.target
        == "papers.stoch3dgs.notebook.stoch3dgs_rgb_l1_ssim_loss"
    )
    assert training_config.loss.target.kwargs["lambda_ssim"] == 0.2
    assert (
        training_config.densification.builders[0].target
        == "papers.stoch3dgs.notebook.Stoch3DGSDensification"
    )
    assert (
        training_config.densification.builders[-1].target
        == "papers.stoch3dgs.notebook.Stoch3DGSFinalCleanup"
    )
    assert (
        training_config.hooks.builders[0].target
        == "papers.stoch3dgs.notebook.Stoch3DGSActiveSHHook"
    )
    assert (
        training_config.checkpoint.output_dir
        == REPO_ROOT
        / "checkpoints/papers/stoch3dgs/garden_stoch/3dgrt.stoch3dgs"
    )


def test_stoch3dgs_script_loader_applies_preset_then_cli_overrides() -> None:
    module = load_stoch3dgs_config_module()

    loaded = load_stoch3dgs_script_config(
        module,
        args=[
            "--preset",
            "garden_stoch",
            "--training.runtime.max-steps",
            "5",
            "--training.render.min-transmittance",
            "0.002",
        ],
    )

    assert isinstance(loaded, module.Stoch3DGSExperimentConfig)
    assert loaded.preset == "garden_stoch"
    assert loaded.training.runtime.max_steps == 5
    assert loaded.training.render.min_transmittance == 0.002

    training_config = module.resolve_training_config(loaded)
    assert training_config.runtime.max_steps == 5
    assert training_config.render.backend_options["min_transmittance"] == 0.002


def test_stoch3dgs_user_config_does_not_expose_runtime_kwargs() -> None:
    module = load_stoch3dgs_config_module()
    config = load_stoch3dgs_preset(module, "garden_stoch")
    serialized = json.dumps(config.model_dump(mode="json"))

    assert '"kwargs"' not in serialized
    assert '"context_kwargs"' not in serialized


def test_stoch3dgs_script_loader_replays_json_config(tmp_path: Path) -> None:
    module = load_stoch3dgs_config_module()
    config = load_stoch3dgs_preset(module, "garden_stoch")
    json_path = tmp_path / "stoch3dgs_config.json"
    json_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))

    loaded = load_stoch3dgs_script_config(
        module,
        args=[str(json_path)],
    )

    assert isinstance(loaded, module.Stoch3DGSExperimentConfig)
    assert loaded == config


def test_stoch3dgs_active_sh_scene_masks_inactive_coefficients() -> None:
    module = load_stoch3dgs_config_module()
    feature = torch.ones((2, 16, 3), dtype=torch.float32)
    scene = ember.GaussianScene3D(
        center_position=torch.zeros((2, 3), dtype=torch.float32),
        log_scales=torch.zeros((2, 3), dtype=torch.float32),
        quaternion_orientation=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        logit_opacity=torch.zeros((2,), dtype=torch.float32),
        feature=feature,
        sh_degree=3,
    )
    model = ember.InitializedModel(
        scene=scene,
        modules={},
        parameters={},
        metadata={"active_sh_degree": 1},
    )

    active_scene = module.stoch3dgs_active_sh_scene(
        model,
        camera=SimpleNamespace(),
    )

    assert active_scene.sh_degree == 1
    torch.testing.assert_close(
        active_scene.feature[:, :4, :], torch.ones(2, 4, 3)
    )
    torch.testing.assert_close(
        active_scene.feature[:, 4:, :],
        torch.zeros(2, 12, 3),
    )


def test_stoch3dgs_step_schedule_matches_upstream_strict_start() -> None:
    module = load_stoch3dgs_config_module()

    assert module.stoch3dgs_step_includes(500, 500, 15000, 100) is False
    assert module.stoch3dgs_step_includes(600, 500, 15000, 100) is True
    assert module.stoch3dgs_step_includes(15000, 500, 15000, 100) is False


def test_stoch3dgs_active_sh_hook_updates_after_scheduled_step() -> None:
    module = load_stoch3dgs_config_module()
    hook = module.Stoch3DGSActiveSHHook(
        max_sh_degree=3,
        sh_start_step=1000,
        sh_step_interval=1000,
    )
    state = SimpleNamespace(model=SimpleNamespace(metadata={}), step=999)

    hook.before_step(state)
    assert state.model.metadata["active_sh_degree"] == 0

    state.step = 1000
    hook.before_step(state)
    assert state.model.metadata["active_sh_degree"] == 0

    state.step = 1001
    hook.before_step(state)
    assert state.model.metadata["active_sh_degree"] == 1

    state.step = 2000
    hook.before_step(state)
    assert state.model.metadata["active_sh_degree"] == 1

    state.step = 2001
    hook.before_step(state)
    assert state.model.metadata["active_sh_degree"] == 2


def test_stoch3dgs_densification_prunes_buffers() -> None:
    module = load_stoch3dgs_config_module()
    densification = module.Stoch3DGSDensification()
    densification.densify_grad_norm_accum = torch.arange(
        4, dtype=torch.float32
    )[:, None]
    densification.densify_grad_norm_denom = torch.arange(
        10, 14, dtype=torch.int32
    )[:, None]
    keep_mask = torch.tensor([True, False, True, False])

    densification._prune_buffers(keep_mask)

    assert densification.densify_grad_norm_accum is not None
    assert densification.densify_grad_norm_denom is not None
    torch.testing.assert_close(
        densification.densify_grad_norm_accum.squeeze(),
        torch.tensor([0.0, 2.0]),
    )
    torch.testing.assert_close(
        densification.densify_grad_norm_denom.squeeze(),
        torch.tensor([10, 12], dtype=torch.int32),
    )
