from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import ember_core as ember
import pytest
import torch
from marimo_config_gui.api import load_script_config
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "nht" / "notebook.py"


def load_nht_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.nht.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_nht_preset(nht_config_module, name: str):
    return load_preset_config(nht_config_module.nht_preset_catalog(), name)


def load_nht_script_config(nht_config_module, args: list[str]):
    return load_script_config(
        nht_config_module.NHTExperimentConfig,
        args=args,
        presets=nht_config_module.nht_preset_catalog(),
    )


def test_nht_resolved_training_config_uses_native_backend() -> None:
    module = load_nht_config_module()
    experiment_config = load_nht_preset(module, "garden_debug_val")

    training_config = module.resolve_training_config(experiment_config)

    assert training_config.render.backend == "nht.3dgut"
    assert training_config.render.return_alpha is True
    assert training_config.render.return_depth is True
    assert (
        training_config.initialization.initializer.target
        == "papers.nht.notebook.initialize_nht_model_from_scene_record"
    )
    assert (
        training_config.render.feature_fn.target
        == "papers.nht.notebook.nht_feature_scene"
    )
    assert (
        training_config.render.postprocess_fn.target
        == "papers.nht.notebook.nht_decode_render"
    )
    shader_spec = training_config.model.modules["deferred_shader"]
    assert shader_spec.target == "papers.nht.notebook.NHTDeferredShader"
    assert shader_spec.kwargs["enable_view_encoding"] is True
    assert shader_spec.kwargs["view_encoding"] == "sh"
    assert training_config.render.backend_options["ray_dir_scale"] == 3.0
    assert training_config.render.backend_options["center_ray_mode"] is False
    assert [
        group.target.scope + ":" + group.target.name
        for group in training_config.optimization.parameter_groups
    ] == [
        "scene:center_position",
        "scene:feature",
        "modules:deferred_shader",
        "scene:logit_opacity",
        "scene:log_scales",
        "scene:quaternion_orientation",
    ]
    for group in (
        training_config.optimization.parameter_groups[0],
        training_config.optimization.parameter_groups[1],
        training_config.optimization.parameter_groups[3],
        training_config.optimization.parameter_groups[4],
        training_config.optimization.parameter_groups[5],
    ):
        assert group.optimizer == "adam"
        assert group.optimizer_kwargs["fused"] is True
        assert group.optimizer_kwargs["eps"] == pytest.approx(1e-15)
        assert group.betas == pytest.approx((0.9, 0.999))
    feature_group = training_config.optimization.parameter_groups[1]
    shader_group = training_config.optimization.parameter_groups[2]
    assert feature_group.scheduler is not None
    assert feature_group.scheduler.target == (
        "torch.optim.lr_scheduler.CosineAnnealingLR"
    )
    assert feature_group.scheduler.kwargs["eta_min"] == pytest.approx(0.0015)
    assert shader_group.scheduler is not None
    assert shader_group.scheduler.target == (
        "torch.optim.lr_scheduler.CosineAnnealingLR"
    )
    assert shader_group.scheduler.kwargs["eta_min"] == pytest.approx(0.000068)
    assert shader_group.optimizer_kwargs == {}
    assert shader_group.betas == pytest.approx((0.9, 0.999))
    assert (
        training_config.loss.target.target
        == "papers.nht.notebook.nht_rgb_l1_dssim_loss"
    )
    assert training_config.loss.target.kwargs["color_refine_start"] == 0
    assert len(training_config.hooks.builders) == 1
    assert (
        training_config.hooks.builders[0].target
        == "papers.nht.notebook.NHTColorRefineAndEMAHook"
    )
    assert training_config.hooks.builders[0].kwargs["color_refine_start"] == 0
    assert training_config.densification is not None
    assert len(training_config.densification.builders) == 1
    assert (
        training_config.densification.builders[0].target
        == "papers.nht.notebook.build_nht_mcmc"
    )
    assert (
        training_config.densification.builders[0].kwargs["end_iteration"]
        == 25_000
    )
    assert (
        training_config.checkpoint.output_dir
        == REPO_ROOT / "checkpoints/papers/nht/garden_debug_val/nht.3dgut"
    )


def test_nht_script_loader_applies_preset_then_cli_overrides() -> None:
    module = load_nht_config_module()

    loaded = load_nht_script_config(
        module,
        args=[
            "--preset",
            "garden_debug_val",
            "--training.runtime.max-steps",
            "5",
            "--training.initialization.feature-dim",
            "12",
            "--training.model.shader.feature-dim",
            "12",
        ],
    )

    assert isinstance(loaded, module.NHTExperimentConfig)
    assert loaded.training.runtime.max_steps == 5
    assert loaded.training.initialization.feature_dim == 12
    assert loaded.training.model.shader.feature_dim == 12


def test_nht_batch_scaled_optimizers_and_shader_ray_scale() -> None:
    module = load_nht_config_module()
    experiment_config = load_nht_preset(module, "garden_debug_val")
    experiment_config.training.batching.batch_size = 4
    experiment_config.training.model.shader.view_encoding = "fourier"

    training_config = module.resolve_training_config(experiment_config)
    groups = training_config.optimization.parameter_groups

    assert training_config.render.backend_options["ray_dir_scale"] == 1.0
    assert groups[0].lr == pytest.approx(3.2e-4)
    assert groups[0].scheduler is not None
    assert groups[0].scheduler.kwargs["final_lr"] == pytest.approx(3.2e-6)
    assert groups[1].lr == pytest.approx(3.0e-2)
    assert groups[2].lr == pytest.approx(1.36e-3)
    assert groups[3].lr == pytest.approx(1.0e-1)
    assert groups[4].lr == pytest.approx(1.0e-2)
    assert groups[5].lr == pytest.approx(2.0e-3)
    assert groups[1].optimizer_kwargs["eps"] == pytest.approx(5e-16)
    assert groups[1].betas == pytest.approx((0.6, 0.996))
    assert groups[2].optimizer_kwargs == {}


def test_nht_script_loader_replays_json_config(tmp_path: Path) -> None:
    module = load_nht_config_module()
    config = load_nht_preset(module, "garden_debug_val")
    json_path = tmp_path / "nht_config.json"
    json_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))

    loaded = load_nht_script_config(module, args=[str(json_path)])

    assert isinstance(loaded, module.NHTExperimentConfig)
    assert loaded == config


def test_nht_deferred_shader_decodes_feature_render() -> None:
    pytest.importorskip("tinycudann")
    if not torch.cuda.is_available():
        pytest.skip("tiny-cuda-nn shader requires CUDA.")
    module = load_nht_config_module()
    shader = module.NHTDeferredShader(
        feature_dim=8,
        hidden_dim=16,
        num_hidden_layers=1,
    ).cuda()
    scene = ember.GaussianScene3D(
        center_position=torch.zeros((2, 3), dtype=torch.float32),
        log_scales=torch.zeros((2, 3), dtype=torch.float32),
        quaternion_orientation=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        logit_opacity=torch.zeros((2,), dtype=torch.float32),
        feature=torch.zeros((2, 4), dtype=torch.float32),
        sh_degree=0,
    )
    model = ember.InitializedModel(
        scene=scene,
        modules={"deferred_shader": shader},
        parameters={},
    )
    camera = ember.CameraState(
        width=torch.tensor([2]),
        height=torch.tensor([2]),
        fov_degrees=torch.tensor([60.0]),
        cam_to_world=torch.eye(4, dtype=torch.float32)[None, :, :],
    )
    raw_output = SimpleNamespace(
        features=torch.zeros((1, 2, 2, 7), dtype=torch.float32, device="cuda"),
        alphas=torch.ones((1, 2, 2), dtype=torch.float32, device="cuda"),
        depth=torch.ones((1, 2, 2), dtype=torch.float32, device="cuda"),
        visibility=torch.ones((2, 1), dtype=torch.float32, device="cuda"),
        weights=torch.ones((2, 1), dtype=torch.float32, device="cuda"),
    )

    decoded = module.nht_decode_render(model, camera, raw_output)

    assert decoded.render.shape == (1, 2, 2, 3)
    assert decoded.features.shape == (1, 2, 2, 7)
    assert decoded.extras is None


def test_nht_knn_initialization_uses_point_distances() -> None:
    module = load_nht_config_module()
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
        colors=torch.zeros((4, 3), dtype=torch.float32),
    )
    scene_record = ember.SceneRecord(
        sensors=(),
        source_format="colmap",
        point_cloud=point_cloud,
    )

    initialized = module.initialize_nht_model_from_scene_record(
        scene_record,
        feature_dim=8,
        initial_scale=0.1,
        initial_opacity=0.5,
    )
    distances = module.nht_root_mean_squared_knn_distances(point_cloud.points)
    expected_scales = torch.log(distances * 0.1).unsqueeze(-1).repeat(1, 3)

    assert torch.allclose(initialized.scene.log_scales, expected_scales)
    assert not torch.allclose(
        initialized.scene.quaternion_orientation[:, 0],
        torch.ones(4),
    )
    assert initialized.scene.feature.shape == (4, 8)


def test_nht_mcmc_sanitizes_nan_opacity_logits(cpu_scene) -> None:
    module = load_nht_config_module()
    mcmc = module.build_nht_mcmc()
    scene = cpu_scene.__class__(
        center_position=cpu_scene.center_position,
        log_scales=cpu_scene.log_scales,
        quaternion_orientation=cpu_scene.quaternion_orientation,
        logit_opacity=torch.tensor([float("nan"), 1.0, 2.0]),
        feature=torch.zeros((3, 8)),
        sh_degree=0,
    )

    mcmc._sanitize_opacity_logits(scene)

    assert torch.isfinite(scene.logit_opacity).all()
    assert scene.logit_opacity[0].item() == 0.0
