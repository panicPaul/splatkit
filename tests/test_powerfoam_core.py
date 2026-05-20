from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import ember_core as ember
import pytest
import torch
from ember_core.densification import DensificationContext
from marimo_config_gui.presets import (
    ConfigPreset,
    ConfigPresetCatalog,
    load_preset_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "powerfoam" / "notebook.py"
POWERFOAM_PACKAGE_SRC = REPO_ROOT / "packages" / "ember-native-powerfoam" / "src"
POWERFOAM_UPSTREAM_SRC = REPO_ROOT / "third_party" / "powerfoam"
sys.path.insert(0, str(POWERFOAM_PACKAGE_SRC))
sys.path.insert(0, str(POWERFOAM_UPSTREAM_SRC))


def make_powerfoam_scene(num_points: int = 16) -> ember.PowerFoamScene:
    return ember.PowerFoamScene(
        points=torch.zeros((num_points, 3), dtype=torch.float32),
        radii=torch.ones((num_points,), dtype=torch.float32),
        quaternions=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).expand(num_points, 4),
        density=torch.ones((num_points,), dtype=torch.float32) * 0.1,
        texel_sites=torch.zeros((num_points, 8, 2), dtype=torch.float32),
        texel_sv_axis=torch.ones((num_points, 8, 24), dtype=torch.float32),
        texel_sv_rgb=torch.zeros((num_points, 8, 24), dtype=torch.float32),
        texel_height=torch.zeros((num_points, 8), dtype=torch.float32),
        adjacency=torch.zeros((0,), dtype=torch.int32),
        adjacency_offsets=torch.zeros((num_points + 1,), dtype=torch.int32),
        sv_dof=8,
        num_texel_sites=8,
    )


def load_powerfoam_preset(powerfoam_config_module, name: str):
    return load_preset_config(
        powerfoam_config_module.powerfoam_preset_catalog(),
        name,
    )


@pytest.fixture
def powerfoam_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.powerfoam.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_powerfoam_scene_contract_exports_core_family() -> None:
    scene = make_powerfoam_scene()

    assert scene.scene_family == "powerfoam"
    assert ember.FOAM.serialized == "core.foam"
    assert ember.POWERFOAM.serialized == "core.powerfoam"
    assert ember.scene_family_id(ember.RADFOAM.serialized) == "foam"
    assert ember.scene_family_id(ember.POWERFOAM.serialized) == "foam"
    assert ember.scene_family_id(scene.scene_family) == "foam"


def test_powerfoam_scene_validates_texel_layout() -> None:
    with pytest.raises(ValueError, match="texel_sv_axis"):
        ember.PowerFoamScene(
            points=torch.zeros((16, 3), dtype=torch.float32),
            radii=torch.ones((16,), dtype=torch.float32),
            quaternions=torch.zeros((16, 4), dtype=torch.float32),
            density=torch.ones((16,), dtype=torch.float32),
            texel_sites=torch.zeros((16, 8, 2), dtype=torch.float32),
            texel_sv_axis=torch.ones((16, 8, 1), dtype=torch.float32),
            texel_sv_rgb=torch.zeros((16, 8, 24), dtype=torch.float32),
            texel_height=torch.zeros((16, 8), dtype=torch.float32),
            adjacency=torch.zeros((0,), dtype=torch.int32),
            adjacency_offsets=torch.zeros((17,), dtype=torch.int32),
            sv_dof=8,
            num_texel_sites=8,
        )


def test_powerfoam_package_import_and_register() -> None:
    import ember_native_powerfoam as powerfoam
    from ember_core.core.registry import BACKEND_REGISTRY

    powerfoam.register()

    assert "powerfoam.rasterize" in BACKEND_REGISTRY


def test_powerfoam_upstream_imports_without_optional_packages() -> None:
    import powerfoam.camera
    import powerfoam.scene

    assert powerfoam.camera.TorchCamera is not None
    assert powerfoam.scene.PowerfoamScene is not None


def test_powerfoam_default_preset_is_base(powerfoam_config_module) -> None:
    catalog = powerfoam_config_module.powerfoam_preset_catalog()
    config = load_preset_config(catalog)

    assert catalog.default == "garden_base"
    assert config.preset == "garden_base"


def test_powerfoam_preset_uses_sibling_path_defaults(
    powerfoam_config_module,
    tmp_path: Path,
) -> None:
    defaults_dir = tmp_path / "defaults"
    scene_root = tmp_path / "scenes"
    defaults_dir.mkdir()
    scene_root.mkdir()
    preset_path = defaults_dir / "garden_debug.json"
    preset_path.write_text(
        json.dumps(
            {
                "preset": "garden_debug",
                "scene": {"path": "dataset/mipnerf360/garden"},
            }
        ),
        encoding="utf-8",
    )
    (defaults_dir / ".path_defaults.json").write_text(
        json.dumps(
            {
                "path_prefixes": {
                    "dataset/mipnerf360": str(scene_root),
                }
            }
        ),
        encoding="utf-8",
    )
    catalog = ConfigPresetCatalog(
        model_cls=powerfoam_config_module.PowerFoamExperimentConfig,
        presets={
            "garden_debug": ConfigPreset(
                name="garden_debug",
                path=preset_path,
                label="Garden debug",
                base_dir=REPO_ROOT,
            )
        },
        default="garden_debug",
    )

    config = load_preset_config(catalog)

    assert config.scene.path == scene_root / "garden"


def test_powerfoam_notebook_resolves_training_config(
    powerfoam_config_module,
) -> None:
    config = load_powerfoam_preset(powerfoam_config_module, "garden_debug")
    training_config = powerfoam_config_module.resolve_training_config(config)

    assert training_config.runtime.device == "cuda"
    assert training_config.runtime.max_steps == 100
    assert training_config.render.backend == "powerfoam.rasterize"
    assert training_config.render.backend_options == {
        "render_objective": None,
        "density_beta": 100.0,
        "radii_beta": 100.0,
        "background_color": [0.0, 0.0, 0.0],
        "clamp_output": False,
        "disable_coop_prim_load": False,
        "disable_coop_adj_load": False,
        "is_pinhole": True,
    }
    assert (
        training_config.render.training_backend_options_builder.target
        == "ember_native_powerfoam.powerfoam_training_backend_options"
    )
    assert (
        training_config.initialization.initializer.target
        == "ember_native_powerfoam.initialize_powerfoam_model_from_scene_record"
    )
    assert training_config.initialization.initializer.kwargs["attr_dtype"] == "float"
    assert [
        group.target.name
        for group in training_config.optimization.parameter_groups
    ] == [
        "points",
        "density",
        "radii",
        "quaternions",
        "texel_sites",
        "texel_sv_axis",
        "texel_sv_rgb",
        "texel_height",
    ]
    assert (
        training_config.loss.target.target
        == "ember_native_powerfoam.powerfoam_training_loss"
    )
    assert training_config.loss.weights["max_steps"] == 100
    assert training_config.densification is not None
    assert (
        training_config.densification.builders[0].target
        == "ember_native_powerfoam.PowerFoamResampling"
    )
    densification_kwargs = training_config.densification.builders[0].kwargs
    assert densification_kwargs["max_steps"] == 100
    assert densification_kwargs["resample_every"] == 25
    assert densification_kwargs["resample_offset"] == 24
    assert densification_kwargs["densify_from"] == 10
    assert densification_kwargs["densify_until"] == 80
    assert densification_kwargs["final_points"] == 4096
    assert training_config.model_dump_json(indent=2)


def test_powerfoam_notebook_builds_scene_and_dataset_configs(
    powerfoam_config_module,
) -> None:
    config = load_powerfoam_preset(powerfoam_config_module, "garden_debug")
    config.data.cache_resized_images = False

    scene_config = powerfoam_config_module.build_scene_load_config(config)
    dataset_config = (
        powerfoam_config_module.build_prepared_frame_dataset_config(config)
    )

    assert scene_config.path == config.scene.path
    assert scene_config.image_root is None
    assert len(scene_config.source_pipes) == 1
    assert dataset_config.camera_sensor_id is None
    assert dataset_config.split is not None
    assert dataset_config.split.target == "train"
    assert dataset_config.split.every_n == 8
    assert dataset_config.materialization is not None
    assert dataset_config.materialization.stage == "prepared"
    assert dataset_config.materialization.mode == "eager"
    assert dataset_config.image_preparation is not None
    assert dataset_config.image_preparation.resize_width_scale == 0.125
    assert dataset_config.image_preparation.interpolation == "bicubic"
    assert config.training.viewer.enabled is True


def test_powerfoam_target_points_matches_upstream_schedule() -> None:
    import ember_native_powerfoam as powerfoam

    kwargs = {
        "initial_num_points": 100,
        "final_points": 800,
        "densify_from": 10,
        "densify_until": 18,
    }

    assert powerfoam.powerfoam_target_points(0, **kwargs) == 100
    assert powerfoam.powerfoam_target_points(10, **kwargs) == 100
    assert powerfoam.powerfoam_target_points(17, **kwargs) == 800
    assert powerfoam.powerfoam_target_points(30, **kwargs) == 800


def test_powerfoam_training_loss_decays_regularizer_weights() -> None:
    import ember_native_powerfoam as powerfoam

    batch = SimpleNamespace(images=torch.zeros((1, 2, 2, 3)))
    render_output = SimpleNamespace(
        render=torch.zeros((1, 2, 2, 3)),
        normal_error=torch.ones((1, 2, 2)),
        contrib=torch.ones((1, 4)),
    )
    weights = {
        "rgb": 1.0,
        "ssim": 0.0,
        "normal": 0.1,
        "contribution": 0.1,
        "interpenetration": 0.0,
        "max_steps": 100.0,
    }

    initial_metrics = powerfoam.powerfoam_training_loss(
        SimpleNamespace(step=0),
        batch,
        render_output,
        weights=weights,
    )
    final_metrics = powerfoam.powerfoam_training_loss(
        SimpleNamespace(step=100),
        batch,
        render_output,
        weights=weights,
    )

    assert initial_metrics["normal_weight"] == pytest.approx(0.1)
    assert initial_metrics["contribution_weight"] == pytest.approx(0.1)
    assert final_metrics["normal_weight"] == pytest.approx(0.01)
    assert final_metrics["contribution_weight"] == pytest.approx(0.0001)


def _make_powerfoam_optimizer_bindings(scene: ember.PowerFoamScene):
    bindings = []
    for name in scene.parameter_field_names:
        parameter = getattr(scene, name)
        optimizer = torch.optim.Adam([parameter], lr=1e-3, eps=1e-15)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        parameter.grad = None
        bindings.append(
            ember.OptimizerBinding(
                target=ember.ParameterTargetSpec(scope="scene", name=name),
                optimizer=optimizer,
                base_parameter=parameter,
                field_name=name,
            )
        )
    return bindings


def _make_powerfoam_cuda_scene() -> ember.PowerFoamScene:
    from ember_native_powerfoam.powerfoam.runtime import (
        build_powerfoam_topology,
    )

    device = torch.device("cuda")
    points = torch.tensor(
        [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.0],
            [0.0, 0.0, 2.3],
            [0.0, 0.3, 2.2],
            [0.3, 0.0, 2.2],
            [-0.3, 0.0, 2.2],
        ],
        dtype=torch.float32,
        device=device,
    )
    radii = torch.full((points.shape[0],), 0.35, device=device)
    topology = build_powerfoam_topology(points, radii)
    quaternions = torch.zeros((points.shape[0], 4), device=device)
    quaternions[:, 0] = 1.0
    return ember.PowerFoamScene(
        points=points.detach().clone().requires_grad_(True),
        radii=radii.detach().clone().requires_grad_(True),
        quaternions=quaternions.detach().clone().requires_grad_(True),
        density=torch.full(
            (points.shape[0],),
            0.1,
            device=device,
        ).requires_grad_(True),
        texel_sites=(
            torch.randn((points.shape[0], 8, 2), device=device) * 0.01
        ).requires_grad_(True),
        texel_sv_axis=torch.randn(
            (points.shape[0], 8, 24),
            device=device,
        ).requires_grad_(True),
        texel_sv_rgb=torch.zeros(
            (points.shape[0], 8, 24),
            device=device,
        ).requires_grad_(True),
        texel_height=torch.zeros(
            (points.shape[0], 8),
            device=device,
        ).requires_grad_(True),
        adjacency=topology.adjacency,
        adjacency_offsets=topology.adjacency_offsets,
        sv_dof=8,
        num_texel_sites=8,
    )


def _make_powerfoam_test_camera(device: torch.device) -> ember.CameraState:
    return ember.CameraState(
        width=torch.tensor([16]),
        height=torch.tensor([16]),
        fov_degrees=torch.tensor([60.0]),
        intrinsics=torch.tensor(
            [
                [
                    [13.8564, 0.0, 8.0],
                    [0.0, 13.8564, 8.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ),
        cam_to_world=torch.eye(4).reshape(1, 4, 4),
        camera_convention="opencv",
    ).to(device)


def test_powerfoam_camera_from_camera_state_can_skip_full_ray_maps() -> None:
    from ember_native_powerfoam.powerfoam.runtime.camera import (
        powerfoam_camera_from_camera_state,
    )

    camera = _make_powerfoam_test_camera(torch.device("cpu"))

    full_camera = powerfoam_camera_from_camera_state(
        camera,
        build_ray_maps=True,
    )
    placeholder_camera = powerfoam_camera_from_camera_state(
        camera,
        build_ray_maps=False,
    )

    assert tuple(full_camera.ray_maps.shape) == (16, 16, 6)
    assert tuple(placeholder_camera.ray_maps.shape) == (1, 1, 6)
    assert placeholder_camera.ray_maps.dtype == torch.float32
    assert placeholder_camera.ray_maps.device.type == "cpu"
    assert placeholder_camera.width == full_camera.width
    assert placeholder_camera.height == full_camera.height
    assert torch.equal(placeholder_camera.eye, full_camera.eye)
    assert torch.equal(placeholder_camera.right, full_camera.right)
    assert torch.equal(placeholder_camera.up, full_camera.up)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_powerfoam_resampling_updates_scene_and_optimizer_state() -> None:
    import ember_native_powerfoam as powerfoam

    device = torch.device("cuda")
    scene = _make_powerfoam_cuda_scene()
    state = ember.TrainState(
        model=ember.InitializedModel(
            scene=scene,
            modules={},
            parameters={},
        ),
        step=9,
        seed=0,
        device=device,
    )
    optimizers = _make_powerfoam_optimizer_bindings(scene)
    stage = powerfoam.PowerFoamResampling(
        max_steps=20,
        resample_every=1,
        resample_offset=0,
        densify_from=0,
        densify_until=10,
        final_points=16,
        stop_fraction=1.0,
    )
    expected_target = powerfoam.powerfoam_target_points(
        9,
        initial_num_points=8,
        final_points=16,
        densify_from=0,
        densify_until=10,
    )
    stage.bind(state, optimizers, None)
    render_output = SimpleNamespace(
        contrib=torch.ones((1, 8), device=device),
        point_error=torch.linspace(0.1, 1.0, 8, device=device).reshape(1, 8),
        prim_visible_mask=torch.ones((1, 8), dtype=torch.bool, device=device),
    )
    context = DensificationContext(
        state=state,
        batch=None,
        render_output=render_output,
        loss_result=None,
        step=9,
        optimizers=optimizers,
        runtime=None,
    )

    stage.post_optimizer_step(context)
    metrics: dict[str, float] = {}
    stage.after_step(context, metrics)

    assert scene.points.shape[0] == expected_target
    assert scene.adjacency_offsets.shape == (expected_target + 1,)
    assert metrics["powerfoam/num_points"] == float(expected_target)
    assert metrics["powerfoam/target_points"] == float(expected_target)
    assert metrics["powerfoam/num_resampled"] == float(expected_target - 8)
    for binding in optimizers:
        parameter = getattr(scene, binding.target.name)
        optimizer_state = binding.optimizer.state[parameter]
        assert binding.base_parameter is parameter
        assert optimizer_state["exp_avg"].shape[0] == expected_target
        assert optimizer_state["exp_avg_sq"].shape[0] == expected_target

    powerfoam.register()
    output = ember.render(
        scene,
        _make_powerfoam_test_camera(device),
        backend="powerfoam.rasterize",
        return_alpha=True,
        options=powerfoam.PowerFoamNativeRenderOptions(clamp_output=False),
    )
    assert tuple(output.render.shape) == (1, 16, 16, 3)
    assert torch.isfinite(output.render).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_powerfoam_cuda_render_smoke() -> None:
    import ember_native_powerfoam as powerfoam
    from ember_native_powerfoam.powerfoam.runtime import (
        build_powerfoam_topology,
    )

    device = torch.device("cuda")
    points = torch.tensor(
        [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.0],
            [0.0, 0.0, 2.3],
            [0.0, 0.3, 2.2],
            [0.3, 0.0, 2.2],
            [-0.3, 0.0, 2.2],
        ],
        dtype=torch.float32,
        device=device,
    )
    radii = torch.full((points.shape[0],), 0.35, device=device)
    topology = build_powerfoam_topology(points, radii)
    quaternions = torch.zeros((points.shape[0], 4), device=device)
    quaternions[:, 0] = 1.0
    scene = ember.PowerFoamScene(
        points=points.detach().clone().requires_grad_(True),
        radii=radii.detach().clone().requires_grad_(True),
        quaternions=quaternions.detach().clone().requires_grad_(True),
        density=torch.full(
            (points.shape[0],),
            0.1,
            device=device,
        ).requires_grad_(True),
        texel_sites=(
            torch.randn((points.shape[0], 8, 2), device=device) * 0.01
        ).requires_grad_(True),
        texel_sv_axis=torch.randn(
            (points.shape[0], 8, 24),
            device=device,
        ).requires_grad_(True),
        texel_sv_rgb=torch.zeros(
            (points.shape[0], 8, 24),
            device=device,
        ).requires_grad_(True),
        texel_height=torch.zeros(
            (points.shape[0], 8),
            device=device,
        ).requires_grad_(True),
        adjacency=topology.adjacency,
        adjacency_offsets=topology.adjacency_offsets,
        sv_dof=8,
        num_texel_sites=8,
    )
    camera = ember.CameraState(
        width=torch.tensor([16]),
        height=torch.tensor([16]),
        fov_degrees=torch.tensor([60.0]),
        intrinsics=torch.tensor(
            [
                [
                    [13.8564, 0.0, 8.0],
                    [0.0, 13.8564, 8.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ),
        cam_to_world=torch.eye(4).reshape(1, 4, 4),
        camera_convention="opencv",
    ).to(device)
    powerfoam.register()

    output = ember.render(
        scene,
        camera,
        backend="powerfoam.rasterize",
        return_alpha=True,
        return_depth=True,
        return_normals=True,
        options=powerfoam.PowerFoamNativeRenderOptions(clamp_output=False),
    )
    loss = (output.render**2).mean()
    loss.backward()

    assert tuple(output.render.shape) == (1, 16, 16, 3)
    assert torch.isfinite(output.render).all()
    assert output.alphas is not None
    assert torch.isfinite(output.alphas).all()
    assert output.depth is not None
    assert torch.isfinite(output.depth).all()
    assert output.normals is not None
    assert torch.isfinite(output.normals).all()
    assert scene.points.grad is not None
    assert torch.isfinite(scene.points.grad).all()
