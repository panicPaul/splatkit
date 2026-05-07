from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from marimo_config_gui.api import load_script_config
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "fastgs" / "notebook.py"


def load_fastgs_preset(fastgs_config_module, name: str):
    return load_preset_config(
        fastgs_config_module.fastgs_preset_catalog(), name
    )


def load_fastgs_script_config(
    fastgs_config_module,
    args: list[str],
):
    return load_script_config(
        fastgs_config_module.FastGSExperimentConfig,
        args=args,
        presets=fastgs_config_module.fastgs_preset_catalog(),
    )


def load_fastgs_config_module():
    spec = importlib.util.spec_from_file_location(
        "papers.fastgs.notebook",
        NOTEBOOK_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fastgs_resolved_training_config_defaults_to_native_backend() -> None:
    module = load_fastgs_config_module()
    experiment_config = load_fastgs_preset(module, "garden_base")

    training_config = module.resolve_training_config(experiment_config)

    assert experiment_config.training.render.backend == "faster_gs.fastgs"
    assert experiment_config.data.materialization_stage == "prepared"
    assert experiment_config.data.materialization_mode == "eager"
    assert experiment_config.data.materialization_num_workers == 8
    assert experiment_config.training.batching.num_workers == 8
    assert training_config.render.backend == "faster_gs.fastgs"
    assert training_config.render.backend_options == {
        "near_plane": 0.2,
        "far_plane": 10_000.0,
        "mip_splatting_screen_filter": False,
        "compact_box_scale": 0.5,
        "background_color": [0.0, 0.0, 0.0],
    }
    assert (
        training_config.initialization.initializer.target
        == "papers.fastgs.notebook.initialize_fastgs_model_from_scene_record"
    )
    assert (
        training_config.densification.builders[0].target
        == "papers.fastgs.notebook.FastGSVanillaDensification"
    )
    assert (
        training_config.densification.builders[-1].target
        == "papers.fastgs.notebook.FastGSFinalCleanup"
    )
    assert (
        training_config.checkpoint.output_dir
        == REPO_ROOT / "checkpoints/papers/fastgs/garden_base/faster_gs.fastgs"
    )


def test_fastgs_garden_big_preset_uses_native_upstream_settings() -> None:
    module = load_fastgs_config_module()
    experiment_config = load_fastgs_preset(module, "garden_big")

    training_config = module.resolve_training_config(experiment_config)

    assert experiment_config.training.render.backend == "faster_gs.fastgs"
    assert experiment_config.data.materialization_stage == "prepared"
    assert experiment_config.data.materialization_mode == "eager"
    assert experiment_config.data.materialization_num_workers == 8
    assert experiment_config.training.batching.num_workers == 8
    assert training_config.render.backend == "faster_gs.fastgs"
    assert (
        training_config.checkpoint.output_dir
        == REPO_ROOT / "checkpoints/papers/fastgs/garden_big/faster_gs.fastgs"
    )
    assert experiment_config.training.optimization.highfeature_lr == 0.02
    assert experiment_config.training.loss.lambda_opacity_regularization == 0.0
    assert experiment_config.training.loss.lambda_scale_regularization == 0.0

    vanilla = experiment_config.training.densification.vanilla
    assert vanilla.refine_every == 100
    assert vanilla.loss_thresh == 0.06
    assert vanilla.grad_abs_threshold == 0.0003
    assert vanilla.dense_fraction == 0.001
    assert vanilla.metric_map_backend == "eager"


def test_fastgs_script_loader_applies_preset_then_cli_overrides() -> None:
    module = load_fastgs_config_module()

    loaded = load_fastgs_script_config(
        module,
        args=[
            "--preset",
            "garden_base",
            "--training.render.backend",
            "adapter.fastgs",
            "--training.runtime.max-steps",
            "5",
        ],
    )

    assert isinstance(loaded, module.FastGSExperimentConfig)
    assert loaded.preset == "garden_base"
    assert loaded.training.render.backend == "adapter.fastgs"
    assert loaded.training.runtime.max_steps == 5

    training_config = module.resolve_training_config(loaded)
    assert training_config.render.backend == "adapter.fastgs"
    assert training_config.render.backend_options == {
        "mult": 0.5,
        "background_color": [0.0, 0.0, 0.0],
    }


def test_fastgs_user_config_does_not_expose_runtime_kwargs() -> None:
    module = load_fastgs_config_module()
    config = load_fastgs_preset(module, "garden_base")
    serialized = json.dumps(config.model_dump(mode="json"))

    assert '"kwargs"' not in serialized
    assert '"context_kwargs"' not in serialized
    assert "proper_antialiasing" not in serialized


def test_fastgs_script_loader_replays_json_config(tmp_path: Path) -> None:
    module = load_fastgs_config_module()
    config = load_fastgs_preset(module, "garden_base")
    json_path = tmp_path / "fastgs_config.json"
    json_path.write_text(json.dumps(config.model_dump(mode="json"), indent=2))

    loaded = load_fastgs_script_config(
        module,
        args=[str(json_path)],
    )

    assert isinstance(loaded, module.FastGSExperimentConfig)
    assert loaded == config


def test_fastgs_scene_loader_materializes_resized_image_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_fastgs_config_module()
    scene_path = tmp_path / "garden"
    prebuilt_image_root = scene_path / "images_4"
    prebuilt_image_root.mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def fake_materialize_fastgs_resized_image_cache(
        *,
        source_root: Path,
        cache_root: Path,
        scale: float,
        interpolation: str,
        max_caches: int,
    ) -> Path:
        calls.append(
            {
                "source_root": source_root,
                "cache_root": cache_root,
                "scale": scale,
                "interpolation": interpolation,
                "max_caches": max_caches,
            }
        )
        return cache_root

    monkeypatch.setattr(
        module,
        "materialize_fastgs_resized_image_cache",
        fake_materialize_fastgs_resized_image_cache,
    )
    config = load_fastgs_preset(module, "garden_base").model_copy(
        update={
            "scene": module.FastGSSceneConfig(path=scene_path),
        }
    )

    scene_config = module.build_scene_load_config(config)

    assert scene_config.image_root == (
        scene_path / "ember_cache/resized_images/scale_0p25_bicubic"
    )
    assert calls == [
        {
            "source_root": scene_path / "images",
            "cache_root": scene_config.image_root,
            "scale": 0.25,
            "interpolation": "bicubic",
            "max_caches": 4,
        }
    ]


def test_fastgs_l1_metric_map_matches_normalized_l1_formula() -> None:
    module = load_fastgs_config_module()
    predicted = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
            [[4.0, 4.0, 4.0], [6.0, 6.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    target = torch.zeros_like(predicted)

    metric_map = module.fastgs_l1_metric_map(
        predicted,
        target,
        loss_thresh=0.5,
    )

    torch.testing.assert_close(
        metric_map,
        torch.tensor([[0, 0], [1, 1]], dtype=torch.int32),
    )


def test_fastgs_l1_metric_map_handles_constant_error() -> None:
    module = load_fastgs_config_module()
    predicted = torch.ones((2, 2, 3), dtype=torch.float32)
    target = torch.zeros_like(predicted)

    metric_map = module.fastgs_l1_metric_map(
        predicted,
        target,
        loss_thresh=0.5,
    )

    torch.testing.assert_close(
        metric_map, torch.zeros((2, 2), dtype=torch.int32)
    )


@pytest.mark.cuda
def test_compiled_fastgs_l1_metric_map_matches_eager() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for compiled FastGS metric-map parity.")
    module = load_fastgs_config_module()
    predicted = torch.rand((8, 8, 3), device="cuda")
    target = torch.rand_like(predicted)

    eager = module.fastgs_l1_metric_map(
        predicted,
        target,
        loss_thresh=0.35,
    )
    compiled = module.compiled_fastgs_l1_metric_map(
        predicted,
        target,
        loss_thresh=0.35,
    )

    torch.testing.assert_close(compiled, eager)


def test_fastgs_densification_accumulators_follow_clone_and_split() -> None:
    module = load_fastgs_config_module()
    densification = module.FastGSVanillaDensification()
    densification.clone_grad_sum = torch.arange(4, dtype=torch.float32)
    densification.split_grad_sum = torch.arange(10, 14, dtype=torch.float32)
    densification.visible_count = torch.ones(4)
    densification.max_screen_radii = torch.arange(20, 24, dtype=torch.float32)

    densification._append_zero_accumulator_values(2)
    split_mask = torch.tensor([False, True, False, False, False, True])
    densification._split_accumulator_values(split_mask, num_children=2)

    assert densification.max_screen_radii is not None
    assert densification.max_screen_radii.shape == (8,)
    torch.testing.assert_close(
        densification.max_screen_radii,
        torch.tensor([20.0, 22.0, 23.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_fastgs_grown_accumulators_match_fused_clone_split_order() -> None:
    module = load_fastgs_config_module()
    densification = module.FastGSVanillaDensification()
    value = torch.arange(20, 24, dtype=torch.float32)
    clone_mask = torch.tensor([True, False, True, False])
    split_mask = torch.tensor([False, True, False, False])

    grown = densification._grown_zero_accumulator_values(
        value,
        clone_mask,
        split_mask,
        num_children=2,
    )

    torch.testing.assert_close(
        grown,
        torch.tensor([20.0, 22.0, 23.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_fastgs_refinement_pruning_pads_scores_after_growth() -> None:
    module = load_fastgs_config_module()
    densification = module.FastGSVanillaDensification()
    prune_mask = torch.ones(6, dtype=torch.bool)
    pruning_score = torch.tensor([0.1, 0.2, 0.3, 0.4])

    torch.manual_seed(0)
    sampled_prune_mask = densification._sample_refinement_prune_mask(
        prune_mask,
        pruning_score,
    )

    assert int(sampled_prune_mask.sum().item()) == 3
    assert not torch.any(sampled_prune_mask[4:])


def test_fastgs_pruning_score_uses_photometric_probe_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_fastgs_config_module()
    densification = module.FastGSVanillaDensification()
    photometric_values = iter(
        (
            torch.tensor(2.0),
            torch.tensor(10.0),
        )
    )
    monkeypatch.setattr(
        densification,
        "_photometric_loss",
        lambda _predicted, _target: next(photometric_values),
    )

    class _State:
        model = type(
            "Model",
            (),
            {
                "scene": type(
                    "Scene",
                    (),
                    {"center_position": torch.zeros((2, 3))},
                )()
            },
        )()

    class _ProbeOutput:
        def __init__(self, render: torch.Tensor) -> None:
            self.render = render

    class _Sample:
        def __init__(self, value: float) -> None:
            self.camera = object()
            self.image = torch.full((2, 2, 3), value)

    class _Attribution:
        def __init__(self) -> None:
            self.values = iter(
                (
                    torch.tensor([1.0, 0.0]),
                    torch.tensor([0.0, 1.0]),
                )
            )

        def attribute_metric_map(self, *_args, **_kwargs) -> torch.Tensor:
            return next(self.values)

    class _Runtime:
        render_options = None

        def __init__(self) -> None:
            self.views = (_Sample(0.0), _Sample(1.0))
            self.renders = iter(
                (
                    torch.ones((1, 2, 2, 3)),
                    torch.zeros((1, 2, 2, 3)),
                )
            )
            self.attribution = _Attribution()

        def sample_views(self, _count: int):
            return self.views

        def render_raw(self, *_args):
            return _ProbeOutput(next(self.renders))

        def resolve_trait(self, _trait_type):
            return self.attribution

    _importance_score, pruning_score = densification.compute_fastgs_scores(
        module.DensificationContext(
            state=_State(),
            batch=None,
            render_output=None,
            loss_result=None,
            step=0,
            optimizers=(),
            runtime=_Runtime(),
        ),
        densify=True,
    )

    torch.testing.assert_close(pruning_score, torch.tensor([0.0, 1.0]))
