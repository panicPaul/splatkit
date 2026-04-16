from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from splatkit.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
    SparseVoxelScene,
)
from splatkit.core.registry import BACKEND_REGISTRY, register_backend
from splatkit.data.contracts import DatasetFrame, PointCloudState, SceneDataset
from splatkit.initialization import InitializedModel
from splatkit.training import (
    CallableSpec,
    LoadedCheckpoint,
    LossResult,
    TrainingConfig,
    build_inference_pipeline,
    build_loss_fn,
    build_optimizer_set,
    build_render_fn,
    initialize_model,
    load_checkpoint_dir,
    run_training,
    save_checkpoint_dir,
    train_step,
)
from splatkit.training.checkpoints import build_checkpoint_metadata
from splatkit.training.config import (
    BatchingConfig,
    CheckpointExportConfig,
    HookConfig,
    InitializationSpec,
    LossConfig,
    ModelSpec,
    OptimizationConfig,
    ParameterGroupConfig,
    ParameterSpec,
    RenderPipelineSpec,
    RuntimeConfig,
)
from splatkit.training.protocols import TrainState
from torch import nn

MODULE_NAME = __name__


def register_test_backend() -> None:
    if "unit_test_backend" in BACKEND_REGISTRY:
        return

    @register_backend(
        name="unit_test_backend",
        default_options=RenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
    )
    def render_unit_test_backend(
        scene: GaussianScene3D,
        camera: CameraState,
        *,
        return_alpha: bool = False,
        return_depth: bool = False,
        return_normals: bool = False,
        return_2d_projections: bool = False,
        return_projective_intersection_transforms: bool = False,
        options: RenderOptions | None = None,
    ) -> RenderOutput:
        del (
            return_alpha,
            return_depth,
            return_normals,
            return_2d_projections,
            return_projective_intersection_transforms,
            options,
        )
        mean_color = scene.feature[:, 0, :].mean(dim=0)
        num_cams = int(camera.width.shape[0])
        height = int(camera.height[0].item())
        width = int(camera.width[0].item())
        render = mean_color.view(1, 1, 1, 3).expand(
            num_cams,
            height,
            width,
            3,
        )
        return RenderOutput(render=render)


def rgb_l2_loss(
    state: TrainState,
    batch: object,
    render_output: RenderOutput,
    *,
    weights: dict[str, float],
) -> LossResult:
    del state, weights
    target = batch.images.permute(0, 2, 3, 1)
    loss = ((render_output.render - target) ** 2).mean()
    return LossResult(
        loss=loss, metrics={"rgb_mse": float(loss.detach().item())}
    )


def apply_color_mlp(
    model: InitializedModel,
    camera: CameraState,
) -> GaussianScene3D:
    del camera
    scene = model.scene
    assert isinstance(scene, GaussianScene3D)
    color_mlp = model.modules["color_mlp"]
    temperature = model.parameters["temperature"]
    base_dc = scene.feature[:, 0, :]
    lifted_dc = torch.sigmoid(color_mlp(base_dc) * temperature)
    feature = scene.feature.clone()
    feature[:, 0, :] = lifted_dc
    return replace(scene, feature=feature)


class CountingHook:
    def __init__(self) -> None:
        self.pre_count = 0
        self.post_count = 0
        self.after_count = 0

    def pre_backward(
        self,
        state: TrainState,
        batch: object,
        render_output: object,
        loss_result: LossResult,
    ) -> None:
        del state, batch, render_output, loss_result
        self.pre_count += 1

    def post_backward(
        self,
        state: TrainState,
        batch: object,
        render_output: object,
        loss_result: LossResult,
    ) -> None:
        del state, batch, render_output, loss_result
        self.post_count += 1

    def after_step(
        self,
        state: TrainState,
        metrics: dict[str, float],
    ) -> None:
        del state, metrics
        self.after_count += 1


def build_sparse_voxel_model(
    dataset: SceneDataset,
    *,
    modules: dict[str, nn.Module],
    parameters: dict[str, nn.Parameter],
) -> InitializedModel:
    del dataset
    scene = SparseVoxelScene(
        backend_name="new_cuda",
        active_sh_degree=0,
        max_num_levels=1,
        scene_center=torch.zeros(3),
        scene_extent=torch.ones(1),
        inside_extent=torch.ones(1),
        octpath=torch.zeros((1, 1), dtype=torch.int64),
        octlevel=torch.zeros((1, 1), dtype=torch.int64),
        geo_grid_pts=torch.zeros((8, 1)),
        sh0=torch.zeros((1, 3)),
        shs=torch.zeros((1, 0, 3)),
    )
    return InitializedModel(
        scene=scene,
        modules=modules,
        parameters=parameters,
        metadata={},
    )


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.fromarray(np.array([[color]], dtype=np.uint8)).save(path)


def build_dataset(tmp_path: Path) -> SceneDataset:
    image_a = tmp_path / "frame_a.png"
    image_b = tmp_path / "frame_b.png"
    _write_image(image_a, (255, 255, 255))
    _write_image(image_b, (128, 64, 32))
    frames = (
        DatasetFrame(
            frame_id="0",
            image_path=image_a,
            camera_index=0,
            width=1,
            height=1,
        ),
        DatasetFrame(
            frame_id="1",
            image_path=image_b,
            camera_index=1,
            width=1,
            height=1,
        ),
    )
    cam_to_world = (
        torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    )
    return SceneDataset(
        frames=frames,
        camera=CameraState(
            width=torch.tensor([1, 1], dtype=torch.int64),
            height=torch.tensor([1, 1], dtype=torch.int64),
            fov_degrees=torch.tensor([60.0, 60.0], dtype=torch.float32),
            cam_to_world=cam_to_world,
        ),
        source_format="colmap",
        root_path=tmp_path,
        point_cloud=PointCloudState(
            points=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            colors=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        ),
    )


def build_config(output_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        runtime=RuntimeConfig(device="cpu", seed=123, max_steps=3),
        batching=BatchingConfig(batch_size=1, shuffle=False, normalize=True),
        initialization=InitializationSpec(
            initializer=CallableSpec(
                target="splatkit.initialization.initialize_gaussian_model_from_dataset",
                kwargs={"sh_degree": 0},
            )
        ),
        model=ModelSpec(),
        render=RenderPipelineSpec(
            backend="unit_test_backend",
            return_alpha=False,
        ),
        optimization=OptimizationConfig(
            parameter_groups=[
                ParameterGroupConfig(selector="scene.feature", lr=0.2),
            ]
        ),
        loss=LossConfig(
            target=CallableSpec(target=f"{MODULE_NAME}.rgb_l2_loss"),
        ),
        hooks=HookConfig(),
        checkpoint=CheckpointExportConfig(output_dir=output_dir),
    )


def test_run_training_writes_checkpoint_directory(tmp_path: Path) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")

    result = run_training(dataset, config)

    checkpoint_dir = Path(result.checkpoint_dir)
    assert result.state.step == 3
    assert len(result.history) == 3
    assert (checkpoint_dir / "config.json").exists()
    assert (checkpoint_dir / "metadata.json").exists()
    assert (checkpoint_dir / "model.ckpt").exists()


def test_train_step_supports_modules_parameters_and_hooks(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = TrainingConfig(
        runtime=RuntimeConfig(device="cpu", seed=7, max_steps=1),
        batching=BatchingConfig(batch_size=1, shuffle=False, normalize=True),
        initialization=InitializationSpec(
            initializer=CallableSpec(
                target="splatkit.initialization.initialize_gaussian_model_from_dataset"
            )
        ),
        model=ModelSpec(
            modules={
                "color_mlp": CallableSpec(
                    target="torch.nn.Linear",
                    kwargs={"in_features": 3, "out_features": 3, "bias": False},
                )
            },
            parameters={
                "temperature": ParameterSpec(
                    shape=(1,),
                    init="constant",
                    value=1.0,
                )
            },
        ),
        render=RenderPipelineSpec(
            backend="unit_test_backend",
            feature_fn=CallableSpec(target=f"{MODULE_NAME}.apply_color_mlp"),
            return_alpha=False,
        ),
        optimization=OptimizationConfig(
            parameter_groups=[
                ParameterGroupConfig(selector="scene.feature", lr=0.1),
                ParameterGroupConfig(selector="modules.color_mlp", lr=0.1),
                ParameterGroupConfig(selector="parameters.temperature", lr=0.1),
            ]
        ),
        loss=LossConfig(
            target=CallableSpec(target=f"{MODULE_NAME}.rgb_l2_loss"),
        ),
        hooks=HookConfig(),
        checkpoint=CheckpointExportConfig(output_dir=tmp_path / "unused"),
    )
    model = initialize_model(dataset, config).to(torch.device("cpu"))
    state = TrainState(
        model=model,
        step=0,
        seed=config.runtime.seed,
        device=torch.device("cpu"),
    )
    batch = next(iter(build_dataset_loader(dataset, config)))
    render_fn = build_render_fn(config)
    loss_fn = build_loss_fn(config)
    optimizers = build_optimizer_set(state, config)
    hook = CountingHook()

    metrics = train_step(
        state,
        batch,
        render_fn=render_fn,
        loss_fn=loss_fn,
        optimizers=optimizers,
        hooks=[hook],
    )

    assert state.step == 1
    assert hook.pre_count == 1
    assert hook.post_count == 1
    assert hook.after_count == 1
    assert "loss" in metrics


def build_dataset_loader(
    dataset: SceneDataset,
    config: TrainingConfig,
) -> torch.utils.data.DataLoader:
    from splatkit.training.runtime import build_dataloader

    return build_dataloader(dataset, config)


def test_checkpoint_roundtrip_rebuilds_render_pipeline(tmp_path: Path) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    result = run_training(dataset, config)

    loaded = load_checkpoint_dir(result.checkpoint_dir)

    assert isinstance(loaded, LoadedCheckpoint)
    batch = next(iter(build_dataset_loader(dataset, config)))
    render_output = loaded.render_fn(loaded.model, batch.camera)
    assert render_output.render.shape == (1, 1, 1, 3)
    assert loaded.metadata.backend_name == "unit_test_backend"


def test_checkpoint_metadata_records_reproducibility_fields(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    model = initialize_model(dataset, config).to(torch.device("cpu"))
    state = TrainState(
        model=model,
        step=2,
        seed=config.runtime.seed,
        device=torch.device("cpu"),
    )

    metadata = build_checkpoint_metadata(state, config, dataset=dataset)

    assert metadata.seed == config.runtime.seed
    assert metadata.backend_name == "unit_test_backend"
    assert f"{MODULE_NAME}.rgb_l2_loss" in metadata.import_paths
    assert metadata.dataset_summary["num_frames"] == 2


def test_export_ply_omits_scene_payload_for_gaussian_scene(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.checkpoint.export_ply = True
    model = initialize_model(dataset, config).to(torch.device("cpu"))
    state = TrainState(
        model=model,
        step=0,
        seed=config.runtime.seed,
        device=torch.device("cpu"),
    )

    checkpoint_dir = save_checkpoint_dir(
        tmp_path / "exported",
        state,
        config,
        dataset=dataset,
    )
    payload = torch.load(checkpoint_dir / "model.ckpt", weights_only=False)

    assert payload["scene"] is None
    assert (checkpoint_dir / "scene.ply").exists()


def test_export_ply_rejects_unsupported_scene_types(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config = TrainingConfig(
        runtime=RuntimeConfig(device="cpu", seed=0, max_steps=1),
        batching=BatchingConfig(batch_size=1, shuffle=False),
        initialization=InitializationSpec(
            initializer=CallableSpec(
                target=f"{MODULE_NAME}.build_sparse_voxel_model"
            )
        ),
        model=ModelSpec(),
        render=RenderPipelineSpec(
            backend="unit_test_backend",
            return_alpha=False,
        ),
        optimization=OptimizationConfig(
            parameter_groups=[
                ParameterGroupConfig(selector="scene.sh0", lr=0.1),
            ]
        ),
        loss=LossConfig(
            target=CallableSpec(target=f"{MODULE_NAME}.rgb_l2_loss"),
        ),
        checkpoint=CheckpointExportConfig(
            output_dir=tmp_path / "run",
            export_ply=True,
        ),
    )
    model = initialize_model(dataset, config).to(torch.device("cpu"))
    state = TrainState(
        model=model,
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="PLY export"):
        save_checkpoint_dir(
            tmp_path / "unsupported", state, config, dataset=dataset
        )


def test_training_is_reproducible_for_same_seed(tmp_path: Path) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config_a = build_config(tmp_path / "run_a")
    config_b = build_config(tmp_path / "run_b")

    result_a = run_training(dataset, config_a)
    result_b = run_training(dataset, config_b)

    assert result_a.history == result_b.history
    assert torch.allclose(
        result_a.state.model.scene.feature,
        result_b.state.model.scene.feature,
    )


def test_build_inference_pipeline_aliases_checkpoint_loader(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    result = run_training(dataset, config)

    loaded = build_inference_pipeline(result.checkpoint_dir)

    assert isinstance(loaded, LoadedCheckpoint)
