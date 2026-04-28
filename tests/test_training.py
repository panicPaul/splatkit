from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
    SparseVoxelScene,
)
from ember_core.core.registry import BACKEND_REGISTRY, register_backend
from ember_core.data.contracts import (
    CameraSensorDataset,
    DatasetFrame,
    PathCameraImageSource,
    PointCloudState,
    SceneRecord,
)
from ember_core.data import PreparedFrameDataset, PreparedFrameDatasetConfig, prepare_frame_dataset
from ember_core.densification import (
    DensificationContext,
    DensificationRenderRequirements,
)
from ember_core.densification.runtime import merge_densification_requirements
from ember_core.initialization import InitializedModel
from ember_core.training import (
    CallableSpec,
    DensificationConfig,
    LoadedCheckpoint,
    LossResult,
    TrainingConfig,
    build_raw_render_fn,
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
from ember_core.training.checkpoints import build_checkpoint_metadata
from ember_core.training.config import (
    BatchingConfig,
    CheckpointExportConfig,
    HookConfig,
    InitializationSpec,
    LossConfig,
    ModelSpec,
    OptimizationConfig,
    ParameterGroupConfig,
    ParameterTargetSpec,
    ParameterSpec,
    RenderPipelineSpec,
    RuntimeConfig,
    TensorSliceSpec,
    TensorViewSpec,
)
from ember_core.training.protocols import TrainState
from torch import nn

MODULE_NAME = __name__


def scene_target(
    name: str,
    *,
    view: TensorViewSpec | None = None,
) -> ParameterTargetSpec:
    return ParameterTargetSpec(scope="scene", name=name, view=view)


def module_target(name: str) -> ParameterTargetSpec:
    return ParameterTargetSpec(scope="modules", name=name)


def parameter_target(
    name: str,
    *,
    view: TensorViewSpec | None = None,
) -> ParameterTargetSpec:
    return ParameterTargetSpec(scope="parameters", name=name, view=view)


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
        return_gaussian_impact_score: bool = False,
        return_normals: bool = False,
        return_2d_projections: bool = False,
        return_projective_intersection_transforms: bool = False,
        options: RenderOptions | None = None,
    ) -> RenderOutput:
        del (
            return_alpha,
            return_depth,
            return_gaussian_impact_score,
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
    target = batch.images
    loss = ((render_output.render - target) ** 2).mean()
    return LossResult(
        loss=loss, metrics={"rgb_mse": float(loss.detach().item())}
    )


def feature_regularization_loss(
    state: TrainState,
    batch: object,
    render_output: RenderOutput,
    *,
    weights: dict[str, float],
) -> LossResult:
    del batch, render_output, weights
    scene = state.model.scene
    assert isinstance(scene, GaussianScene3D)
    loss = scene.feature.square().mean()
    return LossResult(loss=loss)


def test_prepared_frame_materialization_rejects_single_worker() -> None:
    with pytest.raises(ValueError, match="0, None, or >= 2"):
        PreparedFrameDatasetConfig(
            materialization={"num_workers": 1},
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


def one_render_postprocess(
    model: InitializedModel,
    camera: CameraState,
    render_output: RenderOutput,
) -> RenderOutput:
    del model, camera
    return replace(render_output, render=torch.ones_like(render_output.render))


class CountingHook:
    def __init__(self) -> None:
        self.pre_count = 0
        self.post_count = 0
        self.post_optimizer_count = 0
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

    def post_optimizer_step(
        self,
        state: TrainState,
        batch: object,
        render_output: object,
        loss_result: LossResult,
    ) -> None:
        del state, batch, render_output, loss_result
        self.post_optimizer_count += 1


class CloneFirstSplatDensification:
    expected_scene_families = ("gaussian",)

    def __init__(self) -> None:
        self._family_ops = None
        self.post_optimizer_count = 0

    def get_render_requirements(self) -> DensificationRenderRequirements:
        return DensificationRenderRequirements()

    def bind(
        self,
        state: TrainState,
        optimizers: list[object],
        family_ops: object,
    ) -> None:
        del state, optimizers
        self._family_ops = family_ops

    def pre_backward(self, context: DensificationContext) -> None:
        del context

    def post_backward(self, context: DensificationContext) -> None:
        del context

    def post_optimizer_step(self, context: DensificationContext) -> None:
        del context
        self.post_optimizer_count += 1
        assert self._family_ops is not None
        self._family_ops.clone(torch.tensor([True], dtype=torch.bool))

    def after_step(
        self,
        context: DensificationContext,
        metrics: dict[str, float],
    ) -> None:
        del context, metrics


def build_sparse_voxel_model(
    scene_record: SceneRecord,
    *,
    modules: dict[str, nn.Module],
    parameters: dict[str, nn.Parameter],
) -> InitializedModel:
    del scene_record
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


def build_dataset(tmp_path: Path) -> PreparedFrameDataset:
    image_a = tmp_path / "frame_a.png"
    image_b = tmp_path / "frame_b.png"
    _write_image(image_a, (255, 255, 255))
    _write_image(image_b, (128, 64, 32))
    frames = (
        DatasetFrame(
            frame_id="0",
            sensor_id="camera",
            camera_index=0,
            width=1,
            height=1,
        ),
        DatasetFrame(
            frame_id="1",
            sensor_id="camera",
            camera_index=1,
            width=1,
            height=1,
        ),
    )
    cam_to_world = (
        torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    )
    camera_sensor = CameraSensorDataset(
        sensor_id="camera",
        kind="camera",
        frames=frames,
        timestamps_us=(None, None),
        camera=CameraState(
            width=torch.tensor([1, 1], dtype=torch.int64),
            height=torch.tensor([1, 1], dtype=torch.int64),
            fov_degrees=torch.tensor([60.0, 60.0], dtype=torch.float32),
            cam_to_world=cam_to_world,
        ),
        image_source=PathCameraImageSource(
            frame_paths={"0": image_a, "1": image_b}
        ),
    )
    scene_record = SceneRecord(
        sensors=(camera_sensor,),
        source_format="colmap",
        default_camera_sensor_id="camera",
        source_uris=(str(tmp_path),),
        point_cloud=PointCloudState(
            points=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            colors=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        ),
    )
    return prepare_frame_dataset(
        scene_record,
        PreparedFrameDatasetConfig(camera_sensor_id="camera"),
    )


def build_config(output_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        runtime=RuntimeConfig(device="cpu", seed=123, max_steps=3),
        batching=BatchingConfig(batch_size=1, shuffle=False),
        initialization=InitializationSpec(
            initializer=CallableSpec(
                target="ember_core.initialization.initialize_gaussian_model_from_scene_record",
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
                ParameterGroupConfig(target=scene_target("feature"), lr=0.2),
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
        batching=BatchingConfig(batch_size=1, shuffle=False),
        initialization=InitializationSpec(
            initializer=CallableSpec(
                target="ember_core.initialization.initialize_gaussian_model_from_scene_record"
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
                ParameterGroupConfig(target=scene_target("feature"), lr=0.1),
                ParameterGroupConfig(
                    target=module_target("color_mlp"),
                    lr=0.1,
                ),
                ParameterGroupConfig(
                    target=parameter_target("temperature"),
                    lr=0.1,
                ),
            ]
        ),
        loss=LossConfig(
            target=CallableSpec(target=f"{MODULE_NAME}.rgb_l2_loss"),
        ),
        hooks=HookConfig(),
        checkpoint=CheckpointExportConfig(output_dir=tmp_path / "unused"),
    )
    model = initialize_model(dataset.scene_record, config).to(torch.device("cpu"))
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
    assert hook.post_optimizer_count == 1
    assert hook.after_count == 1
    assert "loss" in metrics


def test_train_step_supports_direct_densification_injection(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    model = initialize_model(dataset.scene_record, config).to(torch.device("cpu"))
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
    densification = CloneFirstSplatDensification()
    from ember_core.densification.runtime import bind_densification

    bind_densification(densification, state, optimizers)

    train_step(
        state,
        batch,
        render_fn=render_fn,
        loss_fn=loss_fn,
        optimizers=optimizers,
        densification=densification,
    )

    assert densification.post_optimizer_count == 1
    assert state.model.scene.feature.shape[0] == 2


def test_view_backed_optimizer_updates_only_selected_slice(
    cpu_scene: GaussianScene3D,
) -> None:
    feature = torch.ones_like(cpu_scene.feature).requires_grad_(True)
    state = TrainState(
        model=InitializedModel(
            scene=replace(cpu_scene, feature=feature),
            modules={},
            parameters={},
        ),
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )
    config = TrainingConfig(
        render=RenderPipelineSpec(backend="unit_test_backend", return_alpha=False),
        loss=LossConfig(
            target=CallableSpec(target=f"{MODULE_NAME}.feature_regularization_loss"),
        ),
        optimization=OptimizationConfig(
            parameter_groups=[
                ParameterGroupConfig(
                    target=scene_target(
                        "feature",
                        view=TensorViewSpec(
                            slices=(TensorSliceSpec(axis=1, start=0, stop=1),)
                        ),
                    ),
                    optimizer="sgd",
                    lr=0.1,
                    weight_decay=0.2,
                )
            ]
        ),
    )

    optimizers = build_optimizer_set(state, config)
    before = state.model.scene.feature.detach().clone()
    state.model.scene.feature.square().mean().backward()

    optimizers[0].step()

    after = state.model.scene.feature.detach()
    assert not torch.equal(after[:, :1, :], before[:, :1, :])
    torch.testing.assert_close(after[:, 1:, :], before[:, 1:, :])


def test_build_optimizer_set_rejects_overlapping_views(
    cpu_scene: GaussianScene3D,
) -> None:
    state = TrainState(
        model=InitializedModel(scene=cpu_scene, modules={}, parameters={}),
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )
    config = TrainingConfig(
        render=RenderPipelineSpec(backend="unit_test_backend", return_alpha=False),
        loss=LossConfig(target=CallableSpec(target=f"{MODULE_NAME}.rgb_l2_loss")),
        optimization=OptimizationConfig(
            parameter_groups=[
                ParameterGroupConfig(
                    target=scene_target(
                        "feature",
                        view=TensorViewSpec(
                            slices=(TensorSliceSpec(axis=1, start=0, stop=2),)
                        ),
                    ),
                    lr=0.1,
                ),
                ParameterGroupConfig(
                    target=scene_target(
                        "feature",
                        view=TensorViewSpec(
                            slices=(TensorSliceSpec(axis=1, start=1, stop=3),)
                        ),
                    ),
                    lr=0.1,
                ),
            ]
        ),
    )

    with pytest.raises(ValueError, match="Overlapping optimizer views"):
        build_optimizer_set(state, config)


def test_scheduler_steps_after_optimizer_step(
    cpu_scene: GaussianScene3D,
) -> None:
    feature = cpu_scene.feature.clone().requires_grad_(True)
    state = TrainState(
        model=InitializedModel(
            scene=replace(cpu_scene, feature=feature),
            modules={},
            parameters={},
        ),
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )
    config = TrainingConfig(
        render=RenderPipelineSpec(backend="unit_test_backend", return_alpha=False),
        loss=LossConfig(
            target=CallableSpec(target=f"{MODULE_NAME}.feature_regularization_loss"),
        ),
        optimization=OptimizationConfig(
            parameter_groups=[
                ParameterGroupConfig(
                    target=scene_target("feature"),
                    optimizer="sgd",
                    lr=1.0,
                    scheduler=CallableSpec(
                        target="ember_core.training.schedules.exponential_decay_to",
                        kwargs={"final_lr": 0.01, "max_steps": 2},
                    ),
                )
            ]
        ),
    )

    optimizers = build_optimizer_set(state, config)
    binding = optimizers[0]
    state.model.scene.feature.square().mean().backward()
    binding.step()
    assert binding.current_lr() == pytest.approx(0.1)

    binding.zero_grad()
    state.model.scene.feature.square().mean().backward()
    binding.step()
    assert binding.current_lr() == pytest.approx(0.01)


def test_rgb_l2_loss_uses_nhwc_batch_images(tmp_path: Path) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    model = initialize_model(dataset.scene_record, config).to(torch.device("cpu"))
    state = TrainState(
        model=model,
        step=0,
        seed=config.runtime.seed,
        device=torch.device("cpu"),
    )
    batch = next(iter(build_dataset_loader(dataset, config)))
    render_fn = build_render_fn(config)
    render_output = render_fn(state.model, batch.camera)

    assert batch.images.shape == render_output.render.shape

    loss_result = rgb_l2_loss(state, batch, render_output, weights={})

    expected_loss = ((render_output.render - batch.images) ** 2).mean()
    torch.testing.assert_close(loss_result.loss, expected_loss)


def test_build_raw_render_fn_skips_postprocess(tmp_path: Path) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.render.postprocess_fn = CallableSpec(
        target=f"{MODULE_NAME}.one_render_postprocess"
    )
    model = initialize_model(dataset.scene_record, config).to(torch.device("cpu"))
    batch = next(iter(build_dataset_loader(dataset, config)))
    raw_render_fn = build_raw_render_fn(config)
    render_fn = build_render_fn(config)

    raw_output = raw_render_fn(model, batch.camera)
    postprocessed_output = render_fn(model, batch.camera)

    assert not torch.equal(raw_output.render, postprocessed_output.render)
    assert torch.allclose(
        postprocessed_output.render,
        torch.ones_like(postprocessed_output.render),
    )


def test_run_training_merges_densification_render_requirements(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.densification = DensificationConfig(
        builder=CallableSpec(target="ember_core.densification.Vanilla3DGS")
    )

    with pytest.raises(ValueError, match="2d_projections"):
        run_training(dataset, config)


def test_build_render_fn_rejects_unsupported_gaussian_impact_score(
    tmp_path: Path,
) -> None:
    register_test_backend()
    config = build_config(tmp_path / "run")
    config.render.return_gaussian_impact_score = True

    with pytest.raises(ValueError, match="gaussian_impact_score"):
        build_render_fn(config)


def test_merge_densification_requirements_propagates_gaussian_impact_score(
    tmp_path: Path,
) -> None:
    config = build_config(tmp_path / "run")

    merged = merge_densification_requirements(
        config,
        DensificationRenderRequirements(return_gaussian_impact_score=True),
    )

    assert merged.render.return_gaussian_impact_score is True


def build_dataset_loader(
    dataset: PreparedFrameDataset,
    config: TrainingConfig,
) -> torch.utils.data.DataLoader:
    from ember_core.training.runtime import build_dataloader

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
    config.optimization.parameter_groups[0].scheduler = CallableSpec(
        target="ember_core.training.schedules.exponential_decay_to",
        kwargs={"final_lr": 0.02, "max_steps": 3},
    )
    model = initialize_model(dataset.scene_record, config).to(torch.device("cpu"))
    state = TrainState(
        model=model,
        step=2,
        seed=config.runtime.seed,
        device=torch.device("cpu"),
    )

    metadata = build_checkpoint_metadata(
        state,
        config,
        frame_dataset=dataset,
    )

    assert metadata.seed == config.runtime.seed
    assert metadata.backend_name == "unit_test_backend"
    assert f"{MODULE_NAME}.rgb_l2_loss" in metadata.import_paths
    assert (
        "ember_core.training.schedules.exponential_decay_to"
        in metadata.import_paths
    )
    assert metadata.dataset_summary["num_frames"] == 2
    assert metadata.dataset_summary["source_format"] == "colmap"
    assert metadata.dataset_summary["source_uris"] == [str(tmp_path)]
    assert metadata.dataset_summary["available_camera_sensor_ids"] == [
        "camera"
    ]
    assert metadata.dataset_summary["default_camera_sensor_id"] == "camera"


def test_export_ply_omits_scene_payload_for_gaussian_scene(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.checkpoint.export_ply = True
    model = initialize_model(dataset.scene_record, config).to(torch.device("cpu"))
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
        frame_dataset=dataset,
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
                ParameterGroupConfig(target=scene_target("sh0"), lr=0.1),
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
    model = initialize_model(dataset.scene_record, config).to(torch.device("cpu"))
    state = TrainState(
        model=model,
        step=0,
        seed=0,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="PLY export"):
        save_checkpoint_dir(
            tmp_path / "unsupported",
            state,
            config,
            frame_dataset=dataset,
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
