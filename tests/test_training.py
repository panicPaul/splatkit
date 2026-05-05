from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from ember_core.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
    SparseVoxelScene,
)
from ember_core.core.registry import BACKEND_REGISTRY, register_backend
from ember_core.data import (
    PreparedFrameDataset,
    PreparedFrameDatasetConfig,
    prepare_frame_dataset,
)
from ember_core.data.contracts import (
    CameraSensorDataset,
    DatasetFrame,
    PathCameraImageSource,
    PointCloudState,
    SceneRecord,
)
from ember_core.densification import (
    DensificationContext,
    DensificationLifecycleContext,
    DensificationRenderRequirements,
)
from ember_core.densification.runtime import merge_densification_requirements
from ember_core.initialization import InitializedModel
from ember_core.initialization import (
    initialize_gaussian_model_from_scene_record as ember_core_initialize,
)
from ember_core.training import (
    CallableSpec,
    DensificationConfig,
    LoadedCheckpoint,
    LossResult,
    TrainingConfig,
    TrainingLoggingConfig,
    build_densification_for_context,
    build_inference_pipeline,
    build_loss_fn,
    build_optimizer_set,
    build_raw_render_fn,
    build_render_fn,
    build_training_render_fn,
    build_training_run_context,
    callable_spec,
    checkpoint_log_dir,
    checkpoint_run_dir,
    compute_frame_camera_extent,
    cycle_dataloader,
    densification_config,
    ensure_checkpoint_output_writable,
    initialize_model,
    load_checkpoint_dir,
    loss_config,
    materialize_optimization_config,
    materialize_training_config,
    optimization_config,
    resolve_backend_options,
    run_training,
    save_checkpoint_dir,
    scene_parameter,
    set_torch_seed,
    tensor_slice,
    tensor_view,
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
    ParameterSpec,
    ParameterTargetSpec,
    RenderPipelineSpec,
    RuntimeConfig,
    TensorSliceSpec,
    TensorViewSpec,
    TrainingProfilerConfig,
)
from ember_core.training.protocols import TrainState
from PIL import Image
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
        )
        options = options or RenderOptions()
        mean_color = scene.feature[:, 0, :].mean(dim=0)
        num_cams = int(camera.width.shape[0])
        height = int(camera.height[0].item())
        width = int(camera.width[0].item())
        render = mean_color.view(1, 1, 1, 3).expand(
            num_cams,
            height,
            width,
            3,
        ) + options.background_color.view(1, 1, 1, 3)
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


def test_batching_config_accepts_dataloader_worker_options() -> None:
    batching = BatchingConfig(
        batch_size=1,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    assert batching.model_dump(mode="json") == {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 4,
        "persistent_workers": True,
        "pin_memory": True,
    }


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


def training_backend_options_builder(
    state: TrainState,
    *,
    offset: float,
) -> dict[str, list[float]]:
    value = offset + state.step
    return {"background_color": [value, value, value]}


def context_optimization_builder(
    *,
    lr: float,
    max_steps: int,
) -> OptimizationConfig:
    return OptimizationConfig(
        parameter_groups=[
            ParameterGroupConfig(
                target=scene_target("feature"),
                lr=lr,
                scheduler=CallableSpec(
                    target="ember_core.training.exponential_decay_to",
                    kwargs={"final_lr": lr / 10.0},
                    context_kwargs={"max_steps": "max_steps"},
                ),
            )
        ]
    )


def context_device_initializer(
    scene_record: SceneRecord,
    *,
    modules: dict[str, nn.Module],
    parameters: dict[str, nn.Parameter],
    device: torch.device,
) -> InitializedModel:
    model = ember_core_initialize(
        scene_record,
        modules=modules,
        parameters=parameters,
    )
    model.metadata["initializer_device"] = str(device)
    return model


class CountingHook:
    def __init__(self) -> None:
        self.before_count = 0
        self.pre_count = 0
        self.post_count = 0
        self.post_optimizer_count = 0
        self.after_count = 0

    def before_step(self, state: TrainState) -> None:
        del state
        self.before_count += 1

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

    def get_render_requirements(
        self,
        state: object,
    ) -> DensificationRenderRequirements:
        del state
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

    def before_training(self, context: DensificationLifecycleContext) -> None:
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

    def after_training(self, context: DensificationLifecycleContext) -> None:
        del context


class ContextCameraExtentDensification(CloneFirstSplatDensification):
    def __init__(self, *, camera_extent: float) -> None:
        super().__init__()
        self.camera_extent = camera_extent


class LifecycleRecordingDensification(CloneFirstSplatDensification):
    def post_optimizer_step(self, context: DensificationContext) -> None:
        del context

    def before_training(self, context: DensificationLifecycleContext) -> None:
        context.state.model.metadata["before_training_step"] = (
            context.state.step
        )

    def after_training(self, context: DensificationLifecycleContext) -> None:
        context.state.model.metadata["after_training_step"] = context.state.step


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
    assert checkpoint_dir.name == "run_run_1"
    assert checkpoint_log_dir(checkpoint_dir) == checkpoint_dir / "logs"
    assert list((checkpoint_dir / "logs").glob("events.out.tfevents.*"))
    assert "iterations_per_second" in result.history[-1]
    assert result.history[-1]["iterations_per_second"] > 0.0
    assert result.history[-1]["step_seconds"] > 0.0
    assert result.history[-1]["elapsed_seconds"] > 0.0


class TypedUnitTrainingConfig:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.seen_dataset_length: int | None = None

    def to_training_config(
        self,
        frame_dataset: PreparedFrameDataset | None = None,
    ) -> TrainingConfig:
        assert frame_dataset is not None
        self.seen_dataset_length = len(frame_dataset)
        return build_config(self.output_dir)


def test_run_training_accepts_typed_training_config_source(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    typed_config = TypedUnitTrainingConfig(tmp_path / "typed-run")

    materialized = materialize_training_config(dataset, typed_config)
    result = run_training(dataset, typed_config)

    assert isinstance(materialized, TrainingConfig)
    assert typed_config.seen_dataset_length == len(dataset)
    assert Path(result.checkpoint_dir).name == "typed-run_run_1"


def test_training_utilities_cover_common_notebook_loop_needs(
    tmp_path: Path,
) -> None:
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    dataloader = build_dataset_loader(dataset, config)
    iterator = cycle_dataloader(dataloader)

    first = next(iterator)
    second = next(iterator)

    assert tuple(frame.frame_id for frame in second.frames) == tuple(
        frame.frame_id for frame in first.frames
    )
    assert compute_frame_camera_extent(dataset) == 0.0
    set_torch_seed(123)
    sample_a = torch.rand(1)
    set_torch_seed(123)
    sample_b = torch.rand(1)
    torch.testing.assert_close(sample_a, sample_b)


def test_notebook_spec_helpers_accept_local_callables(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    method = CloneFirstSplatDensification()
    config.densification = densification_config(method)

    densification = build_densification_for_context(
        config,
        context=build_training_run_context(dataset, config),
    )

    assert densification is method
    assert config.densification.model_dump(mode="json") == {"builders": []}

    config.densification = densification_config(
        callable_spec(
            ContextCameraExtentDensification,
            context_kwargs={"camera_extent": "camera_extent"},
        )
    )
    densification = build_densification_for_context(
        config,
        context=build_training_run_context(dataset, config),
    )
    assert isinstance(densification, ContextCameraExtentDensification)
    assert densification.camera_extent == 0.0

    config.loss = loss_config(rgb_l2_loss)
    assert build_loss_fn(config) is not None

    config.optimization = optimization_config(
        scene_parameter(
            "center_position",
            lr=0.1,
            view=tensor_view(tensor_slice(1, start=0, stop=2)),
        )
    )
    assert config.optimization.parameter_groups[0].target.scope == "scene"


def test_resolve_backend_options_applies_updates(tmp_path: Path) -> None:
    register_test_backend()
    config = build_config(tmp_path / "run")

    options = resolve_backend_options(
        config,
        updates={"background_color": [1.0, 0.0, 0.0]},
    )

    assert isinstance(options, RenderOptions)
    torch.testing.assert_close(
        options.background_color,
        torch.tensor([1.0, 0.0, 0.0]),
    )


def test_context_kwargs_materialize_optimization_builder(
    tmp_path: Path,
) -> None:
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.optimization = OptimizationConfig(
        builder=CallableSpec(
            target=f"{MODULE_NAME}.context_optimization_builder",
            kwargs={"lr": 0.2},
            context_kwargs={"max_steps": "max_steps"},
        )
    )
    context = build_training_run_context(dataset, config)

    optimization = materialize_optimization_config(
        config.optimization,
        context=context,
    )

    assert context.max_steps == config.runtime.max_steps
    assert optimization.parameter_groups[0].lr == 0.2
    assert optimization.parameter_groups[0].scheduler.kwargs == {
        "final_lr": 0.02,
    }
    assert optimization.parameter_groups[0].scheduler.context_kwargs == {
        "max_steps": "max_steps",
    }


def test_context_kwargs_materialize_densification_builder(
    tmp_path: Path,
) -> None:
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.densification = DensificationConfig(
        builders=[
            CallableSpec(
                target=f"{MODULE_NAME}.ContextCameraExtentDensification",
                context_kwargs={"camera_extent": "camera_extent"},
            )
        ]
    )

    densification = build_densification_for_context(
        config,
        context=build_training_run_context(dataset, config),
    )

    assert isinstance(densification, ContextCameraExtentDensification)
    assert densification.camera_extent == 0.0


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
    model = initialize_model(dataset.scene_record, config).to(
        torch.device("cpu")
    )
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
    assert hook.before_count == 1
    assert hook.pre_count == 1
    assert hook.post_count == 1
    assert hook.post_optimizer_count == 1
    assert hook.after_count == 1
    assert "loss" in metrics


def test_run_training_supports_runtime_only_hooks(tmp_path: Path) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    hook = CountingHook()

    result = run_training(dataset, config, runtime_hooks=[hook])

    assert len(result.history) == config.runtime.max_steps
    assert hook.before_count == config.runtime.max_steps
    assert hook.after_count == config.runtime.max_steps


def test_run_training_profiler_disabled_adds_no_profile_metrics(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")

    result = run_training(dataset, config)

    assert result.history
    assert not any(
        name.startswith("time_") or name.startswith("cuda_")
        for metrics in result.history
        for name in metrics
    )


def test_run_training_logging_can_be_disabled(tmp_path: Path) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run").model_copy(
        update={"logging": TrainingLoggingConfig(enabled=False)}
    )

    result = run_training(dataset, config)

    assert "iterations_per_second" in result.history[-1]
    assert not (Path(result.checkpoint_dir) / "logs").exists()


def test_run_training_profiler_records_phase_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run").model_copy(
        update={
            "profiler": TrainingProfilerConfig(
                enabled=True,
                log_every=1,
                cuda_memory=False,
                output_path=tmp_path / "profile.jsonl",
            )
        }
    )

    result = run_training(dataset, config)
    captured = capsys.readouterr()

    assert "time_render_ms" in result.history[-1]
    assert "time_backward_ms" in result.history[-1]
    assert result.history[-1]["primitives"] == 1.0
    assert (tmp_path / "profile.jsonl").read_text().count("\n") == 3
    assert '"step": 3' in captured.out


def test_train_step_supports_direct_densification_injection(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    model = initialize_model(dataset.scene_record, config).to(
        torch.device("cpu")
    )
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
        render=RenderPipelineSpec(
            backend="unit_test_backend", return_alpha=False
        ),
        loss=LossConfig(
            target=CallableSpec(
                target=f"{MODULE_NAME}.feature_regularization_loss"
            ),
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
        render=RenderPipelineSpec(
            backend="unit_test_backend", return_alpha=False
        ),
        loss=LossConfig(
            target=CallableSpec(target=f"{MODULE_NAME}.rgb_l2_loss")
        ),
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
        render=RenderPipelineSpec(
            backend="unit_test_backend", return_alpha=False
        ),
        loss=LossConfig(
            target=CallableSpec(
                target=f"{MODULE_NAME}.feature_regularization_loss"
            ),
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
    model = initialize_model(dataset.scene_record, config).to(
        torch.device("cpu")
    )
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
    model = initialize_model(dataset.scene_record, config).to(
        torch.device("cpu")
    )
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


def test_initialize_model_resolves_context_kwargs(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.initialization.initializer = CallableSpec(
        target=f"{MODULE_NAME}.context_device_initializer",
        context_kwargs={"device": "device"},
    )
    run_context = build_training_run_context(dataset, config)

    model = initialize_model(
        dataset.scene_record,
        config,
        context=run_context,
    )

    assert model.metadata["initializer_device"] == "cpu"


def test_run_training_calls_densification_lifecycle_hooks(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.runtime.max_steps = 1
    config.densification = DensificationConfig(
        builders=[
            CallableSpec(
                target=f"{MODULE_NAME}.LifecycleRecordingDensification"
            )
        ]
    )

    result = run_training(dataset, config)

    assert result.state.model.metadata["before_training_step"] == 0
    assert result.state.model.metadata["after_training_step"] == 1


def test_run_training_merges_densification_render_requirements(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.densification = DensificationConfig(
        builders=[CallableSpec(target="ember_core.densification.Vanilla3DGS")]
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


def test_run_training_uses_training_backend_options_builder(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.runtime.max_steps = 1
    config.render.training_backend_options_builder = CallableSpec(
        target=f"{MODULE_NAME}.training_backend_options_builder",
        kwargs={"offset": 0.5},
    )

    model = initialize_model(dataset.scene_record, config).to(
        torch.device("cpu")
    )
    state = TrainState(
        model=model,
        step=0,
        seed=config.runtime.seed,
        device=torch.device("cpu"),
    )
    training_render_fn = build_training_render_fn(config, state)
    regular_render_fn = build_render_fn(config)
    batch = next(iter(build_dataset_loader(dataset, config)))

    training_output = training_render_fn(model, batch.camera)
    regular_output = regular_render_fn(model, batch.camera)
    result = run_training(dataset, config)

    torch.testing.assert_close(
        training_output.render,
        regular_output.render + 0.5,
    )
    assert result.state.step == 1


def test_checkpoint_metadata_records_training_backend_options_builder(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.render.training_backend_options_builder = CallableSpec(
        target=f"{MODULE_NAME}.training_backend_options_builder",
        kwargs={"offset": 0.5},
    )
    model = initialize_model(dataset.scene_record, config).to(
        torch.device("cpu")
    )
    state = TrainState(
        model=model,
        step=0,
        seed=config.runtime.seed,
        device=torch.device("cpu"),
    )

    metadata = build_checkpoint_metadata(
        state,
        config,
        frame_dataset=dataset,
        run_context=build_training_run_context(dataset, config),
    )

    assert (
        f"{MODULE_NAME}.training_backend_options_builder"
        in metadata.import_paths
    )


def test_checkpoint_overwrite_guard_rejects_existing_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "checkpoint"
    output_dir.mkdir()
    (output_dir / "model.ckpt").write_text("existing")

    with pytest.raises(FileExistsError, match=r"checkpoint\.overwrite=true"):
        ensure_checkpoint_output_writable(output_dir, overwrite=False)

    ensure_checkpoint_output_writable(output_dir, overwrite=True)


def test_checkpoint_run_dir_uses_numbered_run_suffixes(tmp_path: Path) -> None:
    output_dir = tmp_path / "checkpoint"
    first_run = tmp_path / "checkpoint_run_1"
    first_run.mkdir()

    assert checkpoint_run_dir(output_dir, overwrite=False) == (
        tmp_path / "checkpoint_run_2"
    )
    assert checkpoint_run_dir(output_dir, overwrite=True) == first_run


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
    model = initialize_model(dataset.scene_record, config).to(
        torch.device("cpu")
    )
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
        run_context=build_training_run_context(dataset, config),
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
    assert metadata.dataset_summary["available_camera_sensor_ids"] == ["camera"]
    assert metadata.dataset_summary["default_camera_sensor_id"] == "camera"
    assert metadata.run_summary["camera_extent"] == 0.0
    assert metadata.run_summary["backend"] == "unit_test_backend"
    assert "ember_core" in metadata.provenance
    assert f"target:{MODULE_NAME}.rgb_l2_loss" in metadata.provenance
    assert "backend:unit_test_backend" in metadata.provenance


def test_export_ply_omits_scene_payload_for_gaussian_scene(
    tmp_path: Path,
) -> None:
    register_test_backend()
    dataset = build_dataset(tmp_path)
    config = build_config(tmp_path / "run")
    config.checkpoint.export_ply = True
    model = initialize_model(dataset.scene_record, config).to(
        torch.device("cpu")
    )
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
    model = initialize_model(dataset.scene_record, config).to(
        torch.device("cpu")
    )
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
