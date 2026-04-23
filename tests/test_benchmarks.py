from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from splatkit import (
    CameraState,
    GaussianScene3D,
    ImagePreparationConfig,
    MaterializationConfig,
    PreparedFrameDataset,
    PreparedFrameDatasetConfig,
    RenderOptions,
    RenderOutput,
    get_sample_scene_path,
    initialize_gaussian_scene_from_scene_record,
    load_colmap_scene_record,
)
from splatkit.benchmarks import (
    RenderBenchmarkResult,
    benchmark_backend_render,
    benchmark_dataloader,
)
from splatkit.core import BACKEND_REGISTRY, register_backend
from splatkit.data import collate_frame_samples
from splatkit.benchmarks.render import _build_comparison_payload
from splatkit.benchmarks.ply_render import _default_repo_point_cloud_path
from torch.utils.data import DataLoader


def _register_unit_test_backend() -> None:
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
        render = mean_color.view(1, 1, 1, 3).expand(
            camera.width.shape[0],
            int(camera.height[0].item()),
            int(camera.width[0].item()),
            3,
        )
        return RenderOutput(render=render)


def _first_camera(camera: CameraState) -> CameraState:
    return CameraState(
        width=camera.width[:1],
        height=camera.height[:1],
        fov_degrees=camera.fov_degrees[:1],
        cam_to_world=camera.cam_to_world[:1],
        intrinsics=None if camera.intrinsics is None else camera.intrinsics[:1],
        camera_convention=camera.camera_convention,
    )


def test_benchmark_dataloader_reports_expected_metrics() -> None:
    scene_record = load_colmap_scene_record(get_sample_scene_path())
    frame_dataset = PreparedFrameDataset(
        scene_record,
        config=PreparedFrameDatasetConfig(
            split=None,
            materialization=MaterializationConfig(
                stage="prepared",
                mode="eager",
                num_workers=0,
            ),
            image_preparation=ImagePreparationConfig(normalize=True),
        ),
    )
    dataloader = DataLoader(
        frame_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_frame_samples,
    )

    result = benchmark_dataloader(
        dataloader,
        warmup_steps=1,
        measured_steps=3,
    )

    assert result.measured_steps == 3
    assert result.initialization_ms >= 0.0
    assert result.mean_ms_per_batch >= 0.0
    assert result.p90_ms_per_batch >= result.p50_ms_per_batch
    assert "iters_per_sec" in asdict(result)


def test_benchmark_backend_render_reports_expected_metrics() -> None:
    _register_unit_test_backend()
    scene_record = load_colmap_scene_record(get_sample_scene_path())
    scene = initialize_gaussian_scene_from_scene_record(scene_record)
    camera = _first_camera(scene_record.resolve_camera_sensor().camera)

    result = benchmark_backend_render(
        scene,
        camera,
        backend="unit_test_backend",
        warmup_steps=1,
        measured_steps=3,
    )

    assert result.backend == "unit_test_backend"
    assert result.image_size == (
        int(camera.width[0].item()),
        int(camera.height[0].item()),
    )
    assert result.num_points == int(scene.center_position.shape[0])
    assert result.first_frame_ms >= 0.0
    assert result.mean_ms_per_frame >= 0.0
    assert result.p90_ms_per_frame >= result.p50_ms_per_frame


def test_render_benchmark_comparison_payload_reports_expected_shape() -> None:
    primary = RenderBenchmarkResult(
        backend="faster_gs_mojo.core",
        device="cuda:0",
        image_size=(640, 480),
        num_points=1024,
        warmup_steps=10,
        measured_steps=100,
        first_frame_ms=1.5,
        mean_ms_per_frame=1.0,
        p50_ms_per_frame=0.9,
        p90_ms_per_frame=1.2,
        fps=1000.0,
    )
    comparison = RenderBenchmarkResult(
        backend="faster_gs.core",
        device="cuda:0",
        image_size=(640, 480),
        num_points=1024,
        warmup_steps=10,
        measured_steps=100,
        first_frame_ms=1.1,
        mean_ms_per_frame=0.8,
        p50_ms_per_frame=0.75,
        p90_ms_per_frame=0.95,
        fps=1250.0,
    )

    payload = _build_comparison_payload(primary, comparison)

    assert payload["primary"]["backend"] == "faster_gs_mojo.core"
    assert payload["comparison"]["backend"] == "faster_gs.core"
    assert payload["delta_ms_per_frame"] == pytest.approx(-0.2)
    assert payload["ratio_vs_primary"] == pytest.approx(0.8)
    assert payload["faster_backend"] == "faster_gs.core"


def test_default_repo_point_cloud_benchmark_asset_exists() -> None:
    path = _default_repo_point_cloud_path()
    assert path.name == "point_cloud.ply"
    assert path.exists()


@pytest.mark.cuda
@pytest.mark.integration
def test_gsplat_render_benchmark_runs_on_bundled_sample() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gsplat benchmark integration tests.")
    gsplat_backend = pytest.importorskip("splatkit_adapter_backends.gsplat")
    gsplat_backend.register()

    scene_record = load_colmap_scene_record(get_sample_scene_path())
    scene = initialize_gaussian_scene_from_scene_record(scene_record).to(
        torch.device("cuda")
    )
    camera = _first_camera(scene_record.resolve_camera_sensor().camera).to(
        torch.device("cuda")
    )

    result = benchmark_backend_render(
        scene,
        camera,
        backend="adapter.gsplat",
        warmup_steps=1,
        measured_steps=2,
    )

    assert result.backend == "adapter.gsplat"
    assert result.device.startswith("cuda")
    assert result.mean_ms_per_frame >= 0.0
