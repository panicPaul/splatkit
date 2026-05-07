from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ember_core.core import BACKEND_REGISTRY, render
from ember_native_nht.threedgut import (
    barycentric_weights,
    harmonic_encode,
    register,
    render_nht_3dgut,
    tetrahedron_vertices,
)

register()

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_native_nht_renderer_does_not_import_reference_gsplat() -> None:
    renderer_source = (
        REPO_ROOT
        / "packages/ember-native-nht/src/ember_native_nht/threedgut/renderer.py"
    ).read_text()

    assert "gsplat.rendering" not in renderer_source
    assert "nht_gsplat_import_path" not in renderer_source


def test_native_nht_runtime_uses_staged_vendored_backend() -> None:
    native_package_root = (
        REPO_ROOT / "packages/ember-native-nht/src/ember_native_nht/threedgut"
    )
    assert not (native_package_root / "native/cuda/_wrapper.py").exists()
    assert not (native_package_root / "native/cuda/_backend.py").exists()
    assert (
        native_package_root / "core/native/nht_rasterizer/upstream/csrc"
    ).is_dir()

    runtime_source = (
        native_package_root / "core/runtime/ops/render.py"
    ).read_text()
    assert "gsplat" not in runtime_source
    assert "native.cuda" not in runtime_source


def test_native_nht_staged_runtime_api_is_importable() -> None:
    from ember_native_nht.threedgut import runtime

    assert callable(runtime.project)
    assert callable(runtime.intersect)
    assert callable(runtime.rasterize_features)
    assert callable(runtime.rasterize_depth)
    assert callable(runtime.render)
    assert callable(runtime.rasterization_nht)


def test_tetrahedron_vertices_interpolate_themselves() -> None:
    vertices = tetrahedron_vertices(
        dtype=torch.float32, device=torch.device("cpu")
    )
    weights = barycentric_weights(vertices)

    assert torch.allclose(weights, torch.eye(4), atol=1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_harmonic_encode_matches_upstream_one_frequency_shape() -> None:
    features = torch.tensor([[0.0, torch.pi / 2.0]], dtype=torch.float32)

    encoded = harmonic_encode(features)

    assert encoded.shape == (1, 4)
    assert torch.allclose(
        encoded,
        torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        atol=1e-6,
    )


def test_render_nht_3dgut_rejects_cpu_scene_before_native_dispatch(
    cpu_scene,
    cpu_camera,
) -> None:
    scene = cpu_scene.__class__(
        center_position=cpu_scene.center_position,
        log_scales=cpu_scene.log_scales,
        quaternion_orientation=cpu_scene.quaternion_orientation,
        logit_opacity=cpu_scene.logit_opacity,
        feature=torch.zeros((3, 8)),
        sh_degree=0,
    )

    with pytest.raises(ValueError, match="requires scene tensors on CUDA"):
        render_nht_3dgut(scene, cpu_camera)


def test_render_nht_3dgut_rejects_non_vertex_feature_dim_on_cuda_inputs(
    cpu_scene,
    cpu_camera,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native NHT validation.")
    scene = cpu_scene.__class__(
        center_position=cpu_scene.center_position.cuda(),
        log_scales=cpu_scene.log_scales.cuda(),
        quaternion_orientation=cpu_scene.quaternion_orientation.cuda(),
        logit_opacity=cpu_scene.logit_opacity.cuda(),
        feature=torch.zeros((3, 5), device="cuda"),
        sh_degree=0,
    )
    camera = cpu_camera.to(torch.device("cuda"))

    with pytest.raises(ValueError, match="divisible by four"):
        render_nht_3dgut(scene, camera)


def test_generic_render_dispatches_to_nht_3dgut_registry() -> None:
    assert BACKEND_REGISTRY["nht.3dgut"].name == "nht.3dgut"


@pytest.mark.cuda
def test_render_nht_3dgut_outputs_encoded_features_and_ray_dirs(
    cpu_scene,
    cpu_camera,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native NHT rendering.")
    scene = cpu_scene.__class__(
        center_position=cpu_scene.center_position.cuda(),
        log_scales=cpu_scene.log_scales.cuda(),
        quaternion_orientation=cpu_scene.quaternion_orientation.cuda(),
        logit_opacity=cpu_scene.logit_opacity.cuda(),
        feature=torch.zeros((3, 8), device="cuda"),
        sh_degree=0,
    )
    camera = cpu_camera.to(torch.device("cuda"))

    output = render(
        scene,
        camera,
        backend="nht.3dgut",
        return_depth=True,
    )

    assert output.features.shape[-1] == 7
    assert output.alphas.shape == (1, 32, 32)
    assert output.depth.shape == (1, 32, 32)
