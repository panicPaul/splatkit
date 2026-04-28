from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from ember_adapter_backends.stoch3dgs import (
    Stoch3DGSAlphaRenderOutput,
    Stoch3DGSRenderOptions,
    Stoch3DGSRenderOutput,
    register,
    render_stoch3dgs,
)
from ember_adapter_backends.stoch3dgs.renderer import _flatten_sh_features
from ember_core.core import BACKEND_REGISTRY, RenderOptions, render

register()


class _FakeTracer:
    def __init__(self) -> None:
        self.build_calls = 0

    def build_acc(self, _scene: Any, rebuild: bool = True) -> None:
        assert rebuild is True
        self.build_calls += 1

    def render(
        self, scene: Any, batch: Any, train: bool = False
    ) -> dict[str, Any]:
        pred_rgb = torch.full_like(batch.rays_dir, 0.25)
        pred_opacity = torch.full(
            (*batch.rays_dir.shape[:-1], 1),
            0.5,
            device=batch.rays_dir.device,
            dtype=batch.rays_dir.dtype,
        )
        pred_rgb, pred_opacity = scene.background(
            batch.T_to_world,
            batch.rays_dir,
            pred_rgb,
            pred_opacity,
            train,
        )
        features = scene.get_features()
        assert features.shape[0] == scene.positions.shape[0]
        pred_dist = torch.full_like(pred_opacity, 3.0)
        return {
            "pred_rgb": pred_rgb,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
        }


@pytest.mark.backend
@pytest.mark.cuda
def test_render_stoch3dgs_rgb_alpha_shapes(
    cuda_scene, cuda_camera, monkeypatch
) -> None:
    fake_tracer = _FakeTracer()
    monkeypatch.setattr(
        "ember_adapter_backends.stoch3dgs.renderer._get_tracer",
        lambda scene, options: fake_tracer,
    )
    monkeypatch.setattr(
        "ember_adapter_backends.stoch3dgs.renderer._import_stoch_runtime",
        lambda: (object, _FakeBatch),
    )

    output = cast(
        Stoch3DGSAlphaRenderOutput,
        render_stoch3dgs(cuda_scene, cuda_camera),
    )

    assert output.render.shape == (1, 32, 32, 3)
    assert output.alphas.shape == (1, 32, 32)
    assert torch.allclose(
        output.render[0, 0, 0],
        torch.tensor([0.25, 0.25, 0.25], device=output.render.device),
    )
    assert torch.allclose(
        output.alphas, torch.full((1, 32, 32), 0.5, device=output.alphas.device)
    )
    assert fake_tracer.build_calls == 1


@pytest.mark.backend
@pytest.mark.cuda
def test_render_stoch3dgs_depth_and_background(
    cuda_scene, cuda_camera, monkeypatch
) -> None:
    monkeypatch.setattr(
        "ember_adapter_backends.stoch3dgs.renderer._get_tracer",
        lambda scene, options: _FakeTracer(),
    )
    monkeypatch.setattr(
        "ember_adapter_backends.stoch3dgs.renderer._import_stoch_runtime",
        lambda: (object, _FakeBatch),
    )

    output = cast(
        Stoch3DGSRenderOutput,
        render_stoch3dgs(
            cuda_scene,
            cuda_camera,
            return_depth=True,
            options=Stoch3DGSRenderOptions(
                background_color=torch.ones(3, device="cuda")
            ),
        ),
    )

    assert output.depth.shape == (1, 32, 32)
    assert torch.allclose(
        output.depth, torch.full((1, 32, 32), 3.0, device=output.depth.device)
    )
    assert torch.allclose(
        output.render[0, 0, 0],
        torch.tensor([0.75, 0.75, 0.75], device=output.render.device),
    )


def test_render_stoch3dgs_rejects_cpu_scene(cpu_scene, cpu_camera) -> None:
    with pytest.raises(ValueError, match="scene tensors on CUDA"):
        render_stoch3dgs(cpu_scene, cpu_camera)


def test_render_stoch3dgs_rejects_non_sh_features(
    cpu_scene, cpu_camera
) -> None:
    bad_scene = type(cpu_scene)(
        center_position=cpu_scene.center_position,
        log_scales=cpu_scene.log_scales,
        quaternion_orientation=cpu_scene.quaternion_orientation,
        logit_opacity=cpu_scene.logit_opacity,
        feature=torch.zeros((3, 3), dtype=torch.float32),
        sh_degree=0,
    )
    if torch.cuda.is_available():
        bad_scene = bad_scene.to(torch.device("cuda"))
        cpu_camera = cpu_camera.to(torch.device("cuda"))
    with pytest.raises(ValueError, match="spherical harmonics"):
        render_stoch3dgs(bad_scene, cpu_camera)


def test_render_stoch3dgs_rejects_2d_projections(cpu_scene, cpu_camera) -> None:
    with pytest.raises(ValueError, match="2D Gaussian projections"):
        render_stoch3dgs(
            cpu_scene,
            cpu_camera,
            return_2d_projections=True,
        )


def test_render_stoch3dgs_beartype_rejects_wrong_options(
    cpu_scene, cpu_camera
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        render_stoch3dgs(
            cpu_scene,
            cpu_camera,
            options=RenderOptions(),  # type: ignore[arg-type]
        )


def test_registry_contains_stoch3dgs() -> None:
    assert "adapter.stoch3dgs" in BACKEND_REGISTRY
    assert isinstance(
        BACKEND_REGISTRY["adapter.stoch3dgs"].default_options,
        Stoch3DGSRenderOptions,
    )


class _FakeBatch:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


@pytest.mark.backend
@pytest.mark.cuda
def test_generic_render_stoch3dgs_returns_depth(
    cuda_scene, cuda_camera, monkeypatch
) -> None:
    monkeypatch.setattr(
        "ember_adapter_backends.stoch3dgs.renderer._get_tracer",
        lambda scene, options: _FakeTracer(),
    )
    monkeypatch.setattr(
        "ember_adapter_backends.stoch3dgs.renderer._import_stoch_runtime",
        lambda: (object, _FakeBatch),
    )

    output = cast(
        Stoch3DGSRenderOutput,
        render(
            cuda_scene,
            cuda_camera,
            backend="adapter.stoch3dgs",
            return_depth=True,
        ),
    )
    assert output.depth.shape == (1, 32, 32)


@pytest.mark.backend
@pytest.mark.cuda
def test_render_stoch3dgs_flattens_sh_features(cuda_scene) -> None:
    feature = torch.tensor(
        [
            [
                [0.9, 0.2, 0.2],
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
            [
                [0.1, 0.2, 0.3],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            [
                [0.4, 0.5, 0.6],
                [12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0],
            ],
        ],
        device=cuda_scene.feature.device,
        dtype=cuda_scene.feature.dtype,
    )
    scene = type(cuda_scene)(
        center_position=cuda_scene.center_position,
        log_scales=cuda_scene.log_scales,
        quaternion_orientation=cuda_scene.quaternion_orientation,
        logit_opacity=cuda_scene.logit_opacity,
        feature=feature,
        sh_degree=2,
    )
    features = _flatten_sh_features(scene)
    assert torch.allclose(
        features[0, :9],
        torch.tensor(
            [0.9, 0.2, 0.2, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            device=features.device,
        ),
    )
