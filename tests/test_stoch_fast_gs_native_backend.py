from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from ember_core.core import BACKEND_REGISTRY
from ember_core.core.registry import resolve_backend_trait
from ember_core.densification import (
    DensificationContext,
    GaussianFastGSSignalProvider,
    GaussianMetricAttribution,
)
from ember_native_3dgrt.stoch_fast_gs import (
    StochFastGSMetricAttribution,
    StochFastGSNativeRenderOptions,
    StochFastGSSignalProvider,
    register,
    render_stoch_fast_gs_native,
)
from ember_splatting_training import GaussianFastGS

register()


def test_gaussian_fastgs_requests_backend_densification_collection() -> None:
    densification = GaussianFastGS(stop_iter=10)

    active = densification.get_render_requirements(SimpleNamespace(step=0))
    stopped = densification.get_render_requirements(SimpleNamespace(step=9))

    assert active.backend_options == {"collect_densification_info": True}
    assert stopped.backend_options == {"collect_densification_info": False}


def test_stoch_fast_gs_backend_registers_densification_traits() -> None:
    backend = BACKEND_REGISTRY["3dgrt.stoch_fast_gs"]

    assert isinstance(backend.default_options, StochFastGSNativeRenderOptions)
    assert backend.default_options.collect_densification_info is False
    assert isinstance(
        resolve_backend_trait(
            "3dgrt.stoch_fast_gs",
            GaussianFastGSSignalProvider,
        ),
        StochFastGSSignalProvider,
    )
    assert isinstance(
        resolve_backend_trait(
            "3dgrt.stoch_fast_gs",
            GaussianMetricAttribution,
        ),
        StochFastGSMetricAttribution,
    )


def test_stoch_fast_gs_signal_provider_collects_cpu_signals(
    cpu_scene,
    cpu_camera,
) -> None:
    provider = StochFastGSSignalProvider()
    cpu_scene.center_position.requires_grad_(True)
    cpu_scene.center_position.grad = torch.ones_like(cpu_scene.center_position)
    render_output = SimpleNamespace(
        weights=torch.tensor([[0.25], [0.0], [0.5]], dtype=torch.float32),
        visibility=torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32),
    )
    context = DensificationContext(
        state=SimpleNamespace(model=SimpleNamespace(scene=cpu_scene)),
        batch=SimpleNamespace(camera=cpu_camera),
        render_output=render_output,
        loss_result=SimpleNamespace(),
        step=0,
        optimizers=(),
    )

    signals = provider.collect_fastgs_signals(context)

    assert signals is not None
    assert signals.visible_count.tolist() == [1.0, 0.0, 1.0]
    assert signals.clone_grad_sum.shape == (3,)
    assert signals.split_grad_sum.shape == (3,)
    assert signals.max_screen_radii.shape == (3,)
    assert torch.all(signals.clone_grad_sum[signals.visible_count == 0] == 0)


def test_stoch_fast_gs_metric_attribution_uses_metric_weight_helper(
    cpu_scene,
    cpu_camera,
    monkeypatch,
) -> None:
    from ember_native_3dgrt.stoch_fast_gs import renderer

    captured_metric_maps: list[torch.Tensor] = []

    def fake_metric_weights(
        *args,
        **_kwargs,
    ) -> torch.Tensor:
        metric_map = args[2]
        captured_metric_maps.append(metric_map.clone())
        return torch.tensor([0.5, 0.25, 0.0], dtype=torch.float32)

    monkeypatch.setattr(
        renderer,
        "stoch_fast_gs_metric_weights",
        fake_metric_weights,
    )
    metric_map = torch.tensor([[1, 0], [1, 0]], dtype=torch.int32)

    attribution = StochFastGSMetricAttribution().attribute_metric_map(
        cpu_scene,
        cpu_camera,
        metric_map,
    )

    assert torch.allclose(
        attribution,
        torch.tensor([0.5, 0.25, 0.0], dtype=torch.float32),
    )
    assert torch.equal(captured_metric_maps[0], metric_map)


@pytest.mark.cuda
def test_stoch_fast_gs_metric_attribution_all_one_matches_render_weights(
    cuda_visible_scene,
    cuda_camera,
) -> None:
    output = render_stoch_fast_gs_native(cuda_visible_scene, cuda_camera)
    metric_map = torch.ones(
        (
            int(cuda_camera.height[0].item()),
            int(cuda_camera.width[0].item()),
        ),
        device=cuda_visible_scene.center_position.device,
        dtype=torch.int32,
    )

    attribution = StochFastGSMetricAttribution().attribute_metric_map(
        cuda_visible_scene,
        cuda_camera,
        metric_map,
    )

    torch.testing.assert_close(
        attribution,
        output.weights.reshape(-1).to(dtype=attribution.dtype),
    )


@pytest.mark.cuda
def test_stoch_fast_gs_metric_attribution_rejects_zero_metric_pixels(
    cuda_visible_scene,
    cuda_camera,
) -> None:
    metric_map = torch.zeros(
        (
            int(cuda_camera.height[0].item()),
            int(cuda_camera.width[0].item()),
        ),
        device=cuda_visible_scene.center_position.device,
        dtype=torch.int32,
    )

    attribution = StochFastGSMetricAttribution().attribute_metric_map(
        cuda_visible_scene,
        cuda_camera,
        metric_map,
    )

    torch.testing.assert_close(attribution, torch.zeros_like(attribution))
