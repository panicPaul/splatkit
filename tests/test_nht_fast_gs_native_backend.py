from __future__ import annotations

from types import SimpleNamespace

import torch
from ember_core.core import BACKEND_REGISTRY
from ember_core.core.registry import resolve_backend_trait
from ember_core.densification import (
    DensificationContext,
    GaussianFastGSSignalProvider,
    GaussianMetricAttribution,
)
from ember_native_nht.threedgut_fast_gs import (
    NHTFastGSMetricAttribution,
    NHTFastGSRenderOptions,
    NHTFastGSSignalProvider,
    nht_fast_gs_metric_counts,
    register,
)

register()


def test_nht_fast_gs_backend_registers_densification_traits() -> None:
    backend = BACKEND_REGISTRY["nht.3dgut_fast_gs"]

    assert isinstance(backend.default_options, NHTFastGSRenderOptions)
    assert backend.default_options.collect_densification_info is False
    assert isinstance(
        resolve_backend_trait(
            "nht.3dgut_fast_gs",
            GaussianFastGSSignalProvider,
        ),
        NHTFastGSSignalProvider,
    )
    assert isinstance(
        resolve_backend_trait(
            "nht.3dgut_fast_gs",
            GaussianMetricAttribution,
        ),
        NHTFastGSMetricAttribution,
    )


def test_nht_fast_gs_signal_provider_collects_cpu_signals(
    cpu_scene,
    cpu_camera,
) -> None:
    provider = NHTFastGSSignalProvider()
    cpu_scene.center_position.requires_grad_(True)
    cpu_scene.center_position.grad = torch.ones_like(cpu_scene.center_position)
    raw_output = SimpleNamespace(
        weights=torch.tensor([[0.25], [0.0], [0.5]], dtype=torch.float32),
        visibility=torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32),
        radii=torch.tensor(
            [[[4, 3], [0, 0], [2, 5]]],
            dtype=torch.int32,
        ),
    )
    context = DensificationContext(
        state=SimpleNamespace(model=SimpleNamespace(scene=cpu_scene)),
        batch=SimpleNamespace(camera=cpu_camera),
        render_output=SimpleNamespace(raw_output=raw_output),
        loss_result=SimpleNamespace(),
        step=0,
        optimizers=(),
    )

    signals = provider.collect_fastgs_signals(context)

    assert signals is not None
    assert signals.visible_count.tolist() == [1.0, 0.0, 1.0]
    assert signals.max_screen_radii.tolist() == [4.0, 0.0, 5.0]
    assert torch.all(signals.clone_grad_sum[signals.visible_count == 0] == 0)


def test_nht_fast_gs_metric_counts_filters_native_contributors(
    monkeypatch,
) -> None:
    from ember_native_nht.threedgut_fast_gs import renderer

    monkeypatch.setattr(
        renderer,
        "rasterize_gaussian_indices",
        lambda **_kwargs: (
            torch.tensor([0, 1, 1, 2], dtype=torch.int64),
            torch.tensor([0, 0, 2, 3], dtype=torch.int64),
        ),
    )
    output = SimpleNamespace(
        render=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        projected_means=torch.zeros((1, 3, 2), dtype=torch.float32),
        conics=torch.zeros((1, 3, 3), dtype=torch.float32),
        tile_offsets=torch.zeros((1, 1, 1), dtype=torch.int32),
        flattened_gaussian_ids=torch.zeros((0,), dtype=torch.int32),
        mip_splatting_screen_filter_compensations=None,
    )
    metric_map = torch.tensor([[1, 0], [1, 0]], dtype=torch.int32)

    counts = nht_fast_gs_metric_counts(
        output=output,
        opacities=torch.ones((3,), dtype=torch.float32),
        metric_map=metric_map,
        tile_size=16,
    )

    assert torch.allclose(counts, torch.tensor([1.0, 2.0, 0.0]))


def test_nht_fast_gs_metric_attribution_uses_exact_metric_counts(
    cpu_scene,
    cpu_camera,
    monkeypatch,
) -> None:
    from ember_native_nht.threedgut_fast_gs import renderer

    monkeypatch.setattr(
        renderer,
        "render_nht_fast_gs",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        renderer,
        "nht_fast_gs_metric_counts",
        lambda **_kwargs: torch.tensor([2.0, 0.0, 1.0]),
    )

    attribution = NHTFastGSMetricAttribution().attribute_metric_map(
        cpu_scene,
        cpu_camera,
        torch.ones((2, 2), dtype=torch.int32),
    )

    assert torch.allclose(attribution, torch.tensor([2.0, 0.0, 1.0]))
