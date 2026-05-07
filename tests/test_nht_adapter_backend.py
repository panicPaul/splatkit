from __future__ import annotations

from typing import Any

import pytest
import torch
from ember_adapter_backends.nht import register, render_nht_adapter
from ember_adapter_backends.nht.renderer import NHTAdapterRenderOptions
from ember_core.core import BACKEND_REGISTRY
from torch import Tensor

register()


def _nht_scene(cpu_scene):
    return cpu_scene.__class__(
        center_position=cpu_scene.center_position,
        log_scales=cpu_scene.log_scales,
        quaternion_orientation=cpu_scene.quaternion_orientation,
        logit_opacity=cpu_scene.logit_opacity,
        feature=torch.zeros((3, 8), dtype=torch.float32),
        sh_degree=0,
    )


def test_adapter_nht_registers_backend() -> None:
    assert BACKEND_REGISTRY["adapter.nht"].name == "adapter.nht"


def test_adapter_nht_rejects_cpu_scene_before_reference_dispatch(
    cpu_scene,
    cpu_camera,
) -> None:
    with pytest.raises(ValueError, match="requires scene tensors on CUDA"):
        render_nht_adapter(_nht_scene(cpu_scene), cpu_camera)


def test_adapter_nht_calls_reference_rasterization_contract(
    monkeypatch,
    cpu_scene,
    cpu_camera,
) -> None:
    import ember_adapter_backends.nht.renderer as nht_renderer

    calls: list[dict[str, Any]] = []

    def fake_rasterization(**kwargs):
        calls.append(kwargs)
        colors = torch.zeros((1, 32, 32, 9), dtype=torch.float32)
        alphas = torch.ones((1, 32, 32, 1), dtype=torch.float32)
        return colors, alphas, {"tiles_per_gauss": torch.ones((1, 3))}

    monkeypatch.setattr(nht_renderer, "_validate_inputs", lambda *_: None)
    monkeypatch.setattr(
        nht_renderer,
        "_import_nht_rasterization",
        lambda: fake_rasterization,
    )

    output = render_nht_adapter(
        _nht_scene(cpu_scene),
        cpu_camera,
        return_depth=True,
        options=NHTAdapterRenderOptions(ray_dir_scale=4.0),
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["nht"] is True
    assert call["with_eval3d"] is True
    assert call["with_ut"] is True
    assert call["sh_degree"] is None
    assert call["render_mode"] == "RGB+ED"
    assert call["ray_dir_scale"] == 4.0
    assert isinstance(call["colors"], Tensor)
    assert call["colors"].shape == (3, 8)
    assert output.features.shape == (1, 32, 32, 8)
    assert output.alphas.shape == (1, 32, 32)
    assert output.depth.shape == (1, 32, 32)
