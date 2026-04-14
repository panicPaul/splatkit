from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from splatkit.core import (
    BACKEND_REGISTRY,
    RenderOptions,
    render,
)
from splatkit_backends.gsplat import (
    GsplatRenderOptions,
    render_gsplat,
)


def test_registry_contains_gsplat() -> None:
    assert "gsplat" in BACKEND_REGISTRY


def test_render_unknown_backend_raises(cpu_scene, cpu_camera) -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        render(cpu_scene, cpu_camera, backend="does-not-exist")


def test_render_rejects_unsupported_output_request(
    cpu_scene, cpu_camera
) -> None:
    original = BACKEND_REGISTRY["gsplat"]
    BACKEND_REGISTRY["gsplat"] = type(original)(
        name=original.name,
        render_fn=original.render_fn,
        default_options=original.default_options,
        supported_outputs=frozenset({"alpha"}),
    )
    try:
        with pytest.raises(
            ValueError, match="does not support requested outputs"
        ):
            render(
                cpu_scene,
                cpu_camera,
                backend="gsplat",
                return_2d_projections=True,
            )
    finally:
        BACKEND_REGISTRY["gsplat"] = original


def test_render_beartype_rejects_wrong_scene(cpu_camera) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        render(cast(Any, "not-a-scene"), cpu_camera, backend="gsplat")


def test_render_gsplat_beartype_rejects_wrong_options(
    cpu_scene, cpu_camera
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        render_gsplat(
            cpu_scene,
            cpu_camera,
            options=RenderOptions(),  # type: ignore[arg-type]
        )


def test_render_default_options_type() -> None:
    assert isinstance(
        BACKEND_REGISTRY["gsplat"].default_options, GsplatRenderOptions
    )
