from typing import Any, Protocol, cast, runtime_checkable

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from splatkit.core import (
    BACKEND_REGISTRY,
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
    render,
    resolve_backend_trait,
)
from splatkit.core.registry import register_backend
from splatkit_adapter_backends.gsplat import (
    GsplatRenderOptions,
    render_gsplat,
    render_gsplat_2dgs,
)


def test_registry_contains_gsplat() -> None:
    assert "adapter.gsplat" in BACKEND_REGISTRY


def test_registry_contains_gsplat_2dgs() -> None:
    assert "adapter.gsplat_2dgs" in BACKEND_REGISTRY


def test_render_unknown_backend_raises(cpu_scene, cpu_camera) -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        render(cpu_scene, cpu_camera, backend="does-not-exist")


def test_render_rejects_unsupported_output_request(
    cpu_scene, cpu_camera
) -> None:
    original = BACKEND_REGISTRY["adapter.gsplat"]
    BACKEND_REGISTRY["adapter.gsplat"] = type(original)(
        name=original.name,
        render_fn=original.render_fn,
        default_options=original.default_options,
        accepted_scene_types=original.accepted_scene_types,
        supported_outputs=frozenset({"alpha"}),
    )
    try:
        with pytest.raises(
            ValueError, match="does not support requested outputs"
        ):
            render(
                cpu_scene,
                cpu_camera,
                backend="adapter.gsplat",
                return_2d_projections=True,
            )
    finally:
        BACKEND_REGISTRY["adapter.gsplat"] = original


def test_render_rejects_unsupported_gaussian_impact_score_request(
    cpu_scene, cpu_camera
) -> None:
    with pytest.raises(
        ValueError, match="does not support requested outputs"
    ):
        render(
            cpu_scene,
            cpu_camera,
            backend="adapter.gsplat",
            return_gaussian_impact_score=True,
        )


def test_render_beartype_rejects_wrong_scene(cpu_camera) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        render(cast(Any, "not-a-scene"), cpu_camera, backend="adapter.gsplat")


def test_render_rejects_incompatible_scene_type(
    cpu_sparse_voxel_scene, cpu_camera
) -> None:
    with pytest.raises(ValueError, match="does not accept scene type"):
        render(cpu_sparse_voxel_scene, cpu_camera, backend="adapter.gsplat")


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
        BACKEND_REGISTRY["adapter.gsplat"].default_options, GsplatRenderOptions
    )
    assert isinstance(
        BACKEND_REGISTRY["adapter.gsplat_2dgs"].default_options,
        GsplatRenderOptions,
    )


def test_render_gsplat_2dgs_beartype_rejects_wrong_options(
    cpu_scene_2d, cpu_camera
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        render_gsplat_2dgs(
            cpu_scene_2d,
            cpu_camera,
            options=RenderOptions(),  # type: ignore[arg-type]
        )


@runtime_checkable
class _HasRegistryTrait(Protocol):
    def ping(self) -> str: ...


class _RegistryTraitProvider:
    def ping(self) -> str:
        return "ok"


def test_resolve_backend_trait_returns_registered_provider() -> None:
    backend_name = "unit_test_registry_trait_backend"

    @register_backend(
        name=backend_name,
        default_options=RenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        trait_providers=(_RegistryTraitProvider(),),
    )
    def _render_unit_test_registry_trait_backend(
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
            scene,
            camera,
            return_alpha,
            return_depth,
            return_gaussian_impact_score,
            return_normals,
            return_2d_projections,
            return_projective_intersection_transforms,
            options,
        )
        raise AssertionError("render should not be called in this test")

    try:
        trait = resolve_backend_trait(backend_name, _HasRegistryTrait)
        assert trait.ping() == "ok"
    finally:
        BACKEND_REGISTRY.pop(backend_name, None)
