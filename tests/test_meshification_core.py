from __future__ import annotations

import torch
from ember_core.core import (
    BACKEND_REGISTRY,
    CameraState,
    GaussianScene2D,
    RenderOptions,
    RenderOutput,
)
from ember_core.core.registry import register_backend
from ember_core.meshification import (
    MeshificationRequest,
    SurfacePointSamples,
    WrappingQueryResult,
    WrappingSurfaceEvidence,
    meshify,
)


class _FakeWrappingProvider:
    def surface_evidence(
        self,
        request: MeshificationRequest,
    ) -> WrappingSurfaceEvidence:
        height = int(request.camera.height[0].item())
        width = int(request.camera.width[0].item())
        render = torch.zeros(
            (1, height, width, 3),
            dtype=torch.float32,
            device=request.camera.cam_to_world.device,
        )
        return WrappingSurfaceEvidence(render=render)

    def sample_surface_points(
        self,
        request: MeshificationRequest,
    ) -> SurfacePointSamples:
        del request
        return SurfacePointSamples(
            points=torch.tensor(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            scales=None,
        )

    def query_wrapping_field(
        self,
        request: MeshificationRequest,
        points: torch.Tensor,
    ) -> WrappingQueryResult:
        del request
        return WrappingQueryResult(
            values=torch.ones(points.shape[0], dtype=points.dtype),
            inside=torch.ones(points.shape[0], dtype=torch.bool),
        )


def test_meshification_uses_wrapping_trait_without_gaussian_3d(
    cpu_scene_2d: GaussianScene2D,
    cpu_camera: CameraState,
) -> None:
    backend_name = "unit_test_wrapping_2dgs_backend"

    @register_backend(
        name=backend_name,
        default_options=RenderOptions(),
        accepted_scene_types=(GaussianScene2D,),
        trait_providers=(_FakeWrappingProvider(),),
    )
    def _render_unit_test_wrapping_2dgs_backend(
        scene: GaussianScene2D,
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
        raise AssertionError("render should not be called")

    try:
        result = meshify(
            cpu_scene_2d,
            cpu_camera,
            backend=backend_name,
            meshifier="wrapping",
        )
        assert result.mesh.vertices.shape == (8, 3)
        assert result.mesh.faces.shape == (12, 3)
        assert result.diagnostics["meshifier"] == "wrapping"
    finally:
        BACKEND_REGISTRY.pop(backend_name, None)
