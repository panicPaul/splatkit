from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    repo_root = _repo_root()
    pythonpath_parts = [
        str(repo_root / "packages" / "ember-native-nht" / "src"),
        str(repo_root / "packages" / "ember-core" / "src"),
        str(repo_root / "third_party" / "neural-harmonic-textures" / "gsplat"),
    ]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _run_reference_check(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_repo_root(),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 17:
        pytest.skip(
            "upstream NHT gsplat reference is not available in this environment"
        )
    if completed.returncode != 0:
        raise AssertionError(
            "NHT reference parity check failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _reference_script(test_body: str) -> str:
    return _COMMON_SCRIPT + dedent(test_body)


_COMMON_SCRIPT = r"""
import sys

import torch
import torch.nn.functional as F

try:
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut as reference_project,
        isect_offset_encode as reference_intersect_offsets,
        isect_tiles as reference_intersect,
        rasterize_to_pixels_eval3d as reference_rasterize_depth,
        rasterize_to_pixels_nht_eval3d as reference_rasterize_features,
    )
    from gsplat.rendering import rasterization as reference_render
except Exception:
    sys.exit(17)

from ember_native_nht.threedgut.core.runtime import (
    intersect as native_intersect,
    project as native_project,
    rasterize_depth as native_rasterize_depth,
    rasterize_features as native_rasterize_features,
    render as native_render,
)
from ember_native_nht.threedgut.runtime import rasterization_nht


if not torch.cuda.is_available():
    sys.exit(17)


def make_inputs(feature_dim=16, *, requires_grad=False):
    torch.manual_seed(20260507)
    device = torch.device("cuda")
    center_positions = torch.tensor(
        [
            [-0.18, -0.10, 0.00],
            [0.14, -0.08, 0.04],
            [0.03, 0.16, -0.02],
            [0.26, 0.18, 0.08],
            [-0.28, 0.12, 0.05],
        ],
        dtype=torch.float32,
        device=device,
    )
    quaternions = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.97, 0.05, 0.12, 0.01],
                [0.94, -0.04, 0.02, 0.20],
                [0.91, 0.18, -0.03, 0.08],
                [0.88, -0.10, 0.11, -0.06],
            ],
            dtype=torch.float32,
            device=device,
        ),
        dim=-1,
    )
    scales = torch.exp(
        torch.tensor(
            [
                [-1.25, -1.35, -1.45],
                [-1.30, -1.15, -1.40],
                [-1.45, -1.25, -1.20],
                [-1.55, -1.35, -1.30],
                [-1.20, -1.50, -1.40],
            ],
            dtype=torch.float32,
            device=device,
        )
    )
    opacities = torch.sigmoid(
        torch.tensor([1.8, 1.4, 1.0, 0.7, 1.2], dtype=torch.float32, device=device)
    )
    features = torch.linspace(
        -0.75,
        0.85,
        steps=center_positions.shape[0] * feature_dim,
        dtype=torch.float32,
        device=device,
    ).reshape(center_positions.shape[0], feature_dim)
    camera_to_world = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    camera_to_world[:, 2, 3] = 3.0
    world_to_camera_matrices = torch.linalg.inv(camera_to_world).contiguous()
    camera_intrinsics = torch.tensor(
        [[[42.0, 0.0, 16.0], [0.0, 43.0, 12.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    if requires_grad:
        center_positions = center_positions.detach().clone().requires_grad_(True)
        quaternions = quaternions.detach().clone().requires_grad_(True)
        scales = scales.detach().clone().requires_grad_(True)
        opacities = opacities.detach().clone().requires_grad_(True)
        features = features.detach().clone().requires_grad_(True)
    return {
        "center_positions": center_positions,
        "quaternions": quaternions,
        "scales": scales,
        "opacities": opacities,
        "features": features,
        "world_to_camera_matrices": world_to_camera_matrices,
        "camera_intrinsics": camera_intrinsics,
        "image_width": 32,
        "image_height": 24,
        "tile_size": 16,
        "eps2d": 0.3,
        "near_plane": 0.01,
        "far_plane": 1.0e10,
        "radius_clip": 0.0,
        "camera_model": "pinhole",
    }


def project_inputs(inputs):
    return {
        "center_positions": inputs["center_positions"],
        "quaternions": inputs["quaternions"],
        "scales": inputs["scales"],
        "opacities": inputs["opacities"],
        "world_to_camera_matrices": inputs["world_to_camera_matrices"],
        "camera_intrinsics": inputs["camera_intrinsics"],
        "image_width": inputs["image_width"],
        "image_height": inputs["image_height"],
        "eps2d": inputs["eps2d"],
        "near_plane": inputs["near_plane"],
        "far_plane": inputs["far_plane"],
        "radius_clip": inputs["radius_clip"],
        "calculate_compensations": False,
        "camera_model": inputs["camera_model"],
    }


def reference_projection(inputs):
    return reference_project(
        means=inputs["center_positions"],
        quats=inputs["quaternions"],
        scales=inputs["scales"],
        opacities=inputs["opacities"],
        viewmats=inputs["world_to_camera_matrices"],
        Ks=inputs["camera_intrinsics"],
        width=inputs["image_width"],
        height=inputs["image_height"],
        eps2d=inputs["eps2d"],
        near_plane=inputs["near_plane"],
        far_plane=inputs["far_plane"],
        radius_clip=inputs["radius_clip"],
        calc_compensations=False,
        camera_model=inputs["camera_model"],
    )


def native_projection(inputs):
    return native_project(**project_inputs(inputs))


def intersections_from_projection(inputs, projection):
    num_cameras = int(inputs["world_to_camera_matrices"].shape[0])
    return native_intersect(
        projected_means=projection.projected_means,
        radii=projection.radii,
        primitive_depths=projection.primitive_depths,
        num_cameras=num_cameras,
        image_width=inputs["image_width"],
        image_height=inputs["image_height"],
        tile_size=inputs["tile_size"],
    )


def native_intersections_from_reference_projection(inputs, projection_tuple):
    radii, projected_means, primitive_depths, _, _ = projection_tuple
    num_cameras = int(inputs["world_to_camera_matrices"].shape[0])
    return native_intersect(
        projected_means=projected_means,
        radii=radii,
        primitive_depths=primitive_depths,
        num_cameras=num_cameras,
        image_width=inputs["image_width"],
        image_height=inputs["image_height"],
        tile_size=inputs["tile_size"],
    )


def reference_intersections_from_projection(inputs, projection_tuple):
    radii, projected_means, primitive_depths, _, _ = projection_tuple
    tile_width = (inputs["image_width"] + inputs["tile_size"] - 1) // inputs["tile_size"]
    tile_height = (inputs["image_height"] + inputs["tile_size"] - 1) // inputs["tile_size"]
    tiles_per_gaussian, tile_intersection_ids, flattened_gaussian_ids = reference_intersect(
        means2d=projected_means,
        radii=radii,
        depths=primitive_depths,
        tile_size=inputs["tile_size"],
        tile_width=tile_width,
        tile_height=tile_height,
        packed=False,
        n_images=int(inputs["world_to_camera_matrices"].shape[0]),
    )
    tile_offsets = reference_intersect_offsets(
        tile_intersection_ids,
        int(inputs["world_to_camera_matrices"].shape[0]),
        tile_width,
        tile_height,
    )
    return tiles_per_gaussian, tile_intersection_ids, flattened_gaussian_ids, tile_offsets


def broadcast_features(features, num_cameras):
    return features.unsqueeze(0).expand(num_cameras, -1, -1).contiguous()


def broadcast_opacities(opacities, num_cameras):
    return opacities.unsqueeze(0).expand(num_cameras, -1).contiguous()
"""


@pytest.mark.cuda
def test_nht_projection_and_intersection_match_upstream() -> None:
    _run_reference_check(
        _reference_script(
            r"""
            inputs = make_inputs(feature_dim=16)

            native_projection_result = native_projection(inputs)
            reference_projection_result = reference_projection(inputs)
            reference_radii, reference_projected_means, reference_depths, reference_conics, reference_compensations = reference_projection_result
            assert torch.max(torch.abs(native_projection_result.radii - reference_radii)) <= 1
            visible_projection_mask = torch.logical_or(
                native_projection_result.radii > 0,
                reference_radii > 0,
            ).any(dim=-1)
            torch.testing.assert_close(
                native_projection_result.projected_means[visible_projection_mask],
                reference_projected_means[visible_projection_mask],
                rtol=1e-4,
                atol=5e-4,
            )
            torch.testing.assert_close(
                native_projection_result.primitive_depths[visible_projection_mask],
                reference_depths[visible_projection_mask],
                rtol=1e-4,
                atol=5e-4,
            )
            torch.testing.assert_close(
                native_projection_result.conics[visible_projection_mask],
                reference_conics[visible_projection_mask],
                rtol=1e-4,
                atol=5e-4,
            )
            assert native_projection_result.compensations is reference_compensations

            native_intersection_result = native_intersections_from_reference_projection(
                inputs,
                reference_projection_result,
            )
            reference_intersection_result = reference_intersections_from_projection(
                inputs,
                reference_projection_result,
            )
            for native_tensor, reference_tensor in zip(
                native_intersection_result.as_tensors(),
                reference_intersection_result,
                strict=True,
            ):
                torch.testing.assert_close(native_tensor, reference_tensor, rtol=0, atol=0)
            """
        )
    )


@pytest.mark.cuda
@pytest.mark.parametrize("feature_dim", [8, 16, 20])
def test_nht_feature_rasterization_forward_matches_upstream(
    feature_dim: int,
) -> None:
    _run_reference_check(
        _reference_script(
            f"""
            inputs = make_inputs(feature_dim={feature_dim})
            projection = native_projection(inputs)
            intersections = intersections_from_projection(inputs, projection)
            num_cameras = int(inputs["world_to_camera_matrices"].shape[0])
            tiled_features = broadcast_features(inputs["features"], num_cameras)
            tiled_opacities = broadcast_opacities(inputs["opacities"], num_cameras)

            native_result = native_rasterize_features(
                center_positions=inputs["center_positions"],
                quaternions=inputs["quaternions"],
                scales=inputs["scales"],
                features=tiled_features,
                opacities=tiled_opacities,
                world_to_camera_matrices=inputs["world_to_camera_matrices"],
                camera_intrinsics=inputs["camera_intrinsics"],
                image_width=inputs["image_width"],
                image_height=inputs["image_height"],
                tile_size=inputs["tile_size"],
                tile_offsets=intersections.tile_offsets,
                flattened_gaussian_ids=intersections.flattened_gaussian_ids,
                camera_model=inputs["camera_model"],
                center_ray_mode=False,
                ray_direction_scale=3.0,
            )
            reference_features, reference_alphas = reference_rasterize_features(
                means=inputs["center_positions"],
                quats=inputs["quaternions"],
                scales=inputs["scales"],
                colors=tiled_features,
                opacities=tiled_opacities,
                viewmats=inputs["world_to_camera_matrices"],
                Ks=inputs["camera_intrinsics"],
                image_width=inputs["image_width"],
                image_height=inputs["image_height"],
                tile_size=inputs["tile_size"],
                isect_offsets=intersections.tile_offsets,
                flatten_ids=intersections.flattened_gaussian_ids,
                camera_model=inputs["camera_model"],
                center_ray_mode=False,
                ray_dir_scale=3.0,
            )
            torch.testing.assert_close(native_result.features, reference_features, rtol=1e-4, atol=5e-4)
            torch.testing.assert_close(native_result.alphas, reference_alphas, rtol=1e-4, atol=5e-4)
            """
        )
    )


@pytest.mark.cuda
def test_nht_depth_rasterization_forward_matches_upstream() -> None:
    _run_reference_check(
        _reference_script(
            r"""
            inputs = make_inputs(feature_dim=16)
            projection = native_projection(inputs)
            intersections = intersections_from_projection(inputs, projection)
            num_cameras = int(inputs["world_to_camera_matrices"].shape[0])
            tiled_opacities = broadcast_opacities(inputs["opacities"], num_cameras)
            depth_features = projection.primitive_depths[..., None]

            native_result = native_rasterize_depth(
                center_positions=inputs["center_positions"],
                quaternions=inputs["quaternions"],
                scales=inputs["scales"],
                depth_features=depth_features,
                opacities=tiled_opacities,
                world_to_camera_matrices=inputs["world_to_camera_matrices"],
                camera_intrinsics=inputs["camera_intrinsics"],
                image_width=inputs["image_width"],
                image_height=inputs["image_height"],
                tile_size=inputs["tile_size"],
                tile_offsets=intersections.tile_offsets,
                flattened_gaussian_ids=intersections.flattened_gaussian_ids,
                camera_model=inputs["camera_model"],
            )
            reference_depths, reference_alphas = reference_rasterize_depth(
                means=inputs["center_positions"],
                quats=inputs["quaternions"],
                scales=inputs["scales"],
                colors=depth_features,
                opacities=tiled_opacities,
                viewmats=inputs["world_to_camera_matrices"],
                Ks=inputs["camera_intrinsics"],
                image_width=inputs["image_width"],
                image_height=inputs["image_height"],
                tile_size=inputs["tile_size"],
                isect_offsets=intersections.tile_offsets,
                flatten_ids=intersections.flattened_gaussian_ids,
                camera_model=inputs["camera_model"],
            )
            torch.testing.assert_close(native_result.depths, reference_depths, rtol=1e-4, atol=5e-4)
            torch.testing.assert_close(native_result.alphas, reference_alphas, rtol=1e-4, atol=5e-4)
            """
        )
    )


@pytest.mark.cuda
def test_nht_composed_render_forward_matches_upstream() -> None:
    _run_reference_check(
        _reference_script(
            r"""
            inputs = make_inputs(feature_dim=16)
            native_result = native_render(
                center_positions=inputs["center_positions"],
                quaternions=inputs["quaternions"],
                scales=inputs["scales"],
                opacities=inputs["opacities"],
                features=inputs["features"],
                world_to_camera_matrices=inputs["world_to_camera_matrices"],
                camera_intrinsics=inputs["camera_intrinsics"],
                image_width=inputs["image_width"],
                image_height=inputs["image_height"],
                tile_size=inputs["tile_size"],
                render_mode="RGB+ED",
                camera_model=inputs["camera_model"],
                center_ray_mode=False,
                ray_direction_scale=3.0,
            )
            compatibility_renders, compatibility_alphas, _ = rasterization_nht(
                means=inputs["center_positions"],
                quats=inputs["quaternions"],
                scales=inputs["scales"],
                opacities=inputs["opacities"],
                colors=inputs["features"],
                viewmats=inputs["world_to_camera_matrices"],
                Ks=inputs["camera_intrinsics"],
                width=inputs["image_width"],
                height=inputs["image_height"],
                tile_size=inputs["tile_size"],
                render_mode="RGB+ED",
                camera_model=inputs["camera_model"],
                center_ray_mode=False,
                ray_dir_scale=3.0,
            )
            reference_renders, reference_alphas, _ = reference_render(
                means=inputs["center_positions"],
                quats=inputs["quaternions"],
                scales=inputs["scales"],
                opacities=inputs["opacities"],
                colors=inputs["features"],
                viewmats=inputs["world_to_camera_matrices"],
                Ks=inputs["camera_intrinsics"],
                width=inputs["image_width"],
                height=inputs["image_height"],
                tile_size=inputs["tile_size"],
                packed=False,
                sparse_grad=False,
                rasterize_mode="classic",
                render_mode="RGB+ED",
                sh_degree=None,
                near_plane=inputs["near_plane"],
                far_plane=inputs["far_plane"],
                radius_clip=inputs["radius_clip"],
                eps2d=inputs["eps2d"],
                camera_model=inputs["camera_model"],
                with_ut=True,
                with_eval3d=True,
                nht=True,
                center_ray_mode=False,
                ray_dir_scale=3.0,
            )
            torch.testing.assert_close(native_result.renders, reference_renders, rtol=1e-4, atol=5e-4)
            torch.testing.assert_close(native_result.alphas, reference_alphas, rtol=1e-4, atol=5e-4)
            torch.testing.assert_close(compatibility_renders, reference_renders, rtol=1e-4, atol=5e-4)
            torch.testing.assert_close(compatibility_alphas, reference_alphas, rtol=1e-4, atol=5e-4)
            """
        )
    )


@pytest.mark.cuda
def test_nht_feature_rasterization_backward_matches_upstream() -> None:
    _run_reference_check(
        _reference_script(
            r"""
            native_inputs = make_inputs(feature_dim=16, requires_grad=True)
            reference_inputs = make_inputs(feature_dim=16, requires_grad=True)
            native_projection_result = native_projection(native_inputs)
            reference_projection_result = native_projection(reference_inputs)
            native_intersections = intersections_from_projection(native_inputs, native_projection_result)
            reference_intersections = intersections_from_projection(reference_inputs, reference_projection_result)
            num_cameras = int(native_inputs["world_to_camera_matrices"].shape[0])

            native_features = broadcast_features(native_inputs["features"], num_cameras)
            native_opacities = broadcast_opacities(native_inputs["opacities"], num_cameras)
            reference_features = broadcast_features(reference_inputs["features"], num_cameras)
            reference_opacities = broadcast_opacities(reference_inputs["opacities"], num_cameras)

            native_result = native_rasterize_features(
                center_positions=native_inputs["center_positions"],
                quaternions=native_inputs["quaternions"],
                scales=native_inputs["scales"],
                features=native_features,
                opacities=native_opacities,
                world_to_camera_matrices=native_inputs["world_to_camera_matrices"],
                camera_intrinsics=native_inputs["camera_intrinsics"],
                image_width=native_inputs["image_width"],
                image_height=native_inputs["image_height"],
                tile_size=native_inputs["tile_size"],
                tile_offsets=native_intersections.tile_offsets,
                flattened_gaussian_ids=native_intersections.flattened_gaussian_ids,
                camera_model=native_inputs["camera_model"],
                center_ray_mode=False,
                ray_direction_scale=3.0,
            )
            reference_rendered_features, reference_alphas = reference_rasterize_features(
                means=reference_inputs["center_positions"],
                quats=reference_inputs["quaternions"],
                scales=reference_inputs["scales"],
                colors=reference_features,
                opacities=reference_opacities,
                viewmats=reference_inputs["world_to_camera_matrices"],
                Ks=reference_inputs["camera_intrinsics"],
                image_width=reference_inputs["image_width"],
                image_height=reference_inputs["image_height"],
                tile_size=reference_inputs["tile_size"],
                isect_offsets=reference_intersections.tile_offsets,
                flatten_ids=reference_intersections.flattened_gaussian_ids,
                camera_model=reference_inputs["camera_model"],
                center_ray_mode=False,
                ray_dir_scale=3.0,
            )
            feature_weights = torch.linspace(
                0.25,
                1.25,
                steps=native_result.features.numel(),
                device="cuda",
                dtype=torch.float32,
            ).reshape_as(native_result.features)
            alpha_weights = torch.linspace(
                1.0,
                1.5,
                steps=native_result.alphas.numel(),
                device="cuda",
                dtype=torch.float32,
            ).reshape_as(native_result.alphas)
            native_loss = (native_result.features * feature_weights).sum() + (
                native_result.alphas * alpha_weights
            ).sum()
            reference_loss = (reference_rendered_features * feature_weights).sum() + (
                reference_alphas * alpha_weights
            ).sum()
            native_loss.backward()
            reference_loss.backward()

            for input_name in ("center_positions", "quaternions", "scales", "opacities", "features"):
                native_grad = native_inputs[input_name].grad
                reference_grad = reference_inputs[input_name].grad
                assert native_grad is not None, input_name
                assert reference_grad is not None, input_name
                assert torch.isfinite(native_grad).all(), input_name
                assert torch.isfinite(reference_grad).all(), input_name
                torch.testing.assert_close(native_grad, reference_grad, rtol=2e-3, atol=2e-3)
            """
        )
    )


@pytest.mark.cuda
def test_nht_depth_rasterization_backward_matches_upstream() -> None:
    _run_reference_check(
        _reference_script(
            r"""
            native_inputs = make_inputs(feature_dim=16, requires_grad=True)
            reference_inputs = make_inputs(feature_dim=16, requires_grad=True)
            native_projection_result = native_projection(native_inputs)
            reference_projection_result = native_projection(reference_inputs)
            native_intersections = intersections_from_projection(native_inputs, native_projection_result)
            reference_intersections = intersections_from_projection(reference_inputs, reference_projection_result)
            num_cameras = int(native_inputs["world_to_camera_matrices"].shape[0])
            native_opacities = broadcast_opacities(native_inputs["opacities"], num_cameras)
            reference_opacities = broadcast_opacities(reference_inputs["opacities"], num_cameras)
            native_depth_features = native_projection_result.primitive_depths.detach().clone().requires_grad_(True)
            reference_depth_features = reference_projection_result.primitive_depths.detach().clone().requires_grad_(True)

            native_result = native_rasterize_depth(
                center_positions=native_inputs["center_positions"],
                quaternions=native_inputs["quaternions"],
                scales=native_inputs["scales"],
                depth_features=native_depth_features[..., None],
                opacities=native_opacities,
                world_to_camera_matrices=native_inputs["world_to_camera_matrices"],
                camera_intrinsics=native_inputs["camera_intrinsics"],
                image_width=native_inputs["image_width"],
                image_height=native_inputs["image_height"],
                tile_size=native_inputs["tile_size"],
                tile_offsets=native_intersections.tile_offsets,
                flattened_gaussian_ids=native_intersections.flattened_gaussian_ids,
                camera_model=native_inputs["camera_model"],
            )
            reference_depths, reference_alphas = reference_rasterize_depth(
                means=reference_inputs["center_positions"],
                quats=reference_inputs["quaternions"],
                scales=reference_inputs["scales"],
                colors=reference_depth_features[..., None],
                opacities=reference_opacities,
                viewmats=reference_inputs["world_to_camera_matrices"],
                Ks=reference_inputs["camera_intrinsics"],
                image_width=reference_inputs["image_width"],
                image_height=reference_inputs["image_height"],
                tile_size=reference_inputs["tile_size"],
                isect_offsets=reference_intersections.tile_offsets,
                flatten_ids=reference_intersections.flattened_gaussian_ids,
                camera_model=reference_inputs["camera_model"],
            )
            depth_weights = torch.linspace(
                0.4,
                1.4,
                steps=native_result.depths.numel(),
                device="cuda",
                dtype=torch.float32,
            ).reshape_as(native_result.depths)
            alpha_weights = torch.linspace(
                0.7,
                1.1,
                steps=native_result.alphas.numel(),
                device="cuda",
                dtype=torch.float32,
            ).reshape_as(native_result.alphas)
            native_loss = (native_result.depths * depth_weights).sum() + (
                native_result.alphas * alpha_weights
            ).sum()
            reference_loss = (reference_depths * depth_weights).sum() + (
                reference_alphas * alpha_weights
            ).sum()
            native_loss.backward()
            reference_loss.backward()

            for input_name in ("center_positions", "quaternions", "scales", "opacities"):
                native_grad = native_inputs[input_name].grad
                reference_grad = reference_inputs[input_name].grad
                assert native_grad is not None, input_name
                assert reference_grad is not None, input_name
                assert torch.isfinite(native_grad).all(), input_name
                assert torch.isfinite(reference_grad).all(), input_name
                torch.testing.assert_close(native_grad, reference_grad, rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(
                native_depth_features.grad,
                reference_depth_features.grad,
                rtol=2e-3,
                atol=2e-3,
            )
            """
        )
    )
