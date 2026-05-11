from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
from ember_native_svraster.core import runtime
from ember_native_svraster.core.runtime.gather import (
    gather_triinterp_geo_params,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SVRASTER_REFERENCE_BACKEND = (
    _REPO_ROOT / "third_party" / "sv_raster" / "backends" / "new_cuda"
)


def _extract_camera_params(
    camera_state,
) -> tuple[int, int, float, float, float, float]:
    intrinsics = camera_state.get_intrinsics()[0]
    return (
        int(camera_state.width[0].item()),
        int(camera_state.height[0].item()),
        float(intrinsics[0, 0].item()),
        float(intrinsics[1, 1].item()),
        float(intrinsics[0, 2].item()),
        float(intrinsics[1, 2].item()),
    )


def _import_reference_renderer():
    sys.path.insert(0, str(_SVRASTER_REFERENCE_BACKEND))
    try:
        return importlib.import_module("new_svraster_cuda.renderer")
    except ImportError as exc:
        pytest.skip(f"upstream SVRaster CUDA reference is unavailable: {exc}")
    finally:
        try:
            sys.path.remove(str(_SVRASTER_REFERENCE_BACKEND))
        except ValueError:
            pass


@pytest.mark.cuda
def test_preprocess_kernel_matches_optional_reference_adapter(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    raw_svraster_renderer = _import_reference_renderer()
    width, height, focal_x, focal_y, center_x, center_y = (
        _extract_camera_params(cuda_camera)
    )
    cam_to_world = cuda_camera.cam_to_world[0]
    tanfovx = (width * 0.5) / focal_x
    tanfovy = (height * 0.5) / focal_y

    native = runtime.preprocess(
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(cam_to_world),
        camera_to_world=cam_to_world,
        near=0.02,
        octree_paths=cuda_sparse_voxel_scene.octpath.reshape(-1),
        voxel_centers=cuda_sparse_voxel_scene.vox_center,
        voxel_lengths=cuda_sparse_voxel_scene.vox_size.reshape(-1),
    )
    reference_duplicates, reference_geom_buffer = (
        raw_svraster_renderer.mark_n_duplicates(
            image_width=width,
            image_height=height,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            cx=center_x,
            cy=center_y,
            w2c_matrix=torch.linalg.inv(cam_to_world),
            c2w_matrix=cam_to_world,
            near=0.02,
            octree_paths=cuda_sparse_voxel_scene.octpath.reshape(-1),
            vox_centers=cuda_sparse_voxel_scene.vox_center,
            vox_lengths=cuda_sparse_voxel_scene.vox_size.reshape(-1),
            return_buffer=True,
        )
    )

    torch.testing.assert_close(native.n_duplicates, reference_duplicates)
    assert native.geom_buffer.shape == reference_geom_buffer.shape
    assert native.geom_buffer.dtype == reference_geom_buffer.dtype
    assert native.geom_buffer.device == reference_geom_buffer.device


@pytest.mark.cuda
def test_sh_eval_kernel_matches_optional_reference_forward_backward(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    raw_svraster_renderer = _import_reference_renderer()
    cam_pos = cuda_camera.cam_to_world[0, :3, 3]
    native_sh0 = cuda_sparse_voxel_scene.sh0.detach().clone().requires_grad_(True)
    native_shs = cuda_sparse_voxel_scene.shs.detach().clone().requires_grad_(True)
    reference_sh0 = (
        cuda_sparse_voxel_scene.sh0.detach().clone().requires_grad_(True)
    )
    reference_shs = (
        cuda_sparse_voxel_scene.shs.detach().clone().requires_grad_(True)
    )

    native = runtime.sh_eval(
        active_sh_degree=cuda_sparse_voxel_scene.active_sh_degree,
        voxel_centers=cuda_sparse_voxel_scene.vox_center,
        camera_position=cam_pos,
        sh0=native_sh0,
        shs=native_shs,
    )
    reference = raw_svraster_renderer.SH_eval.apply(
        cuda_sparse_voxel_scene.active_sh_degree,
        None,
        cuda_sparse_voxel_scene.vox_center,
        cam_pos,
        None,
        reference_sh0,
        reference_shs,
    )
    torch.testing.assert_close(native, reference, rtol=1e-5, atol=1e-6)

    weights = torch.linspace(
        0.25,
        1.0,
        native.numel(),
        device=native.device,
        dtype=native.dtype,
    ).reshape_as(native)
    (native * weights).sum().backward()
    (reference * weights).sum().backward()

    for native_grad, reference_grad in (
        (native_sh0.grad, reference_sh0.grad),
        (native_shs.grad, reference_shs.grad),
    ):
        assert native_grad is not None
        assert reference_grad is not None
        torch.testing.assert_close(
            native_grad,
            reference_grad,
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.cuda
def test_gather_geo_kernel_matches_optional_reference_forward_backward(
    cuda_sparse_voxel_scene,
) -> None:
    raw_svraster_renderer = _import_reference_renderer()
    care_idx = torch.arange(
        cuda_sparse_voxel_scene.num_voxels,
        device=cuda_sparse_voxel_scene.octpath.device,
        dtype=torch.int64,
    )
    native_grid = (
        cuda_sparse_voxel_scene.geo_grid_pts.detach()
        .clone()
        .requires_grad_(True)
    )
    reference_grid = (
        cuda_sparse_voxel_scene.geo_grid_pts.detach()
        .clone()
        .requires_grad_(True)
    )

    native = gather_triinterp_geo_params(
        cuda_sparse_voxel_scene.vox_key,
        care_idx,
        native_grid,
    )
    reference = raw_svraster_renderer.GatherGeoParams.apply(
        cuda_sparse_voxel_scene.vox_key,
        care_idx,
        reference_grid,
    )
    torch.testing.assert_close(native, reference, rtol=1e-5, atol=1e-6)

    weights = torch.linspace(
        0.1,
        0.9,
        native.numel(),
        device=native.device,
        dtype=native.dtype,
    ).reshape_as(native)
    (native * weights).sum().backward()
    (reference * weights).sum().backward()

    assert native_grid.grad is not None
    assert reference_grid.grad is not None
    torch.testing.assert_close(
        native_grid.grad,
        reference_grid.grad,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.cuda
def test_rasterize_kernel_matches_optional_reference_forward_backward(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    raw_svraster_renderer = _import_reference_renderer()
    width, height, focal_x, focal_y, center_x, center_y = (
        _extract_camera_params(cuda_camera)
    )
    cam_to_world = cuda_camera.cam_to_world[0]
    tanfovx = (width * 0.5) / focal_x
    tanfovy = (height * 0.5) / focal_y
    preprocess_result = runtime.preprocess(
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(cam_to_world),
        camera_to_world=cam_to_world,
        near=0.02,
        octree_paths=cuda_sparse_voxel_scene.octpath.reshape(-1),
        voxel_centers=cuda_sparse_voxel_scene.vox_center,
        voxel_lengths=cuda_sparse_voxel_scene.vox_size.reshape(-1),
    )

    native_geos = (
        cuda_sparse_voxel_scene.voxel_geometries.detach()
        .clone()
        .requires_grad_(True)
    )
    native_rgbs = torch.rand(
        (cuda_sparse_voxel_scene.num_voxels, 3),
        device=native_geos.device,
        dtype=native_geos.dtype,
        generator=torch.Generator(device=native_geos.device).manual_seed(3),
        requires_grad=True,
    )
    native_subdiv = torch.ones(
        (cuda_sparse_voxel_scene.num_voxels, 1),
        device=native_geos.device,
        dtype=native_geos.dtype,
        requires_grad=True,
    )
    reference_geos = native_geos.detach().clone().requires_grad_(True)
    reference_rgbs = native_rgbs.detach().clone().requires_grad_(True)
    reference_subdiv = native_subdiv.detach().clone().requires_grad_(True)

    native = runtime.rasterize(
        samples_per_voxel=1,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(cam_to_world),
        camera_to_world=cam_to_world,
        background_color=0.0,
        return_depth=True,
        return_normal=True,
        track_max_weight=True,
        octree_paths=cuda_sparse_voxel_scene.octpath.reshape(-1),
        voxel_centers=cuda_sparse_voxel_scene.vox_center,
        voxel_lengths=cuda_sparse_voxel_scene.vox_size.reshape(-1),
        voxel_geometries=native_geos,
        voxel_colors=native_rgbs,
        subdivision_priority=native_subdiv,
        geometry_buffer=preprocess_result.geom_buffer,
        distortion_weight=0.1,
    )
    raster_settings = raw_svraster_renderer.RasterSettings(
        color_mode="dontcare",
        n_samp_per_vox=1,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        w2c_matrix=torch.linalg.inv(cam_to_world),
        c2w_matrix=cam_to_world,
        bg_color=0.0,
        near=0.02,
        need_depth=True,
        need_normal=True,
        track_max_w=True,
        lambda_dist=0.1,
    )

    def vox_fn(
        _idx: torch.Tensor,
        _cam_pos: torch.Tensor,
        _color_mode: str,
    ) -> dict[str, torch.Tensor]:
        return {
            "geos": reference_geos,
            "rgbs": reference_rgbs,
            "subdiv_p": reference_subdiv,
        }

    color, depth, normal, transmittance, max_weight = (
        raw_svraster_renderer.rasterize_voxels(
            raster_settings,
            cuda_sparse_voxel_scene.octpath.reshape(-1),
            cuda_sparse_voxel_scene.vox_center,
            cuda_sparse_voxel_scene.vox_size.reshape(-1),
            vox_fn,
        )
    )

    torch.testing.assert_close(native.color, color, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native.depth, depth, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native.normal, normal, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        native.transmittance,
        transmittance,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(native.max_weight, max_weight)

    color_weights = torch.linspace(
        0.1,
        1.0,
        native.color.numel(),
        device=native.color.device,
        dtype=native.color.dtype,
    ).reshape_as(native.color)
    native_loss = (
        (native.color * color_weights).sum()
        + native.depth.sum()
        + native.normal.sum()
        + native.transmittance.sum()
    )
    reference_loss = (
        (color * color_weights).sum()
        + depth.sum()
        + normal.sum()
        + transmittance.sum()
    )
    native_loss.backward()
    reference_loss.backward()

    for native_grad, reference_grad in (
        (native_geos.grad, reference_geos.grad),
        (native_rgbs.grad, reference_rgbs.grad),
        (native_subdiv.grad, reference_subdiv.grad),
    ):
        assert native_grad is not None
        assert reference_grad is not None
        torch.testing.assert_close(
            native_grad,
            reference_grad,
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.cuda
def test_runtime_matches_optional_reference_adapter(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    raw_svraster_renderer = _import_reference_renderer()
    width, height, focal_x, focal_y, center_x, center_y = (
        _extract_camera_params(cuda_camera)
    )
    cam_to_world = cuda_camera.cam_to_world[0]
    tanfovx = (width * 0.5) / focal_x
    tanfovy = (height * 0.5) / focal_y
    native = runtime.render(
        active_sh_degree=cuda_sparse_voxel_scene.active_sh_degree,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(cam_to_world),
        camera_to_world=cam_to_world,
        near=0.02,
        background_color=0.0,
        octree_paths=cuda_sparse_voxel_scene.octpath.reshape(-1),
        voxel_centers=cuda_sparse_voxel_scene.vox_center,
        voxel_lengths=cuda_sparse_voxel_scene.vox_size.reshape(-1),
        voxel_geometries=cuda_sparse_voxel_scene.voxel_geometries,
        sh0=cuda_sparse_voxel_scene.sh0,
        shs=cuda_sparse_voxel_scene.shs,
        return_depth=True,
    )

    raster_settings = raw_svraster_renderer.RasterSettings(
        color_mode="sh",
        n_samp_per_vox=1,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        w2c_matrix=torch.linalg.inv(cam_to_world),
        c2w_matrix=cam_to_world,
        bg_color=0.0,
        near=0.02,
        need_depth=True,
    )

    def vox_fn(
        _idx: torch.Tensor,
        cam_pos: torch.Tensor,
        _color_mode: str,
    ) -> dict[str, torch.Tensor]:
        rgbs = raw_svraster_renderer.SH_eval.apply(
            cuda_sparse_voxel_scene.active_sh_degree,
            None,
            cuda_sparse_voxel_scene.vox_center,
            cam_pos,
            None,
            cuda_sparse_voxel_scene.sh0,
            cuda_sparse_voxel_scene.shs,
        )
        subdiv_p = torch.ones(
            (cuda_sparse_voxel_scene.num_voxels, 1),
            dtype=cuda_sparse_voxel_scene.sh0.dtype,
            device=cuda_sparse_voxel_scene.sh0.device,
        )
        return {
            "geos": cuda_sparse_voxel_scene.voxel_geometries,
            "rgbs": rgbs,
            "subdiv_p": subdiv_p,
        }

    color, depth, _normal, transmittance, _max_w = (
        raw_svraster_renderer.rasterize_voxels(
            raster_settings,
            cuda_sparse_voxel_scene.octpath.reshape(-1),
            cuda_sparse_voxel_scene.vox_center,
            cuda_sparse_voxel_scene.vox_size.reshape(-1),
            vox_fn,
        )
    )

    torch.testing.assert_close(native.color, color, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(native.depth, depth, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        native.transmittance,
        transmittance,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.cuda
def test_runtime_backward_matches_optional_reference_adapter(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    raw_svraster_renderer = _import_reference_renderer()
    width, height, focal_x, focal_y, center_x, center_y = (
        _extract_camera_params(cuda_camera)
    )
    cam_to_world = cuda_camera.cam_to_world[0]
    tanfovx = (width * 0.5) / focal_x
    tanfovy = (height * 0.5) / focal_y

    native_scene = cuda_sparse_voxel_scene.detached_copy()
    native_scene.replace_fields_(
        geo_grid_pts=cuda_sparse_voxel_scene.geo_grid_pts.detach()
        .clone()
        .requires_grad_(True),
        sh0=cuda_sparse_voxel_scene.sh0.detach().clone().requires_grad_(True),
        shs=cuda_sparse_voxel_scene.shs.detach().clone().requires_grad_(True),
    )
    adapter_scene = cuda_sparse_voxel_scene.detached_copy()
    adapter_scene.replace_fields_(
        geo_grid_pts=cuda_sparse_voxel_scene.geo_grid_pts.detach()
        .clone()
        .requires_grad_(True),
        sh0=cuda_sparse_voxel_scene.sh0.detach().clone().requires_grad_(True),
        shs=cuda_sparse_voxel_scene.shs.detach().clone().requires_grad_(True),
    )

    native = runtime.render(
        active_sh_degree=native_scene.active_sh_degree,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(cam_to_world),
        camera_to_world=cam_to_world,
        near=0.02,
        background_color=0.0,
        octree_paths=native_scene.octpath.reshape(-1),
        voxel_centers=native_scene.vox_center,
        voxel_lengths=native_scene.vox_size.reshape(-1),
        voxel_geometries=native_scene.voxel_geometries,
        sh0=native_scene.sh0,
        shs=native_scene.shs,
        return_depth=True,
    )
    native_loss = (
        native.color.sum() + native.depth.sum() + native.transmittance.sum()
    )
    native_loss.backward()

    raster_settings = raw_svraster_renderer.RasterSettings(
        color_mode="sh",
        n_samp_per_vox=1,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        w2c_matrix=torch.linalg.inv(cam_to_world),
        c2w_matrix=cam_to_world,
        bg_color=0.0,
        near=0.02,
        need_depth=True,
    )

    def vox_fn(
        _idx: torch.Tensor,
        cam_pos: torch.Tensor,
        _color_mode: str,
    ) -> dict[str, torch.Tensor]:
        rgbs = raw_svraster_renderer.SH_eval.apply(
            adapter_scene.active_sh_degree,
            None,
            adapter_scene.vox_center,
            cam_pos,
            None,
            adapter_scene.sh0,
            adapter_scene.shs,
        )
        subdiv_p = torch.ones(
            (adapter_scene.num_voxels, 1),
            dtype=adapter_scene.sh0.dtype,
            device=adapter_scene.sh0.device,
        )
        return {
            "geos": adapter_scene.voxel_geometries,
            "rgbs": rgbs,
            "subdiv_p": subdiv_p,
        }

    color, depth, _normal, transmittance, _max_w = (
        raw_svraster_renderer.rasterize_voxels(
            raster_settings,
            adapter_scene.octpath.reshape(-1),
            adapter_scene.vox_center,
            adapter_scene.vox_size.reshape(-1),
            vox_fn,
        )
    )
    adapter_loss = color.sum() + depth.sum() + transmittance.sum()
    adapter_loss.backward()

    for native_grad, adapter_grad in (
        (native_scene.geo_grid_pts.grad, adapter_scene.geo_grid_pts.grad),
        (native_scene.sh0.grad, adapter_scene.sh0.grad),
        (native_scene.shs.grad, adapter_scene.shs.grad),
    ):
        assert native_grad is not None
        assert adapter_grad is not None
        assert torch.isfinite(native_grad).all()
        assert torch.isfinite(adapter_grad).all()
        torch.testing.assert_close(
            native_grad,
            adapter_grad,
            rtol=1e-5,
            atol=1e-6,
        )
