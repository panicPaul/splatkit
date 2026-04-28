from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from ember_native_svraster.core import runtime


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


@pytest.mark.cuda
def test_runtime_matches_optional_reference_adapter(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    raw_svraster_renderer = pytest.importorskip("new_svraster_cuda.renderer")
    width, height, focal_x, focal_y, center_x, center_y = _extract_camera_params(
        cuda_camera
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
        w2c_matrix=torch.linalg.inv(cam_to_world),
        c2w_matrix=cam_to_world,
        near=0.02,
        bg_color=0.0,
        octree_paths=cuda_sparse_voxel_scene.octpath.reshape(-1),
        vox_centers=cuda_sparse_voxel_scene.vox_center,
        vox_lengths=cuda_sparse_voxel_scene.vox_size.reshape(-1),
        geos=cuda_sparse_voxel_scene.voxel_geometries,
        sh0=cuda_sparse_voxel_scene.sh0,
        shs=cuda_sparse_voxel_scene.shs,
        need_depth=True,
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
    raw_svraster_renderer = pytest.importorskip("new_svraster_cuda.renderer")
    width, height, focal_x, focal_y, center_x, center_y = _extract_camera_params(
        cuda_camera
    )
    cam_to_world = cuda_camera.cam_to_world[0]
    tanfovx = (width * 0.5) / focal_x
    tanfovy = (height * 0.5) / focal_y

    native_scene = replace(
        cuda_sparse_voxel_scene,
        geo_grid_pts=cuda_sparse_voxel_scene.geo_grid_pts.detach()
        .clone()
        .requires_grad_(True),
        sh0=cuda_sparse_voxel_scene.sh0.detach().clone().requires_grad_(True),
        shs=cuda_sparse_voxel_scene.shs.detach().clone().requires_grad_(True),
    )
    adapter_scene = replace(
        cuda_sparse_voxel_scene,
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
        w2c_matrix=torch.linalg.inv(cam_to_world),
        c2w_matrix=cam_to_world,
        near=0.02,
        bg_color=0.0,
        octree_paths=native_scene.octpath.reshape(-1),
        vox_centers=native_scene.vox_center,
        vox_lengths=native_scene.vox_size.reshape(-1),
        geos=native_scene.voxel_geometries,
        sh0=native_scene.sh0,
        shs=native_scene.shs,
        need_depth=True,
    )
    native_loss = (
        native.color.sum()
        + native.depth.sum()
        + native.transmittance.sum()
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
