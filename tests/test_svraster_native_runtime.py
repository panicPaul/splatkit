from __future__ import annotations

import pytest
import torch
from ember_core.core.sparse_voxel import svraster_octpath_to_ijk
from ember_native_svraster.core import runtime
from ember_native_svraster.core.runtime.gather import (
    gather_triinterp_geo_params,
)
from ember_native_svraster.core.runtime.ops import (
    preprocess_op,
    rasterize_op,
    sh_eval_op,
)
from torch._subclasses.fake_tensor import FakeTensorMode


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
def test_render_matches_explicit_stage_composition(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    width, height, focal_x, focal_y, center_x, center_y = (
        _extract_camera_params(cuda_camera)
    )
    camera_to_world = cuda_camera.cam_to_world[0]
    tanfovx = (width * 0.5) / focal_x
    tanfovy = (height * 0.5) / focal_y
    preprocess_result = runtime.preprocess(
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(camera_to_world),
        camera_to_world=camera_to_world,
        near=0.02,
        octree_paths=cuda_sparse_voxel_scene.octpath.reshape(-1),
        voxel_centers=cuda_sparse_voxel_scene.vox_center,
        voxel_lengths=cuda_sparse_voxel_scene.vox_size.reshape(-1),
    )
    voxel_colors = runtime.sh_eval(
        active_sh_degree=cuda_sparse_voxel_scene.active_sh_degree,
        voxel_centers=cuda_sparse_voxel_scene.vox_center,
        camera_position=camera_to_world[:3, 3],
        sh0=cuda_sparse_voxel_scene.sh0,
        shs=cuda_sparse_voxel_scene.shs,
    )
    explicit = runtime.rasterize(
        samples_per_voxel=1,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(camera_to_world),
        camera_to_world=camera_to_world,
        background_color=0.0,
        return_depth=True,
        return_normal=False,
        track_max_weight=False,
        octree_paths=cuda_sparse_voxel_scene.octpath.reshape(-1),
        voxel_centers=cuda_sparse_voxel_scene.vox_center,
        voxel_lengths=cuda_sparse_voxel_scene.vox_size.reshape(-1),
        voxel_geometries=cuda_sparse_voxel_scene.voxel_geometries,
        voxel_colors=voxel_colors,
        subdivision_priority=torch.ones(
            (cuda_sparse_voxel_scene.num_voxels, 1),
            device=voxel_colors.device,
            dtype=voxel_colors.dtype,
        ),
        geometry_buffer=preprocess_result.geom_buffer,
    )
    combined = runtime.render(
        active_sh_degree=cuda_sparse_voxel_scene.active_sh_degree,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(camera_to_world),
        camera_to_world=camera_to_world,
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

    torch.testing.assert_close(combined.color, explicit.color)
    torch.testing.assert_close(combined.depth, explicit.depth)
    torch.testing.assert_close(
        combined.transmittance,
        explicit.transmittance,
    )


@pytest.mark.cuda
def test_render_backward_produces_finite_gradients(
    cuda_sparse_voxel_scene,
    cuda_camera,
) -> None:
    width, height, focal_x, focal_y, center_x, center_y = (
        _extract_camera_params(cuda_camera)
    )
    camera_to_world = cuda_camera.cam_to_world[0]
    tanfovx = (width * 0.5) / focal_x
    tanfovy = (height * 0.5) / focal_y
    scene = cuda_sparse_voxel_scene.detached_copy()
    scene.replace_fields_(
        geo_grid_pts=cuda_sparse_voxel_scene.geo_grid_pts.detach()
        .clone()
        .requires_grad_(True),
        sh0=cuda_sparse_voxel_scene.sh0.detach().clone().requires_grad_(True),
        shs=cuda_sparse_voxel_scene.shs.detach().clone().requires_grad_(True),
    )

    result = runtime.render(
        active_sh_degree=scene.active_sh_degree,
        image_width=width,
        image_height=height,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(camera_to_world),
        camera_to_world=camera_to_world,
        near=0.02,
        background_color=0.0,
        octree_paths=scene.octpath.reshape(-1),
        voxel_centers=scene.vox_center,
        voxel_lengths=scene.vox_size.reshape(-1),
        voxel_geometries=scene.voxel_geometries,
        sh0=scene.sh0,
        shs=scene.shs,
        return_depth=True,
    )
    (result.color.sum() + result.depth.sum()).backward()

    for grad in (scene.geo_grid_pts.grad, scene.sh0.grad, scene.shs.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()


@pytest.mark.cuda
def test_rasterize_backward_accepts_empty_voxel_gradients(
    cuda_camera,
) -> None:
    width, height, focal_x, focal_y, center_x, center_y = (
        _extract_camera_params(cuda_camera)
    )
    camera_to_world = cuda_camera.cam_to_world[0]
    device = camera_to_world.device
    dtype = torch.float32
    octree_paths = torch.empty((0,), device=device, dtype=torch.int64)
    voxel_centers = torch.empty((0, 3), device=device, dtype=dtype)
    voxel_lengths = torch.empty((0,), device=device, dtype=dtype)
    preprocess_result = runtime.preprocess(
        image_width=width,
        image_height=height,
        tanfovx=(width * 0.5) / focal_x,
        tanfovy=(height * 0.5) / focal_y,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(camera_to_world),
        camera_to_world=camera_to_world,
        near=0.02,
        octree_paths=octree_paths,
        voxel_centers=voxel_centers,
        voxel_lengths=voxel_lengths,
    )
    voxel_geometries = torch.empty(
        (0, 8), device=device, dtype=dtype, requires_grad=True
    )
    voxel_colors = torch.empty(
        (0, 3), device=device, dtype=dtype, requires_grad=True
    )
    subdivision_priority = torch.empty(
        (0, 1), device=device, dtype=dtype, requires_grad=True
    )

    result = runtime.rasterize(
        samples_per_voxel=1,
        image_width=width,
        image_height=height,
        tanfovx=(width * 0.5) / focal_x,
        tanfovy=(height * 0.5) / focal_y,
        cx=center_x,
        cy=center_y,
        world_to_camera=torch.linalg.inv(camera_to_world),
        camera_to_world=camera_to_world,
        background_color=0.0,
        return_depth=True,
        return_normal=False,
        track_max_weight=True,
        octree_paths=octree_paths,
        voxel_centers=voxel_centers,
        voxel_lengths=voxel_lengths,
        voxel_geometries=voxel_geometries,
        voxel_colors=voxel_colors,
        subdivision_priority=subdivision_priority,
        geometry_buffer=preprocess_result.geom_buffer,
        distortion_weight=0.1,
    )
    (result.color.sum() + result.depth.sum()).backward()

    assert voxel_geometries.grad is not None
    assert voxel_geometries.grad.shape == (0, 8)
    assert voxel_colors.grad is not None
    assert voxel_colors.grad.shape == (0, 3)
    assert subdivision_priority.grad is not None
    assert subdivision_priority.grad.shape == (0, 1)


@pytest.mark.cuda
def test_auxiliary_ops_match_sparse_voxel_contract(
    cuda_sparse_voxel_scene,
) -> None:
    care_idx = torch.arange(
        cuda_sparse_voxel_scene.num_voxels,
        device=cuda_sparse_voxel_scene.octpath.device,
        dtype=torch.int64,
    )
    gathered = gather_triinterp_geo_params(
        cuda_sparse_voxel_scene.vox_key,
        care_idx,
        cuda_sparse_voxel_scene.geo_grid_pts,
    )
    torch.testing.assert_close(
        gathered, cuda_sparse_voxel_scene.voxel_geometries
    )

    native_ijk = runtime.utils.octpath_2_ijk(
        cuda_sparse_voxel_scene.octpath.reshape(-1, 1),
        cuda_sparse_voxel_scene.octlevel.reshape(-1, 1).to(torch.int8),
    )
    expected_ijk = svraster_octpath_to_ijk(
        cuda_sparse_voxel_scene.octpath,
        cuda_sparse_voxel_scene.octlevel,
        backend_name=None,
        max_num_levels=cuda_sparse_voxel_scene.max_num_levels,
    )
    torch.testing.assert_close(native_ijk, expected_ijk)


def test_raw_ops_support_fake_tensor_mode(
    cpu_sparse_voxel_scene,
    cpu_camera,
) -> None:
    width, height, focal_x, focal_y, center_x, center_y = (
        _extract_camera_params(cpu_camera)
    )
    camera_to_world = cpu_camera.cam_to_world[0]
    tanfovx = (width * 0.5) / focal_x
    tanfovy = (height * 0.5) / focal_y

    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        octpath = mode.from_tensor(cpu_sparse_voxel_scene.octpath.reshape(-1))
        voxel_centers = mode.from_tensor(cpu_sparse_voxel_scene.vox_center)
        voxel_lengths = mode.from_tensor(
            cpu_sparse_voxel_scene.vox_size.reshape(-1)
        )
        voxel_geometries = mode.from_tensor(
            torch.empty((2, 8), dtype=torch.float32)
        )
        sh0 = mode.from_tensor(cpu_sparse_voxel_scene.sh0)
        shs = mode.from_tensor(cpu_sparse_voxel_scene.shs)
        world_to_camera = mode.from_tensor(torch.linalg.inv(camera_to_world))
        camera_to_world_tensor = mode.from_tensor(camera_to_world)
        preprocess_result = preprocess_op(
            width,
            height,
            tanfovx,
            tanfovy,
            center_x,
            center_y,
            world_to_camera,
            camera_to_world_tensor,
            0.02,
            octpath,
            voxel_centers,
            voxel_lengths,
        )
        voxel_colors = sh_eval_op(
            cpu_sparse_voxel_scene.active_sh_degree,
            mode.from_tensor(torch.empty((0,), dtype=torch.int64)),
            voxel_centers,
            mode.from_tensor(camera_to_world[:3, 3]),
            sh0,
            shs,
        )
        rasterize_result = rasterize_op(
            1,
            width,
            height,
            tanfovx,
            tanfovy,
            center_x,
            center_y,
            world_to_camera,
            camera_to_world_tensor,
            0.0,
            True,
            False,
            False,
            0.0,
            0.0,
            0.0,
            mode.from_tensor(torch.empty((0,), dtype=torch.float32)),
            False,
            octpath,
            voxel_centers,
            voxel_lengths,
            voxel_geometries,
            voxel_colors,
            mode.from_tensor(
                torch.ones((int(octpath.shape[0]), 1), dtype=torch.float32)
            ),
            preprocess_result[1],
        )

    assert preprocess_result[0].shape == (2,)
    assert voxel_colors.shape == (2, 3)
    assert rasterize_result[3].shape == (3, 32, 32)
    assert rasterize_result[4].shape == (3, 32, 32)
