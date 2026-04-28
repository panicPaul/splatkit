import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from ember_core.core import (
    CameraState,
    GaussianScene2D,
    GaussianScene3D,
    SparseVoxelScene,
    camera_params_to_intrinsics,
    intrinsics_to_camera_params,
)
from ember_core.core.sparse_voxel import (
    svraster_build_grid_points_link,
    svraster_octpath_to_ijk,
)


def test_camera_intrinsics_roundtrip_shape() -> None:
    width = torch.tensor([32, 64], dtype=torch.int64)
    height = torch.tensor([24, 48], dtype=torch.int64)
    fov_degrees = torch.tensor([60.0, 45.0], dtype=torch.float32)

    intrinsics = camera_params_to_intrinsics(width, height, fov_degrees)
    params = intrinsics_to_camera_params(intrinsics)

    assert intrinsics.shape == (2, 3, 3)
    assert params.width.shape == width.shape
    assert params.height.shape == height.shape
    assert params.fov_degrees.shape == fov_degrees.shape
    assert torch.equal(params.width, width)
    assert torch.equal(params.height, height)


def test_camera_state_to_moves_all_tensors(cpu_camera: CameraState) -> None:
    moved = cpu_camera.to(torch.device("cpu"))
    assert moved.width.device.type == "cpu"
    assert moved.height.device.type == "cpu"
    assert moved.fov_degrees.device.type == "cpu"
    assert moved.cam_to_world.device.type == "cpu"


def test_camera_state_prefers_explicit_intrinsics() -> None:
    intrinsics = torch.tensor(
        [[[100.0, 0.0, 10.0], [0.0, 120.0, 12.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    camera = CameraState(
        width=torch.tensor([20], dtype=torch.int64),
        height=torch.tensor([24], dtype=torch.int64),
        fov_degrees=torch.tensor([60.0], dtype=torch.float32),
        cam_to_world=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=intrinsics,
    )
    assert torch.equal(camera.get_intrinsics(), intrinsics)


def test_gaussian_scene_to_moves_all_tensors(
    cpu_scene: GaussianScene3D,
) -> None:
    moved = cpu_scene.to(torch.device("cpu"))
    assert moved.center_position.device.type == "cpu"
    assert moved.log_scales.device.type == "cpu"
    assert moved.quaternion_orientation.device.type == "cpu"
    assert moved.logit_opacity.device.type == "cpu"
    assert moved.feature.device.type == "cpu"


def test_sparse_voxel_scene_to_moves_all_tensors(
    cpu_sparse_voxel_scene: SparseVoxelScene,
) -> None:
    moved = cpu_sparse_voxel_scene.to(torch.device("cpu"))
    assert moved.scene_center.device.type == "cpu"
    assert moved.scene_extent.device.type == "cpu"
    assert moved.octpath.device.type == "cpu"
    assert moved.geo_grid_pts.device.type == "cpu"


def test_camera_state_construction_validates_tensor_shape() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        CameraState(
            width=torch.tensor([32], dtype=torch.int64),
            height=torch.tensor([32], dtype=torch.int64),
            fov_degrees=torch.tensor([60.0], dtype=torch.float32),
            cam_to_world=torch.eye(4, dtype=torch.float32),
        )


def test_gaussian_scene_construction_validates_feature_shape() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        GaussianScene3D(
            center_position=torch.zeros((3, 3), dtype=torch.float32),
            log_scales=torch.zeros((3, 3), dtype=torch.float32),
            quaternion_orientation=torch.zeros((3, 3), dtype=torch.float32),
            logit_opacity=torch.zeros((3,), dtype=torch.float32),
            feature=torch.zeros((3, 16), dtype=torch.float32),
            sh_degree=0,
        )


def test_gaussian_scene_2d_validates_scale_dimensionality() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        GaussianScene2D(
            center_position=torch.zeros((3, 3), dtype=torch.float32),
            log_scales=torch.zeros((3, 3), dtype=torch.float32),
            quaternion_orientation=torch.zeros((3, 4), dtype=torch.float32),
            logit_opacity=torch.zeros((3,), dtype=torch.float32),
            feature=torch.zeros((3, 1, 3), dtype=torch.float32),
            sh_degree=0,
        )


@pytest.mark.cuda
def test_svraster_helpers_match_new_cuda_backend_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for SV Raster helper parity test.")

    new_svraster_cuda = pytest.importorskip("new_svraster_cuda")
    octree_utils = pytest.importorskip("sv_raster.new.utils.octree_utils")

    ijk = torch.tensor(
        [
            [0, 0, 0],
            [1, 2, 3],
            [7, 5, 4],
            [31, 12, 9],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    octlevel = torch.tensor(
        [[1], [3], [3], [5]], dtype=torch.int8, device="cuda"
    )
    octpath = new_svraster_cuda.utils.ijk_2_octpath(ijk, octlevel)

    ours_ijk = svraster_octpath_to_ijk(
        octpath,
        octlevel,
        backend_name="new_cuda",
        max_num_levels=5,
    )
    ref_ijk = new_svraster_cuda.utils.octpath_2_ijk(octpath, octlevel)

    assert torch.equal(ours_ijk, ref_ijk)

    ours_grid_pts_key, ours_vox_key = svraster_build_grid_points_link(
        octpath,
        octlevel,
        backend_name="new_cuda",
        max_num_levels=5,
    )
    ref_grid_pts_key, ref_vox_key = octree_utils.build_grid_pts_link(
        octpath,
        octlevel,
        backend_name="new_cuda",
    )

    assert torch.equal(ours_grid_pts_key, ref_grid_pts_key)
    assert torch.equal(ours_vox_key, ref_vox_key)
