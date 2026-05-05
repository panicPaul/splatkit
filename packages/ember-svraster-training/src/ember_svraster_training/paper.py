"""SVRaster paper-style initialization, options, and losses."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as torch_functional
from ember_core.core.contracts import CameraState, SparseVoxelScene
from ember_core.core.sparse_voxel import (
    DEFAULT_SVRASTER_MAX_NUM_LEVELS,
    svraster_build_grid_points_link,
    svraster_octpath_decoding,
    svraster_rgb_to_sh_zero,
)
from ember_core.data import PreparedFrameDataset, SceneRecord
from ember_core.initialization import InitializedModel
from ember_core.training import LossResult, TrainState
from jaxtyping import Float, Int
from torch import Tensor

SceneBoundMode = Literal["default", "forward", "camera_median", "camera_max"]


@dataclass(frozen=True)
class _PaperCamera:
    image_width: int
    image_height: int
    tanfovx: float
    tanfovy: float
    center_x: float
    center_y: float
    world_to_camera: Float[Tensor, " 4 4"]
    camera_to_world: Float[Tensor, " 4 4"]
    position: Float[Tensor, " 3"]
    look_at: Float[Tensor, " 3"]
    pixel_size: float


def _as_paper_cameras(
    cameras: Sequence[CameraState],
    *,
    device: torch.device,
) -> tuple[_PaperCamera, ...]:
    paper_cameras: list[_PaperCamera] = []
    for camera in cameras:
        prepared_camera = camera.to(device)
        intrinsics = prepared_camera.get_intrinsics()
        for camera_index in range(prepared_camera.cam_to_world.shape[0]):
            image_width = int(prepared_camera.width[camera_index].item())
            image_height = int(prepared_camera.height[camera_index].item())
            focal_x = float(intrinsics[camera_index, 0, 0].item())
            focal_y = float(intrinsics[camera_index, 1, 1].item())
            center_x = float(intrinsics[camera_index, 0, 2].item())
            center_y = float(intrinsics[camera_index, 1, 2].item())
            camera_to_world = prepared_camera.cam_to_world[camera_index].to(
                dtype=torch.float32
            )
            paper_cameras.append(
                _PaperCamera(
                    image_width=image_width,
                    image_height=image_height,
                    tanfovx=(image_width * 0.5) / focal_x,
                    tanfovy=(image_height * 0.5) / focal_y,
                    center_x=center_x,
                    center_y=center_y,
                    world_to_camera=torch.linalg.inv(camera_to_world),
                    camera_to_world=camera_to_world,
                    position=camera_to_world[:3, 3],
                    look_at=camera_to_world[:3, 2],
                    pixel_size=2.0 * ((image_width * 0.5) / focal_x) / image_width,
                )
            )
    return tuple(paper_cameras)


def _training_cameras(
    scene_record: SceneRecord,
    frame_dataset: PreparedFrameDataset | None,
    *,
    device: torch.device,
) -> tuple[_PaperCamera, ...]:
    cameras = (
        frame_dataset.prepared_cameras()
        if frame_dataset is not None
        else (scene_record.resolve_camera_sensor().camera,)
    )
    if not cameras:
        raise ValueError("SVRaster paper initialization requires cameras.")
    return _as_paper_cameras(cameras, device=device)


def _camera_positions(
    cameras: Sequence[_PaperCamera],
) -> Float[Tensor, " num_cameras 3"]:
    return torch.stack([camera.position for camera in cameras], dim=0)


def _camera_look_at(
    cameras: Sequence[_PaperCamera],
) -> Float[Tensor, " num_cameras 3"]:
    return torch.stack([camera.look_at for camera in cameras], dim=0)


def _forward_scene_bound(
    cameras: Sequence[_PaperCamera],
    *,
    forward_distance_scale: float,
) -> tuple[Float[Tensor, " 3"], Float[Tensor, " 1"]]:
    camera_positions = _camera_positions(cameras)
    camera_center = camera_positions.mean(dim=0)
    mean_look_at = torch.nn.functional.normalize(
        _camera_look_at(cameras).mean(dim=0),
        dim=0,
    )
    camera_extent = (
        2.0 * (camera_positions - camera_center).norm(dim=-1).max().clamp_min(1e-6)
    )
    scene_center = camera_center + mean_look_at * camera_extent
    inside_extent = (0.8 * forward_distance_scale * camera_extent).reshape(1)
    return scene_center, inside_extent


def _camera_median_scene_bound(
    cameras: Sequence[_PaperCamera],
) -> tuple[Float[Tensor, " 3"], Float[Tensor, " 1"]]:
    camera_positions = _camera_positions(cameras)
    scene_center = camera_positions.median(dim=0).values
    inside_extent = (
        2.0
        * (camera_positions - scene_center).norm(dim=-1).median().clamp_min(1e-6)
    ).reshape(1)
    return scene_center, inside_extent


def _camera_max_scene_bound(
    cameras: Sequence[_PaperCamera],
) -> tuple[Float[Tensor, " 3"], Float[Tensor, " 1"]]:
    camera_positions = _camera_positions(cameras)
    scene_center = camera_positions.mean(dim=0)
    inside_extent = (
        2.0 * (camera_positions - scene_center).norm(dim=-1).max().clamp_min(1e-6)
    ).reshape(1)
    return scene_center, inside_extent


def _paper_scene_bound(
    cameras: Sequence[_PaperCamera],
    *,
    mode: SceneBoundMode,
    forward_distance_scale: float,
    bound_scale: float,
) -> tuple[Float[Tensor, " 3"], Float[Tensor, " 1"]]:
    resolved_mode = mode
    if resolved_mode == "default":
        look_at = _camera_look_at(cameras)
        positions = _camera_positions(cameras)
        dot_products = (look_at[:, None, :] * look_at[None, :, :]).sum(dim=-1)
        displacement = positions[:, None, :] - positions[None, :, :]
        faces_same_direction = bool((dot_products.median() > 0.0).item())
        looks_toward_scene = bool(
            ((displacement * look_at[:, None, :]).sum(dim=-1).median() < 0.0).item()
        )
        resolved_mode = (
            "forward" if faces_same_direction and looks_toward_scene else "camera_median"
        )
    if resolved_mode == "forward":
        scene_center, inside_extent = _forward_scene_bound(
            cameras,
            forward_distance_scale=forward_distance_scale,
        )
    elif resolved_mode == "camera_max":
        scene_center, inside_extent = _camera_max_scene_bound(cameras)
    elif resolved_mode == "camera_median":
        scene_center, inside_extent = _camera_median_scene_bound(cameras)
    else:
        raise ValueError(f"Unsupported SVRaster bound mode: {mode!r}.")
    return scene_center, inside_extent * bound_scale


def _gen_dense_octpaths(
    *,
    outside_level: int,
    initial_inside_level: int,
    max_num_levels: int,
    device: torch.device,
) -> tuple[Int[Tensor, " num_voxels 1"], Int[Tensor, " num_voxels 1"]]:
    octants = torch.arange(8, dtype=torch.int64, device=device)
    octree_paths = octants << (3 * (max_num_levels - 1))
    for level_offset in range(outside_level):
        octree_paths |= (octants ^ 0b111) << (
            3 * (max_num_levels - (level_offset + 2))
        )
    if initial_inside_level > 1:
        dense_paths = torch.arange(
            (2 ** (initial_inside_level - 1)) ** 3,
            dtype=torch.int64,
            device=device,
        )
        dense_paths = dense_paths << (
            3
            * (
                max_num_levels
                - (outside_level + 1)
                - (initial_inside_level - 1)
            )
        )
        octree_paths = octree_paths.view(8, 1) | dense_paths
    octree_paths = octree_paths.reshape(-1, 1)
    octree_levels = torch.full_like(
        octree_paths,
        outside_level + initial_inside_level,
    )
    return octree_paths, octree_levels


def _gen_shell_octpaths(
    *,
    shell_level: int,
    initial_inside_level: int,
    max_num_levels: int,
    device: torch.device,
) -> tuple[Int[Tensor, " num_voxels 1"], Int[Tensor, " num_voxels 1"]]:
    octants = torch.arange(8, dtype=torch.int64, device=device)
    octree_paths = octants << (3 * (max_num_levels - 1))
    for level_offset in range(shell_level - 1):
        octree_paths |= (octants ^ 0b111) << (
            3 * (max_num_levels - (level_offset + 2))
        )
    octree_paths = octree_paths.view(8, 1) | (
        octants << (3 * (max_num_levels - shell_level - 1))
    )
    octree_paths = octree_paths[octants != (octants ^ 0b111).view(8, 1)]
    if initial_inside_level > 1:
        dense_paths = torch.arange(
            (2 ** (initial_inside_level - 1)) ** 3,
            dtype=torch.int64,
            device=device,
        )
        dense_paths = dense_paths << (
            3 * (max_num_levels - shell_level - initial_inside_level)
        )
        octree_paths = octree_paths.view(56, 1) | dense_paths
    octree_paths = octree_paths.reshape(-1, 1)
    octree_levels = torch.full_like(
        octree_paths,
        shell_level + initial_inside_level,
    )
    return octree_paths, octree_levels


def _gen_children(
    octree_paths: Int[Tensor, " num_voxels 1"],
    octree_levels: Int[Tensor, " num_voxels 1"],
    *,
    max_num_levels: int,
) -> tuple[Int[Tensor, " num_children 1"], Int[Tensor, " num_children 1"]]:
    child_levels = octree_levels.to(torch.int64) + 1
    child_ids = torch.arange(8, dtype=torch.int64, device=octree_paths.device)
    child_paths = octree_paths.to(torch.int64).view(-1, 1, 1) | (
        child_ids.view(1, 8, 1)
        << (3 * (max_num_levels - child_levels).view(-1, 1, 1))
    )
    return child_paths.reshape(-1, 1), child_levels.repeat_interleave(8, dim=0)


def _mark_max_sample_rate(
    cameras: Sequence[_PaperCamera],
    *,
    octree_paths: Int[Tensor, " num_voxels 1"],
    voxel_centers: Float[Tensor, " num_voxels 3"],
    voxel_lengths: Float[Tensor, " num_voxels 1"],
    near_plane: float,
) -> Float[Tensor, " num_voxels"]:
    from ember_native_svraster.core import runtime

    sample_rate = torch.zeros(
        (int(octree_paths.shape[0]),),
        dtype=torch.float32,
        device=octree_paths.device,
    )
    for camera in cameras:
        preprocess_result = runtime.preprocess(
            image_width=camera.image_width,
            image_height=camera.image_height,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            cx=camera.center_x,
            cy=camera.center_y,
            world_to_camera=camera.world_to_camera,
            camera_to_world=camera.camera_to_world,
            near=near_plane,
            octree_paths=octree_paths.reshape(-1),
            voxel_centers=voxel_centers,
            voxel_lengths=voxel_lengths.reshape(-1),
        )
        view_distance = ((voxel_centers - camera.position) * camera.look_at).sum(
            dim=-1
        )
        visible_indices = torch.where(
            (preprocess_result.n_duplicates > 0) & (view_distance > near_plane)
        )[0]
        if visible_indices.numel() == 0:
            continue
        sample_interval = (
            view_distance[visible_indices] * camera.pixel_size
        ).clamp_min(1e-12)
        view_sample_rate = voxel_lengths.reshape(-1)[visible_indices] / sample_interval
        sample_rate[visible_indices] = torch.maximum(
            sample_rate[visible_indices],
            view_sample_rate,
        )
    return sample_rate


def _filter_visible_octpaths(
    cameras: Sequence[_PaperCamera],
    *,
    octree_paths: Int[Tensor, " num_voxels 1"],
    octree_levels: Int[Tensor, " num_voxels 1"],
    scene_center: Float[Tensor, " 3"],
    scene_extent: Float[Tensor, " 1"],
    max_num_levels: int,
    near_plane: float,
) -> tuple[Int[Tensor, " visible_voxels 1"], Int[Tensor, " visible_voxels 1"]]:
    voxel_centers, voxel_lengths = svraster_octpath_decoding(
        octree_paths,
        octree_levels,
        scene_center,
        scene_extent,
        backend_name="new_cuda",
        max_num_levels=max_num_levels,
    )
    sample_rate = _mark_max_sample_rate(
        cameras,
        octree_paths=octree_paths,
        voxel_centers=voxel_centers,
        voxel_lengths=voxel_lengths,
        near_plane=near_plane,
    )
    keep_indices = torch.where(sample_rate > 0.0)[0]
    return octree_paths[keep_indices], octree_levels[keep_indices]


def _outside_octpaths(
    cameras: Sequence[_PaperCamera],
    *,
    scene_center: Float[Tensor, " 3"],
    scene_extent: Float[Tensor, " 1"],
    outside_level: int,
    minimum_voxels: int,
    max_level: int,
    max_num_levels: int,
    near_plane: float,
    device: torch.device,
) -> tuple[Int[Tensor, " num_voxels 1"], Int[Tensor, " num_voxels 1"]]:
    if outside_level == 0 or minimum_voxels <= 0:
        empty = torch.empty((0, 1), dtype=torch.int64, device=device)
        return empty, empty
    shell_paths: list[Tensor] = []
    shell_levels: list[Tensor] = []
    for shell_level in range(1, outside_level + 1):
        octree_paths, octree_levels = _gen_shell_octpaths(
            shell_level=shell_level,
            initial_inside_level=1,
            max_num_levels=max_num_levels,
            device=device,
        )
        shell_paths.append(octree_paths)
        shell_levels.append(octree_levels)
    octree_paths = torch.cat(shell_paths, dim=0)
    octree_levels = torch.cat(shell_levels, dim=0)
    while True:
        voxel_centers, voxel_lengths = svraster_octpath_decoding(
            octree_paths,
            octree_levels,
            scene_center,
            scene_extent,
            backend_name="new_cuda",
            max_num_levels=max_num_levels,
        )
        sample_rate = _mark_max_sample_rate(
            cameras,
            octree_paths=octree_paths,
            voxel_centers=voxel_centers,
            voxel_lengths=voxel_lengths,
            near_plane=near_plane,
        )
        visible_indices = torch.where(sample_rate > 0.0)[0]
        octree_paths = octree_paths[visible_indices]
        octree_levels = octree_levels[visible_indices]
        sample_rate = sample_rate[visible_indices]
        if int(octree_paths.shape[0]) == 0:
            break
        can_subdivide = octree_levels.reshape(-1) < max_level
        needed_parent_count = min(
            int(octree_paths.shape[0]),
            round((minimum_voxels - int(octree_paths.shape[0])) / 7),
        )
        if needed_parent_count <= 0 or int(can_subdivide.sum()) == 0:
            break
        ranked_sample_rate = sample_rate * can_subdivide.to(sample_rate.dtype)
        threshold = ranked_sample_rate.sort().values[-needed_parent_count]
        subdivide_mask = ranked_sample_rate >= threshold
        subdivide_mask &= can_subdivide
        if int(subdivide_mask.sum()) > 0:
            quantile_threshold = ranked_sample_rate[subdivide_mask].quantile(0.9)
            subdivide_mask &= ranked_sample_rate >= quantile_threshold
        if int(subdivide_mask.sum()) == 0:
            break
        child_paths, child_levels = _gen_children(
            octree_paths[subdivide_mask],
            octree_levels[subdivide_mask],
            max_num_levels=max_num_levels,
        )
        octree_paths = torch.cat([octree_paths[~subdivide_mask], child_paths], dim=0)
        octree_levels = torch.cat([octree_levels[~subdivide_mask], child_levels], dim=0)
    return _filter_visible_octpaths(
        cameras,
        octree_paths=octree_paths,
        octree_levels=octree_levels,
        scene_center=scene_center,
        scene_extent=scene_extent,
        max_num_levels=max_num_levels,
        near_plane=near_plane,
    )


def initialize_svraster_paper_scene(
    scene_record: SceneRecord,
    *,
    modules: dict[str, torch.nn.Module] | None = None,
    parameters: dict[str, torch.nn.Parameter] | None = None,
    buffers: dict[str, Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
    frame_dataset: PreparedFrameDataset | None = None,
    device: torch.device | None = None,
    backend_name: Literal["new_cuda"] = "new_cuda",
    max_num_levels: int = DEFAULT_SVRASTER_MAX_NUM_LEVELS,
    sh_degree: int = 3,
    initial_sh_degree: int = 3,
    initial_inside_level: int = 6,
    outside_level: int = 5,
    initial_outside_ratio: float = 2.0,
    geometry_initial_value: float = -10.0,
    sh0_initial_rgb: float = 0.5,
    shs_initial_value: float = 0.0,
    bound_mode: SceneBoundMode = "default",
    bound_scale: float = 1.0,
    forward_distance_scale: float = 1.0,
    near_plane: float = 0.02,
    filter_zero_visibility: bool = True,
) -> InitializedModel:
    """Initialize SVRaster's foreground grid and background shell from cameras."""
    if backend_name != "new_cuda":
        raise ValueError("SVRaster paper initialization currently supports new_cuda.")
    target_device = device or torch.device("cpu")
    cameras = _training_cameras(
        scene_record,
        frame_dataset,
        device=target_device,
    )
    scene_center, inside_extent = _paper_scene_bound(
        cameras,
        mode=bound_mode,
        forward_distance_scale=forward_distance_scale,
        bound_scale=bound_scale,
    )
    scene_extent = inside_extent * float(2**outside_level)
    inside_paths, inside_levels = _gen_dense_octpaths(
        outside_level=outside_level,
        initial_inside_level=initial_inside_level,
        max_num_levels=max_num_levels,
        device=target_device,
    )
    if filter_zero_visibility:
        inside_paths, inside_levels = _filter_visible_octpaths(
            cameras,
            octree_paths=inside_paths,
            octree_levels=inside_levels,
            scene_center=scene_center,
            scene_extent=scene_extent,
            max_num_levels=max_num_levels,
            near_plane=near_plane,
        )
    outside_paths, outside_levels = _outside_octpaths(
        cameras,
        scene_center=scene_center,
        scene_extent=scene_extent,
        outside_level=outside_level,
        minimum_voxels=round(int(inside_paths.shape[0]) * initial_outside_ratio),
        max_level=min(outside_level + initial_inside_level, max_num_levels),
        max_num_levels=max_num_levels,
        near_plane=near_plane,
        device=target_device,
    )
    octree_layout = torch.cat(
        [
            torch.cat([inside_paths, inside_levels], dim=1),
            torch.cat([outside_paths, outside_levels], dim=1),
        ],
        dim=0,
    ).unique(dim=0, sorted=True)
    octree_paths = octree_layout[:, :1]
    octree_levels = octree_layout[:, 1:]
    _grid_points_key, voxel_keys = svraster_build_grid_points_link(
        octree_paths,
        octree_levels,
        backend_name=None,
        max_num_levels=max_num_levels,
    )
    num_grid_points = int(voxel_keys.max().item()) + 1 if voxel_keys.numel() else 0
    num_voxels = int(octree_paths.shape[0])
    sh_coefficients = (sh_degree + 1) ** 2 - 1
    scene = SparseVoxelScene(
        backend_name=backend_name,
        active_sh_degree=min(initial_sh_degree, sh_degree),
        max_num_levels=max_num_levels,
        scene_center=scene_center,
        scene_extent=scene_extent.reshape(1),
        inside_extent=inside_extent.reshape(1),
        octpath=octree_paths,
        octlevel=octree_levels,
        geo_grid_pts=torch.full(
            (num_grid_points, 1),
            geometry_initial_value,
            dtype=torch.float32,
            device=target_device,
        ).requires_grad_(True),
        sh0=svraster_rgb_to_sh_zero(
            torch.full(
                (num_voxels, 3),
                sh0_initial_rgb,
                dtype=torch.float32,
                device=target_device,
            )
        ).requires_grad_(True),
        shs=torch.full(
            (num_voxels, sh_coefficients, 3),
            shs_initial_value,
            dtype=torch.float32,
            device=target_device,
        ).requires_grad_(True),
        subdivision_priority=torch.ones(
            (num_voxels, 1),
            dtype=torch.float32,
            device=target_device,
        ).requires_grad_(True),
    )
    return InitializedModel(
        scene=scene,
        modules=dict(modules or {}),
        parameters=dict(parameters or {}),
        buffers=dict(buffers or {}),
        metadata=dict(metadata or {}),
    )


def svraster_paper_training_backend_options(
    state: TrainState | None = None,
    batch: Any | None = None,
    *,
    distortion_weight: float = 0.1,
    distortion_start_step: int = 10000,
    ascending_weight: float = 0.0,
    ascending_start_step: int = 0,
    color_concentration_weight: float = 0.01,
    ss_aug_max: float = 1.5,
    ss_aug_start_step: int = 1000,
) -> dict[str, Any]:
    """Return per-step SVRaster paper backend options."""
    step = 0 if state is None else int(state.step)
    updates: dict[str, Any] = {
        "distortion_weight": (
            distortion_weight if step >= distortion_start_step else 0.0
        ),
        "ascending_weight": (
            ascending_weight if step >= ascending_start_step else 0.0
        ),
        "color_concentration_weight": color_concentration_weight,
    }
    if ss_aug_max > 1.0 and step > ss_aug_start_step:
        rng = random.Random((0 if state is None else int(state.seed)) + step)
        updates["supersampling"] = rng.uniform(1.0, ss_aug_max)
    else:
        updates["supersampling"] = 1.0
    if batch is not None and color_concentration_weight > 0.0:
        updates["ground_truth_color"] = batch.images
    return updates


def _transmittance_concentration_loss(
    transmittance: Tensor,
) -> Tensor:
    return (transmittance.square() * (1.0 - transmittance).square()).mean()


def svraster_paper_rgb_loss(
    state: TrainState,
    batch: Any,
    render_output: Any,
    *,
    lambda_photo: float = 1.0,
    lambda_ssim: float = 0.02,
    use_l1: bool = False,
    use_huber: bool = False,
    huber_threshold: float = 0.03,
    lambda_t_concentration: float = 0.0,
    lambda_t_inside: float = 0.0,
    ssim_backend: str = "cuda",
    weights: dict[str, float] | None = None,
) -> LossResult:
    """Compute the core SVRaster paper RGB reconstruction loss."""
    del state, weights
    prediction = render_output.render
    target = batch.images
    if use_l1:
        photo_loss = (prediction - target).abs().mean()
    elif use_huber:
        photo_loss = torch_functional.huber_loss(
            prediction,
            target,
            delta=huber_threshold,
        )
    else:
        photo_loss = torch_functional.mse_loss(prediction, target)
    loss = lambda_photo * photo_loss
    metrics = {"photo_loss": float(photo_loss.detach().item())}
    if lambda_ssim > 0.0:
        from ember_splatting_training.losses import ssim_score

        ssim_loss = 1.0 - ssim_score(
            prediction,
            target,
            backend=ssim_backend,
        )
        loss = loss + lambda_ssim * ssim_loss
        metrics["ssim_loss"] = float(ssim_loss.detach().item())
    raw_transmittance = getattr(
        render_output,
        "raw_transmittance",
        getattr(render_output, "transmittance", None),
    )
    if raw_transmittance is not None and lambda_t_concentration > 0.0:
        concentration_loss = _transmittance_concentration_loss(raw_transmittance)
        loss = loss + lambda_t_concentration * concentration_loss
        metrics["transmittance_concentration_loss"] = float(
            concentration_loss.detach().item()
        )
    if raw_transmittance is not None and lambda_t_inside > 0.0:
        inside_loss = raw_transmittance.square().mean()
        loss = loss + lambda_t_inside * inside_loss
        metrics["transmittance_inside_loss"] = float(inside_loss.detach().item())
    return LossResult(loss=loss, metrics=metrics)


__all__ = [
    "initialize_svraster_paper_scene",
    "svraster_paper_rgb_loss",
    "svraster_paper_training_backend_options",
]
