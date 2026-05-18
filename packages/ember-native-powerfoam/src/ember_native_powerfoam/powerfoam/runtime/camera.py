"""Camera conversion stages for the PowerFoam runtime."""

from __future__ import annotations

import torch
from ember_core.core.contracts import CameraState
from jaxtyping import Float
from torch import Tensor

from ember_native_powerfoam.powerfoam.native.warp.camera import TorchCamera


def powerfoam_ray_maps(
    camera: CameraState,
    camera_index: int = 0,
    *,
    device: torch.device | None = None,
) -> Float[Tensor, "height width 6"]:
    """Build PowerFoam world-space ray maps from an Ember camera."""
    target_device = device or camera.cam_to_world.device
    intrinsics = camera.get_intrinsics()[camera_index].to(
        device=target_device,
        dtype=torch.float32,
    )
    cam_to_world = camera.cam_to_world[camera_index].to(
        device=target_device,
        dtype=torch.float32,
    )
    width = int(camera.width[camera_index].item())
    height = int(camera.height[camera_index].item())
    xs = torch.arange(width, dtype=torch.float32, device=target_device) + 0.5
    ys = torch.arange(height, dtype=torch.float32, device=target_device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    directions = torch.stack(
        [
            (xx - intrinsics[0, 2]) / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(xx),
        ],
        dim=-1,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    world_dirs = torch.einsum("hwj,ij->hwi", directions, cam_to_world[:3, :3])
    origins = cam_to_world[:3, 3].expand_as(world_dirs)
    return torch.cat([origins, world_dirs], dim=-1).contiguous()


def powerfoam_camera_from_camera_state(
    camera: CameraState,
    camera_index: int = 0,
    *,
    device: torch.device | None = None,
    fov_cos_cutoff: float | None = None,
) -> TorchCamera:
    """Convert a single Ember camera entry to upstream PowerFoam TorchCamera."""
    if camera.camera_convention != "opencv":
        raise ValueError(
            "PowerFoam expects opencv cameras; got "
            f"{camera.camera_convention!r}."
        )
    target_device = device or camera.cam_to_world.device
    intrinsics = camera.get_intrinsics()[camera_index].to(
        device=target_device,
        dtype=torch.float32,
    )
    cam_to_world = camera.cam_to_world[camera_index].to(
        device=target_device,
        dtype=torch.float32,
    )
    width = int(camera.width[camera_index].item())
    height = int(camera.height[camera_index].item())
    right_extent = (float(width) - 1.0) * 0.5 / float(intrinsics[0, 0])
    up_extent = (float(height) - 1.0) * 0.5 / float(intrinsics[1, 1])
    right = cam_to_world[:3, 0] * right_extent
    up = -cam_to_world[:3, 1] * up_extent
    return TorchCamera(
        eye=cam_to_world[:3, 3],
        right=right,
        up=up,
        width=width,
        height=height,
        ray_maps=powerfoam_ray_maps(
            camera,
            camera_index,
            device=target_device,
        ),
        fov_cos_cutoff=fov_cos_cutoff,
    )
