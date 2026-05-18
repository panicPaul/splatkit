"""RADFOAM backend contract surface."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from beartype import beartype
from ember_core.core.contracts import (
    CameraState,
    RadFoamScene,
    RenderOptions,
    RenderOutput,
)
from ember_core.core.registry import output_set, register_backend
from jaxtyping import Float
from torch import Tensor

from ember_native_radfoam.radfoam.runtime import (
    build_aabb_tree,
    nearest_neighbor,
    trace,
)

_SUPPORTED_OUTPUTS = output_set("alpha", "depth")


@beartype
@dataclass(frozen=True, kw_only=True)
class RadFoamNativeRenderOutput(RenderOutput):
    """RADFOAM render output."""

    alphas: Float[Tensor, "num_cams height width"] | None = None


@beartype
@dataclass(frozen=True, kw_only=True)
class RadFoamAlphaDepthRenderOutput(RadFoamNativeRenderOutput):
    """RADFOAM render output with depth."""

    depth: Float[Tensor, "num_cams height width"]


@beartype
@dataclass(frozen=True)
class RadFoamNativeRenderOptions(RenderOptions):
    """RADFOAM-specific render configuration."""

    weight_threshold: float = 0.001
    max_intersections: int = 1024
    density_beta: float = 10.0
    return_contribution: bool = False
    depth_quantiles: tuple[float, ...] = (0.5,)
    clamp_output: bool = True


def _validate_inputs(scene: RadFoamScene, camera: CameraState) -> None:
    if not scene.primal_points.is_cuda:
        raise ValueError("RADFOAM native rendering requires a CUDA scene.")
    if camera.camera_convention != "opencv":
        raise ValueError(
            "RADFOAM expects opencv cameras; got "
            f"{camera.camera_convention!r}."
        )
    if scene.primal_points.dtype != torch.float32:
        raise ValueError(
            "RADFOAM expects float32 primal points; got "
            f"{scene.primal_points.dtype}."
        )
    if len(set(camera.width.detach().cpu().tolist())) != 1:
        raise ValueError("RADFOAM batched rendering requires equal widths.")
    if len(set(camera.height.detach().cpu().tolist())) != 1:
        raise ValueError("RADFOAM batched rendering requires equal heights.")


def _camera_rays(
    camera: CameraState,
    camera_index: int,
    *,
    device: torch.device,
) -> Float[Tensor, "height width 6"]:
    intrinsics = camera.get_intrinsics()[camera_index].to(
        device=device,
        dtype=torch.float32,
    )
    cam_to_world = camera.cam_to_world[camera_index].to(
        device=device,
        dtype=torch.float32,
    )
    width = int(camera.width[camera_index].item())
    height = int(camera.height[camera_index].item())
    xs = torch.arange(width, dtype=torch.float32, device=device) + 0.5
    ys = torch.arange(height, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dirs = torch.stack(
        [
            (xx - intrinsics[0, 2]) / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(xx),
        ],
        dim=-1,
    )
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    world_dirs = torch.einsum("hwj,ij->hwi", dirs, cam_to_world[:3, :3])
    origins = cam_to_world[:3, 3].expand_as(world_dirs)
    return torch.cat([origins, world_dirs], dim=-1).contiguous()


def _radfoam_attributes(
    scene: RadFoamScene,
    options: RadFoamNativeRenderOptions,
) -> Float[Tensor, "num_points attributes"]:
    density = scene.activation_scale * F.softplus(
        scene.density,
        beta=options.density_beta,
    )
    return torch.cat([scene.att_dc, scene.att_sh, density], dim=-1).to(
        scene.att_dc.dtype
    )


def _depth_quantiles(
    rays: Tensor,
    options: RadFoamNativeRenderOptions,
    *,
    return_depth: bool,
) -> Float[Tensor, "height width quantiles"] | None:
    if not return_depth:
        return None
    values = torch.tensor(
        options.depth_quantiles,
        dtype=torch.float32,
        device=rays.device,
    )
    return values.expand(*rays.shape[:-1], values.shape[0]).contiguous()


@beartype
def render_radfoam_native(
    scene: RadFoamScene,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: RadFoamNativeRenderOptions | None = None,
) -> RadFoamNativeRenderOutput:
    """Render a scene with the forked RADFOAM native extension."""
    if return_gaussian_impact_score:
        raise ValueError(
            "The RADFOAM backend does not expose Gaussian impact scores."
        )
    if return_normals:
        raise ValueError("The RADFOAM backend does not expose normals.")
    if return_2d_projections:
        raise ValueError(
            "The RADFOAM backend does not expose 2D Gaussian projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The RADFOAM backend does not expose projective intersection "
            "transforms."
        )

    options = options or RadFoamNativeRenderOptions()
    _validate_inputs(scene, camera)
    attributes = _radfoam_attributes(scene, options)
    aabb_tree = build_aabb_tree(scene.primal_points)
    renders: list[Float[Tensor, "height width 3"]] = []
    alphas: list[Float[Tensor, "height width"]] = []
    depths: list[Float[Tensor, "height width"]] = []

    background = options.background_color.to(
        device=scene.primal_points.device,
        dtype=attributes.dtype,
    )
    for camera_index in range(camera.cam_to_world.shape[0]):
        rays = _camera_rays(
            camera,
            camera_index,
            device=scene.primal_points.device,
        )
        camera_start = nearest_neighbor(
            scene.primal_points,
            aabb_tree,
            rays[..., :3].reshape(-1, 3)[:1],
        ).reshape(())
        start_point = camera_start.expand(rays.shape[:-1]).contiguous()
        result = trace(
            scene.primal_points,
            attributes,
            scene.point_adjacency,
            scene.point_adjacency_offsets,
            rays,
            start_point,
            depth_quantiles=_depth_quantiles(
                rays,
                options,
                return_depth=return_depth,
            ),
            return_contribution=options.return_contribution,
            sh_degree=scene.sh_degree,
            weight_threshold=options.weight_threshold,
            max_intersections=options.max_intersections,
        )
        alpha = result.rgba[..., 3].to(attributes.dtype)
        image = result.rgba[..., :3] + (1.0 - alpha[..., None]) * background
        if options.clamp_output:
            image = image.clamp(0.0, 1.0)
        renders.append(image.contiguous())
        if return_alpha:
            alphas.append(alpha.contiguous())
        if return_depth:
            depths.append(result.depth[..., 0].contiguous())

    render = torch.stack(renders, dim=0)
    alpha_tensor = torch.stack(alphas, dim=0) if return_alpha else None
    if return_depth:
        return RadFoamAlphaDepthRenderOutput(
            render=render,
            alphas=alpha_tensor,
            depth=torch.stack(depths, dim=0),
        )
    return RadFoamNativeRenderOutput(render=render, alphas=alpha_tensor)


def register() -> None:
    """Register the native RADFOAM backend."""
    register_backend(
        name="radfoam.core",
        default_options=RadFoamNativeRenderOptions(),
        accepted_scene_types=(RadFoamScene,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_radfoam_native)


__all__ = [
    "RadFoamAlphaDepthRenderOutput",
    "RadFoamNativeRenderOptions",
    "RadFoamNativeRenderOutput",
    "register",
    "render_radfoam_native",
]
