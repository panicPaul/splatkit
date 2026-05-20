"""PowerFoam backend contract surface."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

import torch
import warp as wp
from beartype import beartype
from ember_core.core.contracts import (
    CameraState,
    PowerFoamScene,
    RenderOptions,
    RenderOutput,
)
from ember_core.core.registry import output_set, register_backend
from jaxtyping import Float
from torch import Tensor

from ember_native_powerfoam.powerfoam.runtime import (
    powerfoam_att_sv,
    powerfoam_camera_from_camera_state,
    powerfoam_density,
    powerfoam_normals,
    powerfoam_radii,
    powerfoam_texel_world_sites,
)
from ember_native_powerfoam.powerfoam.runtime.ops import (
    rasterize_powerfoam,
    spherical_voronoi_colors,
)

_SUPPORTED_OUTPUTS = output_set("alpha", "depth", "normals")


@beartype
@dataclass(frozen=True, kw_only=True)
class PowerFoamNativeRenderOutput(RenderOutput):
    """PowerFoam render output."""

    alphas: Float[Tensor, "num_cams height width"] | None = None
    depth: Float[Tensor, "num_cams height width"] | None = None
    normals: Float[Tensor, "num_cams height width 3"] | None = None
    normal_error: Float[Tensor, "num_cams height width"] | None = None
    contrib: Float[Tensor, "num_cams num_points"] | None = None
    point_error: Float[Tensor, "num_cams num_points"] | None = None
    prim_visible_mask: Tensor | None = None


@beartype
@dataclass(frozen=True)
class PowerFoamNativeRenderOptions(RenderOptions):
    """PowerFoam-specific render configuration."""

    render_objective: Literal["volume", "surface"] | None = None
    attr_dtype: Literal["float", "half"] | None = None
    density_beta: float = 100.0
    radii_beta: float = 100.0
    return_point_err: bool = False
    ray_gt: Tensor | None = None
    depth_quantiles: tuple[float, ...] = (0.5,)
    disable_coop_prim_load: bool = False
    disable_coop_adj_load: bool = False
    is_pinhole: bool = True
    clamp_output: bool = True


def _validate_inputs(scene: PowerFoamScene, camera: CameraState) -> None:
    if not scene.points.is_cuda:
        raise ValueError("PowerFoam native rendering requires a CUDA scene.")
    if camera.camera_convention != "opencv":
        raise ValueError(
            "PowerFoam expects opencv cameras; got "
            f"{camera.camera_convention!r}."
        )
    if scene.points.dtype != torch.float32:
        raise ValueError(
            f"PowerFoam expects float32 points; got {scene.points.dtype}."
        )
    if len(set(camera.width.detach().cpu().tolist())) != 1:
        raise ValueError("PowerFoam batched rendering requires equal widths.")
    if len(set(camera.height.detach().cpu().tolist())) != 1:
        raise ValueError("PowerFoam batched rendering requires equal heights.")


def _powerfoam_args(
    scene: PowerFoamScene,
    options: PowerFoamNativeRenderOptions,
) -> SimpleNamespace:
    return SimpleNamespace(
        render_objective=options.render_objective or scene.render_objective,
        num_texel_sites=scene.num_texel_sites,
        sv_dof=scene.sv_dof,
        disable_coop_prim_load=options.disable_coop_prim_load,
        disable_coop_adj_load=options.disable_coop_adj_load,
        is_pinhole=options.is_pinhole,
    )


def _depth_quantiles(
    camera: CameraState,
    camera_index: int,
    options: PowerFoamNativeRenderOptions,
    *,
    device: torch.device,
    return_depth: bool,
) -> Float[Tensor, "height width quantiles"] | None:
    if not return_depth:
        return None
    values = torch.tensor(
        options.depth_quantiles,
        dtype=torch.float32,
        device=device,
    )
    height = int(camera.height[camera_index].item())
    width = int(camera.width[camera_index].item())
    return values.expand(height, width, values.shape[0]).contiguous()


def _render_single_powerfoam(
    scene: PowerFoamScene,
    camera: CameraState,
    camera_index: int,
    *,
    options: PowerFoamNativeRenderOptions,
    return_depth: bool,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
    Tensor,
    Tensor,
    Tensor | None,
    Tensor,
]:
    device = scene.points.device
    args = _powerfoam_args(scene, options)
    attr_dtype = options.attr_dtype or scene.attr_dtype
    powerfoam_camera = powerfoam_camera_from_camera_state(
        camera,
        camera_index,
        device=device,
        build_ray_maps=not args.is_pinhole,
    )
    radii = powerfoam_radii(scene, beta=options.radii_beta)
    density = powerfoam_density(scene, beta=options.density_beta)
    normals = powerfoam_normals(scene)
    texel_sites = powerfoam_texel_world_sites(scene, radii)
    att_sites, att_values, att_temps = powerfoam_att_sv(scene)
    wp.init()
    texel_rgb = spherical_voronoi_colors(
        texel_sites.view(-1, 3).detach(),
        powerfoam_camera,
        att_sites,
        att_values,
        att_temps,
        sv_dof=scene.sv_dof,
        attr_dtype=attr_dtype,
    )
    texel_rgb = texel_rgb.view(scene.points.shape[0], scene.num_texel_sites, 3)
    texel_height = scene.texel_height * radii[:, None]
    ray_gt = options.ray_gt
    if ray_gt is not None and ray_gt.ndim == 4:
        ray_gt = ray_gt[camera_index]
    if ray_gt is not None:
        ray_gt = ray_gt.to(device=device, dtype=scene.points.dtype)
    return_point_err = options.return_point_err or ray_gt is not None
    result = rasterize_powerfoam(
        powerfoam_camera,
        _depth_quantiles(
            camera,
            camera_index,
            options,
            device=device,
            return_depth=return_depth,
        ),
        scene.points,
        radii,
        density,
        normals,
        texel_sites,
        texel_rgb,
        texel_height,
        scene.adjacency.to(torch.int32).contiguous(),
        scene.adjacency_offsets.to(torch.int32).contiguous(),
        ray_gt,
        return_point_err,
        render_objective=args.render_objective,
        num_texel_sites=args.num_texel_sites,
        sv_dof=args.sv_dof,
        disable_coop_prim_load=args.disable_coop_prim_load,
        disable_coop_adj_load=args.disable_coop_adj_load,
        is_pinhole=args.is_pinhole,
        attr_dtype=attr_dtype,
    )
    color = result[0]
    alpha = result[1]
    normal_error = result[2]
    normal = result[3]
    depth = result[4][..., 0] if return_depth else None
    contrib = result[6]
    point_error = result[7]
    prim_visible_mask = result[8]
    background = options.background_color.to(
        device=device,
        dtype=color.dtype,
    )
    image = color + (1.0 - alpha[..., None]) * background
    if options.clamp_output:
        image = image.clamp(0.0, 1.0)
    return (
        image.contiguous(),
        alpha.contiguous(),
        normal.contiguous(),
        depth,
        normal_error.contiguous(),
        contrib.contiguous(),
        point_error.contiguous() if point_error is not None else None,
        prim_visible_mask.contiguous(),
    )


@beartype
def render_powerfoam_native(
    scene: PowerFoamScene,
    camera: CameraState,
    *,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_gaussian_impact_score: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: PowerFoamNativeRenderOptions | None = None,
) -> PowerFoamNativeRenderOutput:
    """Render a scene with the forked PowerFoam Warp implementation."""
    if return_gaussian_impact_score:
        raise ValueError(
            "The PowerFoam backend does not expose Gaussian impact scores."
        )
    if return_2d_projections:
        raise ValueError(
            "The PowerFoam backend does not expose 2D Gaussian projections."
        )
    if return_projective_intersection_transforms:
        raise ValueError(
            "The PowerFoam backend does not expose projective intersection "
            "transforms."
        )
    options = options or PowerFoamNativeRenderOptions()
    _validate_inputs(scene, camera)
    renders: list[Tensor] = []
    alphas: list[Tensor] = []
    normals: list[Tensor] = []
    depths: list[Tensor] = []
    normal_errors: list[Tensor] = []
    contribs: list[Tensor] = []
    point_errors: list[Tensor] = []
    prim_visible_masks: list[Tensor] = []
    for camera_index in range(camera.cam_to_world.shape[0]):
        (
            image,
            alpha,
            normal,
            depth,
            normal_error,
            contrib,
            point_error,
            prim_visible_mask,
        ) = _render_single_powerfoam(
            scene,
            camera,
            camera_index,
            options=options,
            return_depth=return_depth,
        )
        renders.append(image)
        normal_errors.append(normal_error)
        contribs.append(contrib)
        if point_error is not None:
            point_errors.append(point_error)
        prim_visible_masks.append(prim_visible_mask)
        if return_alpha:
            alphas.append(alpha)
        if return_normals:
            normals.append(normal)
        if return_depth and depth is not None:
            depths.append(depth.contiguous())
    return PowerFoamNativeRenderOutput(
        render=torch.stack(renders, dim=0),
        alphas=torch.stack(alphas, dim=0) if return_alpha else None,
        depth=torch.stack(depths, dim=0) if return_depth else None,
        normals=torch.stack(normals, dim=0) if return_normals else None,
        normal_error=torch.stack(normal_errors, dim=0),
        contrib=torch.stack(contribs, dim=0),
        point_error=torch.stack(point_errors, dim=0) if point_errors else None,
        prim_visible_mask=torch.stack(prim_visible_masks, dim=0),
    )


def register() -> None:
    """Register the native PowerFoam backend."""
    register_backend(
        name="powerfoam.rasterize",
        default_options=PowerFoamNativeRenderOptions(),
        accepted_scene_types=(PowerFoamScene,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_powerfoam_native)


__all__ = [
    "PowerFoamNativeRenderOptions",
    "PowerFoamNativeRenderOutput",
    "register",
    "render_powerfoam_native",
]
