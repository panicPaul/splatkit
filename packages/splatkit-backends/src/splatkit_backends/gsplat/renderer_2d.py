"""2D gsplat renderer implementation."""

from typing import Any, Literal, overload

import torch
from beartype import beartype
from gsplat import rasterization_2dgs
from splatkit.core.contracts import CameraState, GaussianScene2D

from splatkit_backends.gsplat.shared import (
    GsplatAlphaIntersectionRenderOutput,
    GsplatAlphaRenderOutput,
    GsplatIntersectionRenderOutput,
    GsplatRenderOptions,
    GsplatRenderOutput,
    build_backgrounds,
    validate_uniform_camera_shape,
)


@beartype
def _build_rasterization_2dgs_kwargs(
    scene: GaussianScene2D,
    camera: CameraState,
    options: GsplatRenderOptions,
) -> dict[str, Any]:
    """Build shared 2DGS rasterization kwargs."""
    num_splats = int(scene.log_scales.shape[0])
    padded_scales = torch.cat(
        (
            torch.exp(scene.log_scales),
            torch.zeros(
                (num_splats, 1),
                device=scene.log_scales.device,
                dtype=scene.log_scales.dtype,
            ),
        ),
        dim=-1,
    )
    return {
        "means": scene.center_position,
        "quats": scene.quaternion_orientation,
        "scales": padded_scales,
        "opacities": torch.sigmoid(scene.logit_opacity),
        "colors": scene.feature,
        "viewmats": torch.linalg.inv(camera.cam_to_world),
        "Ks": camera.get_intrinsics(),
        "width": int(camera.width[0].item()),
        "height": int(camera.height[0].item()),
        "sh_degree": scene.sh_degree,
        "backgrounds": build_backgrounds(scene, camera, options),
        "packed": options.packed,
        "eps2d": options.eps_2d,
        "sparse_grad": options.sparse_grad,
        "absgrad": options.absgrad,
    }


@overload
def render_gsplat_2dgs(
    scene: GaussianScene2D,
    camera: CameraState,
    *,
    return_alpha: Literal[True] = True,
    return_depth: Literal[False] = False,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[False] = False,
    options: GsplatRenderOptions | None = None,
) -> GsplatAlphaRenderOutput: ...


@overload
def render_gsplat_2dgs(
    scene: GaussianScene2D,
    camera: CameraState,
    *,
    return_alpha: Literal[True] = True,
    return_depth: Literal[True] = True,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[False] = False,
    options: GsplatRenderOptions | None = None,
) -> GsplatRenderOutput: ...


@overload
def render_gsplat_2dgs(
    scene: GaussianScene2D,
    camera: CameraState,
    *,
    return_alpha: Literal[True] = True,
    return_depth: Literal[False] = False,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[True] = True,
    options: GsplatRenderOptions | None = None,
) -> GsplatAlphaIntersectionRenderOutput: ...


@overload
def render_gsplat_2dgs(
    scene: GaussianScene2D,
    camera: CameraState,
    *,
    return_alpha: Literal[True] = True,
    return_depth: Literal[True] = True,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    return_projective_intersection_transforms: Literal[True] = True,
    options: GsplatRenderOptions | None = None,
) -> GsplatIntersectionRenderOutput: ...


@beartype
def render_gsplat_2dgs(
    scene: GaussianScene2D,
    camera: CameraState,
    *,
    return_alpha: bool = True,
    return_depth: bool = False,
    return_normals: bool = False,
    return_2d_projections: bool = False,
    return_projective_intersection_transforms: bool = False,
    options: GsplatRenderOptions | None = None,
) -> (
    GsplatAlphaRenderOutput
    | GsplatRenderOutput
    | GsplatAlphaIntersectionRenderOutput
    | GsplatIntersectionRenderOutput
):
    """Render a 2D Gaussian scene with gsplat's 2DGS rasterizer."""
    del return_alpha

    if camera.camera_convention != "opencv":
        raise ValueError(
            "gsplat_2dgs currently expects cameras in opencv convention; got "
            f"{camera.camera_convention!r}."
        )
    if return_normals:
        raise ValueError("gsplat_2dgs normals are not exposed yet.")
    if return_2d_projections:
        raise ValueError(
            "gsplat_2dgs does not expose 2D projection outputs in the "
            "shared splatkit format."
        )

    options = options or GsplatRenderOptions()
    if options.rasterize_mode != "classic":
        raise ValueError(
            "gsplat_2dgs only supports "
            "GsplatRenderOptions(rasterize_mode='classic')."
        )
    validate_uniform_camera_shape(camera, "gsplat_2dgs")

    render_mode: Literal["RGB", "RGB+D", "RGB+ED"]
    if return_depth:
        if options.depth_render_mode == "D":
            render_mode = "RGB+D"
        else:
            render_mode = "RGB+ED"
    else:
        render_mode = "RGB"

    render_colors, render_alphas, *_rest, metadata = rasterization_2dgs(
        **_build_rasterization_2dgs_kwargs(scene, camera, options),
        render_mode=render_mode,
    )
    projected_means = metadata["means2d"]
    projective_intersection_transforms = metadata["ray_transforms"]
    if return_projective_intersection_transforms and return_depth:
        return GsplatIntersectionRenderOutput(
            render=render_colors[..., :3],
            alphas=render_alphas.squeeze(-1),
            depth=render_colors[..., 3],
            projected_means=projected_means,
            projective_intersection_transforms=(
                projective_intersection_transforms
            ),
        )
    if return_projective_intersection_transforms:
        return GsplatAlphaIntersectionRenderOutput(
            render=render_colors,
            alphas=render_alphas.squeeze(-1),
            projected_means=projected_means,
            projective_intersection_transforms=(
                projective_intersection_transforms
            ),
        )
    if return_depth:
        return GsplatRenderOutput(
            render=render_colors[..., :3],
            alphas=render_alphas.squeeze(-1),
            depth=render_colors[..., 3],
        )
    return GsplatAlphaRenderOutput(
        render=render_colors,
        alphas=render_alphas.squeeze(-1),
    )
