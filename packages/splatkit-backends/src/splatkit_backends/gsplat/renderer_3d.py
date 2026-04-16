"""3D gsplat renderer implementation."""

from typing import Any, Literal, overload

import torch
from beartype import beartype
from gsplat import rasterization
from jaxtyping import Float
from splatkit.core.contracts import CameraState, GaussianScene3D
from torch import Tensor

from splatkit_backends.gsplat.shared import (
    GsplatAlphaProjectionRenderOutput,
    GsplatAlphaRenderOutput,
    GsplatProjectionRenderOutput,
    GsplatRenderOptions,
    GsplatRenderOutput,
    build_backgrounds,
    validate_uniform_camera_shape,
)


@beartype
def _build_rasterization_kwargs(
    scene: GaussianScene3D,
    camera: CameraState,
    options: GsplatRenderOptions,
) -> dict[str, Any]:
    """Build shared 3D gsplat rasterization kwargs."""
    return {
        "means": scene.center_position,
        "quats": scene.quaternion_orientation,
        "scales": torch.exp(scene.log_scales),
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
        "rasterize_mode": options.rasterize_mode,
    }


@beartype
def _extract_2d_projections(
    metadata: dict[str, Any],
    options: GsplatRenderOptions,
) -> tuple[
    Float[Tensor, "num_cams num_splats 2"],
    Float[Tensor, "num_cams num_splats 3"],
]:
    """Extract typed 2D projections from gsplat metadata."""
    if options.packed:
        raise ValueError(
            "2D projection outputs are currently only supported with "
            "GsplatRenderOptions(packed=False)."
        )
    return metadata["means2d"], metadata["conics"]


@beartype
def _render_rgb(
    scene: GaussianScene3D,
    camera: CameraState,
    options: GsplatRenderOptions,
) -> GsplatAlphaRenderOutput:
    """Render RGB and alpha with gsplat."""
    render_colors, render_alphas, _metadata = rasterization(
        **_build_rasterization_kwargs(scene, camera, options),
        render_mode="RGB",
    )
    return GsplatAlphaRenderOutput(
        render=render_colors,
        alphas=render_alphas.squeeze(-1),
    )


@beartype
def _render_rgb_with_2d_projections(
    scene: GaussianScene3D,
    camera: CameraState,
    options: GsplatRenderOptions,
) -> GsplatAlphaProjectionRenderOutput:
    """Render RGB, alpha, and 2D projections with gsplat."""
    render_colors, render_alphas, metadata = rasterization(
        **_build_rasterization_kwargs(scene, camera, options),
        render_mode="RGB",
    )
    projected_means, projected_conics = _extract_2d_projections(
        metadata,
        options,
    )
    return GsplatAlphaProjectionRenderOutput(
        render=render_colors,
        alphas=render_alphas.squeeze(-1),
        projected_means=projected_means,
        projected_conics=projected_conics,
    )


@beartype
def _render_rgb_plus_depth(
    scene: GaussianScene3D,
    camera: CameraState,
    options: GsplatRenderOptions,
) -> GsplatRenderOutput:
    """Render RGB, alpha, and depth with gsplat."""
    render_mode: Literal["RGB+D", "RGB+ED"]
    if options.depth_render_mode == "D":
        render_mode = "RGB+D"
    else:
        render_mode = "RGB+ED"
    render_colors, render_alphas, _metadata = rasterization(
        **_build_rasterization_kwargs(scene, camera, options),
        render_mode=render_mode,
    )
    return GsplatRenderOutput(
        render=render_colors[..., :3],
        alphas=render_alphas.squeeze(-1),
        depth=render_colors[..., 3],
    )


@beartype
def _render_rgb_plus_depth_with_2d_projections(
    scene: GaussianScene3D,
    camera: CameraState,
    options: GsplatRenderOptions,
) -> GsplatProjectionRenderOutput:
    """Render RGB, alpha, depth, and 2D projections with gsplat."""
    render_mode: Literal["RGB+D", "RGB+ED"]
    if options.depth_render_mode == "D":
        render_mode = "RGB+D"
    else:
        render_mode = "RGB+ED"
    render_colors, render_alphas, metadata = rasterization(
        **_build_rasterization_kwargs(scene, camera, options),
        render_mode=render_mode,
    )
    projected_means, projected_conics = _extract_2d_projections(
        metadata,
        options,
    )
    return GsplatProjectionRenderOutput(
        render=render_colors[..., :3],
        alphas=render_alphas.squeeze(-1),
        depth=render_colors[..., 3],
        projected_means=projected_means,
        projected_conics=projected_conics,
    )


@overload
def render_gsplat(
    scene: GaussianScene3D,
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
def render_gsplat(
    scene: GaussianScene3D,
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
def render_gsplat(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[True] = True,
    return_depth: Literal[False] = False,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[True] = True,
    return_projective_intersection_transforms: Literal[False] = False,
    options: GsplatRenderOptions | None = None,
) -> GsplatAlphaProjectionRenderOutput: ...


@overload
def render_gsplat(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[True] = True,
    return_depth: Literal[True] = True,
    return_normals: Literal[False] = False,
    return_2d_projections: Literal[True] = True,
    return_projective_intersection_transforms: Literal[False] = False,
    options: GsplatRenderOptions | None = None,
) -> GsplatProjectionRenderOutput: ...


@beartype
def render_gsplat(
    scene: GaussianScene3D,
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
    | GsplatAlphaProjectionRenderOutput
    | GsplatProjectionRenderOutput
):
    """Render a 3D Gaussian scene with gsplat."""
    del return_alpha

    if scene.log_scales.shape[-1] != 3:
        raise ValueError(
            "gsplat only supports 3D Gaussian scales with shape "
            f"(num_splats, 3); got {tuple(scene.log_scales.shape)}."
        )
    if camera.camera_convention != "opencv":
        raise ValueError(
            "gsplat currently expects cameras in opencv convention; got "
            f"{camera.camera_convention!r}."
        )
    if return_normals:
        raise ValueError("gsplat normals are not exposed yet.")
    if return_projective_intersection_transforms:
        raise ValueError(
            "gsplat does not expose projective intersection transforms."
        )

    options = options or GsplatRenderOptions()
    validate_uniform_camera_shape(camera, "gsplat")

    if return_depth and return_2d_projections:
        return _render_rgb_plus_depth_with_2d_projections(
            scene,
            camera,
            options,
        )

    if return_depth:
        return _render_rgb_plus_depth(scene, camera, options)

    if return_2d_projections:
        return _render_rgb_with_2d_projections(scene, camera, options)

    return _render_rgb(scene, camera, options)
