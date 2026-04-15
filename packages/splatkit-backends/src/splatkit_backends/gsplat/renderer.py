"""Gsplat backend contract surface."""

from dataclasses import dataclass, field
from typing import Any, Literal, overload

import torch
from beartype import beartype
from gsplat import rasterization
from jaxtyping import Float
from splatkit.core.capabilities import (
    Has2DProjections,
    HasAlpha,
    HasDepth,
)
from splatkit.core.contracts import (
    CameraState,
    GaussianScene3D,
    RenderOptions,
    RenderOutput,
)
from splatkit.core.registry import register_backend
from torch import Tensor

_SUPPORTED_OUTPUTS = frozenset({"alpha", "depth", "2d_projections"})


@beartype
@dataclass(frozen=True)
class GsplatAlphaRenderOutput(RenderOutput, HasAlpha):
    """Gsplat render output with alpha."""

    alphas: Float[Tensor, "num_cams height width"]


@beartype
@dataclass(frozen=True)
class GsplatRenderOutput(GsplatAlphaRenderOutput, HasDepth):
    """Gsplat render output with alpha and depth."""

    depth: Float[Tensor, "num_cams height width"]


@beartype
@dataclass(frozen=True)
class GsplatAlphaProjectionRenderOutput(
    GsplatAlphaRenderOutput, Has2DProjections
):
    """Gsplat render output with alpha and 2D Gaussian projections."""

    projected_means: Float[Tensor, "num_cams num_splats 2"]
    projected_conics: Float[Tensor, "num_cams num_splats 3"]


@beartype
@dataclass(frozen=True)
class GsplatProjectionRenderOutput(GsplatRenderOutput, Has2DProjections):
    """Gsplat render output with alpha, depth, and 2D projections."""

    projected_means: Float[Tensor, "num_cams num_splats 2"]
    projected_conics: Float[Tensor, "num_cams num_splats 3"]


@beartype
@dataclass(frozen=True)
class GsplatRenderOptions(RenderOptions):
    """Gsplat-specific render configuration."""

    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    depth_render_mode: Literal["D", "ED"] = "ED"
    eps_2d: float = 0.3
    packed: bool = False
    sparse_grad: bool = False
    absgrad: bool = False
    rasterize_mode_requires_grad: bool = field(default=False, repr=False)


@beartype
def _build_rasterization_kwargs(
    scene: GaussianScene3D,
    camera: CameraState,
    options: GsplatRenderOptions,
) -> dict[str, Any]:
    """Build shared gsplat rasterization kwargs."""
    num_cams = int(camera.width.shape[0])
    background_color = options.background_color.to(
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )
    backgrounds = (
        background_color
        if options.packed
        else background_color.expand(num_cams, -1)
    )
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
        "backgrounds": backgrounds,
        "packed": options.packed,
        "eps2d": options.eps_2d,
        "sparse_grad": options.sparse_grad,
        "absgrad": options.absgrad,
        "rasterize_mode": options.rasterize_mode,
    }


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
        metadata, options
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
        metadata, options
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
    return_2d_projections: Literal[False] = False,
    options: GsplatRenderOptions | None = None,
) -> GsplatAlphaRenderOutput: ...


@overload
def render_gsplat(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[True] = True,
    return_depth: Literal[True] = True,
    return_2d_projections: Literal[False] = False,
    options: GsplatRenderOptions | None = None,
) -> GsplatRenderOutput: ...


@overload
def render_gsplat(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[True] = True,
    return_depth: Literal[False] = False,
    return_2d_projections: Literal[True] = True,
    options: GsplatRenderOptions | None = None,
) -> GsplatAlphaProjectionRenderOutput: ...


@overload
def render_gsplat(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: Literal[True] = True,
    return_depth: Literal[True] = True,
    return_2d_projections: Literal[True] = True,
    options: GsplatRenderOptions | None = None,
) -> GsplatProjectionRenderOutput: ...


@beartype
def render_gsplat(
    scene: GaussianScene3D,
    camera: CameraState,
    *,
    return_alpha: bool = True,
    return_depth: bool = False,
    return_2d_projections: bool = False,
    options: GsplatRenderOptions | None = None,
) -> (
    GsplatAlphaRenderOutput
    | GsplatRenderOutput
    | GsplatAlphaProjectionRenderOutput
    | GsplatProjectionRenderOutput
):
    """Render a scene with gsplat."""
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

    options = options or GsplatRenderOptions()
    if not torch.equal(camera.width, camera.width[:1].expand_as(camera.width)):
        raise ValueError(
            "gsplat requires a uniform image width across cameras."
        )
    if not torch.equal(
        camera.height, camera.height[:1].expand_as(camera.height)
    ):
        raise ValueError(
            "gsplat requires a uniform image height across cameras."
        )

    if return_depth and return_2d_projections:
        return _render_rgb_plus_depth_with_2d_projections(
            scene, camera, options
        )

    if return_depth:
        return _render_rgb_plus_depth(scene, camera, options)

    if return_2d_projections:
        return _render_rgb_with_2d_projections(scene, camera, options)

    return _render_rgb(scene, camera, options)


def register() -> None:
    """Register the gsplat backend in the global splatkit registry."""
    register_backend(
        name="gsplat",
        default_options=GsplatRenderOptions(),
        accepted_scene_types=(GaussianScene3D,),
        supported_outputs=_SUPPORTED_OUTPUTS,
    )(render_gsplat)
