"""Shared gsplat backend types and utilities."""

from dataclasses import dataclass, field
from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float
from splatkit.core.capabilities import (
    Has2DProjections,
    HasAlpha,
    HasDepth,
    HasProjectiveIntersectionTransforms,
)
from splatkit.core.contracts import (
    CameraState,
    GaussianScene,
    RenderOptions,
    RenderOutput,
)
from torch import Tensor

SUPPORTED_OUTPUTS_3D = frozenset({"alpha", "depth", "2d_projections"})
SUPPORTED_OUTPUTS_2D = frozenset(
    {"alpha", "depth", "projective_intersection_transforms"}
)


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
class GsplatAlphaIntersectionRenderOutput(
    GsplatAlphaRenderOutput, HasProjectiveIntersectionTransforms
):
    """Gsplat 2DGS render output with alpha and intersection geometry."""

    projected_means: Float[Tensor, "num_cams num_splats 2"]
    projective_intersection_transforms: Float[Tensor, "num_cams num_splats 3 3"]


@beartype
@dataclass(frozen=True)
class GsplatIntersectionRenderOutput(
    GsplatRenderOutput, HasProjectiveIntersectionTransforms
):
    """Gsplat 2DGS render output with depth and intersection geometry."""

    projected_means: Float[Tensor, "num_cams num_splats 2"]
    projective_intersection_transforms: Float[Tensor, "num_cams num_splats 3 3"]


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
def build_backgrounds(
    scene: GaussianScene,
    camera: CameraState,
    options: GsplatRenderOptions,
) -> Tensor:
    """Build batched background colors for gsplat rasterizers."""
    num_cams = int(camera.width.shape[0])
    background_color = options.background_color.to(
        device=scene.center_position.device,
        dtype=scene.center_position.dtype,
    )
    if options.packed:
        return background_color
    return background_color.expand(num_cams, -1)


@beartype
def validate_uniform_camera_shape(
    camera: CameraState,
    backend_name: str,
) -> None:
    """Validate uniform image dimensions across a camera batch."""
    if not torch.equal(camera.width, camera.width[:1].expand_as(camera.width)):
        raise ValueError(
            f"{backend_name} requires a uniform image width across cameras."
        )
    if not torch.equal(
        camera.height,
        camera.height[:1].expand_as(camera.height),
    ):
        raise ValueError(
            f"{backend_name} requires a uniform image height across cameras."
        )
