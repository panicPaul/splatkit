"""Structural typing for backend output capabilities."""

from typing import Protocol

from jaxtyping import Float
from torch import Tensor


class HasAlpha(Protocol):
    """Output capability for alpha channels."""

    alphas: Float[Tensor, "num_cams height width"]


class HasDepth(Protocol):
    """Output capability for depth."""

    depth: Float[Tensor, "num_cams height width"]


class Has2DProjections(Protocol):
    """Output capability for projected 2D Gaussian geometry."""

    projected_means: Float[Tensor, "num_cams num_splats 2"]
    projected_conics: Float[Tensor, "num_cams num_splats 3"]


class RenderWithAlpha(Protocol):
    """Render output that guarantees alpha."""

    render: Float[Tensor, "num_cams height width 3"]
    alphas: Float[Tensor, "num_cams height width"]


class RenderWithDepth(Protocol):
    """Render output that guarantees depth."""

    render: Float[Tensor, "num_cams height width 3"]
    depth: Float[Tensor, "num_cams height width"]


class RenderWithAlphaDepth(Protocol):
    """Render output that guarantees alpha and depth."""

    render: Float[Tensor, "num_cams height width 3"]
    alphas: Float[Tensor, "num_cams height width"]
    depth: Float[Tensor, "num_cams height width"]


class RenderWith2DProjections(Protocol):
    """Render output that guarantees 2D Gaussian projections."""

    render: Float[Tensor, "num_cams height width 3"]
    projected_means: Float[Tensor, "num_cams num_splats 2"]
    projected_conics: Float[Tensor, "num_cams num_splats 3"]


class RenderWithAlpha2DProjections(RenderWithAlpha, Has2DProjections, Protocol):
    """Render output that guarantees alpha and 2D Gaussian projections."""


class RenderWithDepth2DProjections(RenderWithDepth, Has2DProjections, Protocol):
    """Render output that guarantees depth and 2D Gaussian projections."""


class RenderWithAlphaDepth2DProjections(
    RenderWithAlphaDepth, Has2DProjections, Protocol
):
    """Render output that guarantees alpha, depth, and 2D projections."""
