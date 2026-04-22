"""Structural typing for backend output capabilities."""

from typing import Protocol, runtime_checkable

from jaxtyping import Bool, Float, Int
from torch import Tensor


class HasAlpha(Protocol):
    """Output capability for alpha channels."""

    alphas: Float[Tensor, "num_cams height width"]


class HasDepth(Protocol):
    """Output capability for depth."""

    depth: Float[Tensor, "num_cams height width"]


class HasGaussianImpactScore(Protocol):
    """Output capability for backend-defined per-Gaussian impact scores."""

    gaussian_impact_score: Float[Tensor, "num_cams num_splats"]


@runtime_checkable
class HasViewspacePoints(Protocol):
    """Output capability for per-Gaussian view-space point buffers."""

    viewspace_points: Float[Tensor, "num_cams num_splats 4"]


@runtime_checkable
class HasVisibilityFilter(Protocol):
    """Output capability for per-Gaussian visibility masks."""

    visibility_filter: Bool[Tensor, "num_cams num_splats"]


@runtime_checkable
class HasScreenSpaceRadii(Protocol):
    """Output capability for per-Gaussian screen-space radii."""

    radii: Int[Tensor, "num_cams num_splats"]


@runtime_checkable
class HasScreenSpaceDensificationSignals(
    HasViewspacePoints,
    HasVisibilityFilter,
    HasScreenSpaceRadii,
    Protocol,
):
    """Output capability for densification-relevant screen-space signals."""


class HasNormals(Protocol):
    """Output capability for normals."""

    normals: Float[Tensor, "num_cams height width 3"]


class HasProjectedMeans(Protocol):
    """Output capability for 2D projected Gaussian centers."""

    projected_means: Float[Tensor, "num_cams num_splats 2"]


class HasProjectedConics(Protocol):
    """Output capability for projected 2D Gaussian conics."""

    projected_conics: Float[Tensor, "num_cams num_splats 3"]


class Has2DProjections(HasProjectedMeans, HasProjectedConics, Protocol):
    """Output capability for projected 2D Gaussian geometry."""


class HasProjectiveIntersectionTransforms(HasProjectedMeans, Protocol):
    """Output capability for 2DGS projective intersection geometry.

    `projective_intersection_transforms` stores the 2DGS-specific projected
    intersection transform for each camera/primitive pair. Unlike
    `projected_conics`, which is a compact 2D ellipse representation,
    these matrices retain the richer projective geometry used by 2DGS
    rasterization.
    """

    projective_intersection_transforms: Float[Tensor, "num_cams num_splats 3 3"]


class RenderWithAlpha(Protocol):
    """Render output that guarantees alpha."""

    render: Float[Tensor, "num_cams height width 3"]
    alphas: Float[Tensor, "num_cams height width"]


class RenderWithDepth(Protocol):
    """Render output that guarantees depth."""

    render: Float[Tensor, "num_cams height width 3"]
    depth: Float[Tensor, "num_cams height width"]


class RenderWithGaussianImpactScore(Protocol):
    """Render output that guarantees Gaussian impact scores."""

    render: Float[Tensor, "num_cams height width 3"]
    gaussian_impact_score: Float[Tensor, "num_cams num_splats"]


class RenderWithNormals(Protocol):
    """Render output that guarantees normals."""

    render: Float[Tensor, "num_cams height width 3"]
    normals: Float[Tensor, "num_cams height width 3"]


class RenderWithAlphaDepth(Protocol):
    """Render output that guarantees alpha and depth."""

    render: Float[Tensor, "num_cams height width 3"]
    alphas: Float[Tensor, "num_cams height width"]
    depth: Float[Tensor, "num_cams height width"]


class RenderWithDepthGaussianImpactScore(
    RenderWithDepth, HasGaussianImpactScore, Protocol
):
    """Render output that guarantees depth and Gaussian impact scores."""


class RenderWith2DProjections(Protocol):
    """Render output that guarantees 2D Gaussian projections."""

    render: Float[Tensor, "num_cams height width 3"]
    projected_means: Float[Tensor, "num_cams num_splats 2"]
    projected_conics: Float[Tensor, "num_cams num_splats 3"]


class RenderWithProjectiveIntersectionTransforms(Protocol):
    """Render output that guarantees projective intersection transforms."""

    render: Float[Tensor, "num_cams height width 3"]
    projected_means: Float[Tensor, "num_cams num_splats 2"]
    projective_intersection_transforms: Float[Tensor, "num_cams num_splats 3 3"]


class RenderWithAlpha2DProjections(RenderWithAlpha, Has2DProjections, Protocol):
    """Render output that guarantees alpha and 2D Gaussian projections."""


class RenderWithDepth2DProjections(RenderWithDepth, Has2DProjections, Protocol):
    """Render output that guarantees depth and 2D Gaussian projections."""


class RenderWithAlphaDepth2DProjections(
    RenderWithAlphaDepth, Has2DProjections, Protocol
):
    """Render output that guarantees alpha, depth, and 2D projections."""


class RenderWithAlphaProjectiveIntersectionTransforms(
    RenderWithAlpha, HasProjectiveIntersectionTransforms, Protocol
):
    """Render output that guarantees alpha and projective intersections."""


class RenderWithDepthProjectiveIntersectionTransforms(
    RenderWithDepth, HasProjectiveIntersectionTransforms, Protocol
):
    """Render output that guarantees depth and projective intersections."""


class RenderWithAlphaDepthProjectiveIntersectionTransforms(
    RenderWithAlphaDepth, HasProjectiveIntersectionTransforms, Protocol
):
    """Render output that guarantees alpha, depth, and intersections."""


class RenderWithAlphaNormals(RenderWithAlpha, HasNormals, Protocol):
    """Render output that guarantees alpha and normals."""


class RenderWithDepthNormals(RenderWithDepth, HasNormals, Protocol):
    """Render output that guarantees depth and normals."""


class RenderWithAlphaDepthNormals(RenderWithAlphaDepth, HasNormals, Protocol):
    """Render output that guarantees alpha, depth, and normals."""
