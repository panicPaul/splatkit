"""Primitive-agnostic mesh extraction contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ember_core.core.contracts import BackendName, CameraState, Scene


@beartype
@dataclass(frozen=True)
class MeshArtifact:
    """A triangle mesh produced by a meshification method."""

    vertices: Float[Tensor, " num_vertices 3"]
    faces: Int[Tensor, " num_faces 3"]
    vertex_colors: Float[Tensor, " num_vertices 3"] | None = None
    vertex_normals: Float[Tensor, " num_vertices 3"] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@beartype
@dataclass(frozen=True)
class MeshificationResult:
    """Meshification output with optional method diagnostics."""

    mesh: MeshArtifact
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@beartype
@dataclass(frozen=True)
class SurfacePointSamples:
    """Candidate surface samples exposed by a backend trait provider."""

    points: Float[Tensor, " num_points 3"]
    scales: Float[Tensor, " num_points"] | Float[Tensor, " num_points 1"] | None
    attributes: Mapping[str, Tensor] = field(default_factory=dict)


@beartype
@dataclass(frozen=True)
class WrappingSurfaceEvidence:
    """View-space evidence needed by wrapping-style mesh extraction."""

    render: Float[Tensor, "num_cams height width 3"]
    alpha: Float[Tensor, "num_cams height width"] | None = None
    depth: Float[Tensor, "num_cams height width"] | None = None
    median_depth: Float[Tensor, "num_cams height width"] | None = None
    expected_depth: Float[Tensor, "num_cams height width"] | None = None
    normals: Float[Tensor, "num_cams height width 3"] | None = None
    valid_mask: Bool[Tensor, "num_cams height width"] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@beartype
@dataclass(frozen=True)
class WrappingQueryResult:
    """Wrapping-field query result for arbitrary 3D points."""

    values: Float[Tensor, " num_points"]
    inside: Bool[Tensor, " num_points"] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@beartype
@dataclass(frozen=True)
class MeshificationOptions:
    """Base class for meshification configuration."""


@beartype
@dataclass(frozen=True)
class MeshificationRequest:
    """Inputs shared by meshification methods."""

    scene: Scene
    camera: CameraState
    backend: BackendName
    backend_options: Any | None = None
    device: torch.device | None = None


@runtime_checkable
class WrappingSurfaceProvider(Protocol):
    """Backend trait for wrapping-style surface extraction.

    The trait is intentionally phrased in terms of evidence and queries rather
    than a concrete primitive family. Gaussian, 2DGS, or future scene types can
    participate as long as their backend can provide these methods.
    """

    def surface_evidence(
        self,
        request: MeshificationRequest,
    ) -> WrappingSurfaceEvidence:
        """Return image-space evidence used by wrapping extraction."""

    def sample_surface_points(
        self,
        request: MeshificationRequest,
    ) -> SurfacePointSamples:
        """Return candidate world-space points for topology construction."""

    def query_wrapping_field(
        self,
        request: MeshificationRequest,
        points: Float[Tensor, " num_points 3"],
    ) -> WrappingQueryResult:
        """Evaluate the provider's wrapping field at world-space points."""


class Meshifier(Protocol):
    """Callable mesh extraction method."""

    def __call__(
        self,
        request: MeshificationRequest,
        *,
        options: MeshificationOptions | None,
        provider: object | None,
    ) -> MeshificationResult:
        """Extract a mesh from a request and an optional backend trait."""
