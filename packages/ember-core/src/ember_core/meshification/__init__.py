"""Meshification contracts and registry."""

from ember_core.meshification.contracts import (
    MeshArtifact,
    MeshificationOptions,
    MeshificationRequest,
    MeshificationResult,
    Meshifier,
    SurfacePointSamples,
    WrappingQueryResult,
    WrappingSurfaceEvidence,
    WrappingSurfaceProvider,
)
from ember_core.meshification.registry import (
    MESHIFIER_REGISTRY,
    MeshifierName,
    RegisteredMeshifier,
    meshify,
    register_meshifier,
    resolve_meshifier,
)
from ember_core.meshification.wrapping import (
    WrappingMeshificationOptions,
    extract_wrapping_mesh,
)

__all__ = [
    "MESHIFIER_REGISTRY",
    "MeshArtifact",
    "MeshificationOptions",
    "MeshificationRequest",
    "MeshificationResult",
    "Meshifier",
    "MeshifierName",
    "RegisteredMeshifier",
    "SurfacePointSamples",
    "WrappingMeshificationOptions",
    "WrappingQueryResult",
    "WrappingSurfaceEvidence",
    "WrappingSurfaceProvider",
    "extract_wrapping_mesh",
    "meshify",
    "register_meshifier",
    "resolve_meshifier",
]
