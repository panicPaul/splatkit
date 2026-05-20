"""Registry for primitive-agnostic meshification methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ember_core.core.contracts import BackendName, CameraState, Scene
from ember_core.core.registry import resolve_backend_trait
from ember_core.meshification.contracts import (
    MeshificationOptions,
    MeshificationRequest,
    MeshificationResult,
    Meshifier,
    WrappingSurfaceProvider,
)

MeshifierName = str


@dataclass(frozen=True)
class RegisteredMeshifier:
    """Registry entry for a mesh extraction method."""

    name: MeshifierName
    extract_fn: Meshifier
    default_options: MeshificationOptions | None = None
    required_trait: type[object] | None = None


MESHIFIER_REGISTRY: dict[MeshifierName, RegisteredMeshifier] = {}


def _infer_scene_device(scene: Scene) -> torch.device | None:
    for tensor in scene.parameters(recurse=False):
        return tensor.device
    for tensor in scene.buffers(recurse=False):
        return tensor.device
    return None


def register_meshifier(
    *,
    name: MeshifierName,
    default_options: MeshificationOptions | None = None,
    required_trait: type[object] | None = WrappingSurfaceProvider,
) -> Callable[[Meshifier], Meshifier]:
    """Register a meshification method as a decorator."""

    def decorator(extract_fn: Meshifier) -> Meshifier:
        MESHIFIER_REGISTRY[name] = RegisteredMeshifier(
            name=name,
            extract_fn=extract_fn,
            default_options=default_options,
            required_trait=required_trait,
        )
        return extract_fn

    return decorator


def resolve_meshifier(name: MeshifierName) -> RegisteredMeshifier:
    """Resolve a meshification method by name."""
    registered_meshifier = MESHIFIER_REGISTRY.get(name)
    if registered_meshifier is not None:
        return registered_meshifier
    available = ", ".join(sorted(MESHIFIER_REGISTRY)) or "<none>"
    raise ValueError(
        f"Unknown meshifier {name!r}. Available meshifiers: {available}."
    )


def meshify(
    scene: Scene,
    camera: CameraState,
    *,
    backend: BackendName,
    meshifier: MeshifierName,
    options: MeshificationOptions | None = None,
    backend_options: object | None = None,
) -> MeshificationResult:
    """Extract a mesh from a scene through a registered backend trait."""
    registered_meshifier = resolve_meshifier(meshifier)
    provider: object | None = None
    if registered_meshifier.required_trait is not None:
        provider = resolve_backend_trait(
            backend,
            registered_meshifier.required_trait,
        )
    request = MeshificationRequest(
        scene=scene,
        camera=camera,
        backend=backend,
        backend_options=backend_options,
        device=_infer_scene_device(scene),
    )
    resolved_options = (
        options if options is not None else registered_meshifier.default_options
    )
    return registered_meshifier.extract_fn(
        request,
        options=resolved_options,
        provider=provider,
    )
