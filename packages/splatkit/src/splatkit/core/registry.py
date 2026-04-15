"""Backend registry and generic render wrapper."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, overload

from beartype import beartype

from splatkit.core.capabilities import (
    RenderWith2DProjections,
    RenderWithAlpha,
    RenderWithAlpha2DProjections,
    RenderWithAlphaDepth,
    RenderWithAlphaDepth2DProjections,
    RenderWithDepth,
    RenderWithDepth2DProjections,
)
from splatkit.core.contracts import (
    BackendName,
    CameraState,
    OutputName,
    RenderOptions,
    RenderOutput,
    Scene,
)


@dataclass(frozen=True)
class RegisteredBackend:
    """Registry entry for a concrete backend render function."""

    name: BackendName
    render_fn: Callable[..., RenderOutput]
    default_options: RenderOptions
    accepted_scene_types: tuple[type[Scene], ...]
    supported_outputs: frozenset[OutputName] = frozenset()


BACKEND_REGISTRY: dict[BackendName, RegisteredBackend] = {}


def register_backend(
    *,
    name: BackendName,
    default_options: RenderOptions,
    accepted_scene_types: tuple[type[Scene], ...],
    supported_outputs: frozenset[OutputName] = frozenset(),
) -> Callable[[Callable[..., RenderOutput]], Callable[..., RenderOutput]]:
    """Register a backend render function as a decorator."""

    def decorator(
        render_fn: Callable[..., RenderOutput],
    ) -> Callable[..., RenderOutput]:
        BACKEND_REGISTRY[name] = RegisteredBackend(
            name=name,
            render_fn=render_fn,
            default_options=default_options,
            accepted_scene_types=accepted_scene_types,
            supported_outputs=supported_outputs,
        )
        return render_fn

    return decorator


@overload
def render(
    scene: Scene,
    camera: CameraState,
    *,
    backend: BackendName,
    return_alpha: Literal[False] = False,
    return_depth: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    options: RenderOptions | None = None,
) -> RenderOutput: ...


@overload
def render(
    scene: Scene,
    camera: CameraState,
    *,
    backend: BackendName,
    return_alpha: Literal[True],
    return_depth: Literal[False] = False,
    return_2d_projections: Literal[False] = False,
    options: RenderOptions | None = None,
) -> RenderWithAlpha: ...


@overload
def render(
    scene: Scene,
    camera: CameraState,
    *,
    backend: BackendName,
    return_alpha: Literal[False] = False,
    return_depth: Literal[True] = True,
    return_2d_projections: Literal[False] = False,
    options: RenderOptions | None = None,
) -> RenderWithDepth: ...


@overload
def render(
    scene: Scene,
    camera: CameraState,
    *,
    backend: BackendName,
    return_alpha: Literal[True],
    return_depth: Literal[True],
    return_2d_projections: Literal[False] = False,
    options: RenderOptions | None = None,
) -> RenderWithAlphaDepth: ...


@overload
def render(
    scene: Scene,
    camera: CameraState,
    *,
    backend: BackendName,
    return_alpha: Literal[False] = False,
    return_depth: Literal[False] = False,
    return_2d_projections: Literal[True] = True,
    options: RenderOptions | None = None,
) -> RenderWith2DProjections: ...


@overload
def render(
    scene: Scene,
    camera: CameraState,
    *,
    backend: BackendName,
    return_alpha: Literal[True],
    return_depth: Literal[False] = False,
    return_2d_projections: Literal[True] = True,
    options: RenderOptions | None = None,
) -> RenderWithAlpha2DProjections: ...


@overload
def render(
    scene: Scene,
    camera: CameraState,
    *,
    backend: BackendName,
    return_alpha: Literal[False] = False,
    return_depth: Literal[True] = True,
    return_2d_projections: Literal[True] = True,
    options: RenderOptions | None = None,
) -> RenderWithDepth2DProjections: ...


@overload
def render(
    scene: Scene,
    camera: CameraState,
    *,
    backend: BackendName,
    return_alpha: Literal[True],
    return_depth: Literal[True],
    return_2d_projections: Literal[True] = True,
    options: RenderOptions | None = None,
) -> RenderWithAlphaDepth2DProjections: ...


@beartype
def render(
    scene: Scene,
    camera: CameraState,
    *,
    backend: BackendName,
    return_alpha: bool = False,
    return_depth: bool = False,
    return_2d_projections: bool = False,
    options: RenderOptions | None = None,
) -> RenderOutput:
    """Render through a named backend."""
    registered_backend = BACKEND_REGISTRY.get(backend)
    if registered_backend is None:
        available = ", ".join(sorted(BACKEND_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown backend {backend!r}. Available backends: {available}."
        )
    if not isinstance(scene, registered_backend.accepted_scene_types):
        accepted_names = ", ".join(
            scene_type.__name__
            for scene_type in registered_backend.accepted_scene_types
        )
        raise ValueError(
            f"Backend {backend!r} does not accept scene type "
            f"{type(scene).__name__}. Accepted scene types: {accepted_names}."
        )

    requested_outputs: frozenset[OutputName] = frozenset(
        name
        for enabled, name in (
            (return_alpha, "alpha"),
            (return_depth, "depth"),
            (return_2d_projections, "2d_projections"),
        )
        if enabled
    )
    unsupported_outputs = (
        requested_outputs - registered_backend.supported_outputs
    )
    if unsupported_outputs:
        requested = ", ".join(sorted(unsupported_outputs))
        supported = (
            ", ".join(sorted(registered_backend.supported_outputs)) or "<none>"
        )
        raise ValueError(
            f"Backend {backend!r} does not support requested outputs: {requested}. "
            f"Supported outputs: {supported}."
        )

    resolved_options = options or registered_backend.default_options
    return registered_backend.render_fn(
        scene,
        camera,
        return_alpha=return_alpha,
        return_depth=return_depth,
        return_2d_projections=return_2d_projections,
        options=resolved_options,
    )
