"""Lightweight backend bundle helpers for viewer pipelines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from marimo_3dv.pipeline.gui import PipelineItem, ViewerPipeline
from marimo_3dv.viewer.defaults import (
    ViewerControlsConfig,
    viewer_controls_config,
)
from marimo_3dv.viewer.widget import ViewerState

SceneT = TypeVar("SceneT")
RenderViewT = TypeVar("RenderViewT")
CompiledViewT = TypeVar("CompiledViewT")


@dataclass(frozen=True)
class ViewerBackendBundle(Generic[SceneT, RenderViewT, CompiledViewT]):
    """Reusable backend family bundle with optional defaults."""

    name: str
    render_view_factory: Callable[[SceneT], RenderViewT]
    compile_view: Callable[[RenderViewT], CompiledViewT]
    default_render_items: tuple[PipelineItem, ...] = field(
        default_factory=tuple
    )
    default_effect_items: tuple[PipelineItem, ...] = field(
        default_factory=tuple
    )
    viewer_controls_transform: (
        Callable[[ViewerControlsConfig], ViewerControlsConfig] | None
    ) = None

    def pipeline(self) -> ViewerPipeline[SceneT, RenderViewT, CompiledViewT]:
        """Build a pipeline preloaded with the bundle's optional default items."""
        pipeline = ViewerPipeline(
            view_factory=self.render_view_factory,
            compile_view=self.compile_view,
        )
        for item in self.default_render_items:
            pipeline = pipeline.render(item)
        for item in self.default_effect_items:
            pipeline = pipeline.effect(item)
        return pipeline

    def viewer_controls(
        self, viewer_state: ViewerState
    ) -> ViewerControlsConfig:
        """Return bundle-aware default viewer controls."""
        config = viewer_controls_config(viewer_state)
        if self.viewer_controls_transform is None:
            return config
        return self.viewer_controls_transform(config)


def backend_bundle(
    *,
    name: str,
    render_view_factory: Callable[[SceneT], RenderViewT],
    compile_view: Callable[[RenderViewT], CompiledViewT],
    default_render_items: Sequence[PipelineItem] = (),
    default_effect_items: Sequence[PipelineItem] = (),
    viewer_controls_transform: (
        Callable[[ViewerControlsConfig], ViewerControlsConfig] | None
    ) = None,
) -> ViewerBackendBundle[SceneT, RenderViewT, CompiledViewT]:
    """Create a lightweight backend bundle."""
    return ViewerBackendBundle(
        name=name,
        render_view_factory=render_view_factory,
        compile_view=compile_view,
        default_render_items=tuple(default_render_items),
        default_effect_items=tuple(default_effect_items),
        viewer_controls_transform=viewer_controls_transform,
    )


__all__ = ["ViewerBackendBundle", "backend_bundle"]
