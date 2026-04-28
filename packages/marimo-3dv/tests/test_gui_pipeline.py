"""Tests for the declarative viewer pipeline."""

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from marimo_3dv.ops.gs import gs_backend_bundle
from marimo_3dv.pipeline.bundle import ViewerBackendBundle, backend_bundle
from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.pipeline.gui import (
    AbstractRenderView,
    PipelineGroup,
    RenderResult,
    ViewerPipeline,
    effect_node,
    render_node,
)
from marimo_3dv.viewer.widget import ViewerState


@dataclass(frozen=True)
class _ListRenderView(AbstractRenderView[list[str]]):
    tokens: tuple[str, ...] = ()

    def with_token(self, token: str) -> "_ListRenderView":
        return _ListRenderView(
            source_scene=self.source_scene,
            backend_key=self.backend_key,
            capabilities=self.capabilities,
            extensions=self.extensions,
            tokens=(*self.tokens, token),
        )


def _view_factory(scene: list[str]) -> _ListRenderView:
    return _ListRenderView(source_scene=scene, tokens=tuple(scene))


def _make_viewer_state() -> ViewerState:
    return ViewerState()


def _dummy_image(h: int = 4, w: int = 4) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_empty_pipeline_builds() -> None:
    pipeline = ViewerPipeline(view_factory=_view_factory)
    result = pipeline.build(source_scene=[], viewer_state=_make_viewer_state())
    assert result is not None


def test_backend_bundle_builds_pipeline_with_default_items() -> None:
    render_item = render_node(
        name="append",
        apply=lambda view, config, context: view.with_token("x"),
    )
    bundle = backend_bundle(
        name="list",
        render_view_factory=_view_factory,
        compile_view=lambda view: view,
        default_render_items=(PipelineGroup("defaults", render_item),),
    )

    result = bundle.pipeline().build(
        source_scene=[],
        viewer_state=_make_viewer_state(),
    )

    def backend(camera_state, compiled_view: _ListRenderView) -> RenderResult:
        del camera_state
        return RenderResult(
            image=_dummy_image(),
            metadata={"tokens": compiled_view.tokens},
        )

    from marimo_3dv.viewer.widget import CameraState

    render_fn = result.bind(result.default_config, backend_fn=backend)
    output = render_fn(CameraState.default())
    assert output.metadata["tokens"] == ("x",)


def test_backend_bundle_exposes_viewer_controls_defaults() -> None:
    def _transform(config):
        return config.model_copy(
            update={
                "overlays": config.overlays.model_copy(
                    update={"show_stats": True}
                )
            }
        )

    bundle = backend_bundle(
        name="list",
        render_view_factory=_view_factory,
        compile_view=lambda view: view,
        viewer_controls_transform=_transform,
    )

    config = bundle.viewer_controls(_make_viewer_state())

    assert config.overlays.show_stats is True


def test_gs_backend_bundle_returns_pipeline_bundle() -> None:
    bundle = gs_backend_bundle()

    assert isinstance(bundle, ViewerBackendBundle)
    assert bundle.name == "gsplat"


def test_nested_group_config_is_exposed() -> None:
    class ThresholdConfig(BaseModel):
        threshold: float = 0.5

    node = render_node(
        name="threshold",
        config_model=ThresholdConfig,
        default_config=ThresholdConfig(),
        apply=lambda view, config, context: view,
    )
    pipeline = ViewerPipeline(view_factory=_view_factory).render(
        PipelineGroup("filtering", node)
    )

    result = pipeline.build(source_scene=[], viewer_state=_make_viewer_state())

    assert "filtering" in result.config_model.model_fields
    assert result.default_config.filtering.threshold == 0.5


def test_duplicate_top_level_field_raises() -> None:
    class A(BaseModel):
        value: int = 1

    node_a = render_node(
        name="shared",
        config_model=A,
        default_config=A(),
        apply=lambda view, config, context: view,
    )
    node_b = render_node(
        name="shared",
        config_model=A,
        default_config=A(),
        apply=lambda view, config, context: view,
    )

    pipeline = (
        ViewerPipeline(view_factory=_view_factory).render(node_a).render(node_b)
    )

    try:
        pipeline.build(source_scene=[], viewer_state=_make_viewer_state())
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Expected duplicate config field to raise.")


def test_render_nodes_run_before_backend_and_effects_after() -> None:
    log: list[str] = []

    def render_apply(
        view: _ListRenderView,
        config: BaseModel,
        context: ViewerContext,
    ) -> _ListRenderView:
        del config, context
        log.append("render")
        return view.with_token("prepared")

    def effect_apply(
        result: RenderResult,
        config: BaseModel,
        context: ViewerContext,
        runtime_state: None,
    ) -> RenderResult:
        del config, context, runtime_state
        log.append("effect")
        return result

    pipeline = (
        ViewerPipeline(view_factory=_view_factory)
        .render(render_node(name="prep", apply=render_apply))
        .effect(effect_node(name="overlay", apply=effect_apply))
    )
    result = pipeline.build(source_scene=[], viewer_state=_make_viewer_state())

    backend_inputs: list[tuple[str, ...]] = []

    def backend(camera_state, compiled_view: _ListRenderView) -> RenderResult:
        del camera_state
        log.append("backend")
        backend_inputs.append(compiled_view.tokens)
        return RenderResult(image=_dummy_image(), metadata={})

    render_fn = result.bind(result.default_config, backend_fn=backend)
    from marimo_3dv.viewer.widget import CameraState

    render_fn(CameraState.default())

    assert log == ["render", "backend", "effect"]
    assert backend_inputs == [("prepared",)]


def test_compiled_view_is_cached_across_camera_renders() -> None:
    compile_calls: list[int] = []

    def compile_view(view: _ListRenderView) -> tuple[str, ...]:
        compile_calls.append(1)
        return view.tokens

    pipeline = ViewerPipeline(
        view_factory=_view_factory,
        compile_view=compile_view,
    ).render(
        render_node(
            name="append",
            apply=lambda view, config, context: view.with_token("x"),
        )
    )
    result = pipeline.build(source_scene=[], viewer_state=_make_viewer_state())

    def backend(camera_state, compiled_view: tuple[str, ...]) -> RenderResult:
        del camera_state, compiled_view
        return RenderResult(image=_dummy_image(), metadata={})

    from marimo_3dv.viewer.widget import CameraState

    render_fn = result.bind(result.default_config, backend_fn=backend)
    render_fn(CameraState.default())
    render_fn(CameraState.default(width=640, height=360))

    assert compile_calls == [1]


def test_render_config_change_invalidates_compiled_view_cache() -> None:
    compile_calls: list[int] = []

    class ThresholdConfig(BaseModel):
        value: int = 1

    def append_token(
        view: _ListRenderView,
        config: ThresholdConfig,
        context: ViewerContext,
    ) -> _ListRenderView:
        del context
        return view.with_token(str(config.value))

    pipeline = ViewerPipeline(
        view_factory=_view_factory,
        compile_view=lambda view: compile_calls.append(1) or view.tokens,
    ).render(
        PipelineGroup(
            "filtering",
            render_node(
                name="threshold",
                config_model=ThresholdConfig,
                default_config=ThresholdConfig(),
                apply=append_token,
            ),
        )
    )
    result = pipeline.build(source_scene=[], viewer_state=_make_viewer_state())

    def backend(camera_state, compiled_view: tuple[str, ...]) -> RenderResult:
        del camera_state, compiled_view
        return RenderResult(image=_dummy_image(), metadata={})

    from marimo_3dv.viewer.widget import CameraState

    render_fn = result.bind(result.default_config, backend_fn=backend)
    render_fn(CameraState.default())

    updated_config = result.config_model(
        filtering=result.default_config.filtering.model_copy(
            update={"value": 2}
        )
    )
    updated_render_fn = result.bind(updated_config, backend_fn=backend)
    updated_render_fn(CameraState.default())

    assert compile_calls == [1, 1]


def test_effect_runtime_state_persists_across_renders() -> None:
    @dataclass
    class CounterState:
        count: int = 0

    def effect_apply(
        result: RenderResult,
        config: BaseModel,
        context: ViewerContext,
        runtime_state: CounterState,
    ) -> RenderResult:
        del config, context
        runtime_state.count += 1
        return result

    pipeline = ViewerPipeline(view_factory=_view_factory).effect(
        effect_node(
            name="counter",
            apply=effect_apply,
            state_factory=CounterState,
        )
    )
    result = pipeline.build(source_scene=[], viewer_state=_make_viewer_state())

    def backend(camera_state, compiled_view: _ListRenderView) -> RenderResult:
        del camera_state, compiled_view
        return RenderResult(image=_dummy_image(), metadata={})

    from marimo_3dv.viewer.widget import CameraState

    render_fn = result.bind(result.default_config, backend_fn=backend)
    render_fn(CameraState.default())
    render_fn(CameraState.default())
    render_fn(CameraState.default())

    assert result.runtime_state["counter"].count == 3


def test_viewer_context_carries_viewer_state() -> None:
    captured: list[ViewerContext] = []

    def effect_apply(
        result: RenderResult,
        config: BaseModel,
        context: ViewerContext,
        runtime_state: None,
    ) -> RenderResult:
        del config, runtime_state
        captured.append(context)
        return result

    pipeline = ViewerPipeline(view_factory=_view_factory).effect(
        effect_node(name="ctx", apply=effect_apply)
    )
    viewer_state = _make_viewer_state()
    result = pipeline.build(source_scene=[], viewer_state=viewer_state)

    def backend(camera_state, compiled_view: _ListRenderView) -> RenderResult:
        del camera_state, compiled_view
        return RenderResult(image=_dummy_image(), metadata={})

    from marimo_3dv.viewer.widget import CameraState

    render_fn = result.bind(result.default_config, backend_fn=backend)
    render_fn(CameraState.default())

    assert len(captured) == 1
    assert captured[0].viewer_state is viewer_state
