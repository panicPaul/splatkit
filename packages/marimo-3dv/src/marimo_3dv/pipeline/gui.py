"""Declarative viewer pipeline primitives."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, create_model

from marimo_3dv.pipeline.context import ViewerContext
from marimo_3dv.viewer.widget import CameraState, ViewerState

SceneT = TypeVar("SceneT")
RenderViewT = TypeVar("RenderViewT", bound="AbstractRenderView[Any]")
CompiledViewT = TypeVar("CompiledViewT")
StateT = TypeVar("StateT")


@dataclass(frozen=True)
class AbstractRenderView(Generic[SceneT]):
    """Backend-neutral immutable render view."""

    source_scene: SceneT
    backend_key: str = "generic"
    capabilities: frozenset[str] = frozenset()
    extensions: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RenderResult:
    """Structured output from a backend render."""

    image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class EmptyConfig(BaseModel):
    """Empty config model used by nodes without controls."""


_EMPTY_CONFIG = EmptyConfig()


@dataclass(frozen=True)
class RenderNode(Generic[RenderViewT]):
    """Pure render-view transform."""

    name: str
    apply: Callable[[RenderViewT, BaseModel, ViewerContext], RenderViewT]
    config_model: type[BaseModel] | None = None
    default_config: BaseModel | None = None


@dataclass(frozen=True)
class EffectNode(Generic[CompiledViewT, StateT]):
    """Post-render effect with optional runtime state."""

    name: str
    apply: Callable[
        [RenderResult, BaseModel, ViewerContext, StateT], RenderResult
    ]
    config_model: type[BaseModel] | None = None
    default_config: BaseModel | None = None
    state_factory: Callable[[], StateT] | None = None


@dataclass(frozen=True)
class PipelineGroup:
    """Named group of nodes used to build nested config trees."""

    name: str
    items: tuple[PipelineItem, ...]

    def __init__(self, name: str, *items: PipelineItem) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "items", tuple(items))


PipelineItem = RenderNode[Any] | EffectNode[Any, Any] | PipelineGroup


def render_node(
    *,
    name: str,
    apply: Callable[[RenderViewT, BaseModel, ViewerContext], RenderViewT],
    config_model: type[BaseModel] | None = None,
    default_config: BaseModel | None = None,
) -> RenderNode[RenderViewT]:
    """Create a render-view node."""
    if (config_model is None) != (default_config is None):
        raise ValueError(
            "Render nodes must provide both config_model and default_config "
            "or neither."
        )
    return RenderNode(
        name=name,
        apply=apply,
        config_model=config_model,
        default_config=default_config,
    )


def effect_node(
    *,
    name: str,
    apply: Callable[
        [RenderResult, BaseModel, ViewerContext, StateT], RenderResult
    ],
    config_model: type[BaseModel] | None = None,
    default_config: BaseModel | None = None,
    state_factory: Callable[[], StateT] | None = None,
) -> EffectNode[Any, StateT]:
    """Create a post-render effect node."""
    if (config_model is None) != (default_config is None):
        raise ValueError(
            "Effect nodes must provide both config_model and default_config "
            "or neither."
        )
    return EffectNode(
        name=name,
        apply=apply,
        config_model=config_model,
        default_config=default_config,
        state_factory=state_factory,
    )


@dataclass(frozen=True)
class _ConfiguredRenderNode(Generic[RenderViewT]):
    node: RenderNode[RenderViewT]
    config_path: tuple[str, ...] | None
    config_model: type[BaseModel] | None = None
    flattened_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ConfiguredEffectNode(Generic[StateT]):
    node: EffectNode[Any, StateT]
    config_path: tuple[str, ...] | None
    config_model: type[BaseModel] | None = None
    flattened_fields: tuple[str, ...] = ()


@dataclass
class _PipelineRuntimeState(Generic[SceneT, RenderViewT, CompiledViewT]):
    render_nodes: list[_ConfiguredRenderNode[RenderViewT]]
    effect_nodes: list[_ConfiguredEffectNode[Any]]
    effect_states: dict[str, Any]
    source_scene: SceneT
    viewer_state: ViewerState
    view_factory: Callable[[SceneT], RenderViewT]
    compile_view: Callable[[RenderViewT], CompiledViewT]
    compiled_view: CompiledViewT | None = None
    compiled_view_key: str | None = None


@dataclass
class ViewerPipelineResult(Generic[SceneT, RenderViewT, CompiledViewT]):
    """Built viewer pipeline ready to bind to a backend renderer."""

    config_model: type[BaseModel]
    default_config: BaseModel
    runtime_state: dict[str, Any]
    _pipeline_state: _PipelineRuntimeState[SceneT, RenderViewT, CompiledViewT]

    def bind(
        self,
        config: BaseModel,
        backend_fn: Callable[[CameraState, CompiledViewT], RenderResult],
    ) -> Callable[[CameraState], RenderResult]:
        """Bind a config and backend renderer to a single render function."""

        def render(camera_state: CameraState) -> RenderResult:
            viewer_context = ViewerContext(
                viewer_state=self._pipeline_state.viewer_state,
                last_click=self._pipeline_state.viewer_state.last_click,
            )
            compiled_view = _get_compiled_view(
                config=config,
                context=viewer_context,
                pipeline_state=self._pipeline_state,
            )
            result = backend_fn(camera_state, compiled_view)
            for configured_effect in self._pipeline_state.effect_nodes:
                effect_config = _config_for_node(
                    config,
                    configured_effect.config_path,
                    configured_effect.config_model,
                    configured_effect.flattened_fields,
                )
                effect_state = self._pipeline_state.effect_states.get(
                    configured_effect.node.name
                )
                result = configured_effect.node.apply(
                    result,
                    effect_config,
                    viewer_context,
                    effect_state,
                )
            return result

        return render


def _get_compiled_view(
    *,
    config: BaseModel,
    context: ViewerContext,
    pipeline_state: _PipelineRuntimeState[SceneT, RenderViewT, CompiledViewT],
) -> CompiledViewT:
    render_config_payload = _render_config_payload(
        config,
        pipeline_state.render_nodes,
    )
    compiled_key = json.dumps(
        {
            "scene_id": id(pipeline_state.source_scene),
            "render_config": render_config_payload,
        },
        sort_keys=True,
        default=str,
    )
    if (
        pipeline_state.compiled_view is not None
        and pipeline_state.compiled_view_key == compiled_key
    ):
        return pipeline_state.compiled_view

    render_view = pipeline_state.view_factory(pipeline_state.source_scene)
    for configured_node in pipeline_state.render_nodes:
        node_config = _config_for_node(
            config,
            configured_node.config_path,
            configured_node.config_model,
            configured_node.flattened_fields,
        )
        render_view = configured_node.node.apply(
            render_view,
            node_config,
            context,
        )

    compiled_view = pipeline_state.compile_view(render_view)
    pipeline_state.compiled_view = compiled_view
    pipeline_state.compiled_view_key = compiled_key
    return compiled_view


def _render_config_payload(
    config: BaseModel, render_nodes: Sequence[_ConfiguredRenderNode[Any]]
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for configured_node in render_nodes:
        if configured_node.config_model is None:
            continue
        payload_key = ".".join(configured_node.config_path or ())
        payload[payload_key] = _config_for_node(
            config,
            configured_node.config_path,
            configured_node.config_model,
            configured_node.flattened_fields,
        ).model_dump(mode="json")
    return payload


def _config_for_path(
    config: BaseModel, path: tuple[str, ...] | None
) -> BaseModel:
    if path is None:
        return _EMPTY_CONFIG

    value: Any = config
    for part in path:
        value = getattr(value, part)
    if not isinstance(value, BaseModel):
        raise TypeError(f"Expected BaseModel at config path {path!r}.")
    return value


def _config_for_node(
    config: BaseModel,
    config_path: tuple[str, ...] | None,
    config_model: type[BaseModel] | None,
    flattened_fields: tuple[str, ...],
) -> BaseModel:
    if config_model is None:
        return _EMPTY_CONFIG
    if not flattened_fields:
        return _config_for_path(config, config_path)
    container = _config_for_path(config, config_path)
    return config_model(
        **{
            field_name: getattr(container, field_name)
            for field_name in flattened_fields
        }
    )


@dataclass(frozen=True)
class _ConfigEntry:
    field_name: str
    model: type[BaseModel]
    default_value: BaseModel


def _build_group_config(
    item: PipelineItem,
    *,
    path: tuple[str, ...],
    render_nodes: list[_ConfiguredRenderNode[Any]],
    effect_nodes: list[_ConfiguredEffectNode[Any]],
) -> _ConfigEntry | None:
    if isinstance(item, PipelineGroup):
        group_fields: dict[str, tuple[Any, Any]] = {}
        group_defaults: dict[str, Any] = {}
        for child in item.items:
            if isinstance(child, RenderNode):
                if child.config_model is None or child.default_config is None:
                    render_nodes.append(
                        _ConfiguredRenderNode(node=child, config_path=None)
                    )
                    continue
                child_defaults = child.default_config.model_dump()
                flattened_fields = tuple(child.config_model.model_fields)
                for (
                    field_name,
                    field_info,
                ) in child.config_model.model_fields.items():
                    if field_name in group_fields:
                        raise ValueError(
                            f"Duplicate config field {field_name!r} "
                            f"inside group {item.name!r}."
                        )
                    group_fields[field_name] = (
                        field_info.annotation,
                        field_info,
                    )
                    group_defaults[field_name] = child_defaults.get(field_name)
                render_nodes.append(
                    _ConfiguredRenderNode(
                        node=child,
                        config_path=(*path, item.name),
                        config_model=child.config_model,
                        flattened_fields=flattened_fields,
                    )
                )
                continue
            if isinstance(child, EffectNode):
                if child.config_model is None or child.default_config is None:
                    effect_nodes.append(
                        _ConfiguredEffectNode(node=child, config_path=None)
                    )
                    continue
                child_defaults = child.default_config.model_dump()
                flattened_fields = tuple(child.config_model.model_fields)
                for (
                    field_name,
                    field_info,
                ) in child.config_model.model_fields.items():
                    if field_name in group_fields:
                        raise ValueError(
                            f"Duplicate config field {field_name!r} "
                            f"inside group {item.name!r}."
                        )
                    group_fields[field_name] = (
                        field_info.annotation,
                        field_info,
                    )
                    group_defaults[field_name] = child_defaults.get(field_name)
                effect_nodes.append(
                    _ConfiguredEffectNode(
                        node=child,
                        config_path=(*path, item.name),
                        config_model=child.config_model,
                        flattened_fields=flattened_fields,
                    )
                )
                continue
            child_entry = _build_group_config(
                child,
                path=(*path, item.name),
                render_nodes=render_nodes,
                effect_nodes=effect_nodes,
            )
            if child_entry is None:
                continue
            if child_entry.field_name in group_fields:
                raise ValueError(
                    f"Duplicate config field {child_entry.field_name!r} "
                    f"inside group {item.name!r}."
                )
            group_fields[child_entry.field_name] = (
                child_entry.model,
                child_entry.default_value,
            )
            group_defaults[child_entry.field_name] = child_entry.default_value

        if not group_fields:
            return None

        group_model = create_model(
            f"{_title_case(item.name)}ConfigGroup",
            **group_fields,
        )
        return _ConfigEntry(
            field_name=item.name,
            model=group_model,
            default_value=group_model(**group_defaults),
        )

    if isinstance(item, RenderNode):
        config_path = None if item.config_model is None else (*path, item.name)
        render_nodes.append(
            _ConfiguredRenderNode(
                node=item,
                config_path=config_path,
                config_model=item.config_model,
            )
        )
        if item.config_model is None or item.default_config is None:
            return None
        return _ConfigEntry(
            field_name=item.name,
            model=item.config_model,
            default_value=item.default_config,
        )

    config_path = None if item.config_model is None else (*path, item.name)
    effect_nodes.append(
        _ConfiguredEffectNode(
            node=item,
            config_path=config_path,
            config_model=item.config_model,
        )
    )
    if item.config_model is None or item.default_config is None:
        return None
    return _ConfigEntry(
        field_name=item.name,
        model=item.config_model,
        default_value=item.default_config,
    )


def _title_case(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_"))


def _build_config_model(
    render_items: Sequence[PipelineItem],
    effect_items: Sequence[PipelineItem],
) -> tuple[
    type[BaseModel],
    BaseModel,
    list[_ConfiguredRenderNode[Any]],
    list[_ConfiguredEffectNode[Any]],
]:
    render_nodes: list[_ConfiguredRenderNode[Any]] = []
    effect_nodes: list[_ConfiguredEffectNode[Any]] = []
    top_level_fields: dict[str, tuple[type[BaseModel], BaseModel]] = {}

    for item in (*render_items, *effect_items):
        config_entry = _build_group_config(
            item,
            path=(),
            render_nodes=render_nodes,
            effect_nodes=effect_nodes,
        )
        if config_entry is None:
            continue
        if config_entry.field_name in top_level_fields:
            raise ValueError(
                f"Duplicate top-level config field {config_entry.field_name!r}."
            )
        top_level_fields[config_entry.field_name] = (
            config_entry.model,
            config_entry.default_value,
        )

    root_model = create_model("ViewerPipelineConfig", **top_level_fields)
    defaults = {
        field_name: default_value
        for field_name, (_, default_value) in top_level_fields.items()
    }
    return root_model, root_model(**defaults), render_nodes, effect_nodes


class ViewerPipeline(Generic[SceneT, RenderViewT, CompiledViewT]):
    """Declarative pipeline for viewer render views and post-render effects."""

    def __init__(
        self,
        *,
        view_factory: Callable[[SceneT], RenderViewT],
        compile_view: Callable[[RenderViewT], CompiledViewT] | None = None,
    ) -> None:
        self._view_factory = view_factory
        self._compile_view = (
            compile_view if compile_view is not None else lambda view: view  # type: ignore[return-value]
        )
        self._render_items: list[PipelineItem] = []
        self._effect_items: list[PipelineItem] = []

    def render(
        self, item: RenderNode[RenderViewT] | PipelineGroup
    ) -> ViewerPipeline[SceneT, RenderViewT, CompiledViewT]:
        """Return a new pipeline with a render-view item appended."""
        next_pipeline = ViewerPipeline(
            view_factory=self._view_factory,
            compile_view=self._compile_view,
        )
        next_pipeline._render_items = [*self._render_items, item]
        next_pipeline._effect_items = list(self._effect_items)
        return next_pipeline

    def effect(
        self, item: EffectNode[CompiledViewT, Any] | PipelineGroup
    ) -> ViewerPipeline[SceneT, RenderViewT, CompiledViewT]:
        """Return a new pipeline with an effect item appended."""
        next_pipeline = ViewerPipeline(
            view_factory=self._view_factory,
            compile_view=self._compile_view,
        )
        next_pipeline._render_items = list(self._render_items)
        next_pipeline._effect_items = [*self._effect_items, item]
        return next_pipeline

    def build(
        self,
        source_scene: SceneT,
        viewer_state: ViewerState,
    ) -> ViewerPipelineResult[SceneT, RenderViewT, CompiledViewT]:
        """Build the pipeline for one source scene."""
        (
            config_model,
            default_config,
            configured_render_nodes,
            configured_effect_nodes,
        ) = _build_config_model(self._render_items, self._effect_items)

        effect_states: dict[str, Any] = {}
        for configured_effect in configured_effect_nodes:
            if configured_effect.node.state_factory is not None:
                effect_states[configured_effect.node.name] = (
                    configured_effect.node.state_factory()
                )

        pipeline_state = _PipelineRuntimeState(
            render_nodes=configured_render_nodes,
            effect_nodes=configured_effect_nodes,
            effect_states=effect_states,
            source_scene=source_scene,
            viewer_state=viewer_state,
            view_factory=self._view_factory,
            compile_view=self._compile_view,
        )
        return ViewerPipelineResult(
            config_model=config_model,
            default_config=default_config,
            runtime_state=effect_states,
            _pipeline_state=pipeline_state,
        )


__all__ = [
    "AbstractRenderView",
    "EffectNode",
    "EmptyConfig",
    "PipelineGroup",
    "RenderNode",
    "RenderResult",
    "ViewerPipeline",
    "ViewerPipelineResult",
    "effect_node",
    "render_node",
]
