"""Generate marimo forms from Pydantic models."""

from __future__ import annotations

import asyncio
import html
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

import annotated_types
import marimo as mo
import numpy as np
import torch
import tyro
from jaxtyping import AbstractArray
from marimo._plugins.core.web_component import JSONType
from marimo._plugins.ui._core.ui_element import UIElement
from marimo._runtime.commands import UpdateUIElementCommand
from marimo._runtime.context import ContextNotInitializedError, get_context
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from tyro import _docstrings as _tyro_docstrings

ModelT = TypeVar("ModelT", bound=BaseModel)

_MAX_MATRIX_CELLS = 100
_DEFAULT_SLIDER_STEPS = 100
_FORM_TAB = "Form"
_JSON_TAB = "JSON"
_JSON_EDITOR_KEY = "__json_editor__"
_TABS_KEY = "__tabs__"
_DIRECT_JSON_EDITOR_KEY = "__direct_json_editor__"
_NULLABLE_ENABLED_KEY = "__enabled__"
_NULLABLE_VALUE_KEY = "__value__"
_NULLABLE_NONE_TAB = "None"
_NULLABLE_SET_TAB = "Configure"
_UNION_ACTIVE_KEY = "__active__"
_UNION_KIND_KEY = "__kind__"

_RenderMode = Literal["auto", "json", "widget"]
_WidgetMode = Literal["auto", "slider"]


@dataclass(frozen=True)
class _NumericBounds:
    lower: int | float | None
    upper: int | float | None
    step: int | float | None
    strict_lower: bool
    strict_upper: bool


@dataclass(frozen=True)
class _ArrayShape:
    ndim: int
    fixed_shape: tuple[int, ...] | None


@dataclass(frozen=True)
class ConfigBindings(Generic[ModelT]):
    """Setter and metadata bindings for config GUI helpers.

    Attributes:
        model_cls: The Pydantic model type being edited.
        set_form_gui_state: Setter for the structured form GUI state.
        set_json_gui_state: Setter for the JSON draft text state.
    """

    model_cls: type[ModelT]
    set_form_gui_state: Callable[[dict[str, Any]], None]
    set_json_gui_state: Callable[[str], None]


@dataclass(frozen=True)
class _JsonConfigSource:
    path: Annotated[Path, tyro.conf.Positional]


ScriptConfigLoader = Callable[
    [type[BaseModel], BaseModel | dict[str, Any] | None, Sequence[str] | None],
    BaseModel,
]


@dataclass(frozen=True)
class _FieldGuiConfig:
    render: _RenderMode = "auto"
    flat: bool = False
    widget: _WidgetMode = "auto"


@dataclass(frozen=True)
class _FieldSpec:
    model_cls: type[BaseModel]
    name: str
    annotation: Any
    effective_annotation: Any
    info: FieldInfo
    is_optional: bool
    is_nested_model: bool
    is_model_union: bool
    render_mode: _RenderMode
    flat: bool
    widget_mode: _WidgetMode

    def label(self) -> str:
        return self.info.title or self.name.replace("_", " ").capitalize()

    def help_text(self) -> str | None:
        if self.info.description:
            return self.info.description
        return _tyro_help_text_for_field(self.model_cls, self.name)

    @property
    def is_structural_field(self) -> bool:
        return self.is_nested_model or self.is_model_union

    @property
    def force_json_editor(self) -> bool:
        return self.render_mode == "json"

    def parse_frontend_value(
        self,
        element: UIElement[Any, Any],
        frontend_value: JSONType,
        *,
        update_children: bool,
    ) -> Any:
        if isinstance(element, NullableGui):
            return element._parse_frontend_value(
                frontend_value,
                update_children=update_children,
            )

        if isinstance(element, ModelUnionGui):
            return element._parse_frontend_value(
                frontend_value,
                update_children=update_children,
            )

        if isinstance(element, ModelTupleGui):
            return element._parse_frontend_value(
                frontend_value,
                update_children=update_children,
            )

        if isinstance(element, PydanticGui):
            payload, _ = element._payload_from_frontend(
                frontend_value,
                update_children=update_children,
                force_json=False,
            )
            return payload

        return _parse_element_frontend_value(
            self,
            element,
            frontend_value,
            update_children=update_children,
        )

    def to_model_value(self, value: Any) -> Any:
        if self.is_nested_model or self.is_model_union:
            return value
        if _is_model_tuple_type(self.effective_annotation):
            return tuple(value)
        if self.effective_annotation is Path:
            return _coerce_path_value(self.info, value)
        if _is_array_annotation(self.effective_annotation):
            return _coerce_array_value(self.effective_annotation, value)
        if _uses_text_fallback(self.effective_annotation) and isinstance(
            value, str
        ):
            return _maybe_parse_json_text(value)
        return value


class NullableGui(UIElement[dict[str, JSONType], Any]):
    """Wrap a child control with a nullable on/off switch."""

    _name = "marimo-dict"

    def __init__(
        self,
        spec: _FieldSpec,
        child: UIElement[Any, Any],
        *,
        enabled: bool,
        on_change: Any | None = None,
    ) -> None:
        self._spec = spec
        self._child = child
        self._toggle = mo.ui.tabs(
            {
                _NULLABLE_NONE_TAB: mo.md(""),
                _NULLABLE_SET_TAB: mo.callout(child, kind="neutral"),
            },
            value=_NULLABLE_SET_TAB if enabled else _NULLABLE_NONE_TAB,
            label="",
        )
        label = mo.md(
            "<span style="
            '"font-weight: 500;"'
            ">"
            f"{html.escape(spec.label())}"
            "</span>"
        )
        layout_html = (
            '<div style="'
            "display: flex; "
            "align-items: center; "
            "justify-content: flex-start; "
            "gap: 0.5rem; "
            'width: 100%;">'
            f'<div style="flex: 0 0 7rem;">{label.text}</div>'
            '<div style="flex: 1 1 0; min-width: 0; width: 100%;">'
            f"{self._toggle.text}"
            "</div>"
            "</div>"
        )
        self._draft_frontend_value = _current_element_frontend_value(child)
        self._elements = {
            _NULLABLE_ENABLED_KEY: self._toggle,
            _NULLABLE_VALUE_KEY: child,
        }
        super().__init__(
            component_name=self._name,
            initial_value=self._current_frontend_value(),
            label="",
            args={
                "element-ids": {
                    element._id: name
                    for name, element in self._elements.items()
                }
            },
            slotted_html=layout_html,
            on_change=on_change,
        )
        for name, element in self._elements.items():
            element._register_as_view(parent=self, key=name)

    @property
    def elements(self) -> dict[str, UIElement[Any, Any]]:
        return self._elements

    def _clone(self) -> NullableGui:
        return type(self)(
            self._spec,
            self._child._clone(),
            enabled=self._toggle.value == _NULLABLE_SET_TAB,
            on_change=self._on_change,
        )

    def _current_frontend_value(self) -> dict[str, JSONType]:
        return {
            _NULLABLE_ENABLED_KEY: _current_element_frontend_value(
                self._toggle
            ),
            _NULLABLE_VALUE_KEY: self._draft_frontend_value,
        }

    def _convert_value(self, value: dict[str, JSONType]) -> Any:
        return self._parse_frontend_value(value, update_children=True)

    def _parse_frontend_value(
        self,
        value: JSONType,
        *,
        update_children: bool,
    ) -> Any:
        merged = self._current_frontend_value()
        if isinstance(value, dict):
            merged.update(value)

        enabled_frontend = merged.get(
            _NULLABLE_ENABLED_KEY,
            _current_element_frontend_value(self._toggle),
        )
        if isinstance(enabled_frontend, bool):
            enabled_frontend = self._tabs_frontend_value(enabled_frontend)
        child_frontend = merged.get(
            _NULLABLE_VALUE_KEY,
            _current_element_frontend_value(self._child),
        )

        if update_children:
            if self._toggle._value_frontend != enabled_frontend:
                _set_local_frontend_value(self._toggle, enabled_frontend)
            if self._child._value_frontend != child_frontend:
                _set_local_frontend_value(self._child, child_frontend)
            self._draft_frontend_value = child_frontend

        enabled = self._is_enabled_frontend_value(enabled_frontend)
        if not enabled:
            return None
        return self._parse_child_frontend_value(
            child_frontend,
            update_children=update_children,
        )

    def _parse_child_frontend_value(
        self,
        frontend_value: JSONType,
        *,
        update_children: bool,
    ) -> Any:
        return _parse_structural_frontend_value(
            self._spec,
            self._child,
            frontend_value,
            update_children=update_children,
        )

    def _frontend_value_from_payload(self, value: Any) -> dict[str, JSONType]:
        if value is None:
            child_frontend = self._draft_frontend_value
        else:
            child_frontend = _frontend_value_for_element(
                self._spec,
                self._child,
                value,
            )
            self._draft_frontend_value = child_frontend
        return {
            _NULLABLE_ENABLED_KEY: self._tabs_frontend_value(value is not None),
            _NULLABLE_VALUE_KEY: child_frontend,
        }

    def _is_enabled_frontend_value(self, frontend_value: JSONType) -> bool:
        if isinstance(frontend_value, bool):
            return frontend_value
        return self._toggle._convert_value(frontend_value) == _NULLABLE_SET_TAB

    def _tabs_frontend_value(self, enabled: bool) -> JSONType:
        return 1 if enabled else 0


class ModelUnionGui(UIElement[dict[str, JSONType], Any]):
    """Render a union of nested Pydantic models as selectable alternatives."""

    _name = "marimo-dict"

    def __init__(
        self,
        spec: _FieldSpec,
        branch_models: tuple[type[BaseModel], ...],
        children: tuple[UIElement[Any, Any], ...],
        *,
        active_index: int = 0,
        on_change: Any | None = None,
    ) -> None:
        self._spec = spec
        self._branch_models = branch_models
        self._children = children
        self._branch_keys = tuple(str(index) for index in range(len(children)))
        self._branch_labels = _disambiguate_labels(
            [_humanize_model_name(model_cls) for model_cls in branch_models]
        )
        self._branch_kinds = tuple(
            _union_kind_for_model(branch_models, index)
            for index, _ in enumerate(branch_models)
        )
        self._key_to_index = {
            key: index for index, key in enumerate(self._branch_keys)
        }
        tabs = {
            label: mo.callout(child, kind="neutral")
            for label, child in zip(self._branch_labels, children, strict=False)
        }
        initial_label = self._branch_labels[active_index]
        self._selector = mo.ui.tabs(
            tabs,
            value=initial_label,
            lazy=False,
            label="",
        )
        label = mo.md(
            "<span style="
            '"font-weight: 500;"'
            ">"
            f"{html.escape(spec.label())}"
            "</span>"
        )
        layout_html = (
            '<div style="'
            "display: flex; "
            "align-items: flex-start; "
            "justify-content: flex-start; "
            "gap: 0.5rem; "
            'width: 100%;">'
            f'<div style="flex: 0 0 7rem; padding-top: 0.125rem;">{label.text}</div>'
            '<div style="flex: 1 1 0; min-width: 0; width: 100%;">'
            f"{self._selector.text}"
            "</div>"
            "</div>"
        )
        self._elements = {_UNION_ACTIVE_KEY: self._selector}
        self._elements.update(
            {
                key: child
                for key, child in zip(self._branch_keys, children, strict=False)
            }
        )
        super().__init__(
            component_name=self._name,
            initial_value=self._current_frontend_value(),
            label="",
            args={
                "element-ids": {
                    element._id: name
                    for name, element in self._elements.items()
                }
            },
            slotted_html=layout_html,
            on_change=on_change,
        )
        for name, element in self._elements.items():
            element._register_as_view(parent=self, key=name)

    @property
    def elements(self) -> dict[str, UIElement[Any, Any]]:
        return self._elements

    def _clone(self) -> ModelUnionGui:
        return type(self)(
            self._spec,
            self._branch_models,
            tuple(child._clone() for child in self._children),
            active_index=self._active_index_from_frontend(
                _current_element_frontend_value(self._selector)
            ),
            on_change=self._on_change,
        )

    def _current_frontend_value(self) -> dict[str, JSONType]:
        current = {
            _UNION_ACTIVE_KEY: _current_element_frontend_value(self._selector)
        }
        for key, child in zip(self._branch_keys, self._children, strict=False):
            current[key] = child._value_frontend
        return current

    def _convert_value(self, value: dict[str, JSONType]) -> Any:
        return self._parse_frontend_value(value, update_children=True)

    def _parse_frontend_value(
        self,
        value: JSONType,
        *,
        update_children: bool,
    ) -> Any:
        merged = self._current_frontend_value()
        if isinstance(value, dict):
            merged.update(value)

        active_frontend = merged.get(
            _UNION_ACTIVE_KEY,
            _current_element_frontend_value(self._selector),
        )
        active_index = self._active_index_from_frontend(active_frontend)
        if (
            update_children
            and self._selector._value_frontend != active_frontend
        ):
            _set_local_frontend_value(self._selector, active_frontend)

        for key, child in zip(self._branch_keys, self._children, strict=False):
            child_frontend = merged.get(
                key, _current_element_frontend_value(child)
            )
            if update_children and child._value_frontend != child_frontend:
                _apply_structural_child_frontend(
                    self._spec,
                    child,
                    child_frontend,
                )

        active_key = self._branch_keys[active_index]
        active_child = self._children[active_index]
        active_frontend_value = merged.get(
            active_key,
            _current_element_frontend_value(active_child),
        )
        payload = _parse_structural_frontend_value(
            self._spec,
            active_child,
            active_frontend_value,
            update_children=update_children,
        )
        return _union_payload(self._branch_kinds[active_index], payload)

    def _frontend_value_from_payload(self, value: Any) -> dict[str, JSONType]:
        active_index = self._select_active_index(value)
        frontend = {_UNION_ACTIVE_KEY: self._tabs_frontend_value(active_index)}
        for index, (key, child) in enumerate(
            zip(self._branch_keys, self._children, strict=False)
        ):
            if index == active_index:
                frontend[key] = _frontend_value_for_structural_child(
                    child,
                    self._branch_models[index],
                    value,
                )
            else:
                frontend[key] = child._value_frontend
        return frontend

    def _active_index_from_frontend(self, frontend_value: JSONType) -> int:
        label = self._selector._convert_value(frontend_value)
        return self._branch_labels.index(label)

    def _tabs_frontend_value(self, index: int) -> JSONType:
        return index

    def _select_active_index(self, value: Any) -> int:
        current_active = self._active_index_from_frontend(
            _current_element_frontend_value(self._selector)
        )
        for index in [current_active, *range(len(self._branch_models))]:
            if self._branch_accepts_value(index, value):
                return index
        return 0

    def _branch_accepts_value(self, index: int, value: Any) -> bool:
        return _union_value_matches_branch(
            self._branch_models,
            index,
            value,
        )


class ModelTupleGui(UIElement[dict[str, JSONType], Any]):
    """Render a fixed tuple of nested Pydantic models as an ordered sequence."""

    _name = "marimo-dict"

    def __init__(
        self,
        spec: _FieldSpec,
        children: tuple[PydanticGui[Any], ...],
        *,
        on_change: Any | None = None,
    ) -> None:
        self._spec = spec
        self._children = children
        steps = [
            mo.vstack(
                [
                    mo.md(
                        f"**Step {index + 1}: "
                        f"{_humanize_model_name(child._model_cls)}**"
                    ),
                    mo.callout(child, kind="neutral"),
                ],
                align="stretch",
            )
            for index, child in enumerate(children)
        ]
        content = mo.vstack(steps, align="stretch")
        label = mo.md(
            "<span style="
            '"font-weight: 500;"'
            ">"
            f"{html.escape(spec.label())}"
            "</span>"
        )
        layout_html = (
            '<div style="'
            "display: flex; "
            "align-items: flex-start; "
            "justify-content: flex-start; "
            "gap: 0.5rem; "
            'width: 100%;">'
            f'<div style="flex: 0 0 7rem; padding-top: 0.125rem;">{label.text}</div>'
            '<div style="flex: 1 1 0; min-width: 0; width: 100%;">'
            f"{content.text}"
            "</div>"
            "</div>"
        )
        self._elements = {
            str(index): child for index, child in enumerate(children)
        }
        super().__init__(
            component_name=self._name,
            initial_value=self._current_frontend_value(),
            label="",
            args={
                "element-ids": {
                    element._id: name
                    for name, element in self._elements.items()
                }
            },
            slotted_html=layout_html,
            on_change=on_change,
        )
        for name, element in self._elements.items():
            element._register_as_view(parent=self, key=name)

    @property
    def elements(self) -> dict[str, UIElement[Any, Any]]:
        return self._elements

    def _clone(self) -> ModelTupleGui:
        return type(self)(
            self._spec,
            tuple(child._clone() for child in self._children),
            on_change=self._on_change,
        )

    def _current_frontend_value(self) -> dict[str, JSONType]:
        return {
            name: _current_element_frontend_value(element)
            for name, element in self._elements.items()
        }

    def _convert_value(self, value: dict[str, JSONType]) -> Any:
        return self._parse_frontend_value(value, update_children=True)

    def _parse_frontend_value(
        self,
        value: JSONType,
        *,
        update_children: bool,
    ) -> tuple[Any, ...]:
        merged = self._current_frontend_value()
        if isinstance(value, dict):
            merged.update(value)

        payloads: list[Any] = []
        for name, child in self._elements.items():
            child_frontend = merged.get(
                name, _current_element_frontend_value(child)
            )
            if update_children and child._value_frontend != child_frontend:
                _set_local_frontend_value(child, child_frontend)
            payload, _ = child._payload_from_frontend(
                child_frontend,
                update_children=update_children,
                force_json=False,
            )
            payloads.append(payload)
        return tuple(payloads)

    def _frontend_value_from_payload(self, value: Any) -> dict[str, JSONType]:
        values: tuple[Any, ...]
        if isinstance(value, tuple):
            values = value
        elif isinstance(value, list):
            values = tuple(value)
        else:
            values = tuple()

        frontend: dict[str, JSONType] = {}
        for index, child in enumerate(self._children):
            payload = values[index] if index < len(values) else {}
            frontend[str(index)] = child._frontend_value_from_payload(
                payload if isinstance(payload, dict) else {}
            )
        return frontend


class PydanticGui(
    UIElement[dict[str, JSONType], ModelT | None], Generic[ModelT]
):
    """Internal marimo UI element for Pydantic-backed forms."""

    _name = "marimo-dict"

    def __init__(
        self,
        model_cls: type[ModelT],
        *,
        value: ModelT | dict[str, Any] | None = None,
        label: str = "",
        include_json_editor: bool = True,
        bordered: bool = False,
        nested_models_multiple_open: bool = True,
        nested_models_flat_after_level: int | None = None,
        current_level: int = 0,
        on_change: Any | None = None,
    ) -> None:
        self._model_cls = model_cls
        self._label = label
        self._include_json_editor = include_json_editor
        self._bordered = bordered
        self._nested_models_multiple_open = nested_models_multiple_open
        self._nested_models_flat_after_level = nested_models_flat_after_level
        self._current_level = current_level
        self._last_active_tab = _FORM_TAB if include_json_editor else ""
        self._last_json_error: str | None = None
        self._initial_payload = _resolve_initial_payload(model_cls, value)
        self._last_payload = self._initial_payload
        self._sync_task: asyncio.Task[Any] | None = None

        field_specs, field_elements, form_layout = _build_model_gui(
            model_cls=model_cls,
            payload=self._last_payload,
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            current_level=current_level,
        )
        self._field_specs = field_specs
        self._field_elements = field_elements
        self._field_names = list(field_elements)

        elements: dict[str, UIElement[Any, Any]] = dict(field_elements)
        layout_blocks: list[Any] = []
        self._json_editor: UIElement[Any, Any] | None = None
        self._tabs: UIElement[Any, Any] | None = None

        if include_json_editor:
            json_editor = mo.ui.code_editor(
                value=_payload_to_json(self._last_payload),
                language="json",
                show_copy_button=True,
                debounce=False,
                label="",
            )
            tabs = mo.ui.tabs(
                {
                    _FORM_TAB: form_layout,
                    _JSON_TAB: json_editor,
                },
                value=_FORM_TAB,
                label="",
            )
            elements[_JSON_EDITOR_KEY] = json_editor
            elements[_TABS_KEY] = tabs
            self._json_editor = json_editor
            self._tabs = tabs
            layout_blocks.append(tabs)
        else:
            layout_blocks.append(form_layout)

        layout = mo.vstack(layout_blocks, align="stretch")
        slotted_html = layout.text
        if bordered:
            slotted_html = _wrap_live_update_layout(slotted_html)

        self._elements = elements
        super().__init__(
            component_name=self._name,
            initial_value={
                name: element._initial_value_frontend
                for name, element in self._elements.items()
            },
            label=label,
            args={
                "element-ids": {
                    element._id: name
                    for name, element in self._elements.items()
                }
            },
            slotted_html=slotted_html,
            on_change=on_change,
        )
        for name, element in self._elements.items():
            element._register_as_view(parent=self, key=name)

    @property
    def elements(self) -> dict[str, UIElement[Any, Any]]:
        """Return child UI elements keyed by field name."""
        return self._elements

    def _clone(self) -> PydanticGui[ModelT]:
        return type(self)(
            self._model_cls,
            value=self._last_payload,
            label=self._label,
            include_json_editor=self._include_json_editor,
            bordered=self._bordered,
            nested_models_multiple_open=self._nested_models_multiple_open,
            nested_models_flat_after_level=self._nested_models_flat_after_level,
            current_level=self._current_level,
            on_change=self._on_change,
        )

    def _convert_value(self, value: dict[str, JSONType]) -> ModelT | None:
        self._apply_non_field_partials(value)
        merged_value = self._merged_frontend_value(value)
        active_tab = self._active_tab_name(merged_value)
        tab_switched = (
            self._include_json_editor
            and _TABS_KEY in value
            and bool(self._last_active_tab)
            and active_tab != self._last_active_tab
        )
        source_tab = self._source_tab_name(merged_value, value)
        payload, json_error = self._payload_from_frontend(
            merged_value,
            update_children=True,
            force_json=source_tab == _JSON_TAB,
        )
        self._last_payload = payload
        model_value, validation_error = _validate_payload_with_error(
            self._model_cls,
            payload,
        )
        current_error = json_error or validation_error

        if self._include_json_editor:
            if (
                tab_switched
                and current_error is not None
                and self._tabs is not None
            ):
                self._sync_elements(
                    [
                        (
                            self._tabs,
                            self._tabs_frontend_value(self._last_active_tab),
                        )
                    ],
                    force=True,
                )
                self._last_json_error = json_error
                return None
            if source_tab == _FORM_TAB:
                self._sync_json_editor(payload, force=tab_switched)
            elif json_error is None:
                self._sync_field_controls(payload, force=tab_switched)
            self._last_active_tab = active_tab
            self._last_json_error = json_error

        if current_error is not None:
            return None
        return model_value

    def _payload_from_frontend(
        self,
        value: dict[str, JSONType],
        *,
        update_children: bool,
        force_json: bool,
    ) -> tuple[dict[str, Any], str | None]:
        payload = self._field_payload_from_frontend(
            value,
            update_children=update_children,
        )
        if not self._include_json_editor:
            return payload, None

        active_tab = self._active_tab_name(value)
        should_use_json = (
            force_json
            or active_tab == _JSON_TAB
            or self._last_active_tab == _JSON_TAB
        )
        if not should_use_json:
            return payload, None
        return self._merge_json_payload(value[_JSON_EDITOR_KEY], payload)

    def _field_payload_from_frontend(
        self,
        value: dict[str, JSONType],
        *,
        update_children: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self._field_names:
            spec = self._field_specs[name]
            frontend_value = value.get(
                name,
                _current_element_frontend_value(self._field_elements[name]),
            )
            payload[name] = spec.parse_frontend_value(
                self._field_elements[name],
                frontend_value,
                update_children=update_children,
            )
        return payload

    def _merged_frontend_value(
        self,
        value: dict[str, JSONType],
    ) -> dict[str, JSONType]:
        merged: dict[str, JSONType] = self._current_frontend_value()
        merged.update(value)
        return merged

    def _current_frontend_value(self) -> dict[str, JSONType]:
        current: dict[str, JSONType] = {}
        for name, element in self._elements.items():
            current[name] = _current_element_frontend_value(element)
        return current

    def _apply_non_field_partials(self, value: dict[str, JSONType]) -> None:
        if self._include_json_editor and self._json_editor is not None:
            if _JSON_EDITOR_KEY in value:
                _set_local_frontend_value(
                    self._json_editor,
                    value[_JSON_EDITOR_KEY],
                )
        if self._include_json_editor and self._tabs is not None:
            if _TABS_KEY in value:
                _set_local_frontend_value(
                    self._tabs,
                    value[_TABS_KEY],
                )

    def _merge_json_payload(
        self,
        json_text: JSONType,
        base_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        if not isinstance(json_text, str):
            return base_payload, "JSON editor value must be a string."

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return base_payload, f"json: {exc.msg}"

        if not isinstance(parsed, dict):
            return base_payload, "json: top-level JSON value must be an object."

        merged = dict(base_payload)
        for name in self._field_names:
            if name in parsed:
                merged[name] = _merge_json_value(merged.get(name), parsed[name])
        return merged, None

    def validate_frontend_value(
        self, value: dict[str, JSONType] | None
    ) -> str | None:
        """Validate a frontend payload without mutating the committed value."""
        if value is None:
            return None

        merged_value = self._merged_frontend_value(value)

        payload, json_error = self._payload_from_frontend(
            merged_value,
            update_children=False,
            force_json=True,
        )
        if json_error is not None:
            return json_error

        _, error = _validate_payload_with_error(self._model_cls, payload)
        return error

    def get_model(self) -> ModelT | None:
        """Return the current validated model value, if any."""
        return self.value

    def json(self) -> Any | None:
        """Return the current validated value serialized as JSON, if any."""
        if self.value is None:
            return _invalid_json_output(
                self.validate_frontend_value(self._current_frontend_value())
            )
        return _json_output(_jsonify_model_value(self.value))

    def error(self) -> Any:
        """Return the current validation output."""
        return _validation_output(
            self.validate_frontend_value(self._current_frontend_value())
        )

    def view(self) -> Any:
        """Return a notebook-friendly view with validation output."""
        return mo.vstack([self.error(), self], align="stretch")

    def _sync_json_editor(
        self, payload: dict[str, Any], *, force: bool = False
    ) -> None:
        if self._json_editor is None:
            return
        json_text = _payload_to_json(payload)
        self._sync_elements([(self._json_editor, json_text)], force=force)

    def _active_tab_name(self, value: dict[str, JSONType]) -> str:
        if not self._include_json_editor or self._tabs is None:
            return _FORM_TAB
        raw_value = value.get(_TABS_KEY, self._tabs._value_frontend)
        return self._tabs._convert_value(raw_value)

    def _source_tab_name(
        self,
        merged_value: dict[str, JSONType],
        incoming_value: dict[str, JSONType],
    ) -> str:
        active_tab = self._active_tab_name(merged_value)
        if (
            _TABS_KEY in incoming_value
            and active_tab != self._last_active_tab
            and self._last_active_tab
        ):
            return self._last_active_tab
        return active_tab

    def _tabs_frontend_value(self, tab_name: str) -> JSONType:
        return 0 if tab_name == _FORM_TAB else 1

    def _sync_field_controls(
        self,
        payload: dict[str, Any],
        *,
        force: bool = False,
    ) -> None:
        updates: list[tuple[UIElement[Any, Any], JSONType]] = []
        for name in self._field_names:
            element = self._field_elements[name]
            self._collect_sync_updates(
                self._field_specs[name],
                element,
                payload.get(name),
                updates,
            )
        self._sync_elements(updates, force=force)

    def _collect_sync_updates(
        self,
        spec: _FieldSpec,
        element: UIElement[Any, Any],
        value: Any,
        updates: list[tuple[UIElement[Any, Any], JSONType]],
    ) -> None:
        if isinstance(element, NullableGui):
            nullable_frontend = element._frontend_value_from_payload(value)
            _set_local_frontend_value(element, nullable_frontend)
            updates.append(
                (
                    element.elements[_NULLABLE_ENABLED_KEY],
                    nullable_frontend[_NULLABLE_ENABLED_KEY],
                )
            )
            if value is not None:
                self._collect_sync_updates(
                    spec,
                    element.elements[_NULLABLE_VALUE_KEY],
                    value,
                    updates,
                )
            return

        if isinstance(element, ModelUnionGui):
            union_frontend = element._frontend_value_from_payload(value)
            _set_local_frontend_value(element, union_frontend)
            updates.append(
                (
                    element.elements[_UNION_ACTIVE_KEY],
                    union_frontend[_UNION_ACTIVE_KEY],
                )
            )
            active_index = element._active_index_from_frontend(
                union_frontend[_UNION_ACTIVE_KEY]
            )
            active_child = element._children[active_index]
            active_model = element._branch_models[active_index]
            self._collect_sync_updates(
                spec,
                active_child,
                _payload_for_branch_model(active_model, value),
                updates,
            )
            return

        if isinstance(element, PydanticGui):
            nested_payload = value if isinstance(value, dict) else {}
            nested_frontend = element._frontend_value_from_payload(
                nested_payload
            )
            _set_local_frontend_value(element, nested_frontend)
            for child_name in element._field_names:
                element._collect_sync_updates(
                    element._field_specs[child_name],
                    element._field_elements[child_name],
                    nested_payload.get(child_name),
                    updates,
                )
            return

        frontend_value = _frontend_value_for_element(
            spec,
            element,
            value,
        )
        updates.append((element, frontend_value))

    def _frontend_value_from_payload(
        self, payload: dict[str, Any]
    ) -> dict[str, JSONType]:
        frontend_value: dict[str, JSONType] = {}
        for name in self._field_names:
            frontend_value[name] = _frontend_value_for_element(
                self._field_specs[name],
                self._field_elements[name],
                payload.get(name),
            )

        if (
            self._include_json_editor
            and self._json_editor is not None
            and self._tabs is not None
        ):
            frontend_value[_JSON_EDITOR_KEY] = _payload_to_json(payload)
            frontend_value[_TABS_KEY] = self._tabs._value_frontend
        return frontend_value

    def _sync_elements(
        self,
        updates: list[tuple[UIElement[Any, Any], JSONType]],
        *,
        force: bool = False,
    ) -> None:
        deduped: dict[str, tuple[UIElement[Any, Any], JSONType]] = {}
        for element, frontend_value in updates:
            if (
                not force
                and _current_element_frontend_value(element) == frontend_value
            ):
                continue
            deduped[element._id] = (element, frontend_value)

        if not deduped:
            return

        for element, frontend_value in deduped.values():
            _set_local_frontend_value(element, frontend_value)

        try:
            ctx = get_context()
            kernel = ctx._kernel
            loop = asyncio.get_running_loop()
        except (ContextNotInitializedError, RuntimeError, AttributeError):
            return

        command = UpdateUIElementCommand.from_ids_and_values(
            [
                (element._id, frontend_value)
                for element, frontend_value in deduped.values()
            ]
        )
        self._sync_task = loop.create_task(kernel.set_ui_element_value(command))


class PydanticJsonGui(UIElement[Any, ModelT | None], Generic[ModelT]):
    """Internal config/tree editor for Pydantic-backed forms."""

    _name = "marimo-dict"

    def __init__(
        self,
        model_cls: type[ModelT],
        *,
        value: ModelT | dict[str, Any] | None = None,
        label: str = "",
        nested_models_multiple_open: bool = True,
        nested_models_flat_after_level: int | None = None,
        force_direct_json: bool = False,
        render_mode: Literal["hybrid", "json"] = "json",
        current_level: int = 0,
        on_change: Any | None = None,
    ) -> None:
        self._model_cls = model_cls
        self._label = label
        self._nested_models_multiple_open = nested_models_multiple_open
        self._nested_models_flat_after_level = nested_models_flat_after_level
        self._force_direct_json = force_direct_json
        self._render_mode = render_mode
        self._current_level = current_level
        self._initial_payload = _resolve_initial_payload(model_cls, value)
        self._last_payload = self._initial_payload
        self._last_error: str | None = None
        self._field_specs: dict[str, _FieldSpec] = {}
        self._field_names: list[str] = []
        self._direct_field_names: list[str] = []
        self._elements: dict[str, UIElement[Any, Any]] = {}
        self._editor: UIElement[Any, Any] | None = None
        self._tabs: UIElement[Any, Any] | None = None
        self._composite_mode = False
        self._sync_task: asyncio.Task[Any] | None = None
        layout_blocks: list[Any] = []
        self._elements = {}

        if force_direct_json:
            editor = mo.ui.code_editor(
                value=_payload_to_json(self._initial_payload),
                language="json",
                show_copy_button=True,
                debounce=False,
                label="",
            )
            self._editor = editor
            self._elements[_DIRECT_JSON_EDITOR_KEY] = editor
            layout_blocks.append(editor)
        else:
            (
                self._field_specs,
                self._field_names,
                self._direct_field_names,
                child_elements,
                layout,
            ) = _build_model_config_gui(
                model_cls=model_cls,
                payload=self._initial_payload,
                nested_models_multiple_open=nested_models_multiple_open,
                nested_models_flat_after_level=nested_models_flat_after_level,
                gui_mode=render_mode,
                current_level=current_level,
            )
            self._elements.update(child_elements)
            self._composite_mode = len(self._direct_field_names) != len(
                self._model_cls.model_fields
            )
            self._editor = self._elements.get(_DIRECT_JSON_EDITOR_KEY)
            self._tabs = self._elements.get(_TABS_KEY)
            layout_blocks.append(layout)
        layout = mo.vstack(layout_blocks, align="stretch")
        super().__init__(
            component_name="marimo-dict",
            initial_value={
                name: element._initial_value_frontend
                for name, element in self._elements.items()
            },
            label=label,
            args={
                "element-ids": {
                    element._id: name
                    for name, element in self._elements.items()
                }
            },
            slotted_html=layout.text,
            on_change=on_change,
        )
        for name, element in self._elements.items():
            element._register_as_view(parent=self, key=name)

    @property
    def elements(self) -> dict[str, UIElement[Any, Any]]:
        """Return child UI elements keyed by field name."""
        return self._elements

    def _clone(self) -> PydanticJsonGui[ModelT]:
        return type(self)(
            self._model_cls,
            value=self._last_payload,
            label=self._label,
            nested_models_multiple_open=self._nested_models_multiple_open,
            nested_models_flat_after_level=self._nested_models_flat_after_level,
            force_direct_json=self._force_direct_json,
            render_mode=self._render_mode,
            current_level=self._current_level,
            on_change=self._on_change,
        )

    def json(self) -> Any | None:
        """Return the current validated value serialized as JSON, if any."""
        if self.value is None:
            return _invalid_json_output(
                self.validate_frontend_value(self._current_frontend_value())
            )
        return _json_output(_jsonify_model_value(self.value))

    def error(self) -> Any:
        """Return the current validation output."""
        return _validation_output(
            self.validate_frontend_value(self._current_frontend_value())
        )

    def view(self) -> Any:
        """Return a notebook-friendly view with validation output."""
        return mo.vstack([self.error(), self], align="stretch")

    def _convert_value(self, value: Any) -> ModelT | None:
        if self._force_direct_json:
            editor_value = (
                value.get(_DIRECT_JSON_EDITOR_KEY, self._editor._value_frontend)
                if isinstance(value, dict)
                else value
            )
            payload, error = _json_text_to_payload(editor_value)
            if self._editor is not None and isinstance(value, dict):
                if _DIRECT_JSON_EDITOR_KEY in value:
                    _set_local_frontend_value(self._editor, editor_value)
        elif self._composite_mode:
            self._apply_non_field_partials(value)
            merged_value = self._merged_frontend_value(value)
            payload, error = self._payload_from_frontend(
                merged_value,
                update_children=True,
            )
        else:
            editor_value = (
                value.get(_DIRECT_JSON_EDITOR_KEY, self._editor._value_frontend)
                if isinstance(value, dict)
                else value
            )
            if isinstance(value, dict):
                self._apply_non_field_partials(value)
            payload, error = _json_text_to_payload(editor_value)
        self._last_error = error
        if error is not None:
            return None
        self._last_payload = payload
        model_value, validation_error = _validate_payload_with_error(
            self._model_cls,
            payload,
        )
        if validation_error is not None:
            return None
        return model_value

    def validate_frontend_value(self, value: Any | None) -> str | None:
        """Validate a frontend payload without mutating the committed value."""
        if value is None:
            return None
        if self._force_direct_json:
            editor_value = (
                value.get(_DIRECT_JSON_EDITOR_KEY, self._editor._value_frontend)
                if isinstance(value, dict)
                else value
            )
            payload, error = _json_text_to_payload(editor_value)
        elif self._composite_mode:
            merged_value = self._merged_frontend_value(value)
            payload, error = self._payload_from_frontend(
                merged_value,
                update_children=False,
            )
        else:
            editor_value = (
                value.get(_DIRECT_JSON_EDITOR_KEY, self._editor._value_frontend)
                if isinstance(value, dict)
                else value
            )
            payload, error = _json_text_to_payload(editor_value)
        if error is not None:
            return error
        _, validation_error = _validate_payload_with_error(
            self._model_cls,
            payload,
        )
        return validation_error

    def _payload_from_frontend(
        self,
        value: dict[str, JSONType],
        *,
        update_children: bool,
    ) -> tuple[dict[str, Any], str | None]:
        payload: dict[str, Any] = {}

        if self._editor is not None:
            editor_value = value.get(
                _DIRECT_JSON_EDITOR_KEY,
                self._editor._value_frontend,
            )
            direct_payload, error = _json_text_to_payload(editor_value)
            if error is not None:
                return {}, error
            unexpected_keys = sorted(
                set(direct_payload) - set(self._direct_field_names)
            )
            if unexpected_keys:
                return (
                    {},
                    "json: structural fields must be edited in their own nested editors.",
                )
            if update_children and self._editor._value_frontend != editor_value:
                _set_local_frontend_value(self._editor, editor_value)
            payload.update(direct_payload)

        for name in self._field_names:
            if name in self._direct_field_names:
                continue
            spec = self._field_specs[name]
            element = self._elements[name]
            frontend_value = value.get(
                name,
                _current_element_frontend_value(element),
            )
            try:
                payload[name] = _parse_nested_json_value(
                    spec,
                    element,
                    frontend_value,
                    update_children=update_children,
                )
            except ValueError as exc:
                return {}, f"{name}.{exc}"
        return payload, None

    def _merged_frontend_value(
        self,
        value: dict[str, JSONType],
    ) -> dict[str, JSONType]:
        merged = self._current_frontend_value()
        merged.update(value)
        return merged

    def _current_frontend_value(self) -> dict[str, JSONType]:
        if self._force_direct_json or not self._composite_mode:
            assert self._editor is not None
            return {
                _DIRECT_JSON_EDITOR_KEY: _current_element_frontend_value(
                    self._editor
                ),
            }
        current = {
            name: _current_element_frontend_value(element)
            for name, element in self._elements.items()
        }
        return current

    def _frontend_value_from_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, JSONType] | str:
        if self._force_direct_json or not self._composite_mode:
            return {
                _DIRECT_JSON_EDITOR_KEY: _payload_to_json(payload),
            }

        frontend_value: dict[str, JSONType] = {}
        if self._editor is not None:
            direct_payload = {
                name: payload[name]
                for name in self._direct_field_names
                if name in payload
            }
            frontend_value[_DIRECT_JSON_EDITOR_KEY] = _payload_to_json(
                direct_payload
            )

        for name in self._field_names:
            if name in self._direct_field_names:
                continue
            frontend_value[name] = _frontend_value_for_element(
                self._field_specs[name],
                self._elements[name],
                payload.get(name),
            )
        return frontend_value

    def _apply_non_field_partials(self, value: Any) -> None:
        if not isinstance(value, dict):
            return
        if (
            self._force_direct_json or not self._composite_mode
        ) and self._editor is not None:
            if _DIRECT_JSON_EDITOR_KEY in value:
                _set_local_frontend_value(
                    self._editor, value[_DIRECT_JSON_EDITOR_KEY]
                )
            return
        if self._tabs is not None and _TABS_KEY in value:
            _set_local_frontend_value(self._tabs, value[_TABS_KEY])

    def _sync_elements(
        self,
        updates: list[tuple[UIElement[Any, Any], JSONType]],
        *,
        force: bool = False,
    ) -> None:
        deduped: dict[str, tuple[UIElement[Any, Any], JSONType]] = {}
        for element, frontend_value in updates:
            if (
                not force
                and _current_element_frontend_value(element) == frontend_value
            ):
                continue
            deduped[element._id] = (element, frontend_value)

        if not deduped:
            return

        for element, frontend_value in deduped.values():
            _set_local_frontend_value(element, frontend_value)

        try:
            ctx = get_context()
            kernel = ctx._kernel
            loop = asyncio.get_running_loop()
        except (ContextNotInitializedError, RuntimeError, AttributeError):
            return

        command = UpdateUIElementCommand.from_ids_and_values(
            [
                (element._id, frontend_value)
                for element, frontend_value in deduped.values()
            ]
        )
        self._sync_task = loop.create_task(kernel.set_ui_element_value(command))


def _json_editor_error(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return f"json: {error.msg}"
    return str(error)


def _parse_json_editor_payload(
    model_cls: type[BaseModel],
    json_text: str,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, _json_editor_error(exc)

    if not isinstance(parsed, dict):
        return None, "json: top-level JSON value must be an object."

    return _order_payload_for_model(model_cls, parsed), None


def _build_error_view(error: str | None) -> Any:
    if error is None:
        return mo.md("")
    return mo.callout(error, kind="warn")


def _current_config_error(
    model_cls: type[BaseModel],
    *,
    form_gui_state: Any,
    json_gui_state: Any,
) -> str | None:
    json_text = json_gui_state()
    parsed_payload, parse_error = _parse_json_editor_payload(model_cls, json_text)
    if parse_error is not None or parsed_payload is None:
        return parse_error
    _, validation_error = _validate_payload_with_error(model_cls, parsed_payload)
    if validation_error is not None:
        return validation_error
    _, payload_validation_error = _validate_payload_with_error(
        model_cls,
        form_gui_state(),
    )
    return payload_validation_error


def _initial_config_payload(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not mo.running_in_notebook():
        if script_loader is None:
            parsed = load_script_config(
                model_cls,
                value=value,
                args=script_args,
            )
        else:
            parsed = script_loader(model_cls, value, script_args)
        resolved_value: ModelT | dict[str, Any] | None = parsed
    else:
        resolved_value = value
    return _order_payload_for_model(
        model_cls,
        _resolve_initial_payload(model_cls, resolved_value),
    )


def _resolve_bound_model_cls(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
) -> type[ModelT]:
    return (
        state_or_model_cls.model_cls
        if isinstance(state_or_model_cls, ConfigBindings)
        else state_or_model_cls
    )


def load_script_config(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    args: Sequence[str] | None = None,
) -> ModelT:
    """Load a config in script mode via tyro CLI or JSON file.

    Args:
        model_cls: Pydantic model type to load.
        value: Optional default value for the `cli` subcommand.
        args: Optional CLI argument sequence. When omitted, uses `sys.argv`.

    Returns:
        The validated model instance loaded from either the `cli` or `json`
        tyro subcommand.
    """
    default = _resolve_cli_default(model_cls, value)
    script_input_type = (
        Annotated[model_cls, tyro.conf.subcommand("cli", default=default)]
        | Annotated[_JsonConfigSource, tyro.conf.subcommand("json")]
    )
    parsed = tyro.cli(script_input_type, args=args)
    if isinstance(parsed, _JsonConfigSource):
        json_payload = json.loads(parsed.path.read_text())
        return model_cls.model_validate(json_payload)
    return parsed


@overload
def create_config_state(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
) -> tuple[Any, Any, ConfigBindings[ModelT]]: ...


def create_config_state(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
) -> Any:
    """Create reactive state for form and JSON config GUIs.

    Args:
        model_cls: Pydantic model type to edit.
        value: Optional initial model instance or payload override.
        script_loader: Optional override for script-mode config loading. When
            given, it replaces the default `load_script_config(...)` behavior.
        script_args: Optional CLI args forwarded to the script loader in
            script mode. When omitted, loaders should typically fall back to
            `sys.argv`.

    Returns:
        A tuple of `(form_gui_state, json_gui_state, config_bindings)`, where
        `form_gui_state` is the canonical structured config draft,
        `json_gui_state` is the current JSON editor text, and
        `config_bindings` contains the model type and setter callbacks used by
        the helper constructors.
    """
    initial_payload = _initial_config_payload(
        model_cls,
        value=value,
        script_loader=script_loader,
        script_args=script_args,
    )
    payload_state, set_payload_state = mo.state(
        initial_payload,
        allow_self_loops=True,
    )
    json_text_state, set_json_text_state = mo.state(
        _payload_to_json(initial_payload),
        allow_self_loops=True,
    )
    return (
        payload_state,
        json_text_state,
        ConfigBindings(
            model_cls=model_cls,
            set_form_gui_state=set_payload_state,
            set_json_gui_state=set_json_text_state,
        ),
    )


@overload
def create_committed_config_state(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
) -> tuple[Any, Callable[[dict[str, Any]], None]]: ...


def create_committed_config_state(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
) -> tuple[Any, Callable[[dict[str, Any]], None]]:
    """Create reactive state for the last committed config payload."""
    initial_payload = _initial_config_payload(
        model_cls,
        value=value,
        script_loader=script_loader,
        script_args=script_args,
    )
    committed_state, set_committed_state = mo.state(
        initial_payload,
        allow_self_loops=True,
    )
    return committed_state, set_committed_state


@overload
def config_gui(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    label: str = "",
    return_error_element: Literal[True] = True,
    return_form: Literal[True] = True,
    return_json: Literal[False] = False,
    nested_models_multiple_open: bool = True,
    nested_models_flat_after_level: int | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
) -> tuple[Any, PydanticGui[ModelT]]: ...


@overload
def config_gui(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    label: str = "",
    return_error_element: Literal[True],
    return_form: Literal[False],
    return_json: Literal[True],
    nested_models_multiple_open: bool = True,
    nested_models_flat_after_level: int | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
) -> tuple[Any, UIElement[Any, Any]]: ...


@overload
def config_gui(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    label: str = "",
    return_error_element: Literal[False],
    return_form: Literal[True],
    return_json: Literal[False] = False,
    nested_models_multiple_open: bool = True,
    nested_models_flat_after_level: int | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
) -> PydanticGui[ModelT]: ...


@overload
def config_gui(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    label: str = "",
    return_error_element: Literal[False],
    return_form: Literal[False],
    return_json: Literal[True],
    nested_models_multiple_open: bool = True,
    nested_models_flat_after_level: int | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
) -> UIElement[Any, Any]: ...


def config_gui(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    label: str = "",
    return_error_element: bool = True,
    return_form: bool = True,
    return_json: bool = False,
    nested_models_multiple_open: bool = True,
    nested_models_flat_after_level: int | None = None,
    script_loader: ScriptConfigLoader[ModelT] | None = None,
    script_args: Sequence[str] | None = None,
) -> Any:
    """Create renderable config GUI elements over shared reactive state.

    For multi-cell notebook wiring, use ``create_config_state(...)`` together with
    ``config_form(...)``, ``config_json(...)``, and related helpers.
    """
    if not return_form and not return_json:
        raise ValueError(
            "config_gui requires at least one of return_form or return_json."
        )

    (
        payload_state,
        json_text_state,
        bindings,
    ) = create_config_state(
        model_cls,
        value=value,
        script_loader=script_loader,
        script_args=script_args,
    )

    outputs: list[Any] = []
    if return_error_element:
        outputs.append(
            config_error(
                model_cls,
                form_gui_state=payload_state,
                json_gui_state=json_text_state,
            )
        )
    if return_form:
        outputs.append(
            config_form(
                bindings,
                form_gui_state=payload_state,
                label=label,
                nested_models_multiple_open=nested_models_multiple_open,
                nested_models_flat_after_level=nested_models_flat_after_level,
            )
        )
    if return_json:
        outputs.append(
            config_json(
                bindings,
                form_gui_state=payload_state,
                json_gui_state=json_text_state,
                label=label if not return_form else "",
            )
        )
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def config_form(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    form_gui_state: Any,
    set_form_gui_state: Callable[[dict[str, Any]], None] | None = None,
    set_json_gui_state: Callable[[str], None] | None = None,
    label: str = "",
    nested_models_multiple_open: bool = True,
    nested_models_flat_after_level: int | None = None,
) -> PydanticGui[ModelT]:
    """Build the form GUI for a config model.

    Args:
        state_or_model_cls: Either config bindings returned by
            `create_config_state(...)` or the raw model class.
        form_gui_state: Reactive structured state backing the form GUI.
        set_form_gui_state: Raw setter for `form_gui_state` when not using
            bindings.
        set_json_gui_state: Raw setter for the JSON GUI state when not using
            bindings.
        label: Optional UI label.
        nested_models_multiple_open: Whether nested accordions allow multiple
            sections open.
        nested_models_flat_after_level: Optional nesting depth after which
            nested models render flat.

    Returns:
        A `PydanticGui` bound to the provided reactive state.
    """
    if isinstance(state_or_model_cls, ConfigBindings):
        model_cls = state_or_model_cls.model_cls
        set_form_gui_state = state_or_model_cls.set_form_gui_state
        set_json_gui_state = state_or_model_cls.set_json_gui_state
    else:
        model_cls = state_or_model_cls
        if (
            set_form_gui_state is None
            or set_json_gui_state is None
        ):
            raise TypeError(
                "config_form requires setters when not given ConfigBindings."
            )
    form_ref: dict[str, PydanticGui[ModelT]] = {}

    def _on_form_change(_: Any) -> None:
        form = form_ref["form"]
        next_payload = _order_payload_for_model(model_cls, form._last_payload)
        set_form_gui_state(next_payload)
        set_json_gui_state(_payload_to_json(next_payload))

    form = PydanticGui(
        model_cls,
        value=form_gui_state(),
        label=label,
        include_json_editor=False,
        bordered=False,
        nested_models_multiple_open=nested_models_multiple_open,
        nested_models_flat_after_level=nested_models_flat_after_level,
        on_change=_on_form_change,
    )
    form_ref["form"] = form
    return form


def config_json(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    form_gui_state: Any,
    set_form_gui_state: Callable[[dict[str, Any]], None] | None = None,
    json_gui_state: Any,
    set_json_gui_state: Callable[[str], None] | None = None,
    label: str = "",
) -> UIElement[Any, Any]:
    """Build the JSON editor for a config model.

    Args:
        state_or_model_cls: Either config bindings returned by
            `create_config_state(...)` or the raw model class.
        form_gui_state: Reactive structured form GUI state.
        set_form_gui_state: Raw setter for `form_gui_state` when not using
            bindings.
        json_gui_state: Reactive JSON draft state.
        set_json_gui_state: Raw setter for `json_gui_state` when not using
            bindings.
        label: Optional UI label.

    Returns:
        A marimo code editor bound to the provided reactive state.
    """
    del form_gui_state
    if isinstance(state_or_model_cls, ConfigBindings):
        model_cls = state_or_model_cls.model_cls
        set_form_gui_state = state_or_model_cls.set_form_gui_state
        set_json_gui_state = state_or_model_cls.set_json_gui_state
    else:
        model_cls = state_or_model_cls
        if (
            set_form_gui_state is None
            or set_json_gui_state is None
        ):
            raise TypeError(
                "config_json requires setters when not given ConfigBindings."
            )

    def _on_json_change(next_text: str) -> None:
        set_json_gui_state(next_text)
        next_payload, parse_error = _parse_json_editor_payload(
            model_cls, next_text
        )
        if parse_error is not None or next_payload is None:
            return
        _, validation_error = _validate_payload_with_error(
            model_cls,
            next_payload,
        )
        if validation_error is not None:
            return
        normalized_payload = _order_payload_for_model(model_cls, next_payload)
        set_form_gui_state(normalized_payload)
        set_json_gui_state(_payload_to_json(normalized_payload))

    return mo.ui.code_editor(
        value=json_gui_state(),
        language="json",
        show_copy_button=True,
        debounce=False,
        label=label,
        on_change=_on_json_change,
    )


def config_error(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    form_gui_state: Any,
    json_gui_state: Any,
) -> Any:
    """Render the current config error output.

    Args:
        state_or_model_cls: Either config bindings returned by
            `create_config_state(...)` or the raw model class.
        form_gui_state: Reactive structured form GUI state.
        json_gui_state: Reactive JSON draft state.

    Returns:
        An empty markdown node when valid, otherwise a warning callout.
    """
    model_cls = _resolve_bound_model_cls(state_or_model_cls)
    return _build_error_view(
        _current_config_error(
            model_cls,
            form_gui_state=form_gui_state,
            json_gui_state=json_gui_state,
        )
    )


def config_value(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    form_gui_state: Any,
    json_gui_state: Any,
) -> ModelT | None:
    """Validate and return the current config value.

    Args:
        state_or_model_cls: Either config bindings returned by
            `create_config_state(...)` or the raw model class.
        form_gui_state: Reactive structured form GUI state.
        json_gui_state: Reactive JSON draft state.

    Returns:
        The validated model instance if valid, otherwise `None`.
    """
    model_cls = _resolve_bound_model_cls(state_or_model_cls)
    current_error = _current_config_error(
        model_cls,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    if current_error is not None:
        return None
    value, _ = _validate_payload_with_error(model_cls, form_gui_state())
    return value


def config_commit_button(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    form_gui_state: Any,
    json_gui_state: Any,
    committed_state: Any,
    set_committed_state: Callable[[dict[str, Any]], None],
    label: str = "Apply config",
) -> Any:
    """Build a button that snapshots a valid draft into committed state."""
    model_cls = _resolve_bound_model_cls(state_or_model_cls)
    current_error = _current_config_error(
        model_cls,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    draft_payload = _order_payload_for_model(model_cls, form_gui_state())
    committed_payload = _order_payload_for_model(model_cls, committed_state())
    is_dirty = draft_payload != committed_payload
    is_disabled = current_error is not None or not is_dirty
    tooltip: str | None = None
    if current_error is not None:
        tooltip = "Fix config errors before applying."
    elif not is_dirty:
        tooltip = "No unapplied config changes."

    return mo.ui.button(
        value=0,
        label=label,
        disabled=is_disabled,
        tooltip=tooltip,
        on_click=lambda value: (
            set_committed_state(draft_payload),
            (0 if value is None else int(value)) + 1,
        )[1],
    )


def config_committed_value(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    committed_state: Any,
) -> ModelT | None:
    """Validate and return the current committed config value."""
    model_cls = _resolve_bound_model_cls(state_or_model_cls)
    value, validation_error = _validate_payload_with_error(
        model_cls,
        _order_payload_for_model(model_cls, committed_state()),
    )
    if value is None or validation_error is not None:
        return None
    return value


def config_json_output(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    form_gui_state: Any,
    json_gui_state: Any,
) -> Any:
    """Render the validated config as JSON output.

    Args:
        state_or_model_cls: Either config bindings returned by
            `create_config_state(...)` or the raw model class.
        form_gui_state: Reactive structured form GUI state.
        json_gui_state: Reactive JSON draft state.

    Returns:
        A `mo.json(...)` view when valid, otherwise an invalid-config output.
    """
    model_cls = _resolve_bound_model_cls(state_or_model_cls)
    current_error = _current_config_error(
        model_cls,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    if current_error is not None:
        return _invalid_json_output(current_error)
    value, validation_error = _validate_payload_with_error(
        model_cls,
        form_gui_state(),
    )
    if value is None or validation_error is not None:
        return _invalid_json_output(validation_error)
    return _json_output(_order_payload_for_model(model_cls, form_gui_state()))


def config_require_valid(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    form_gui_state: Any,
    json_gui_state: Any,
    output: object | None = None,
) -> ModelT:
    """Return the validated config or stop notebook execution.

    Args:
        state_or_model_cls: Either config bindings returned by
            `create_config_state(...)` or the raw model class.
        form_gui_state: Reactive structured form GUI state.
        json_gui_state: Reactive JSON draft state.
        output: Optional custom output to pass to `mo.stop(...)`.

    Returns:
        The validated model instance.
    """
    model_cls = _resolve_bound_model_cls(state_or_model_cls)
    value = config_value(
        model_cls,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    if value is not None:
        return value
    current_error = _current_config_error(
        model_cls,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    mo.stop(
        True,
        output=output
        if output is not None
        else _validation_output(current_error or "Not a valid config."),
    )
    raise AssertionError("mo.stop should prevent evaluation from continuing.")


def form_gui(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    label: str = "",
    submit_label: str = "Submit",
    live_update: bool = False,
    nested_models_multiple_open: bool = True,
    nested_models_flat_after_level: int | None = None,
) -> Any:
    """Backwards-compatible form-only wrapper."""
    del submit_label, live_update
    return PydanticGui(
        model_cls,
        value=value,
        label=label,
        include_json_editor=False,
        nested_models_multiple_open=nested_models_multiple_open,
        nested_models_flat_after_level=nested_models_flat_after_level,
    )


def json_gui(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    label: str = "",
    submit_label: str = "Submit",
    nested_models_multiple_open: bool = True,
    nested_models_flat_after_level: int | None = None,
) -> Any:
    """Backwards-compatible JSON-only wrapper."""
    del submit_label, nested_models_multiple_open, nested_models_flat_after_level
    initial_payload = _order_payload_for_model(
        model_cls,
        _resolve_initial_payload(model_cls, value),
    )
    return mo.ui.code_editor(
        value=_payload_to_json(initial_payload),
        language="json",
        show_copy_button=True,
        debounce=False,
        label=label,
    )


def _build_model_gui(
    *,
    model_cls: type[BaseModel],
    payload: dict[str, Any],
    nested_models_multiple_open: bool,
    nested_models_flat_after_level: int | None,
    current_level: int,
) -> tuple[
    dict[str, _FieldSpec], dict[str, UIElement[Any, Any]], UIElement[Any, Any]
]:
    field_specs: dict[str, _FieldSpec] = {}
    elements: dict[str, UIElement[Any, Any]] = {}
    direct_controls: list[Any] = []
    nested_sections: list[tuple[str, Any]] = []

    for name, info in model_cls.model_fields.items():
        spec = _make_field_spec(model_cls, name, info, gui_mode="form")
        field_value = payload[name]
        element = _build_field_element(
            spec,
            field_value,
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            current_level=current_level,
        )
        field_specs[name] = spec
        elements[name] = element
        if spec.is_nested_model and _nested_model_field_uses_collapsible_layout(
            current_level=current_level,
            nested_models_flat_after_level=nested_models_flat_after_level,
        ):
            nested_sections.append(
                (
                    spec.label(),
                    _with_help_text(element, spec.help_text()),
                )
            )
        else:
            direct_controls.append(_with_help_text(element, spec.help_text()))

    nested_layout = _nested_model_layout(
        nested_sections,
        multiple_open=nested_models_multiple_open,
        current_level=current_level,
    )
    if nested_layout is not None:
        direct_controls.append(nested_layout)

    if not direct_controls:
        layout = mo.md("")
    elif len(direct_controls) == 1:
        layout = direct_controls[0]
    else:
        layout = mo.vstack(direct_controls)

    return field_specs, elements, layout


def _resolve_cli_default(
    model_cls: type[ModelT],
    value: ModelT | dict[str, Any] | None,
) -> ModelT | Any:
    if value is None:
        return tyro.MISSING_NONPROP
    if isinstance(value, model_cls):
        return value
    return model_cls.model_validate(value)


def _wrap_live_update_layout(slotted_html: str) -> str:
    """Wrap a live-updating form in a bordered yellow container."""
    return (
        '<div style="'
        "border: 1px solid #f2c94c; "
        "background: rgba(242, 201, 76, 0.12); "
        "border-radius: 8px; "
        "padding: 0.75rem; "
        '">'
        f"{slotted_html}"
        "</div>"
    )


def _build_model_config_gui(
    *,
    model_cls: type[BaseModel],
    payload: dict[str, Any],
    nested_models_multiple_open: bool,
    nested_models_flat_after_level: int | None,
    gui_mode: Literal["hybrid", "json"],
    current_level: int,
) -> tuple[
    dict[str, _FieldSpec],
    list[str],
    list[str],
    dict[str, UIElement[Any, Any]],
    UIElement[Any, Any],
]:
    field_specs: dict[str, _FieldSpec] = {}
    field_names: list[str] = []
    direct_field_names: list[str] = []
    elements: dict[str, UIElement[Any, Any]] = {}
    direct_controls: list[Any] = []
    nested_sections: list[tuple[str, Any]] = []

    direct_payload: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        spec = _make_field_spec(model_cls, name, info, gui_mode=gui_mode)
        field_specs[name] = spec
        field_names.append(name)
        if _field_uses_direct_json_editor(spec, gui_mode=gui_mode):
            direct_field_names.append(name)
            direct_payload[name] = payload[name]
            continue

        if _field_uses_config_child_editor(spec, gui_mode=gui_mode):
            element = _build_config_field_element(
                spec,
                payload[name],
                nested_models_multiple_open=nested_models_multiple_open,
                nested_models_flat_after_level=nested_models_flat_after_level,
                gui_mode=gui_mode,
                current_level=current_level,
            )
            elements[name] = element
            content = _with_help_text(element, spec.help_text())
            if (
                spec.is_nested_model
                and not spec.flat
                and _nested_model_field_uses_collapsible_layout(
                    current_level=current_level,
                    nested_models_flat_after_level=nested_models_flat_after_level,
                )
            ):
                nested_sections.append((spec.label(), content))
            else:
                direct_controls.append(content)

    if direct_field_names:
        editor = mo.ui.code_editor(
            value=_payload_to_json(direct_payload),
            language="json",
            show_copy_button=True,
            debounce=False,
            label="",
        )
        elements[_DIRECT_JSON_EDITOR_KEY] = editor
        direct_controls.append(editor)

    nested_layout = _nested_model_layout(
        nested_sections,
        multiple_open=nested_models_multiple_open,
        current_level=current_level,
    )
    if nested_layout is not None:
        direct_controls.append(nested_layout)

    if not direct_controls:
        layout = mo.md("")
    elif len(direct_controls) == 1:
        layout = direct_controls[0]
    else:
        layout = mo.vstack(direct_controls)

    return field_specs, field_names, direct_field_names, elements, layout


def _field_uses_direct_json_editor(
    spec: _FieldSpec,
    *,
    gui_mode: Literal["hybrid", "json"],
) -> bool:
    if spec.render_mode == "json":
        return True
    if gui_mode == "json":
        return not spec.is_structural_field
    if spec.render_mode == "widget":
        return False
    return not (
        spec.is_structural_field
        or spec.effective_annotation is Path
        or _is_literal_type(spec.effective_annotation)
        or _is_enum_type(spec.effective_annotation)
    )


def _field_uses_config_child_editor(
    spec: _FieldSpec,
    *,
    gui_mode: Literal["hybrid", "json"],
) -> bool:
    return not _field_uses_direct_json_editor(spec, gui_mode=gui_mode)


def _nested_model_field_uses_collapsible_layout(
    *,
    current_level: int,
    nested_models_flat_after_level: int | None,
) -> bool:
    if nested_models_flat_after_level is None:
        return True
    return current_level + 1 <= nested_models_flat_after_level


def _build_config_field_element(
    spec: _FieldSpec,
    value: Any,
    *,
    nested_models_multiple_open: bool,
    nested_models_flat_after_level: int | None,
    gui_mode: Literal["hybrid", "json"],
    current_level: int,
) -> UIElement[Any, Any]:
    if spec.is_optional:
        child = _build_concrete_config_field_element(
            spec,
            _optional_child_value(spec, value),
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            gui_mode=gui_mode,
            current_level=current_level,
        )
        return NullableGui(spec, child, enabled=value is not None)
    return _build_concrete_config_field_element(
        spec,
        value,
        nested_models_multiple_open=nested_models_multiple_open,
        nested_models_flat_after_level=nested_models_flat_after_level,
        gui_mode=gui_mode,
        current_level=current_level,
    )


def _build_concrete_config_field_element(
    spec: _FieldSpec,
    value: Any,
    *,
    nested_models_multiple_open: bool,
    nested_models_flat_after_level: int | None,
    gui_mode: Literal["hybrid", "json"],
    current_level: int,
) -> UIElement[Any, Any]:
    if spec.is_nested_model:
        nested_payload = value if isinstance(value, dict) else {}
        return PydanticJsonGui(
            spec.effective_annotation,
            value=nested_payload,
            label="",
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            force_direct_json=spec.force_json_editor,
            render_mode=gui_mode,
            current_level=current_level + 1,
        )

    if spec.is_model_union:
        return _build_union_json_field_element(
            spec,
            value,
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            gui_mode=gui_mode,
            current_level=current_level,
        )

    return _build_concrete_field_element(
        spec,
        value,
        label=spec.label(),
        nested_models_multiple_open=nested_models_multiple_open,
        nested_models_flat_after_level=nested_models_flat_after_level,
        current_level=current_level,
    )


def _build_union_field_element(
    spec: _FieldSpec,
    value: Any,
    *,
    nested_models_multiple_open: bool,
    nested_models_flat_after_level: int | None,
    current_level: int,
) -> UIElement[Any, Any]:
    branch_models = _union_model_types(spec.effective_annotation)
    active_index = _select_union_branch_index(
        branch_models,
        value,
    )
    children = tuple(
        PydanticGui(
            model_cls,
            value=_initial_payload_for_union_branch(
                branch_models,
                index,
                value,
                active_index=active_index,
            ),
            label="",
            include_json_editor=False,
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            current_level=current_level + 1,
        )
        for index, model_cls in enumerate(branch_models)
    )
    return ModelUnionGui(
        spec,
        branch_models,
        children,
        active_index=active_index,
    )


def _build_union_json_field_element(
    spec: _FieldSpec,
    value: Any,
    *,
    nested_models_multiple_open: bool,
    nested_models_flat_after_level: int | None,
    gui_mode: Literal["hybrid", "json"],
    current_level: int,
) -> UIElement[Any, Any]:
    branch_models = _union_model_types(spec.effective_annotation)
    active_index = _select_union_branch_index(branch_models, value)
    children = tuple(
        PydanticJsonGui(
            model_cls,
            value=_initial_payload_for_union_branch(
                branch_models,
                index,
                value,
                active_index=active_index,
            ),
            label="",
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            render_mode=gui_mode,
            current_level=current_level + 1,
        )
        for index, model_cls in enumerate(branch_models)
    )
    return ModelUnionGui(
        spec,
        branch_models,
        children,
        active_index=active_index,
    )


def _select_union_branch_index(
    branch_models: tuple[type[BaseModel], ...],
    value: Any,
) -> int:
    for index in range(len(branch_models)):
        if _union_value_matches_branch(branch_models, index, value):
            return index
    return 0


def _initial_payload_for_union_branch(
    branch_models: tuple[type[BaseModel], ...],
    index: int,
    value: Any,
    *,
    active_index: int,
) -> dict[str, Any]:
    model_cls = branch_models[index]
    if index != active_index:
        return _resolve_initial_payload(model_cls, None)

    payload = _payload_for_branch_model(model_cls, value)
    if not payload:
        return _resolve_initial_payload(model_cls, None)
    try:
        validated = model_cls.model_validate(payload)
    except ValidationError:
        merged = _resolve_initial_payload(model_cls, None)
        merged.update(payload)
        return merged
    return _resolve_initial_payload(model_cls, validated)


def _build_field_element(
    spec: _FieldSpec,
    value: Any,
    *,
    nested_models_multiple_open: bool,
    nested_models_flat_after_level: int | None,
    current_level: int,
) -> UIElement[Any, Any]:
    label = spec.label()
    if spec.is_optional:
        child = _build_concrete_field_element(
            spec,
            _optional_child_value(spec, value),
            label="",
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            current_level=current_level,
        )
        return NullableGui(spec, child, enabled=value is not None)
    return _build_concrete_field_element(
        spec,
        value,
        label=label,
        nested_models_multiple_open=nested_models_multiple_open,
        nested_models_flat_after_level=nested_models_flat_after_level,
        current_level=current_level,
    )


def _build_concrete_field_element(
    spec: _FieldSpec,
    value: Any,
    *,
    label: str,
    nested_models_multiple_open: bool,
    nested_models_flat_after_level: int | None,
    current_level: int,
) -> UIElement[Any, Any]:
    annotation = spec.effective_annotation

    if _is_model_union_type(annotation):
        return _build_union_field_element(
            spec,
            value,
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            current_level=current_level,
        )

    if _is_union_type(annotation):
        raise NotImplementedError(
            f"Union fields are not supported yet: {spec.name}"
        )

    if annotation is bool:
        return mo.ui.checkbox(value=bool(value), label=label)

    if annotation is str:
        return mo.ui.text(value=str(value), label=label)

    if annotation is Path:
        return mo.ui.file_browser(
            initial_path=_initial_browser_path(value),
            selection_mode="file",
            multiple=False,
            label=label,
        )

    if annotation in (int, float):
        return _build_numeric_element(
            annotation,
            spec.info,
            value,
            label,
            prefer_slider=spec.widget_mode == "slider",
        )

    if _is_literal_type(annotation):
        options = list(get_args(annotation))
        return mo.ui.dropdown(options=options, value=value, label=label)

    if _is_enum_type(annotation):
        options = list(annotation)
        return mo.ui.dropdown(options=options, value=value, label=label)

    if _is_model_type(annotation):
        nested_payload = value if isinstance(value, dict) else {}
        if spec.force_json_editor:
            return PydanticJsonGui(
                annotation,
                value=nested_payload,
                label=label,
                nested_models_multiple_open=nested_models_multiple_open,
                nested_models_flat_after_level=nested_models_flat_after_level,
                force_direct_json=True,
                current_level=current_level + 1,
            )
        return PydanticGui(
            annotation,
            value=nested_payload,
            label="",
            include_json_editor=False,
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            current_level=current_level + 1,
        )

    if _is_model_tuple_type(annotation):
        model_types = _model_tuple_types(annotation)
        values: tuple[Any, ...]
        if isinstance(value, tuple):
            values = value
        elif isinstance(value, list):
            values = tuple(value)
        else:
            values = tuple()
        children = tuple(
            PydanticGui(
                model_type,
                value=(
                    values[index]
                    if index < len(values) and isinstance(values[index], dict)
                    else {}
                ),
                label="",
                include_json_editor=False,
                nested_models_flat_after_level=nested_models_flat_after_level,
                current_level=current_level + 1,
            )
            for index, model_type in enumerate(model_types)
        )
        return ModelTupleGui(spec, children)

    if _is_array_annotation(annotation):
        return _build_array_element(annotation, spec.info, value, label)

    return mo.ui.text(value=_text_value(value), label=label)


def _build_numeric_element(
    annotation: type[int] | type[float],
    info: FieldInfo,
    value: int | float,
    label: str,
    *,
    prefer_slider: bool,
) -> UIElement[Any, Any]:
    bounds = _numeric_bounds(info)
    if prefer_slider and bounds.lower is not None and bounds.upper is not None:
        start, stop = _slider_limits(annotation, bounds)
        step = bounds.step
        if step is None:
            inferred_step = _infer_numeric_step(annotation, value)
            if inferred_step is None:
                if annotation is int:
                    step = 1
                else:
                    step = max((stop - start) / _DEFAULT_SLIDER_STEPS, 1e-6)
            else:
                step = inferred_step
        return mo.ui.slider(
            start=start,
            stop=stop,
            step=step,
            value=value,
            label=label,
        )

    step = bounds.step
    if step is None and annotation is int:
        step = 1
    return mo.ui.number(
        start=bounds.lower,
        stop=bounds.upper,
        step=step,
        value=value,
        label=label,
    )


def _build_array_element(
    annotation: Any,
    info: FieldInfo,
    value: Any,
    label: str,
) -> UIElement[Any, Any]:
    matrix_value = _normalize_matrix_value(annotation, value)
    total_cells = _matrix_total_cells(matrix_value)
    if total_cells > _MAX_MATRIX_CELLS:
        raise ValueError(
            f"Array field {label!r} has {total_cells} cells, which exceeds the "
            f"supported limit of {_MAX_MATRIX_CELLS}."
        )
    min_value, max_value, step = _array_widget_bounds(info, matrix_value)

    return mo.ui.matrix(
        value=matrix_value,
        min_value=min_value,
        max_value=max_value,
        step=step,
        label=label,
    )


def _with_help_text(
    element: UIElement[Any, Any],
    help_text: str | None,
) -> Any:
    if not help_text:
        return element
    return mo.hstack(
        [
            element,
            mo.md(
                "<span style="
                '"color: var(--mo-foreground-muted, #6b7280);'
                " font-style: italic;"
                ' font-size: 0.875em;">'
                f"{html.escape(help_text)}"
                "</span>"
            ),
        ],
        align="start",
        justify="start",
    )


def _nested_model_layout(
    sections: list[tuple[str, Any]],
    *,
    multiple_open: bool,
    current_level: int,
) -> Any | None:
    if not sections:
        return None

    labels = _disambiguate_labels([label for label, _ in sections])
    if len(sections) == 1:
        label, content = sections[0]
        layout = mo.vstack(
            [
                mo.md(f"**{label}**"),
                mo.callout(content, kind="neutral"),
            ],
            align="stretch",
        )
        return _indent_nested_layout(layout, current_level=current_level)

    items = {
        label: content
        for label, (_, content) in zip(labels, sections, strict=False)
    }
    layout = mo.accordion(items, multiple=multiple_open, lazy=False)
    return _indent_nested_layout(layout, current_level=current_level)


def _indent_nested_layout(layout: Any, *, current_level: int) -> Any:
    if current_level <= 0:
        return layout
    indent_rem = min(0.85 * current_level, 2.55)
    return mo.Html(
        
            '<div style="'
            f"margin-left: {indent_rem:.2f}rem; "
            "padding-left: 0.75rem; "
            "border-left: 1px solid rgba(127, 127, 127, 0.28);"
            '">'
            f"{layout.text}"
            "</div>"
        
    )


def _order_payload_for_model(
    model_cls: type[BaseModel],
    payload: dict[str, Any],
) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        if name not in payload:
            continue
        spec = _make_field_spec(model_cls, name, info, gui_mode="form")
        value = payload[name]
        if isinstance(value, dict):
            if spec.is_nested_model:
                ordered[name] = _order_payload_for_model(
                    spec.effective_annotation,
                    value,
                )
                continue
            if spec.is_model_union:
                ordered[name] = _order_union_payload(
                    spec.effective_annotation,
                    value,
                )
                continue
        ordered[name] = value

    for name, value in payload.items():
        if name not in ordered:
            ordered[name] = value
    return ordered


def _order_union_payload(annotation: Any, payload: dict[str, Any]) -> dict[str, Any]:
    branch_models = _union_model_types(annotation)
    branch_index = _union_branch_index_from_kind(branch_models, payload)
    if branch_index is not None:
        model_cls = branch_models[branch_index]
        ordered = _order_payload_for_model(model_cls, _strip_union_kind(payload))
        return _union_payload(
            _union_kind_for_model(branch_models, branch_index),
            ordered,
        )

    for index, model_cls in enumerate(branch_models):
        try:
            model_cls.model_validate(_strip_union_kind(payload))
        except ValidationError:
            continue
        ordered = _order_payload_for_model(model_cls, _strip_union_kind(payload))
        return _union_payload(
            _union_kind_for_model(branch_models, index),
            ordered,
        )
    return payload


def _resolve_initial_payload(
    model_cls: type[BaseModel],
    value: BaseModel | dict[str, Any] | None,
) -> dict[str, Any]:
    raw: dict[str, Any]
    if isinstance(value, BaseModel):
        raw = value.model_dump()
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raw = {}

    payload: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        spec = _make_field_spec(model_cls, name, info, gui_mode="form")
        if name in raw:
            field_value = raw[name]
            if field_value is not None and spec.is_nested_model:
                payload[name] = _resolve_initial_payload(
                    spec.effective_annotation,
                    field_value,
                )
            elif field_value is not None and spec.is_model_union:
                payload[name] = _resolve_union_initial_payload(
                    spec.effective_annotation,
                    field_value,
                )
            else:
                payload[name] = field_value
        else:
            payload[name] = _initial_field_value(
                name, info, model_cls=model_cls
            )
    return payload


def _resolve_materialized_payload(
    model_cls: type[BaseModel],
    payload: dict[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        if name not in payload:
            continue
        spec = _make_field_spec(model_cls, name, info, gui_mode="form")
        field_value = payload[name]
        if field_value is None:
            resolved[name] = None
            continue
        if spec.is_nested_model:
            if isinstance(field_value, BaseModel):
                resolved[name] = field_value
            else:
                resolved[name] = spec.effective_annotation.model_validate(
                    _resolve_materialized_payload(
                        spec.effective_annotation,
                        field_value,
                    )
                )
            continue
        if spec.is_model_union:
            resolved[name] = _materialize_union_value(
                spec.effective_annotation,
                field_value,
            )
            continue
        resolved[name] = field_value
    return resolved


def _initial_field_value(
    name: str,
    info: FieldInfo,
    *,
    model_cls: type[BaseModel] | None = None,
) -> Any:
    spec = _make_field_spec(model_cls, name, info, gui_mode="form")
    if not info.is_required():
        default = info.get_default(call_default_factory=True)
        if default is None:
            return None
        if spec.is_nested_model:
            return _resolve_initial_payload(spec.effective_annotation, default)
        if spec.is_model_union:
            return _resolve_union_initial_payload(
                spec.effective_annotation,
                default,
            )
        return default

    if spec.is_optional:
        return None

    annotation = spec.effective_annotation
    if spec.is_model_union:
        return _resolve_initial_payload(_union_model_types(annotation)[0], None)
    if _is_union_type(annotation):
        raise NotImplementedError(f"Union fields are not supported yet: {name}")
    return _default_value_for_annotation(name, annotation, info)


def _validate_payload(
    model_cls: type[ModelT],
    payload: dict[str, Any],
) -> ModelT | None:
    try:
        return model_cls.model_validate(
            _resolve_materialized_payload(model_cls, payload)
        )
    except (ValidationError, ValueError):
        return None


def _validate_payload_with_error(
    model_cls: type[ModelT],
    payload: dict[str, Any],
) -> tuple[ModelT | None, str | None]:
    try:
        value = model_cls.model_validate(
            _resolve_materialized_payload(model_cls, payload)
        )
    except (ValidationError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            return None, _format_validation_error(exc)
        return None, str(exc)
    return value, None


def _numeric_bounds(info: FieldInfo) -> _NumericBounds:
    lower: int | float | None = None
    upper: int | float | None = None
    step: int | float | None = None
    strict_lower = False
    strict_upper = False

    for metadata in info.metadata:
        if isinstance(metadata, annotated_types.Ge):
            lower = metadata.ge
        elif isinstance(metadata, annotated_types.Gt):
            lower = metadata.gt
            strict_lower = True
        elif isinstance(metadata, annotated_types.Le):
            upper = metadata.le
        elif isinstance(metadata, annotated_types.Lt):
            upper = metadata.lt
            strict_upper = True
        elif isinstance(metadata, annotated_types.MultipleOf):
            step = metadata.multiple_of

    return _NumericBounds(
        lower=lower,
        upper=upper,
        step=step,
        strict_lower=strict_lower,
        strict_upper=strict_upper,
    )


def _infer_numeric_step(
    annotation: type[int] | type[float],
    value: int | float,
) -> int | float | None:
    if value == 0:
        return 1 if annotation is int else None

    magnitude = abs(float(value))
    exponent = math.floor(math.log10(magnitude)) - 1

    if annotation is int:
        return max(1, 10**exponent)
    return 10.0**exponent


def _array_widget_bounds(
    info: FieldInfo,
    matrix_value: Any,
) -> tuple[int | float | None, int | float | None, int | float]:
    extras = info.json_schema_extra or {}
    if not isinstance(extras, dict):
        extras = {}

    min_value = extras.get("matrix_min")
    max_value = extras.get("matrix_max")
    step = extras.get("matrix_step")

    if min_value is not None or max_value is not None or step is not None:
        return min_value, max_value, step if step is not None else 1.0

    bounds = _numeric_bounds(info)
    resolved_min: int | float | None = None
    resolved_max: int | float | None = None

    if bounds.lower is not None:
        resolved_min = bounds.lower
        if bounds.strict_lower:
            resolved_min += (
                bounds.step
                if bounds.step is not None
                else 1
                if np.asarray(matrix_value).dtype.kind in {"i", "u"}
                else 0.1
            )

    if bounds.upper is not None:
        resolved_max = bounds.upper
        if bounds.strict_upper:
            resolved_max -= (
                bounds.step
                if bounds.step is not None
                else 1
                if np.asarray(matrix_value).dtype.kind in {"i", "u"}
                else 0.1
            )

    return (
        resolved_min,
        resolved_max,
        (bounds.step if bounds.step is not None else 1.0),
    )


def _tyro_help_text_for_field(
    model_cls: type[BaseModel],
    field_name: str,
) -> str | None:
    try:
        return _tyro_docstrings.get_field_docstring(model_cls, field_name, ())
    except Exception:
        return None


def _field_gui_config(
    info: FieldInfo,
    gui_mode: ConfigGuiMode,
) -> _FieldGuiConfig:
    extras = info.json_schema_extra or {}
    if not isinstance(extras, dict):
        return _FieldGuiConfig()

    raw_config = extras.get("marimo_config_gui")
    if not isinstance(raw_config, dict):
        return _FieldGuiConfig()

    base = _coerce_field_gui_config(raw_config)
    raw_modes = raw_config.get("modes")
    if not isinstance(raw_modes, dict):
        return base

    override = raw_modes.get(gui_mode)
    if not isinstance(override, dict):
        return base

    mode_config = _coerce_field_gui_config(override)
    return _FieldGuiConfig(
        render=mode_config.render
        if mode_config.render != "auto"
        else base.render,
        flat=mode_config.flat if "flat" in override else base.flat,
        widget=mode_config.widget
        if mode_config.widget != "auto"
        else base.widget,
    )


def _coerce_field_gui_config(raw_config: dict[str, Any]) -> _FieldGuiConfig:
    render = raw_config.get("render", "auto")
    if render not in {"auto", "json", "widget"}:
        render = "auto"

    widget = raw_config.get("widget", "auto")
    if widget not in {"auto", "slider"}:
        widget = "auto"

    return _FieldGuiConfig(
        render=cast(_RenderMode, render),
        flat=bool(raw_config.get("flat", False)),
        widget=cast(_WidgetMode, widget),
    )


def _humanize_model_name(model_cls: type[BaseModel]) -> str:
    title = getattr(model_cls, "model_config", {}).get("title")
    if isinstance(title, str) and title:
        return title
    name = model_cls.__name__
    for suffix in ("Config", "Model"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip()


def _disambiguate_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for label in labels:
        count = counts.get(label, 0) + 1
        counts[label] = count
        result.append(label if count == 1 else f"{label} ({count})")
    return result


def _slider_limits(
    annotation: type[int] | type[float],
    bounds: _NumericBounds,
) -> tuple[int | float, int | float]:
    assert bounds.lower is not None
    assert bounds.upper is not None

    if annotation is int:
        start = int(bounds.lower) + (1 if bounds.strict_lower else 0)
        stop = int(bounds.upper) - (1 if bounds.strict_upper else 0)
        return start, stop

    start = float(bounds.lower)
    stop = float(bounds.upper)
    step = (
        float(bounds.step)
        if bounds.step is not None
        else max(
            (stop - start) / _DEFAULT_SLIDER_STEPS,
            1e-6,
        )
    )
    if bounds.strict_lower:
        start += step
    if bounds.strict_upper:
        stop -= step
    return start, stop


def _make_field_spec(
    model_cls: type[BaseModel] | None,
    name: str,
    info: FieldInfo,
    *,
    gui_mode: ConfigGuiMode = "form",
) -> _FieldSpec:
    annotation = info.annotation
    optional = _is_optional_type(annotation)
    effective_annotation = _strip_optional_type(annotation)
    field_config = _field_gui_config(info, gui_mode)
    return _FieldSpec(
        model_cls=model_cls or BaseModel,
        name=name,
        annotation=annotation,
        effective_annotation=effective_annotation,
        info=info,
        is_optional=optional,
        is_nested_model=_is_model_type(effective_annotation),
        is_model_union=_is_model_union_type(effective_annotation),
        render_mode=field_config.render,
        flat=field_config.flat,
        widget_mode=field_config.widget,
    )


def _default_value_for_annotation(
    name: str,
    annotation: Any,
    info: FieldInfo,
) -> Any:
    if annotation is bool:
        return False
    if annotation is str:
        return ""
    if annotation is Path:
        return ""
    if annotation is int:
        bounds = _numeric_bounds(info)
        lower = bounds.lower if bounds.lower is not None else 0
        return int(lower) + (1 if bounds.strict_lower else 0)
    if annotation is float:
        bounds = _numeric_bounds(info)
        lower = bounds.lower if bounds.lower is not None else 0.0
        if bounds.strict_lower:
            step = bounds.step if bounds.step is not None else 0.1
            return float(lower) + float(step)
        return float(lower)
    if _is_literal_type(annotation):
        options = get_args(annotation)
        if not options:
            raise ValueError(
                f"Literal field {name!r} does not define any options."
            )
        return options[0]
    if _is_enum_type(annotation):
        options = list(annotation)
        if not options:
            raise ValueError(
                f"Enum field {name!r} does not define any options."
            )
        return options[0]
    if _is_model_type(annotation):
        return _resolve_initial_payload(annotation, None)
    if _is_model_union_type(annotation):
        return _resolve_initial_payload(_union_model_types(annotation)[0], None)
    if _is_model_tuple_type(annotation):
        return tuple(
            _resolve_initial_payload(model_type, None)
            for model_type in _model_tuple_types(annotation)
        )
    if _is_array_annotation(annotation):
        return _default_array_value(annotation)
    return ""


def _optional_child_value(spec: _FieldSpec, value: Any) -> Any:
    if value is not None:
        return value
    return _default_value_for_annotation(
        spec.name,
        spec.effective_annotation,
        spec.info,
    )


def _is_literal_type(annotation: Any) -> bool:
    return get_origin(annotation) is Literal


def _is_union_type(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin is UnionType or origin is Union


def _is_optional_type(annotation: Any) -> bool:
    if not _is_union_type(annotation):
        return False
    args = get_args(annotation)
    return any(arg is type(None) for arg in args)


def _strip_optional_type(annotation: Any) -> Any:
    if not _is_union_type(annotation):
        return annotation
    non_none_args = tuple(
        arg for arg in get_args(annotation) if arg is not type(None)
    )
    if len(non_none_args) == len(get_args(annotation)):
        return annotation
    if len(non_none_args) == 1:
        return non_none_args[0]
    return Union[non_none_args]


def _is_model_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _union_model_types(annotation: Any) -> tuple[type[BaseModel], ...]:
    if not _is_union_type(annotation):
        return ()
    args = get_args(annotation)
    if not args or any(not _is_model_type(arg) for arg in args):
        return ()
    return args


def _is_model_union_type(annotation: Any) -> bool:
    return len(_union_model_types(annotation)) >= 2


def _union_kind_for_model(
    branch_models: tuple[type[BaseModel], ...],
    index: int,
) -> str:
    names = [model_cls.__name__ for model_cls in branch_models]
    name = names[index]
    if names.count(name) == 1:
        return name
    model_cls = branch_models[index]
    return f"{model_cls.__module__}:{model_cls.__qualname__}"


def _union_payload(kind: str, payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        merged = dict(payload)
    elif isinstance(payload, BaseModel):
        merged = payload.model_dump()
    else:
        merged = {}
    merged[_UNION_KIND_KEY] = kind
    return merged


def _strip_union_kind(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    stripped = dict(value)
    stripped.pop(_UNION_KIND_KEY, None)
    return stripped


def _union_branch_index_from_kind(
    branch_models: tuple[type[BaseModel], ...],
    value: Any,
) -> int | None:
    if not isinstance(value, dict):
        return None
    kind = value.get(_UNION_KIND_KEY)
    if not isinstance(kind, str):
        return None
    for index in range(len(branch_models)):
        if _union_kind_for_model(branch_models, index) == kind:
            return index
    return None


def _union_value_matches_branch(
    branch_models: tuple[type[BaseModel], ...],
    index: int,
    value: Any,
) -> bool:
    model_cls = branch_models[index]
    if isinstance(value, BaseModel):
        return isinstance(value, model_cls)
    if isinstance(value, dict):
        kind_index = _union_branch_index_from_kind(branch_models, value)
        if kind_index is not None:
            return kind_index == index
        try:
            model_cls.model_validate(_strip_union_kind(value))
        except ValidationError:
            return False
        return True
    return False


def _materialize_union_value(
    annotation: Any,
    value: Any,
) -> BaseModel:
    branch_models = _union_model_types(annotation)
    if isinstance(value, BaseModel):
        return value
    if not isinstance(value, dict):
        raise ValueError("Union payload must be an object.")
    kind_index = _union_branch_index_from_kind(branch_models, value)
    payload = _strip_union_kind(value)
    if kind_index is not None:
        model_cls = branch_models[kind_index]
        return model_cls.model_validate(
            _resolve_materialized_payload(model_cls, payload)
        )
    for model_cls in branch_models:
        try:
            return model_cls.model_validate(
                _resolve_materialized_payload(model_cls, payload)
            )
        except ValidationError:
            continue
    return branch_models[0].model_validate(
        _resolve_materialized_payload(branch_models[0], payload)
    )


def _resolve_union_initial_payload(
    annotation: Any, value: Any
) -> dict[str, Any]:
    branch_models = _union_model_types(annotation)
    if not branch_models:
        raise ValueError("Expected a union of BaseModel branches.")

    if isinstance(value, BaseModel):
        for index, model_cls in enumerate(branch_models):
            if isinstance(value, model_cls):
                return _union_payload(
                    _union_kind_for_model(branch_models, index),
                    _resolve_initial_payload(model_cls, value),
                )

    if isinstance(value, dict):
        kind_index = _union_branch_index_from_kind(branch_models, value)
        if kind_index is not None:
            model_cls = branch_models[kind_index]
            stripped = _strip_union_kind(value)
            return _union_payload(
                _union_kind_for_model(branch_models, kind_index),
                _resolve_initial_payload(model_cls, stripped),
            )
        for model_cls in branch_models:
            try:
                validated = model_cls.model_validate(_strip_union_kind(value))
            except ValidationError:
                continue
            return _union_payload(
                _union_kind_for_model(
                    branch_models,
                    branch_models.index(model_cls),
                ),
                _resolve_initial_payload(model_cls, validated),
            )

        merged = _resolve_initial_payload(branch_models[0], None)
        merged.update(_strip_union_kind(value))
        return _union_payload(
            _union_kind_for_model(branch_models, 0),
            merged,
        )

    return _union_payload(
        _union_kind_for_model(branch_models, 0),
        _resolve_initial_payload(branch_models[0], None),
    )


def _model_tuple_types(annotation: Any) -> tuple[type[BaseModel], ...]:
    if get_origin(annotation) is not tuple:
        return ()
    args = get_args(annotation)
    if not args or any(arg is Ellipsis for arg in args):
        return ()
    if not all(_is_model_type(arg) for arg in args):
        return ()
    return args


def _is_model_tuple_type(annotation: Any) -> bool:
    return bool(_model_tuple_types(annotation))


def _is_enum_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, Enum)


def _is_array_annotation(annotation: Any) -> bool:
    if annotation in (np.ndarray, torch.Tensor):
        return True
    return isinstance(annotation, type) and issubclass(
        annotation, AbstractArray
    )


def _uses_text_fallback(annotation: Any) -> bool:
    return not (
        annotation in (str, bool, int, float, Path)
        or _is_literal_type(annotation)
        or _is_enum_type(annotation)
        or _is_model_type(annotation)
        or _is_model_union_type(annotation)
        or _is_array_annotation(annotation)
    )


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    try:
        return json.dumps(_jsonify(value))
    except TypeError:
        return str(value)


def _maybe_parse_json_text(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in '[{"' and stripped not in {"true", "false", "null"}:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _coerce_path_value(info: FieldInfo, value: Any) -> Path | str:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value) if value else ""
    if value:
        selected = value[0]
        path = getattr(selected, "path", None)
        if isinstance(path, Path):
            return path
        if isinstance(selected, dict):
            maybe_path = selected.get("path")
            if isinstance(maybe_path, str):
                return Path(maybe_path)
    if not info.is_required():
        default = info.get_default(call_default_factory=True)
        if isinstance(default, Path):
            return default
    return ""


def _initial_browser_path(value: Any) -> Path:
    if isinstance(value, Path):
        if value.exists():
            return value if value.is_dir() else value.parent
        return Path.cwd()
    if isinstance(value, str) and value:
        path = Path(value)
        if path.exists():
            return path if path.is_dir() else path.parent
    return Path.cwd()


def _frontend_value_for_element(
    spec: _FieldSpec,
    element: UIElement[Any, Any],
    value: Any,
) -> JSONType:
    if isinstance(element, NullableGui):
        return element._frontend_value_from_payload(value)
    if isinstance(element, ModelUnionGui):
        return element._frontend_value_from_payload(value)
    if isinstance(element, ModelTupleGui):
        return element._frontend_value_from_payload(value)
    if isinstance(element, PydanticGui):
        return element._frontend_value_from_payload(
            _payload_for_branch_model(element._model_cls, value)
        )
    if isinstance(element, PydanticJsonGui):
        return _frontend_value_for_structural_child(
            element,
            element._model_cls,
            value,
        )

    annotation = spec.effective_annotation
    if annotation is Path:
        return _file_browser_frontend_value(value)
    if _is_array_annotation(annotation):
        return _normalize_matrix_value(annotation, value)
    if _is_enum_type(annotation) or _is_literal_type(annotation):
        return [] if value is None else [_dropdown_key(value)]
    if annotation in (bool, int, float, str):
        return value
    return _text_value(value)


def _parse_element_frontend_value(
    spec: _FieldSpec,
    element: UIElement[Any, Any],
    frontend_value: JSONType,
    *,
    update_children: bool,
) -> Any:
    if update_children and element._value_frontend != frontend_value:
        _set_local_frontend_value(element, frontend_value)
        python_value = element._value
    else:
        python_value = element._convert_value(frontend_value)
    return spec.to_model_value(python_value)


def _parse_nested_json_value(
    spec: _FieldSpec,
    element: UIElement[Any, Any],
    frontend_value: JSONType,
    *,
    update_children: bool,
) -> Any:
    try:
        return _parse_structural_frontend_value(
            spec,
            element,
            frontend_value,
            update_children=update_children,
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def _parse_structural_frontend_value(
    spec: _FieldSpec,
    element: UIElement[Any, Any],
    frontend_value: JSONType,
    *,
    update_children: bool,
) -> Any:
    if isinstance(element, NullableGui):
        return element._parse_frontend_value(
            frontend_value,
            update_children=update_children,
        )
    if isinstance(element, ModelUnionGui):
        return element._parse_frontend_value(
            frontend_value,
            update_children=update_children,
        )
    if isinstance(element, PydanticGui):
        payload, _ = element._payload_from_frontend(
            frontend_value,
            update_children=update_children,
            force_json=False,
        )
        return payload
    if isinstance(element, PydanticJsonGui):
        if element._composite_mode:
            if not isinstance(frontend_value, dict):
                raise ValueError(f"{spec.name}: Expected an object.")
            payload, error = element._payload_from_frontend(
                element._merged_frontend_value(frontend_value),
                update_children=update_children,
            )
        else:
            if isinstance(frontend_value, dict):
                frontend_value = frontend_value.get(
                    _DIRECT_JSON_EDITOR_KEY,
                    element._current_frontend_value().get(
                        _DIRECT_JSON_EDITOR_KEY,
                        "",
                    ),
                )
            payload, error = _json_text_to_payload(frontend_value)
        if error is not None:
            raise ValueError(error)
        _, validation_error = _validate_payload_with_error(
            element._model_cls,
            payload,
        )
        if update_children and element._value_frontend != frontend_value:
            _set_local_frontend_value(element, frontend_value)
        return payload
    return _parse_element_frontend_value(
        spec,
        element,
        frontend_value,
        update_children=update_children,
    )


def _apply_structural_child_frontend(
    spec: _FieldSpec,
    element: UIElement[Any, Any],
    frontend_value: JSONType,
) -> None:
    try:
        _parse_structural_frontend_value(
            spec,
            element,
            frontend_value,
            update_children=True,
        )
        element._value_frontend = _current_element_frontend_value(element)
    except ValueError:
        _set_local_frontend_value(element, frontend_value)


def _frontend_value_for_structural_child(
    element: UIElement[Any, Any],
    model_cls: type[BaseModel],
    value: Any,
) -> JSONType:
    payload = _payload_for_branch_model(model_cls, value)
    if isinstance(element, PydanticGui):
        return element._frontend_value_from_payload(payload)
    if isinstance(element, PydanticJsonGui):
        return element._frontend_value_from_payload(payload)
    if isinstance(element, ModelUnionGui):
        return element._frontend_value_from_payload(value)
    return _current_element_frontend_value(element)


def _payload_for_branch_model(
    model_cls: type[BaseModel],
    value: Any,
) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        if isinstance(value, model_cls):
            return value.model_dump()
        return {}
    if isinstance(value, dict):
        kind = value.get(_UNION_KIND_KEY)
        if isinstance(kind, str):
            valid_kinds = {
                model_cls.__name__,
                f"{model_cls.__module__}:{model_cls.__qualname__}",
            }
            if kind not in valid_kinds:
                return {}
            return dict(_strip_union_kind(value))
        return dict(value)
    return {}


def _dropdown_key(value: Any) -> str:
    if isinstance(value, Enum):
        return repr(value)
    return str(value)


def _file_browser_frontend_value(value: Any) -> list[dict[str, Any]]:
    path: Path | None = None
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value:
        path = Path(value)
    if path is None or not path.exists():
        return []
    return [
        {
            "id": str(path),
            "name": path.name,
            "path": str(path),
            "is_directory": path.is_dir(),
        }
    ]


def _set_local_frontend_value(
    element: UIElement[Any, Any],
    frontend_value: JSONType,
) -> None:
    element._value_frontend = frontend_value
    try:
        element._value = element._convert_value(frontend_value)
    except Exception:
        pass


def _current_element_frontend_value(element: UIElement[Any, Any]) -> JSONType:
    if isinstance(element, NullableGui):
        return element._current_frontend_value()
    if isinstance(element, ModelUnionGui):
        return element._current_frontend_value()
    if isinstance(element, ModelTupleGui):
        return element._current_frontend_value()
    if isinstance(element, PydanticGui):
        return element._current_frontend_value()
    if isinstance(element, PydanticJsonGui):
        return element._current_frontend_value()
    return element._value_frontend


def _merge_json_value(current: Any, incoming: Any) -> Any:
    if isinstance(current, dict) and isinstance(incoming, dict):
        merged = dict(current)
        for key, value in incoming.items():
            merged[key] = _merge_json_value(merged.get(key), value)
        return merged
    return incoming


def _json_text_to_payload(json_text: str) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {}, f"json: {exc.msg}"
    if not isinstance(parsed, dict):
        return {}, "json: top-level JSON value must be an object."
    return parsed, None


def _payload_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(_jsonify(payload), indent=2)


def _jsonify_model_value(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    raise TypeError("Expected a Pydantic model or payload dict.")


def _json_output(payload: dict[str, Any]) -> Any:
    if mo.running_in_notebook():
        return mo.json(_jsonify(payload))
    return _payload_to_json(payload)


def _empty_validation_output() -> Any:
    if mo.running_in_notebook():
        return mo.md("")
    return ""


def _validation_output(error: str | None) -> Any:
    if not error:
        return _empty_validation_output()
    if mo.running_in_notebook():
        return mo.callout(error, kind="warn")
    return error


def _invalid_json_output(error: str | None = None) -> Any:
    message = "Not a valid config."
    if error:
        message = f"Not a valid config: {error}"
    if mo.running_in_notebook():
        return mo.callout(message, kind="warn")
    return message


def _validation_output_for_form(form: Any) -> Any:
    error = form.element.validate_frontend_value(
        form.element._current_frontend_value()
    )
    return _validation_output(error)


def _jsonify(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonify(value.model_dump())
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _array_shape(annotation: Any, value: Any | None = None) -> _ArrayShape:
    if isinstance(annotation, type) and issubclass(annotation, AbstractArray):
        fixed_sizes: list[int] = []
        for dim in annotation.dims:
            size = getattr(dim, "size", None)
            if size is None:
                return _ArrayShape(ndim=len(annotation.dims), fixed_shape=None)
            fixed_sizes.append(int(size))
        return _ArrayShape(
            ndim=len(annotation.dims),
            fixed_shape=tuple(fixed_sizes),
        )

    if value is not None:
        array = _to_numpy_array(value)
        if array.ndim in (1, 2):
            return _ArrayShape(ndim=array.ndim, fixed_shape=tuple(array.shape))

    raise ValueError(
        "Array fields without a fixed shape need a default value or explicit "
        "initial value."
    )


def _default_array_value(annotation: Any) -> np.ndarray | torch.Tensor:
    shape = _array_shape(annotation)
    if shape.fixed_shape is None:
        raise ValueError(
            "Array fields without a fixed shape need a default value or "
            "explicit initial value."
        )
    data = np.zeros(shape.fixed_shape, dtype=np.float64)
    return _coerce_array_value(annotation, data)


def _normalize_matrix_value(annotation: Any, value: Any) -> Any:
    array = _to_numpy_array(value)
    shape = _array_shape(annotation, value)
    if shape.ndim not in (1, 2):
        raise NotImplementedError(
            "Only 1D and 2D arrays are supported by the matrix widget."
        )
    if (
        shape.fixed_shape is not None
        and tuple(array.shape) != shape.fixed_shape
    ):
        raise ValueError(
            f"Expected array shape {shape.fixed_shape}, got {tuple(array.shape)}."
        )
    return array.tolist()


def _coerce_array_value(
    annotation: Any, value: Any
) -> np.ndarray | torch.Tensor:
    array = _to_numpy_array(value)
    shape = _array_shape(annotation, array)
    if shape.ndim not in (1, 2):
        raise NotImplementedError(
            "Only 1D and 2D arrays are supported by the matrix widget."
        )
    if (
        shape.fixed_shape is not None
        and tuple(array.shape) != shape.fixed_shape
    ):
        raise ValueError(
            f"Expected array shape {shape.fixed_shape}, got {tuple(array.shape)}."
        )

    if annotation is torch.Tensor or (
        isinstance(annotation, type)
        and issubclass(annotation, AbstractArray)
        and annotation.array_type is torch.Tensor
    ):
        dtype = torch.float32 if array.dtype.kind == "f" else torch.int64
        return torch.tensor(array.tolist(), dtype=dtype)
    return np.asarray(array)


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _matrix_total_cells(value: Any) -> int:
    array = np.asarray(value)
    return int(array.size)


def _format_validation_error(exc: ValidationError) -> str:
    first_error = exc.errors()[0]
    location = ".".join(str(part) for part in first_error["loc"])
    if location:
        return f"{location}: {first_error['msg']}"
    return str(first_error["msg"])
