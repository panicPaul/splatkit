"""Composite UI elements for Pydantic config GUIs."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

import marimo as mo
from marimo._plugins.core.web_component import JSONType
from marimo._plugins.ui._core.ui_element import UIElement
from pydantic import BaseModel

from marimo_config_gui.constants import (
    DIRECT_JSON_EDITOR_KEY,
    FORM_TAB,
    JSON_EDITOR_KEY,
    JSON_TAB,
    NULLABLE_ENABLED_KEY,
    NULLABLE_NONE_TAB,
    NULLABLE_SET_TAB,
    NULLABLE_VALUE_KEY,
    TABS_KEY,
    UNION_ACTIVE_KEY,
)
from marimo_config_gui.labels import (
    _disambiguate_labels,
    _humanize_model_name,
)
from marimo_config_gui.state import ModelT

if TYPE_CHECKING:
    from marimo_config_gui.widgets import _FieldSpec

CONFIG_FORM_VIEW_KEY = "__config_form__"
CONFIG_JSON_VIEW_KEY = "__config_json__"
CONFIG_PRESET_VIEW_KEY = "__config_preset__"
DEFAULT_BACKGROUND = object()
ConfigBackground = (
    Literal["neutral", "warn", "success", "info", "danger"]
    | Mapping[str, str]
    | None
)

_BACKGROUND_BASE_STYLE: dict[str, str] = {
    "border-radius": "8px",
    "padding": "0.75rem",
}
_BACKGROUND_STYLES: dict[str, dict[str, str]] = {
    "neutral": {
        **_BACKGROUND_BASE_STYLE,
        "background": "rgba(113, 113, 122, 0.10)",
        "border": "1px solid rgba(113, 113, 122, 0.28)",
    },
    "warn": {
        **_BACKGROUND_BASE_STYLE,
        "background": "rgba(250, 204, 21, 0.14)",
        "border": "1px solid rgba(202, 138, 4, 0.35)",
    },
    "success": {
        **_BACKGROUND_BASE_STYLE,
        "background": "rgba(34, 197, 94, 0.12)",
        "border": "1px solid rgba(22, 163, 74, 0.32)",
    },
    "info": {
        **_BACKGROUND_BASE_STYLE,
        "background": "rgba(59, 130, 246, 0.11)",
        "border": "1px solid rgba(37, 99, 235, 0.30)",
    },
    "danger": {
        **_BACKGROUND_BASE_STYLE,
        "background": "rgba(239, 68, 68, 0.11)",
        "border": "1px solid rgba(220, 38, 38, 0.30)",
    },
}


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
                NULLABLE_NONE_TAB: mo.md(""),
                NULLABLE_SET_TAB: mo.callout(child, kind="neutral"),
            },
            value=NULLABLE_SET_TAB if enabled else NULLABLE_NONE_TAB,
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
        self._elements: dict[str, UIElement[Any, Any]] = {
            NULLABLE_ENABLED_KEY: self._toggle,
            NULLABLE_VALUE_KEY: child,
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
        """Return child UI elements keyed by frontend payload field."""
        return self._elements

    def _clone(self) -> NullableGui:
        return type(self)(
            self._spec,
            self._child._clone(),
            enabled=self._toggle.value == NULLABLE_SET_TAB,
            on_change=self._on_change,
        )

    def _current_frontend_value(self) -> dict[str, JSONType]:
        return {
            NULLABLE_ENABLED_KEY: _current_element_frontend_value(self._toggle),
            NULLABLE_VALUE_KEY: self._draft_frontend_value,
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
            merged.update(cast(dict[str, JSONType], value))

        enabled_frontend = merged.get(
            NULLABLE_ENABLED_KEY,
            _current_element_frontend_value(self._toggle),
        )
        if isinstance(enabled_frontend, bool):
            enabled_frontend = self._tabs_frontend_value(enabled_frontend)
        child_frontend = merged.get(
            NULLABLE_VALUE_KEY,
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
            NULLABLE_ENABLED_KEY: self._tabs_frontend_value(value is not None),
            NULLABLE_VALUE_KEY: child_frontend,
        }

    def _is_enabled_frontend_value(self, frontend_value: JSONType) -> bool:
        if isinstance(frontend_value, bool):
            return frontend_value
        return (
            cast(Any, self._toggle)._convert_value(frontend_value)
            == NULLABLE_SET_TAB
        )

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
        tabs: dict[str, Any] = {
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
        self._elements: dict[str, UIElement[Any, Any]] = {
            UNION_ACTIVE_KEY: self._selector
        }
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
        """Return child UI elements keyed by frontend payload field."""
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
            UNION_ACTIVE_KEY: _current_element_frontend_value(self._selector)
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
            merged.update(cast(dict[str, JSONType], value))

        active_frontend = merged.get(
            UNION_ACTIVE_KEY,
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
        frontend = {UNION_ACTIVE_KEY: self._tabs_frontend_value(active_index)}
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
        label = cast(Any, self._selector)._convert_value(frontend_value)
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
        self._elements: dict[str, UIElement[Any, Any]] = {
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
        """Return child UI elements keyed by frontend payload field."""
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
            merged.update(cast(dict[str, JSONType], value))

        payloads: list[Any] = []
        for name, child in self._elements.items():
            child_frontend = merged.get(
                name, _current_element_frontend_value(child)
            )
            if update_children and child._value_frontend != child_frontend:
                _set_local_frontend_value(child, child_frontend)
            payload, _ = cast(PydanticGui[Any], child)._payload_from_frontend(
                cast(dict[str, JSONType], child_frontend),
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


class PrimitiveTupleGui(UIElement[dict[str, JSONType], tuple[Any, ...]]):
    """Render a short fixed tuple of primitive values side by side."""

    _name = "marimo-dict"

    def __init__(
        self,
        spec: _FieldSpec,
        item_types: tuple[type[int] | type[float] | type[str], ...],
        children: tuple[UIElement[Any, Any], ...],
        *,
        on_change: Any | None = None,
    ) -> None:
        self._spec = spec
        self._item_types = item_types
        self._children = children
        self._elements = {
            str(index): child for index, child in enumerate(children)
        }
        controls = mo.hstack(children, align="start", justify="start")
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
            f"{controls.text}"
            "</div>"
            "</div>"
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
        """Return child UI elements keyed by tuple index."""
        return self._elements

    def _clone(self) -> PrimitiveTupleGui:
        return type(self)(
            self._spec,
            self._item_types,
            tuple(child._clone() for child in self._children),
            on_change=self._on_change,
        )

    def _current_frontend_value(self) -> dict[str, JSONType]:
        return {
            name: _current_element_frontend_value(element)
            for name, element in self._elements.items()
        }

    def _convert_value(self, value: dict[str, JSONType]) -> tuple[Any, ...]:
        return self._parse_frontend_value(value, update_children=True)

    def _parse_frontend_value(
        self,
        value: JSONType,
        *,
        update_children: bool,
    ) -> tuple[Any, ...]:
        merged = self._current_frontend_value()
        if isinstance(value, dict):
            merged.update(cast(dict[str, JSONType], value))

        items: list[Any] = []
        for index, (name, child) in enumerate(self._elements.items()):
            child_frontend = merged.get(
                name, _current_element_frontend_value(child)
            )
            if update_children and child._value_frontend != child_frontend:
                _set_local_frontend_value(child, child_frontend)
                python_value = child._value
            else:
                python_value = child._convert_value(child_frontend)
            items.append(self._item_types[index](python_value))
        return tuple(items)

    def _frontend_value_from_payload(self, value: Any) -> dict[str, JSONType]:
        values: tuple[Any, ...]
        if isinstance(value, tuple):
            values = value
        elif isinstance(value, list):
            values = tuple(value)
        else:
            values = tuple()

        frontend: dict[str, JSONType] = {}
        for index, item_type in enumerate(self._item_types):
            default = "" if item_type is str else 0
            frontend[str(index)] = (
                values[index] if index < len(values) else default
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
        exclude_fields: set[str] | frozenset[str] = frozenset(),
        current_level: int = 0,
        force_frozen: bool = False,
        on_change: Any | None = None,
    ) -> None:
        self._model_cls = model_cls
        self._label = label
        self._include_json_editor = include_json_editor
        self._bordered = bordered
        self._nested_models_multiple_open = nested_models_multiple_open
        self._nested_models_flat_after_level = nested_models_flat_after_level
        self._exclude_fields = frozenset(exclude_fields)
        self._current_level = current_level
        self._force_frozen = force_frozen
        self._last_active_tab = FORM_TAB if include_json_editor else ""
        self._last_json_error: str | None = None
        self._initial_payload = _resolve_initial_payload(model_cls, value)
        self._last_payload = self._initial_payload

        field_specs, field_elements, form_layout = _build_model_gui(
            model_cls=model_cls,
            payload=self._last_payload,
            exclude_fields=self._exclude_fields,
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            current_level=current_level,
            force_frozen=force_frozen,
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
                    FORM_TAB: form_layout,
                    JSON_TAB: json_editor,
                },
                value=FORM_TAB,
                label="",
            )
            elements[JSON_EDITOR_KEY] = json_editor
            elements[TABS_KEY] = tabs
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
            exclude_fields=self._exclude_fields,
            current_level=self._current_level,
            force_frozen=self._force_frozen,
            on_change=self._on_change,
        )

    def _convert_value(self, value: dict[str, JSONType]) -> ModelT | None:
        self._apply_non_field_partials(value)
        merged_value = self._merged_frontend_value(value)
        active_tab = self._active_tab_name(merged_value)
        tab_switched = (
            self._include_json_editor
            and TABS_KEY in value
            and bool(self._last_active_tab)
            and active_tab != self._last_active_tab
        )
        source_tab = self._source_tab_name(merged_value, value)
        payload, json_error = self._payload_from_frontend(
            merged_value,
            update_children=True,
            force_json=source_tab == JSON_TAB,
        )
        self._last_payload = payload
        model_value, validation_error = _validate_payload_with_error(
            self._model_cls,
            payload,
        )
        frozen_error = _frozen_payload_error(
            self._model_cls,
            self._initial_payload,
            payload,
        )
        current_error = json_error or frozen_error or validation_error

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
            if source_tab == FORM_TAB:
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
            return self._restore_excluded_fields(payload), None

        active_tab = self._active_tab_name(value)
        should_use_json = (
            force_json
            or active_tab == JSON_TAB
            or self._last_active_tab == JSON_TAB
        )
        if not should_use_json:
            return self._restore_excluded_fields(payload), None
        return self._merge_json_payload(value[JSON_EDITOR_KEY], payload)

    def _restore_excluded_fields(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._exclude_fields:
            return payload
        restored = dict(payload)
        for name in self._exclude_fields:
            if name in self._last_payload:
                restored[name] = self._last_payload[name]
        return restored

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
            if JSON_EDITOR_KEY in value:
                _set_local_frontend_value(
                    self._json_editor,
                    value[JSON_EDITOR_KEY],
                )
        if self._include_json_editor and self._tabs is not None:
            if TABS_KEY in value:
                _set_local_frontend_value(
                    self._tabs,
                    value[TABS_KEY],
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
        frozen_error = _frozen_payload_error(
            self._model_cls,
            self._initial_payload,
            payload,
        )
        if frozen_error is not None:
            return frozen_error

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
            return FORM_TAB
        raw_value = value.get(TABS_KEY, self._tabs._value_frontend)
        return self._tabs._convert_value(raw_value)

    def _source_tab_name(
        self,
        merged_value: dict[str, JSONType],
        incoming_value: dict[str, JSONType],
    ) -> str:
        active_tab = self._active_tab_name(merged_value)
        if (
            TABS_KEY in incoming_value
            and active_tab != self._last_active_tab
            and self._last_active_tab
        ):
            return self._last_active_tab
        return active_tab

    def _tabs_frontend_value(self, tab_name: str) -> JSONType:
        return 0 if tab_name == FORM_TAB else 1

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
                    element.elements[NULLABLE_ENABLED_KEY],
                    nullable_frontend[NULLABLE_ENABLED_KEY],
                )
            )
            if value is not None:
                self._collect_sync_updates(
                    spec,
                    element.elements[NULLABLE_VALUE_KEY],
                    value,
                    updates,
                )
            return

        if isinstance(element, ModelUnionGui):
            union_frontend = element._frontend_value_from_payload(value)
            _set_local_frontend_value(element, union_frontend)
            updates.append(
                (
                    element.elements[UNION_ACTIVE_KEY],
                    union_frontend[UNION_ACTIVE_KEY],
                )
            )
            active_index = element._active_index_from_frontend(
                union_frontend[UNION_ACTIVE_KEY]
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

        if isinstance(element, PrimitiveTupleGui):
            tuple_frontend = element._frontend_value_from_payload(value)
            _set_local_frontend_value(element, tuple_frontend)
            for name, child in element.elements.items():
                updates.append((child, tuple_frontend[name]))
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
            frontend_value[JSON_EDITOR_KEY] = _payload_to_json(payload)
            frontend_value[TABS_KEY] = self._tabs._value_frontend
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

        for element, frontend_value in deduped.values():
            _send_frontend_value_update(element, frontend_value)


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
        force_frozen: bool = False,
        on_change: Any | None = None,
    ) -> None:
        self._model_cls = model_cls
        self._label = label
        self._nested_models_multiple_open = nested_models_multiple_open
        self._nested_models_flat_after_level = nested_models_flat_after_level
        self._force_direct_json = force_direct_json
        self._render_mode = render_mode
        self._current_level = current_level
        self._force_frozen = force_frozen
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
        layout_blocks: list[Any] = []
        self._elements = {}

        if force_direct_json:
            editor = mo.ui.code_editor(
                value=_payload_to_json(self._initial_payload),
                language="json",
                show_copy_button=True,
                debounce=False,
                label="",
                disabled=force_frozen,
            )
            self._editor = editor
            self._elements[DIRECT_JSON_EDITOR_KEY] = editor
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
                force_frozen=force_frozen,
            )
            self._elements.update(child_elements)
            self._composite_mode = len(self._direct_field_names) != len(
                self._model_cls.model_fields
            )
            self._editor = self._elements.get(DIRECT_JSON_EDITOR_KEY)
            self._tabs = self._elements.get(TABS_KEY)
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
            force_frozen=self._force_frozen,
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
            assert self._editor is not None
            editor_value = (
                value.get(DIRECT_JSON_EDITOR_KEY, self._editor._value_frontend)
                if isinstance(value, dict)
                else value
            )
            payload, error = _json_text_to_payload(editor_value)
            if self._editor is not None and isinstance(value, dict):
                if DIRECT_JSON_EDITOR_KEY in value:
                    _set_local_frontend_value(self._editor, editor_value)
        elif self._composite_mode:
            self._apply_non_field_partials(value)
            merged_value = self._merged_frontend_value(value)
            payload, error = self._payload_from_frontend(
                merged_value,
                update_children=True,
            )
        else:
            assert self._editor is not None
            editor_value = (
                value.get(DIRECT_JSON_EDITOR_KEY, self._editor._value_frontend)
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
        frozen_error = _frozen_payload_error(
            self._model_cls,
            self._initial_payload,
            payload,
        )
        if frozen_error is not None:
            self._last_error = frozen_error
            return None
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
            assert self._editor is not None
            editor_value = (
                value.get(DIRECT_JSON_EDITOR_KEY, self._editor._value_frontend)
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
            assert self._editor is not None
            editor_value = (
                value.get(DIRECT_JSON_EDITOR_KEY, self._editor._value_frontend)
                if isinstance(value, dict)
                else value
            )
            payload, error = _json_text_to_payload(editor_value)
        if error is not None:
            return error
        frozen_error = _frozen_payload_error(
            self._model_cls,
            self._initial_payload,
            payload,
        )
        if frozen_error is not None:
            return frozen_error
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
                DIRECT_JSON_EDITOR_KEY,
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
                DIRECT_JSON_EDITOR_KEY: _current_element_frontend_value(
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
                DIRECT_JSON_EDITOR_KEY: _payload_to_json(payload),
            }

        frontend_value: dict[str, JSONType] = {}
        if self._editor is not None:
            direct_payload = {
                name: payload[name]
                for name in self._direct_field_names
                if name in payload
            }
            frontend_value[DIRECT_JSON_EDITOR_KEY] = _payload_to_json(
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
            if DIRECT_JSON_EDITOR_KEY in value:
                _set_local_frontend_value(
                    self._editor, value[DIRECT_JSON_EDITOR_KEY]
                )
            return
        if self._tabs is not None and TABS_KEY in value:
            _set_local_frontend_value(self._tabs, value[TABS_KEY])

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

        for element, frontend_value in deduped.values():
            _send_frontend_value_update(element, frontend_value)


class ConfigGui(UIElement[dict[str, JSONType], ModelT | None], Generic[ModelT]):
    """Owning config GUI that keeps form, JSON, status, and value in sync."""

    _name = "marimo-dict"

    def __init__(
        self,
        model_cls: type[ModelT],
        *,
        value: ModelT | dict[str, Any],
        background: ConfigBackground = "neutral",
        presets: Any | None = None,
        label: str = "",
        nested_models_multiple_open: bool = True,
        nested_models_flat_after_level: int | None = None,
        exclude_fields: set[str] | frozenset[str] = frozenset(),
        path_defaults: Sequence[Any] = (),
        path_base_dir: Path | None = None,
    ) -> None:
        self._model_cls = model_cls
        self._background = background
        self._presets = presets
        self._preset_option_label_by_name: dict[str, str] = {}
        self._preset_selector: UIElement[Any, Any] | None = None
        initial_payload = _order_payload_for_model(
            model_cls,
            _resolve_initial_payload(model_cls, value),
        )
        self._selected_preset_name = self._preset_name_from_payload(
            initial_payload
        )
        self._label = label
        self._nested_models_multiple_open = nested_models_multiple_open
        self._nested_models_flat_after_level = nested_models_flat_after_level
        self._exclude_fields = frozenset(exclude_fields)
        self._path_defaults = tuple(path_defaults)
        self._path_base_dir = path_base_dir
        self._payload = initial_payload
        self._initial_payload = deepcopy(self._payload)
        self._json_text = _payload_to_json(self._payload)
        self._error: str | None = None
        self._validated_config = self._validate_payload(self._payload)

        self._form = PydanticGui(
            model_cls,
            value=self._payload,
            label=label,
            include_json_editor=False,
            bordered=False,
            nested_models_multiple_open=nested_models_multiple_open,
            nested_models_flat_after_level=nested_models_flat_after_level,
            exclude_fields=self._exclude_fields,
        )
        self._json_editor = mo.ui.code_editor(
            value=self._json_text,
            language="json",
            show_copy_button=True,
            debounce=False,
            label=label,
        )
        self._elements: dict[str, UIElement[Any, Any]] = {
            CONFIG_FORM_VIEW_KEY: self._form,
            CONFIG_JSON_VIEW_KEY: self._json_editor,
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
            slotted_html="",
            on_change=None,
        )
        for name, element in self._elements.items():
            element._register_as_view(parent=self, key=name)

    @property
    def elements(self) -> dict[str, UIElement[Any, Any]]:
        """Return child views keyed by frontend payload field."""
        return self._elements

    def _clone(self) -> ConfigGui[ModelT]:
        return type(self)(
            self._model_cls,
            value=self._payload,
            background=self._background,
            presets=self._presets,
            label=self._label,
            nested_models_multiple_open=self._nested_models_multiple_open,
            nested_models_flat_after_level=self._nested_models_flat_after_level,
            exclude_fields=self._exclude_fields,
            path_defaults=self._path_defaults,
            path_base_dir=self._path_base_dir,
        )

    def _current_frontend_value(self) -> dict[str, JSONType]:
        value = {
            CONFIG_FORM_VIEW_KEY: _current_element_frontend_value(self._form),
            CONFIG_JSON_VIEW_KEY: _current_element_frontend_value(
                self._json_editor
            ),
        }
        if self._preset_selector is not None:
            value[CONFIG_PRESET_VIEW_KEY] = _current_element_frontend_value(
                self._preset_selector
            )
        return value

    def _convert_value(self, value: dict[str, JSONType]) -> ModelT | None:
        if not getattr(self, "_initialized", False):
            return self._validated_config
        if CONFIG_PRESET_VIEW_KEY in value:
            preset_name = self._preset_name_from_frontend(
                value[CONFIG_PRESET_VIEW_KEY]
            )
            if preset_name != self._selected_preset_name:
                self._apply_preset_name(preset_name, sync_selector=True)
                return self._validated_config
        if CONFIG_FORM_VIEW_KEY in value:
            self._apply_form_update(value[CONFIG_FORM_VIEW_KEY])
        if CONFIG_JSON_VIEW_KEY in value:
            self._apply_json_update(value[CONFIG_JSON_VIEW_KEY])
        return self._validated_config

    def gui_panel(self, *, background: Any = DEFAULT_BACKGROUND) -> Any:
        """Return the synchronized form view."""
        return self._with_background(self._form, background)

    def json_editor(self, *, background: Any = DEFAULT_BACKGROUND) -> Any:
        """Return the synchronized JSON editor view."""
        return self._with_background(self._json_editor, background)

    def status_panel(self, *, background: Any = DEFAULT_BACKGROUND) -> Any:
        """Return the current validation status output."""
        if self._error is None:
            return mo.md("")
        return self._with_background(
            mo.callout(
                self._error,
                kind="warn",
            ),
            background,
        )

    def preset_selector(
        self,
        *,
        label: str = "Preset",
        background: Any = DEFAULT_BACKGROUND,
    ) -> Any:
        """Return a preset selector bound to this GUI owner."""
        if self._presets is None:
            raise ValueError(
                "preset_selector() requires create_config_gui(..., presets=...)."
            )
        if self._preset_selector is None:
            presets = self._presets
            options = {
                preset.label or preset.name: name
                for name, preset in presets.presets.items()
            }
            self._preset_option_label_by_name = {
                name: option_label for option_label, name in options.items()
            }
            current_label = self._preset_option_label_by_name[
                self._selected_preset_name or presets.default
            ]

            def _on_change(name: str) -> None:
                self._apply_preset_name(
                    name,
                    sync_selector=True,
                    notify_owner=True,
                )

            self._preset_selector = mo.ui.dropdown(
                options=options,
                value=current_label,
                label=label,
                on_change=_on_change,
            )
            self._elements[CONFIG_PRESET_VIEW_KEY] = self._preset_selector
            self._preset_selector._register_as_view(
                parent=self,
                key=CONFIG_PRESET_VIEW_KEY,
            )
            self._value_frontend = self._current_frontend_value()
        return self._with_background(self._preset_selector, background)

    def stacked(
        self,
        *,
        background: Any = DEFAULT_BACKGROUND,
        background_scope: Literal["panels", "stack"] = "panels",
        widths: Literal["equal"] | Sequence[float] | None = (1, 1),
        gap: float = 0.5,
    ) -> Any:
        """Return the default stacked form, JSON editor, and status layout."""
        resolved_background = self._resolve_background(background)
        if background_scope == "panels":
            stack = mo.vstack(
                [
                    mo.hstack(
                        [
                            self.gui_panel(background=resolved_background),
                            self.json_editor(background=resolved_background),
                        ],
                        widths=widths,
                        align="start",
                        gap=gap,
                    ),
                    self.status_panel(background=resolved_background),
                ],
                gap=gap,
            )
            return stack
        if background_scope == "stack":
            stack = mo.vstack(
                [
                    mo.hstack(
                        [
                            self.gui_panel(background=None),
                            self.json_editor(background=None),
                        ],
                        widths=widths,
                        align="start",
                        gap=gap,
                    ),
                    self.status_panel(background=None),
                ],
                gap=gap,
            )
            return _with_config_background(stack, resolved_background)
        raise ValueError("background_scope must be either 'panels' or 'stack'.")

    def validated_config(self) -> ModelT:
        """Return the valid config or stop the current consumer cell."""
        if self._validated_config is not None:
            return self._validated_config
        error = self.validation_error() or "Config is invalid."
        if mo.running_in_notebook():
            mo.stop(True, self.status_panel())
        raise ValueError(error)

    def is_valid(self) -> bool:
        """Return whether the current draft is valid without stopping."""
        return self._validated_config is not None

    def validation_error(self) -> str | None:
        """Return the current validation error without stopping."""
        return self._error

    def _resolve_background(self, background: Any) -> ConfigBackground:
        if background is DEFAULT_BACKGROUND:
            return self._background
        return background

    def _with_background(self, item: Any, background: Any) -> Any:
        return _with_config_background(
            item,
            self._resolve_background(background),
        )

    def _apply_form_update(self, form_value: JSONType) -> None:
        if not isinstance(form_value, dict):
            return
        self._form._convert_value(cast(dict[str, JSONType], form_value))
        next_payload = _order_payload_for_model(
            self._model_cls,
            self._form._last_payload,
        )
        self._replace_payload(next_payload, sync_form=False)

    def _apply_json_update(self, json_value: JSONType) -> None:
        if not isinstance(json_value, str):
            self._json_text = str(json_value)
            self._error = "JSON editor value must be a string."
            self._validated_config = None
            return

        _set_local_frontend_value(self._json_editor, json_value)
        self._json_text = json_value
        try:
            parsed = json.loads(json_value)
        except json.JSONDecodeError as exc:
            self._error = f"json: {exc.msg}"
            self._validated_config = None
            return

        if not isinstance(parsed, dict):
            self._error = "json: top-level JSON value must be an object."
            self._validated_config = None
            return

        try:
            next_payload = _order_payload_for_model(self._model_cls, parsed)
        except ValueError as exc:
            self._error = str(exc)
            self._validated_config = None
            return

        validated = self._validate_payload(next_payload)
        if validated is None:
            self._validated_config = None
            return

        self._payload = next_payload
        self._json_text = _payload_to_json(next_payload)
        self._validated_config = validated
        self._sync_json_editor(self._json_text)
        self._sync_form_controls(next_payload)

    def _replace_payload(
        self,
        payload: dict[str, Any],
        *,
        sync_form: bool = True,
        selected_preset_name: str | None = None,
    ) -> None:
        self._payload = payload
        self._selected_preset_name = (
            selected_preset_name
            if selected_preset_name is not None
            else self._preset_name_from_payload(payload)
        )
        self._json_text = _payload_to_json(payload)
        self._validated_config = self._validate_payload(payload)
        self._sync_json_editor(self._json_text)
        if sync_form:
            self._sync_form_controls(payload)
        self._sync_preset_selector()

    def _validate_payload(self, payload: dict[str, Any]) -> ModelT | None:
        frozen_error = _frozen_payload_error(
            self._model_cls,
            self._initial_payload,
            payload,
        )
        if frozen_error is not None:
            self._error = frozen_error
            return None
        value, error = _validate_payload_with_error(self._model_cls, payload)
        self._error = error
        if value is None:
            return None
        return self._resolve_config_paths(value, payload)

    def _resolve_config_paths(
        self,
        value: ModelT,
        payload: dict[str, Any],
    ) -> ModelT:
        path_defaults = self._path_defaults
        base_dir = self._path_base_dir
        should_resolve_paths = bool(path_defaults or base_dir is not None)
        if self._presets is not None:
            should_resolve_paths = True
            presets = self._presets
            preset_field = presets.preset_field or "preset"
            preset_name = payload.get(preset_field, presets.default)
            preset = presets.presets.get(str(preset_name))
            if preset is not None:
                preset_path = preset.path.expanduser().resolve()
                path_defaults = (
                    preset_path.parent / ".path_defaults.json",
                    *tuple(presets.path_defaults),
                    *path_defaults,
                )
                base_dir = (
                    preset.base_dir.expanduser().resolve()
                    if preset.base_dir is not None
                    else None
                )
        if not should_resolve_paths:
            return value
        from marimo_config_gui.presets import resolve_config_paths

        return resolve_config_paths(
            value,
            base_dir=base_dir,
            path_defaults=path_defaults,
        )

    def _sync_json_editor(self, json_text: str) -> None:
        self._sync_elements([(self._json_editor, json_text)])

    def _sync_form_controls(self, payload: dict[str, Any]) -> None:
        frontend_value = self._form._frontend_value_from_payload(payload)
        _set_local_frontend_value(self._form, frontend_value)
        self._form._last_payload = payload
        self._form._sync_field_controls(payload, force=True)

    def _sync_preset_selector(self) -> None:
        if self._preset_selector is None or self._selected_preset_name is None:
            return
        frontend_label = self._preset_option_label_by_name.get(
            self._selected_preset_name,
            self._selected_preset_name,
        )
        self._sync_elements([(self._preset_selector, [frontend_label])])

    def _sync_owner_frontend_value(self) -> None:
        frontend_value = self._current_frontend_value()
        self._value_frontend = frontend_value
        self._value = self._validated_config
        _send_frontend_value_update(self, frontend_value)

    def _apply_preset_name(
        self,
        name: str,
        *,
        sync_selector: bool = False,
        notify_owner: bool = False,
    ) -> None:
        if self._presets is None:
            raise ValueError("ConfigGui has no preset catalog.")
        if name not in self._presets.presets:
            raise ValueError(f"Unknown config preset: {name!r}.")
        config = _load_preset_config(self._presets, name)
        next_payload = _order_payload_for_model(
            self._model_cls,
            _payload_for_config(config),
        )
        self._replace_payload(next_payload, selected_preset_name=name)
        if sync_selector:
            self._sync_preset_selector()
        if notify_owner:
            self._sync_owner_frontend_value()

    def _preset_name_from_payload(
        self,
        payload: dict[str, Any],
    ) -> str | None:
        if self._presets is None:
            return None
        preset_field = self._presets.preset_field
        if preset_field is not None:
            preset_name = str(payload.get(preset_field, ""))
            if preset_name in self._presets.presets:
                return preset_name
        return self._presets.default

    def _preset_name_from_frontend(self, frontend_value: JSONType) -> str:
        if self._preset_selector is None:
            return str(frontend_value)
        if self._preset_selector._value_frontend == frontend_value:
            return str(self._preset_selector.value)
        return str(self._preset_selector._convert_value(frontend_value))

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

        for element, frontend_value in deduped.values():
            _send_frontend_value_update(element, frontend_value)


def _with_config_background(
    item: Any,
    background: ConfigBackground,
) -> Any:
    if background is None:
        return item
    if isinstance(background, str) and background in _BACKGROUND_STYLES:
        style = _BACKGROUND_STYLES[background]
    elif isinstance(background, Mapping):
        style = dict(background)
    else:
        names = ", ".join(sorted(_BACKGROUND_STYLES))
        raise ValueError(
            f"background must be one of {names}; a CSS style mapping; or None."
        )
    return item.style(style)


def _send_frontend_value_update(
    element: UIElement[Any, Any],
    frontend_value: JSONType,
) -> None:
    element._send_message(
        {
            "type": "marimo-ui-value-update",
            "value": frontend_value,
        },
        buffers=None,
    )


def _build_model_gui(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _build_model_gui as impl

    return impl(*args, **kwargs)


def _build_model_config_gui(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _build_model_config_gui as impl

    return impl(*args, **kwargs)


def _order_payload_for_model(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _order_payload_for_model as impl

    return impl(*args, **kwargs)


def _resolve_initial_payload(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _resolve_initial_payload as impl

    return impl(*args, **kwargs)


def _validate_payload_with_error(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _validate_payload_with_error as impl

    return impl(*args, **kwargs)


def _frozen_payload_error(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _frozen_payload_error as impl

    return impl(*args, **kwargs)


def _payload_to_json(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _payload_to_json as impl

    return impl(*args, **kwargs)


def _load_preset_config(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.presets import load_preset_config as impl

    return impl(*args, **kwargs)


def _payload_for_config(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.presets import payload_for_config as impl

    return impl(*args, **kwargs)


def _json_text_to_payload(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _json_text_to_payload as impl

    return impl(*args, **kwargs)


def _merge_json_value(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _merge_json_value as impl

    return impl(*args, **kwargs)


def _current_element_frontend_value(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import (
        _current_element_frontend_value as impl,
    )

    return impl(*args, **kwargs)


def _set_local_frontend_value(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _set_local_frontend_value as impl

    return impl(*args, **kwargs)


def _wrap_live_update_layout(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _wrap_live_update_layout as impl

    return impl(*args, **kwargs)


def _frontend_value_for_element(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _frontend_value_for_element as impl

    return impl(*args, **kwargs)


def _frontend_value_for_structural_child(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import (
        _frontend_value_for_structural_child as impl,
    )

    return impl(*args, **kwargs)


def _payload_for_branch_model(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _payload_for_branch_model as impl

    return impl(*args, **kwargs)


def _apply_structural_child_frontend(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import (
        _apply_structural_child_frontend as impl,
    )

    return impl(*args, **kwargs)


def _parse_nested_json_value(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _parse_nested_json_value as impl

    return impl(*args, **kwargs)


def _parse_structural_frontend_value(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import (
        _parse_structural_frontend_value as impl,
    )

    return impl(*args, **kwargs)


def _union_kind_for_model(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _union_kind_for_model as impl

    return impl(*args, **kwargs)


def _union_payload(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _union_payload as impl

    return impl(*args, **kwargs)


def _union_value_matches_branch(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _union_value_matches_branch as impl

    return impl(*args, **kwargs)


def _jsonify_model_value(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _jsonify_model_value as impl

    return impl(*args, **kwargs)


def _json_output(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _json_output as impl

    return impl(*args, **kwargs)


def _validation_output(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _validation_output as impl

    return impl(*args, **kwargs)


def _invalid_json_output(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _invalid_json_output as impl

    return impl(*args, **kwargs)
