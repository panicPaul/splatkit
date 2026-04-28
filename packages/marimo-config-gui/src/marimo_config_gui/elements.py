"""Composite UI elements for Pydantic config GUIs."""

from __future__ import annotations

import asyncio
import html
import json
from typing import Any, Generic, Literal

import marimo as mo
from marimo._plugins.core.web_component import JSONType
from marimo._plugins.ui._core.ui_element import UIElement
from marimo._runtime.commands import UpdateUIElementCommand
from marimo._runtime.context import ContextNotInitializedError, get_context
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
from marimo_config_gui.state import ModelT


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
        self._elements = {
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
            merged.update(value)

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
        return self._toggle._convert_value(frontend_value) == NULLABLE_SET_TAB

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
        self._elements = {UNION_ACTIVE_KEY: self._selector}
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
            merged.update(value)

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
        self._last_active_tab = FORM_TAB if include_json_editor else ""
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
            current_level=self._current_level,
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
            return payload, None

        active_tab = self._active_tab_name(value)
        should_use_json = (
            force_json
            or active_tab == JSON_TAB
            or self._last_active_tab == JSON_TAB
        )
        if not should_use_json:
            return payload, None
        return self._merge_json_payload(value[JSON_EDITOR_KEY], payload)

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
            editor_value = (
                value.get(DIRECT_JSON_EDITOR_KEY, self._editor._value_frontend)
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


def _payload_to_json(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _payload_to_json as impl

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


def _frontend_value_for_element(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _frontend_value_for_element as impl

    return impl(*args, **kwargs)


def _apply_structural_child_frontend(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import (
        _apply_structural_child_frontend as impl,
    )

    return impl(*args, **kwargs)


def _parse_nested_json_value(*args: Any, **kwargs: Any) -> Any:
    from marimo_config_gui.widgets import _parse_nested_json_value as impl

    return impl(*args, **kwargs)
