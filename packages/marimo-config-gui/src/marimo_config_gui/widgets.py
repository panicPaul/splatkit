"""Generate marimo forms from Pydantic models."""

from __future__ import annotations

import html
import json
import math
import operator
import re
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    Literal,
    cast,
    get_args,
    get_origin,
)

import annotated_types
import marimo as mo
import numpy as np
import torch
import tyro
from jaxtyping import AbstractArray
from marimo._plugins.core.web_component import JSONType
from marimo._plugins.ui._core.ui_element import UIElement
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from tyro import _docstrings as _tyro_docstrings

from marimo_config_gui.constants import (
    DEFAULT_SLIDER_STEPS,
    DIRECT_JSON_EDITOR_KEY,
    MAX_MATRIX_CELLS,
    UNION_KIND_KEY,
    RenderMode,
    WidgetMode,
)
from marimo_config_gui.elements import (
    ModelTupleGui,
    ModelUnionGui,
    NullableGui,
    PydanticGui,
    PydanticJsonGui,
)
from marimo_config_gui.state import (
    ModelT,
)


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
class _FieldGuiConfig:
    render: RenderMode = "auto"
    flat: bool = False
    widget: WidgetMode = "auto"


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
    render_mode: RenderMode
    flat: bool
    widget_mode: WidgetMode

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
        elements[DIRECT_JSON_EDITOR_KEY] = editor
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
                    step = max((stop - start) / DEFAULT_SLIDER_STEPS, 1e-6)
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
    if total_cells > MAX_MATRIX_CELLS:
        raise ValueError(
            f"Array field {label!r} has {total_cells} cells, which exceeds the "
            f"supported limit of {MAX_MATRIX_CELLS}."
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


def _order_union_payload(
    annotation: Any, payload: dict[str, Any]
) -> dict[str, Any]:
    branch_models = _union_model_types(annotation)
    branch_index = _union_branch_index_from_kind(branch_models, payload)
    if branch_index is not None:
        model_cls = branch_models[branch_index]
        ordered = _order_payload_for_model(
            model_cls, _strip_union_kind(payload)
        )
        return _union_payload(
            _union_kind_for_model(branch_models, branch_index),
            ordered,
        )

    for index, model_cls in enumerate(branch_models):
        try:
            model_cls.model_validate(_strip_union_kind(payload))
        except ValidationError:
            continue
        ordered = _order_payload_for_model(
            model_cls, _strip_union_kind(payload)
        )
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
        render=cast(RenderMode, render),
        flat=bool(raw_config.get("flat", False)),
        widget=cast(WidgetMode, widget),
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
            (stop - start) / DEFAULT_SLIDER_STEPS,
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
    return origin is UnionType


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
    return reduce(operator.or_, non_none_args)


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
    merged[UNION_KIND_KEY] = kind
    return merged


def _strip_union_kind(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    stripped = dict(value)
    stripped.pop(UNION_KIND_KEY, None)
    return stripped


def _union_branch_index_from_kind(
    branch_models: tuple[type[BaseModel], ...],
    value: Any,
) -> int | None:
    if not isinstance(value, dict):
        return None
    kind = value.get(UNION_KIND_KEY)
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
                    DIRECT_JSON_EDITOR_KEY,
                    element._current_frontend_value().get(
                        DIRECT_JSON_EDITOR_KEY,
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
        if validation_error is not None:
            raise ValueError(validation_error)
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
        kind = value.get(UNION_KIND_KEY)
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
