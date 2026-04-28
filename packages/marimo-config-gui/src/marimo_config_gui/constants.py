"""Shared constants for Pydantic config GUI controls."""

from __future__ import annotations

from typing import Literal

MAX_MATRIX_CELLS = 100
DEFAULT_SLIDER_STEPS = 100
FORM_TAB = "Form"
JSON_TAB = "JSON"
JSON_EDITOR_KEY = "__json_editor__"
TABS_KEY = "__tabs__"
DIRECT_JSON_EDITOR_KEY = "__direct_json_editor__"
NULLABLE_ENABLED_KEY = "__enabled__"
NULLABLE_VALUE_KEY = "__value__"
NULLABLE_NONE_TAB = "None"
NULLABLE_SET_TAB = "Configure"
UNION_ACTIVE_KEY = "__active__"
UNION_KIND_KEY = "__kind__"

RenderMode = Literal["auto", "json", "widget"]
WidgetMode = Literal["auto", "slider"]
