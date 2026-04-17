from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field

import marimo_config_gui._pydantic as pgui
from marimo_config_gui import (
    PydanticGui,
    config_error,
    config_form,
    config_gui,
    config_state,
    config_json,
    config_json_output,
    config_require_valid,
    config_value,
    form_gui,
    json_gui,
)


class _RequiredModel(BaseModel):
    title: str = "demo"
    count: int = Field(0, ge=0)


class _OrderModel(BaseModel):
    zeta: int = 1
    alpha: int = 2
    middle: int = 3


class _NestedLeafModel(BaseModel):
    enabled: bool = True


class _NestedRootModel(BaseModel):
    left: _NestedLeafModel = _NestedLeafModel()
    right: _NestedLeafModel = _NestedLeafModel()


class _PathModel(BaseModel):
    source: Path = Path("README.md")


class _EnumMode(Enum):
    FAST = "fast"
    QUALITY = "quality"


class _EnumModel(BaseModel):
    mode: Literal["fast", "quality"] = "fast"
    enum_mode: _EnumMode = _EnumMode.FAST


class _UnionA(BaseModel):
    value: int = 1


class _UnionB(BaseModel):
    title: str = "b"


class _UnionRoot(BaseModel):
    item: _UnionA | _UnionB = Field(default_factory=_UnionA)


@pytest.fixture
def notebook_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: True)


def _dispatch_form_change(form: PydanticGui[BaseModel], patch: dict[str, object]) -> None:
    form._convert_value(patch)
    form._on_change(None)


def _dispatch_json_change(json_view: object, text: str) -> None:
    json_view._on_change(text)


def _make_state(model_cls: type[BaseModel]):
    return config_state(model_cls)


def test_config_state_returns_state_tuple(notebook_runtime: None) -> None:
    generated = _make_state(_RequiredModel)

    assert isinstance(generated, tuple)
    assert len(generated) == 6
    payload_state, _set_payload, json_text_state, _set_json, json_error_state, _set_error = generated
    assert payload_state() == {"title": "demo", "count": 0}
    assert '"count": 0' in json_text_state()
    assert json_error_state() is None


def test_config_gui_defaults_to_error_and_form(
    notebook_runtime: None,
) -> None:
    error_view, form_view = config_gui(_RequiredModel)

    assert isinstance(form_view, PydanticGui)
    assert error_view.text == pgui.mo.md("").text


def test_config_gui_supports_json_only(notebook_runtime: None) -> None:
    error_view, json_view = config_gui(
        _RequiredModel,
        return_form=False,
        return_json=True,
    )

    assert error_view.text == pgui.mo.md("").text
    assert '"count": 0' in json_view.value


def test_config_gui_supports_form_only_without_error(
    notebook_runtime: None,
) -> None:
    form_view = config_gui(
        _RequiredModel,
        return_error_element=False,
        return_form=True,
        return_json=False,
    )

    assert isinstance(form_view, PydanticGui)


def test_config_gui_requires_a_rendered_view(notebook_runtime: None) -> None:
    with pytest.raises(ValueError, match="at least one of return_form or return_json"):
        config_gui(
            _RequiredModel,
            return_form=False,
            return_json=False,
        )


def test_config_form_updates_payload_and_json_text(notebook_runtime: None) -> None:
    (
        payload_state,
        set_payload_state,
        _json_text_state,
        set_json_text_state,
        _json_error_state,
        set_json_error_state,
    ) = _make_state(_RequiredModel)
    form = config_form(
        _RequiredModel,
        payload_state=payload_state,
        set_payload_state=set_payload_state,
        set_json_text_state=set_json_text_state,
        set_json_error_state=set_json_error_state,
    )

    _dispatch_form_change(form, {"count": 3})

    assert payload_state() == {"title": "demo", "count": 3}
    assert config_value(
        _RequiredModel,
        payload_state=payload_state,
        json_error_state=_json_error_state,
    ) == _RequiredModel(count=3)


def test_config_json_updates_payload_only_on_valid_model(
    notebook_runtime: None,
) -> None:
    (
        payload_state,
        set_payload_state,
        json_text_state,
        set_json_text_state,
        json_error_state,
        set_json_error_state,
    ) = _make_state(_RequiredModel)
    json_view = config_json(
        _RequiredModel,
        payload_state=payload_state,
        set_payload_state=set_payload_state,
        json_text_state=json_text_state,
        set_json_text_state=set_json_text_state,
        json_error_state=json_error_state,
        set_json_error_state=set_json_error_state,
    )

    _dispatch_json_change(json_view, '{"title": "updated", "count": -1}')
    assert payload_state() == {"title": "demo", "count": 0}
    assert json_error_state() == "count: Input should be greater than or equal to 0"

    _dispatch_json_change(json_view, "{")
    assert payload_state() == {"title": "demo", "count": 0}
    assert json_error_state() == "json: Expecting property name enclosed in double quotes"

    _dispatch_json_change(json_view, '{"title": "updated", "count": 5}')
    assert payload_state() == {"title": "updated", "count": 5}
    assert json_error_state() is None


def test_config_error_prefers_json_errors_then_validation(
    notebook_runtime: None,
) -> None:
    (
        payload_state,
        _set_payload_state,
        _json_text_state,
        _set_json_text_state,
        json_error_state,
        set_json_error_state,
    ) = _make_state(_RequiredModel)

    empty = config_error(
        _RequiredModel,
        payload_state=payload_state,
        json_error_state=json_error_state,
    )
    assert empty.text == pgui.mo.md("").text

    set_json_error_state("json: broken")
    errored = config_error(
        _RequiredModel,
        payload_state=payload_state,
        json_error_state=json_error_state,
    )
    assert "json: broken" in errored.text


def test_config_value_and_json_output(notebook_runtime: None) -> None:
    (
        payload_state,
        _set_payload_state,
        _json_text_state,
        _set_json_text_state,
        json_error_state,
        set_json_error_state,
    ) = _make_state(_RequiredModel)

    assert config_value(
        _RequiredModel,
        payload_state=payload_state,
        json_error_state=json_error_state,
    ) == _RequiredModel()
    assert "&quot;count&quot;:0" in config_json_output(
        _RequiredModel,
        payload_state=payload_state,
        json_error_state=json_error_state,
    ).text

    set_json_error_state("json: broken")
    assert config_value(
        _RequiredModel,
        payload_state=payload_state,
        json_error_state=json_error_state,
    ) is None
    assert "Not a valid config" in config_json_output(
        _RequiredModel,
        payload_state=payload_state,
        json_error_state=json_error_state,
    ).text


def test_config_require_valid_stops_when_invalid(
    notebook_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        payload_state,
        _set_payload_state,
        _json_text_state,
        _set_json_text_state,
        json_error_state,
        set_json_error_state,
    ) = _make_state(_RequiredModel)
    set_json_error_state("json: broken")

    captured: dict[str, object] = {}

    def _fake_stop(condition: bool, output: object | None = None) -> None:
        captured["condition"] = condition
        captured["output"] = output
        raise RuntimeError("stopped")

    monkeypatch.setattr(pgui.mo, "stop", _fake_stop)

    with pytest.raises(RuntimeError, match="stopped"):
        config_require_valid(
            _RequiredModel,
            payload_state=payload_state,
            json_error_state=json_error_state,
        )

    assert captured["condition"] is True
    assert "json: broken" in captured["output"].text


def test_script_mode_uses_tyro(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: False)
    monkeypatch.setattr(
        pgui.tyro,
        "cli",
        lambda model_cls, default: model_cls(title="cli", count=4),
    )

    payload_state, *_rest = config_state(_RequiredModel)
    assert payload_state() == {"title": "cli", "count": 4}


def test_form_gui_returns_raw_form(notebook_runtime: None) -> None:
    generated = form_gui(_RequiredModel)

    assert isinstance(generated, PydanticGui)
    assert generated.value == _RequiredModel()


def test_json_gui_returns_raw_json_editor(notebook_runtime: None) -> None:
    generated = json_gui(_RequiredModel)
    assert '"count": 0' in generated.value


def test_form_builder_keeps_path_and_enum_widgets() -> None:
    path_gui = PydanticGui(_PathModel, include_json_editor=False)
    enum_gui = PydanticGui(_EnumModel, include_json_editor=False)

    assert type(path_gui.elements["source"]).__name__ == "file_browser"
    assert type(enum_gui.elements["mode"]).__name__ == "dropdown"
    assert type(enum_gui.elements["enum_mode"]).__name__ == "dropdown"


def test_nested_models_use_accordion_until_flat_level() -> None:
    collapsed = PydanticGui(
        _NestedRootModel,
        include_json_editor=False,
        nested_models_flat_after_level=0,
    )
    sectioned = PydanticGui(
        _NestedRootModel,
        include_json_editor=False,
        nested_models_flat_after_level=1,
    )

    assert "marimo-accordion" not in collapsed.text
    assert "marimo-accordion" in sectioned.text


def test_union_json_serialization_keeps_kind(notebook_runtime: None) -> None:
    (
        payload_state,
        set_payload_state,
        json_text_state,
        set_json_text_state,
        json_error_state,
        set_json_error_state,
    ) = _make_state(_UnionRoot)
    json_view = config_json(
        _UnionRoot,
        payload_state=payload_state,
        set_payload_state=set_payload_state,
        json_text_state=json_text_state,
        set_json_text_state=set_json_text_state,
        json_error_state=json_error_state,
        set_json_error_state=set_json_error_state,
    )

    _dispatch_json_change(
        json_view,
        '{\n  "item": {\n    "__kind__": "_UnionB",\n    "title": "switched"\n  }\n}',
    )

    assert json_error_state() is None
    assert "&quot;__kind__&quot;" in config_json_output(
        _UnionRoot,
        payload_state=payload_state,
        json_error_state=json_error_state,
    ).text


def test_json_editor_uses_model_field_order(notebook_runtime: None) -> None:
    (
        payload_state,
        set_payload_state,
        json_text_state,
        set_json_text_state,
        json_error_state,
        set_json_error_state,
    ) = _make_state(_OrderModel)
    json_view = config_json(
        _OrderModel,
        payload_state=payload_state,
        set_payload_state=set_payload_state,
        json_text_state=json_text_state,
        set_json_text_state=set_json_text_state,
        json_error_state=json_error_state,
        set_json_error_state=set_json_error_state,
    )
    assert json_view.value.index('"zeta"') < json_view.value.index('"alpha"')
    assert json_view.value.index('"alpha"') < json_view.value.index('"middle"')
