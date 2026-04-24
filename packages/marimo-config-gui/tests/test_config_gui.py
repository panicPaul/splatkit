from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import marimo_config_gui._pydantic as pgui
import pytest
from marimo_config_gui import (
    ConfigBindings,
    PydanticGui,
    config_commit_button,
    config_committed_value,
    config_error,
    config_form,
    config_gui,
    config_json,
    config_json_output,
    config_require_valid,
    config_value,
    create_committed_config_state,
    create_config_state,
    form_gui,
    json_gui,
    load_script_config,
)
from pydantic import BaseModel, Field


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
    return create_config_state(model_cls)


def _dispatch_button_click(button: object) -> None:
    next_value = button._on_click(button.value)
    button._value = next_value
    button._value_frontend = button._convert_value(next_value)


def test_config_state_returns_state_tuple(notebook_runtime: None) -> None:
    generated = _make_state(_RequiredModel)

    assert isinstance(generated, tuple)
    assert len(generated) == 3
    form_gui_state, json_gui_state, bindings = generated
    assert form_gui_state() == {"title": "demo", "count": 0}
    assert '"count": 0' in json_gui_state()
    assert isinstance(bindings, ConfigBindings)


def test_committed_config_state_matches_initial_payload(
    notebook_runtime: None,
) -> None:
    committed_state, _set_committed_state = create_committed_config_state(
        _RequiredModel
    )

    assert committed_state() == {"title": "demo", "count": 0}


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
        form_gui_state,
        _json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)
    form = config_form(
        bindings,
        form_gui_state=form_gui_state,
    )

    _dispatch_form_change(form, {"count": 3})

    assert form_gui_state() == {"title": "demo", "count": 3}
    assert config_value(
        _RequiredModel,
        form_gui_state=form_gui_state,
        json_gui_state=_json_gui_state,
    ) == _RequiredModel(count=3)


def test_config_json_updates_payload_only_on_valid_model(
    notebook_runtime: None,
) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)
    json_view = config_json(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )

    _dispatch_json_change(json_view, '{"title": "updated", "count": -1}')
    assert form_gui_state() == {"title": "demo", "count": 0}
    assert config_error(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    ).text != pgui.mo.md("").text

    _dispatch_json_change(json_view, "{")
    assert form_gui_state() == {"title": "demo", "count": 0}
    assert "Expecting property name enclosed in double quotes" in config_error(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    ).text

    _dispatch_json_change(json_view, '{"title": "updated", "count": 5}')
    assert form_gui_state() == {"title": "updated", "count": 5}
    assert config_error(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    ).text == pgui.mo.md("").text


def test_config_error_prefers_json_errors_then_validation(
    notebook_runtime: None,
) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)

    empty = config_error(bindings, form_gui_state=form_gui_state, json_gui_state=json_gui_state)
    assert empty.text == pgui.mo.md("").text

    bindings.set_json_gui_state("{")
    errored = config_error(bindings, form_gui_state=form_gui_state, json_gui_state=json_gui_state)
    assert "Expecting property name enclosed in double quotes" in errored.text


def test_config_value_and_json_output(notebook_runtime: None) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)

    assert config_value(bindings, form_gui_state=form_gui_state, json_gui_state=json_gui_state) == _RequiredModel()
    assert "&quot;count&quot;:0" in config_json_output(bindings, form_gui_state=form_gui_state, json_gui_state=json_gui_state).text

    bindings.set_json_gui_state("{")
    assert config_value(bindings, form_gui_state=form_gui_state, json_gui_state=json_gui_state) is None
    assert "Not a valid config" in config_json_output(bindings, form_gui_state=form_gui_state, json_gui_state=json_gui_state).text


def test_config_commit_button_commits_valid_dirty_draft(
    notebook_runtime: None,
) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)
    committed_state, set_committed_state = create_committed_config_state(
        _RequiredModel
    )
    bindings.set_form_gui_state({"title": "demo", "count": 5})
    bindings.set_json_gui_state('{\n  "title": "demo",\n  "count": 5\n}')

    commit_button = config_commit_button(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
        committed_state=committed_state,
        set_committed_state=set_committed_state,
    )

    assert commit_button._component_args["disabled"] is False
    _dispatch_button_click(commit_button)
    assert committed_state() == {"title": "demo", "count": 5}
    assert config_committed_value(
        bindings,
        committed_state=committed_state,
    ) == _RequiredModel(count=5)


def test_config_commit_button_stays_disabled_when_not_dirty(
    notebook_runtime: None,
) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)
    committed_state, set_committed_state = create_committed_config_state(
        _RequiredModel
    )

    commit_button = config_commit_button(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
        committed_state=committed_state,
        set_committed_state=set_committed_state,
    )

    assert commit_button._component_args["disabled"] is True
    assert commit_button._component_args["tooltip"] == "No unapplied config changes."


def test_config_commit_button_stays_disabled_when_invalid(
    notebook_runtime: None,
) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)
    committed_state, set_committed_state = create_committed_config_state(
        _RequiredModel
    )
    bindings.set_json_gui_state("{")

    commit_button = config_commit_button(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
        committed_state=committed_state,
        set_committed_state=set_committed_state,
    )

    assert commit_button._component_args["disabled"] is True
    assert commit_button._component_args["tooltip"] == "Fix config errors before applying."
    assert committed_state() == {"title": "demo", "count": 0}


def test_config_committed_value_returns_none_for_invalid_payload(
    notebook_runtime: None,
) -> None:
    committed_state, set_committed_state = create_committed_config_state(
        _RequiredModel
    )
    set_committed_state({"title": "demo", "count": -1})

    assert (
        config_committed_value(
            _RequiredModel,
            committed_state=committed_state,
        )
        is None
    )


def test_config_require_valid_stops_when_invalid(
    notebook_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)
    bindings.set_json_gui_state("{")

    captured: dict[str, object] = {}

    def _fake_stop(condition: bool, output: object | None = None) -> None:
        captured["condition"] = condition
        captured["output"] = output
        raise RuntimeError("stopped")

    monkeypatch.setattr(pgui.mo, "stop", _fake_stop)

    with pytest.raises(RuntimeError, match="stopped"):
        config_require_valid(bindings, form_gui_state=form_gui_state, json_gui_state=json_gui_state)

    assert captured["condition"] is True
    assert "Expecting property name enclosed in double quotes" in captured["output"].text


def test_script_mode_uses_tyro(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: False)
    monkeypatch.setattr(
        pgui,
        "load_script_config",
        lambda model_cls, value=None, args=None: model_cls(
            title="cli", count=4
        ),
    )

    form_gui_state, *_rest = create_config_state(_RequiredModel)
    assert form_gui_state() == {"title": "cli", "count": 4}


def test_script_mode_forwards_explicit_script_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: False)
    captured: dict[str, object] = {}

    def _fake_loader(model_cls, value=None, args=None):
        captured["model_cls"] = model_cls
        captured["value"] = value
        captured["args"] = args
        return model_cls(title="cli", count=6)

    monkeypatch.setattr(pgui, "load_script_config", _fake_loader)

    form_gui_state, *_rest = create_config_state(
        _RequiredModel,
        script_args=["cli", "--title", "cli", "--count", "6"],
    )

    assert captured["model_cls"] is _RequiredModel
    assert captured["args"] == ["cli", "--title", "cli", "--count", "6"]
    assert form_gui_state() == {"title": "cli", "count": 6}


def test_script_mode_supports_custom_script_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: False)

    def _custom_loader(model_cls, value=None, args=None):
        del args
        assert value == {"title": "base", "count": 2}
        return model_cls(title="preset", count=9)

    form_gui_state, *_rest = create_config_state(
        _RequiredModel,
        value={"title": "base", "count": 2},
        script_loader=_custom_loader,
        script_args=["preset", "demo"],
    )

    assert form_gui_state() == {"title": "preset", "count": 9}


def test_committed_script_state_supports_custom_script_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: False)

    def _custom_loader(model_cls, value=None, args=None):
        del value, args
        return model_cls(title="committed", count=8)

    committed_state, _set_committed_state = create_committed_config_state(
        _RequiredModel,
        script_loader=_custom_loader,
    )

    assert committed_state() == {"title": "committed", "count": 8}


def test_config_gui_supports_custom_script_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: False)

    def _custom_loader(model_cls, value=None, args=None):
        del value, args
        return model_cls(title="gui", count=3)

    error_view, form_view = config_gui(
        _RequiredModel,
        script_loader=_custom_loader,
    )

    assert error_view.text == pgui.mo.md("").text
    assert form_view.value == _RequiredModel(title="gui", count=3)


def test_load_script_config_supports_cli_subcommand() -> None:
    loaded = load_script_config(
        _RequiredModel,
        args=["cli", "--title", "cli", "--count", "5"],
    )

    assert loaded == _RequiredModel(title="cli", count=5)


def test_load_script_config_supports_json_subcommand(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"title": "json", "count": 7}')

    loaded = load_script_config(
        _RequiredModel,
        args=["json", str(config_path)],
    )

    assert loaded == _RequiredModel(title="json", count=7)


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


def test_indent_nested_layout_wraps_nested_content() -> None:
    layout = pgui.mo.md("demo")

    unindented = pgui._indent_nested_layout(layout, current_level=0)
    indented = pgui._indent_nested_layout(layout, current_level=2)

    assert unindented is layout
    assert "border-left:" in indented.text
    assert "margin-left:" in indented.text


def test_union_json_serialization_keeps_kind(notebook_runtime: None) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_UnionRoot)
    json_view = config_json(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )

    _dispatch_json_change(
        json_view,
        '{\n  "item": {\n    "__kind__": "_UnionB",\n    "title": "switched"\n  }\n}',
    )

    assert "&quot;__kind__&quot;" in config_json_output(bindings, form_gui_state=form_gui_state, json_gui_state=json_gui_state).text


def test_json_editor_uses_model_field_order(notebook_runtime: None) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_OrderModel)
    json_view = config_json(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    assert json_view.value.index('"zeta"') < json_view.value.index('"alpha"')
    assert json_view.value.index('"alpha"') < json_view.value.index('"middle"')
