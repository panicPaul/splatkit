from __future__ import annotations

import json
from enum import Enum, Flag, IntFlag, auto
from pathlib import Path
from typing import Any, Literal

import marimo_config_gui.api as pgui
import marimo_config_gui.labels as labels
import marimo_config_gui.widgets as widgets
import pytest
from marimo._runtime.control_flow import MarimoStopError
from marimo_config_gui import (
    ConfigFile,
    ConfigPreset,
    ConfigPresetCatalog,
    create_config_gui,
)
from marimo_config_gui.api import load_script_config
from marimo_config_gui.elements import (
    CONFIG_FORM_VIEW_KEY,
    CONFIG_JSON_VIEW_KEY,
    PydanticGui,
)
from marimo_config_gui.presets import load_json_config, load_preset_config
from marimo_config_gui.state import ConfigBindings
from pydantic import BaseModel, ConfigDict, Field

config_gui_panel = pgui.config_gui_panel
config_json_editor = pgui.config_json_editor
config_preset_selector = pgui.config_preset_selector
config_status_panel = pgui.config_status_panel
create_config_state = pgui.create_config_state
validated_config = pgui.validated_config


class _RequiredModel(BaseModel):
    title: str = "demo"
    count: int = Field(0, ge=0)


class _BoolModel(BaseModel):
    flag: bool = False


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


class _DictModel(BaseModel):
    payload: dict[str, Any] = Field(
        default_factory=lambda: {"alpha": 1, "enabled": True}
    )


class _PrimitiveSequenceModel(BaseModel):
    opacity_range: tuple[float, float] = (0.1, 0.9)
    crop_origin: tuple[int, int, int] = (1, 2, 3)
    weights: tuple[float, float, float, float, float] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    )
    labels: list[str] = Field(default_factory=lambda: ["train", "eval"])


class _VariableTupleModel(BaseModel):
    values: tuple[int, ...] = (1, 2)


class _LongTupleModel(BaseModel):
    values: tuple[int, int, int, int, int, int] = (1, 2, 3, 4, 5, 6)


class _OptionalModel(BaseModel):
    source: Path | None = Path("README.md")


class _PresetModel(BaseModel):
    preset: str = "base"
    title: str = "demo"
    count: int = Field(0, ge=0)
    path: Path = Path("data/input")


class _EnumMode(Enum):
    FAST = "fast"
    QUALITY = "quality"


class _ColorSpace(Enum):
    SRGB = "srgb"
    LINEAR_RGB = "linear_rgb"


class _FeatureFlag(Flag):
    NONE = 0
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()


class _PermissionFlag(IntFlag):
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 4


class _EnumModel(BaseModel):
    mode: Literal["fast", "quality"] = "fast"
    enum_mode: _EnumMode = _EnumMode.FAST
    color_space: _ColorSpace = _ColorSpace.SRGB


class _FlagModel(BaseModel):
    features: _FeatureFlag = _FeatureFlag.READ | _FeatureFlag.WRITE
    permissions: _PermissionFlag = _PermissionFlag.READ


class _OptionalFlagModel(BaseModel):
    features: _FeatureFlag | None = None


class _UnionA(BaseModel):
    value: int = 1


class _UnionB(BaseModel):
    title: str = "b"


class _UnionRoot(BaseModel):
    item: _UnionA | _UnionB = Field(default_factory=_UnionA)


class _UnionFieldDefaultRoot(BaseModel):
    enabled: bool = True
    item: _UnionA | _UnionB = Field(_UnionA())


class ResNet50(BaseModel):
    model_config = ConfigDict(title="ResNet-50")

    depth: int = 50


class ContextNet(BaseModel):
    model_config = ConfigDict(title="Context Network")

    width: int = 64


class CustomNetwork(BaseModel):
    width: int = 128


class _UnionBranchLabelRoot(BaseModel):
    subconfig: ResNet50 | ContextNet | CustomNetwork = Field(
        default_factory=ResNet50
    )


class _TitledResNetConfig(BaseModel):
    model_config = ConfigDict(title="Custom ResNet-50")


class _TitledRuntimeConfig(BaseModel):
    model_config = ConfigDict(title="Runtime Settings")

    seed: int = 0


class PlainRuntimeConfig(BaseModel):
    seed: int = 0


class _LabelPrecedenceRoot(BaseModel):
    titled_field: _TitledRuntimeConfig = Field(
        default_factory=_TitledRuntimeConfig,
        title="Field Runtime",
    )
    titled_model: _TitledRuntimeConfig = Field(
        default_factory=_TitledRuntimeConfig
    )
    plain_model: PlainRuntimeConfig = Field(default_factory=PlainRuntimeConfig)
    plain_value: int = 0


@pytest.fixture
def notebook_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: True)


def test_public_package_surface_is_intentional() -> None:
    import marimo_config_gui

    expected = {
        "ConfigPreset",
        "ConfigPresetCatalog",
        "ConfigFile",
        "__version__",
        "create_config_gui",
    }
    legacy = {
        "ConfigBindings",
        "PydanticGui",
        "config_commit_button",
        "config_committed_value",
        "config_error",
        "config_form",
        "config_gui",
        "config_json",
        "config_json_output",
        "config_gui_panel",
        "config_json_editor",
        "config_preset_selector",
        "config_status_panel",
        "config_require_valid",
        "config_value",
        "create_config_state",
        "create_committed_config_state",
        "form_gui",
        "json_gui",
        "load_json_config",
        "load_preset_config",
        "load_script_config",
        "validated_config",
    }

    assert set(marimo_config_gui.__all__) == expected
    assert all(hasattr(marimo_config_gui, name) for name in expected)
    assert all(not hasattr(marimo_config_gui, name) for name in legacy)


def _dispatch_form_change(
    form: PydanticGui[BaseModel], patch: dict[str, object]
) -> None:
    form._convert_value(patch)
    form._on_change(None)


def _dispatch_json_change(json_view: object, text: str) -> None:
    json_view._on_change(text)


def _make_state(model_cls: type[BaseModel]):
    return create_config_state(model_cls)


def test_config_gui_owner_initializes_views_and_value(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(_RequiredModel, background=None)

    assert config_gui.validated_config() == _RequiredModel()
    assert type(config_gui.gui_panel()).__name__ == "PydanticGui"
    assert type(config_gui.json_editor()).__name__ == "code_editor"
    assert config_gui.status_panel().text == pgui.mo.md("").text


def test_config_gui_owner_registers_child_views_for_wrapping(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(_BoolModel, background=None)
    form = config_gui.gui_panel()
    json_editor = config_gui.json_editor()

    assert form._lens is not None
    assert form._lens.parent_id == config_gui._id
    assert form._lens.key == CONFIG_FORM_VIEW_KEY
    assert json_editor._lens is not None
    assert json_editor._lens.parent_id == config_gui._id
    assert json_editor._lens.key == CONFIG_JSON_VIEW_KEY


def test_config_gui_owner_syncs_bool_form_to_json(
    notebook_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_gui = create_config_gui(_BoolModel, background=None)
    sent_messages: list[dict[str, object]] = []

    def _record_json_message(
        message: dict[str, object], buffers: object
    ) -> None:
        del buffers
        sent_messages.append(message)

    monkeypatch.setattr(
        config_gui.json_editor(),
        "_send_message",
        _record_json_message,
    )

    config_gui._convert_value({CONFIG_FORM_VIEW_KEY: {"flag": True}})

    assert config_gui.validated_config() == _BoolModel(flag=True)
    assert '"flag": true' in config_gui.json_editor().value
    assert sent_messages[-1]["type"] == "marimo-ui-value-update"
    assert '"flag": true' in str(sent_messages[-1]["value"])
    assert config_gui.status_panel().text == pgui.mo.md("").text


def test_config_gui_owner_syncs_bool_json_to_form(
    notebook_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_gui = create_config_gui(_BoolModel, background=None)
    sent_messages: list[dict[str, object]] = []
    flag_control = config_gui.gui_panel().elements["flag"]

    def _record_flag_message(
        message: dict[str, object], buffers: object
    ) -> None:
        del buffers
        sent_messages.append(message)

    monkeypatch.setattr(flag_control, "_send_message", _record_flag_message)

    config_gui._convert_value({CONFIG_JSON_VIEW_KEY: '{"flag": true}'})

    assert config_gui.validated_config() == _BoolModel(flag=True)
    assert config_gui.gui_panel().elements["flag"]._value_frontend is True
    assert sent_messages[-1] == {
        "type": "marimo-ui-value-update",
        "value": True,
    }
    assert '"flag": true' in config_gui.json_editor().value
    assert config_gui.status_panel().text == pgui.mo.md("").text


def test_config_gui_owner_syncs_flags_both_directions(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(_FlagModel, background=None)

    config_gui._convert_value(
        {
            CONFIG_FORM_VIEW_KEY: {
                "features": ["execute"],
                "permissions": ["read", "write"],
            }
        }
    )

    assert config_gui.validated_config() == _FlagModel(
        features=_FeatureFlag.EXECUTE,
        permissions=_PermissionFlag.READ | _PermissionFlag.WRITE,
    )
    assert '"features": [\n    "EXECUTE"\n  ]' in config_gui.json_editor().value

    config_gui._convert_value(
        {
            CONFIG_JSON_VIEW_KEY: (
                '{"features": ["READ", "WRITE"], "permissions": ["EXECUTE"]}'
            )
        }
    )

    assert config_gui.validated_config() == _FlagModel(
        features=_FeatureFlag.READ | _FeatureFlag.WRITE,
        permissions=_PermissionFlag.EXECUTE,
    )
    assert config_gui.gui_panel().elements["features"]._value_frontend == [
        "read",
        "write",
    ]
    assert config_gui.gui_panel().elements["permissions"]._value_frontend == [
        "execute"
    ]


def test_config_gui_owner_invalid_json_does_not_corrupt_form(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(_BoolModel, background=None)
    config_gui._convert_value({CONFIG_FORM_VIEW_KEY: {"flag": True}})

    config_gui._convert_value({CONFIG_JSON_VIEW_KEY: "{"})

    assert config_gui.is_valid() is False
    assert "Expecting property name enclosed in double quotes" in (
        config_gui.validation_error() or ""
    )
    with pytest.raises(MarimoStopError):
        config_gui.validated_config()
    assert "Expecting property name enclosed in double quotes" in (
        config_gui.status_panel().text
    )
    assert config_gui.gui_panel().elements["flag"]._value_frontend is True


def test_config_gui_owner_invalid_flag_name_reports_status(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(_FlagModel, background=None)

    config_gui._convert_value({CONFIG_JSON_VIEW_KEY: '{"features": ["GREN"]}'})

    assert config_gui.is_valid() is False
    assert "GREN" in (config_gui.validation_error() or "")
    with pytest.raises(MarimoStopError):
        config_gui.validated_config()
    assert "GREN" in config_gui.status_panel().text
    assert "_FeatureFlag" in config_gui.status_panel().text


def test_config_gui_owner_uses_default_neutral_background(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(_BoolModel)

    assert "rgba(113, 113, 122, 0.10)" in config_gui.gui_panel().text
    assert "rgba(113, 113, 122, 0.10)" in config_gui.json_editor().text
    assert type(config_gui.gui_panel(background=None)).__name__ == "PydanticGui"


def test_config_gui_owner_rejects_removed_yellow_background(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(_BoolModel)

    with pytest.raises(
        ValueError, match="danger, info, neutral, success, warn"
    ):
        config_gui.gui_panel(background="yellow")


def test_config_gui_owner_accepts_custom_background(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(
        _BoolModel,
        background={"background": "white", "padding": "1rem"},
    )

    assert "background:white" in config_gui.gui_panel().text
    assert "padding:1rem" in config_gui.json_editor().text


def test_config_gui_owner_accepts_named_backgrounds(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(_BoolModel, background="success")

    assert "rgba(34, 197, 94, 0.12)" in config_gui.gui_panel().text
    assert "rgba(34, 197, 94, 0.12)" in config_gui.json_editor().text
    assert "rgba(59, 130, 246, 0.11)" in (
        config_gui.gui_panel(background="info").text
    )


def test_config_gui_owner_stacked_layout_background_scopes(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(_BoolModel)

    panel_stack = config_gui.stacked()
    outer_stack = config_gui.stacked(background_scope="stack")
    unstyled_stack = config_gui.stacked(background=None)

    assert panel_stack.text.count("rgba(113, 113, 122, 0.10)") == 2
    assert outer_stack.text.count("rgba(113, 113, 122, 0.10)") == 1
    assert "marimo-ui-element" in panel_stack.text
    assert unstyled_stack.text.count("rgba(113, 113, 122, 0.10)") == 0


def test_config_state_returns_state_tuple(notebook_runtime: None) -> None:
    generated = _make_state(_RequiredModel)

    assert isinstance(generated, tuple)
    assert len(generated) == 3
    form_gui_state, json_gui_state, bindings = generated
    assert form_gui_state() == {"title": "demo", "count": 0}
    assert '"count": 0' in json_gui_state()
    assert isinstance(bindings, ConfigBindings)


def test_config_gui_panel_updates_payload_and_json_text(
    notebook_runtime: None,
) -> None:
    (
        form_gui_state,
        _json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)
    form = config_gui_panel(
        bindings,
        form_gui_state=form_gui_state,
    )

    _dispatch_form_change(form, {"count": 3})

    assert form_gui_state() == {"title": "demo", "count": 3}
    assert validated_config(
        _RequiredModel,
        form_gui_state=form_gui_state,
        json_gui_state=_json_gui_state,
    ) == _RequiredModel(count=3)


def test_config_json_editor_updates_payload_only_on_valid_model(
    notebook_runtime: None,
) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)
    json_view = config_json_editor(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )

    _dispatch_json_change(json_view, '{"title": "updated", "count": -1}')
    assert form_gui_state() == {"title": "demo", "count": 0}
    assert (
        config_status_panel(
            bindings,
            form_gui_state=form_gui_state,
            json_gui_state=json_gui_state,
        ).text
        != pgui.mo.md("").text
    )

    _dispatch_json_change(json_view, "{")
    assert form_gui_state() == {"title": "demo", "count": 0}
    assert (
        "Expecting property name enclosed in double quotes"
        in config_status_panel(
            bindings,
            form_gui_state=form_gui_state,
            json_gui_state=json_gui_state,
        ).text
    )

    _dispatch_json_change(json_view, '{"title": "updated", "count": 5}')
    assert form_gui_state() == {"title": "updated", "count": 5}
    assert (
        config_status_panel(
            bindings,
            form_gui_state=form_gui_state,
            json_gui_state=json_gui_state,
        ).text
        == pgui.mo.md("").text
    )


def test_config_status_panel_prefers_json_errors_then_validation(
    notebook_runtime: None,
) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)

    empty = config_status_panel(
        bindings, form_gui_state=form_gui_state, json_gui_state=json_gui_state
    )
    assert empty.text == pgui.mo.md("").text

    bindings.set_json_gui_state("{")
    errored = config_status_panel(
        bindings, form_gui_state=form_gui_state, json_gui_state=json_gui_state
    )
    assert "Expecting property name enclosed in double quotes" in errored.text


def test_legacy_validated_config_returns_model_or_stops(
    notebook_runtime: None,
) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_RequiredModel)

    assert (
        validated_config(
            bindings,
            form_gui_state=form_gui_state,
            json_gui_state=json_gui_state,
        )
        == _RequiredModel()
    )

    bindings.set_json_gui_state("{")
    with pytest.raises(MarimoStopError):
        validated_config(
            bindings,
            form_gui_state=form_gui_state,
            json_gui_state=json_gui_state,
        )


def test_config_gui_owner_invalid_script_mode_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pgui.mo, "running_in_notebook", lambda: False)
    config_gui = create_config_gui(_BoolModel, background=None, script_args=[])
    config_gui._convert_value({CONFIG_JSON_VIEW_KEY: "{"})

    assert config_gui.is_valid() is False
    with pytest.raises(ValueError, match="Expecting property name"):
        config_gui.validated_config()


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
        script_args=["--title", "cli", "--count", "6"],
    )

    assert captured["model_cls"] is _RequiredModel
    assert captured["args"] == ["--title", "cli", "--count", "6"]
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
        script_args=["--preset", "demo"],
    )

    assert form_gui_state() == {"title": "preset", "count": 9}


def test_load_script_config_supports_default_cli_overrides() -> None:
    loaded = load_script_config(
        _RequiredModel,
        args=["--title", "cli", "--count", "5"],
    )

    assert loaded == _RequiredModel(title="cli", count=5)


def test_load_script_config_supports_flag_cli_values() -> None:
    loaded = load_script_config(
        _FlagModel,
        args=[
            "--features",
            "READ",
            "WRITE",
            "--permissions",
            "EXECUTE",
        ],
    )

    assert loaded == _FlagModel(
        features=_FeatureFlag.READ | _FeatureFlag.WRITE,
        permissions=_PermissionFlag.EXECUTE,
    )


def test_load_script_config_supports_json_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"title": "json", "count": 7}')

    loaded = load_script_config(
        _RequiredModel,
        args=[str(config_path)],
    )

    assert loaded == _RequiredModel(title="json", count=7)


def test_load_script_config_supports_json_path_overrides(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"title": "json", "count": 7}')

    loaded = load_script_config(
        _RequiredModel,
        args=[str(config_path), "--title", "server", "--count", "11"],
    )

    assert loaded == _RequiredModel(title="server", count=11)


def test_load_script_config_applies_json_overlays_before_cli(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    overlay_path = tmp_path / "overlay.json"
    config_path.write_text('{"title": "json", "count": 7}')
    overlay_path.write_text('{"title": "overlay", "count": 9}')

    loaded = load_script_config(
        _RequiredModel,
        args=[
            str(config_path),
            "--overlay",
            str(overlay_path),
            "--count",
            "11",
        ],
    )

    assert loaded == _RequiredModel(title="overlay", count=11)


def test_load_json_config_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "config.json"
    config_path.parent.mkdir()
    config_path.write_text(
        '{"preset": "file", "title": "json", "count": 2, "path": "data"}'
    )

    loaded = load_json_config(_PresetModel, config_path)

    assert loaded == _PresetModel(
        preset="file",
        title="json",
        count=2,
        path=config_path.parent / "data",
    )


def test_load_json_config_applies_sibling_path_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"preset": "file", "title": "json", "count": 2, "path": "data/input"}'
    )
    (tmp_path / ".path_defaults.json").write_text(
        '{"path_prefixes": {"data": "/mnt/datasets"}}'
    )

    loaded = load_json_config(_PresetModel, config_path)

    assert loaded.path == Path("/mnt/datasets/input")


def test_path_defaults_only_apply_to_path_fields(tmp_path: Path) -> None:
    class _StringPathModel(BaseModel):
        path: Path = Path("data/input")
        label: str = "data/input"

    config_path = tmp_path / "config.json"
    config_path.write_text('{"path": "data/input", "label": "data/input"}')
    (tmp_path / ".path_defaults.json").write_text(
        '{"path_prefixes": {"data": "/mnt/data"}}'
    )

    loaded = load_json_config(_StringPathModel, config_path)

    assert loaded.path == Path("/mnt/data/input")
    assert loaded.label == "data/input"


def test_path_defaults_remap_point_cloud_ply_load_config(
    tmp_path: Path,
) -> None:
    class _PlyLoadConfig(BaseModel):
        ply_path: Path = Path("point_cloud.ply")

    local_ply = tmp_path / "scenes" / "garden" / "point_cloud.ply"
    config_path = tmp_path / "viewer.json"
    config_path.write_text('{"ply_path": "point_cloud.ply"}')
    (tmp_path / ".path_defaults.json").write_text(
        json.dumps({"fields": {"ply_path": str(local_ply)}})
    )

    loaded = load_json_config(_PlyLoadConfig, config_path)

    assert loaded.ply_path == local_ply


def test_explicit_path_defaults_tuple_can_be_required(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"preset": "file", "title": "json", "count": 2, "path": "data/input"}'
    )

    with pytest.raises(FileNotFoundError):
        load_json_config(
            _PresetModel,
            config_path,
            path_defaults=[("missing_paths.json", True)],
        )


def test_config_overlay_file_entries_support_required_defaults(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"title": "json", "count": 7}')

    with pytest.raises(FileNotFoundError):
        load_json_config(_RequiredModel, config_path, overlays=["missing.json"])

    loaded = load_json_config(
        _RequiredModel,
        config_path,
        overlays=[ConfigFile(path=Path("missing.json"), required=False)],
    )
    assert loaded == _RequiredModel(title="json", count=7)


def test_load_json_config_supports_flag_name_lists(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"features": ["READ", "WRITE"], "permissions": ["EXECUTE"]}'
    )

    loaded = load_json_config(_FlagModel, config_path)

    assert loaded == _FlagModel(
        features=_FeatureFlag.READ | _FeatureFlag.WRITE,
        permissions=_PermissionFlag.EXECUTE,
    )


def test_config_json_editor_normalizes_flag_name_lists(
    notebook_runtime: None,
) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_FlagModel)
    json_view = config_json_editor(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )

    _dispatch_json_change(
        json_view,
        '{"features": ["EXECUTE"], "permissions": ["READ", "WRITE"]}',
    )

    assert form_gui_state() == {
        "features": _FeatureFlag.EXECUTE,
        "permissions": _PermissionFlag.READ | _PermissionFlag.WRITE,
    }
    assert '"features": [\n    "EXECUTE"\n  ]' in json_gui_state()


def test_preset_catalog_loads_default_and_sets_preset_field(
    tmp_path: Path,
) -> None:
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(
        '{"preset": "ignored", "title": "preset", "count": 4, "path": "scene"}'
    )
    catalog = ConfigPresetCatalog(
        model_cls=_PresetModel,
        presets={
            "demo": ConfigPreset(
                name="demo",
                path=preset_path,
                base_dir=tmp_path,
            )
        },
        default="demo",
    )

    loaded = load_preset_config(catalog)

    assert loaded == _PresetModel(
        preset="demo",
        title="preset",
        count=4,
        path=tmp_path / "scene",
    )


def test_config_state_initializes_from_preset_catalog(
    notebook_runtime: None,
    tmp_path: Path,
) -> None:
    preset_path = tmp_path / "preset.json"
    preset_path.write_text(
        '{"preset": "ignored", "title": "preset", "count": 4, "path": "scene"}'
    )
    catalog = ConfigPresetCatalog(
        model_cls=_PresetModel,
        presets={
            "demo": ConfigPreset(
                name="demo",
                path=preset_path,
                base_dir=tmp_path,
            )
        },
        default="demo",
    )

    form_gui_state, json_gui_state, _bindings = create_config_state(
        _PresetModel,
        presets=catalog,
    )

    assert form_gui_state()["preset"] == "demo"
    assert form_gui_state()["path"] == tmp_path / "scene"
    assert '"preset": "demo"' in json_gui_state()


def test_load_script_config_supports_preset_cli_selection(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.json"
    quality_path = tmp_path / "quality.json"
    base_path.write_text(
        '{"preset": "ignored", "title": "base", "count": 1, "path": "base"}'
    )
    quality_path.write_text(
        '{"preset": "ignored", "title": "quality", "count": 2, "path": "q"}'
    )
    catalog = ConfigPresetCatalog(
        model_cls=_PresetModel,
        presets={
            "base": ConfigPreset(
                name="base",
                path=base_path,
                label="Base preset",
                base_dir=tmp_path,
            ),
            "quality": ConfigPreset(
                name="quality",
                path=quality_path,
                label="Quality preset",
                base_dir=tmp_path,
            ),
        },
        default="base",
    )

    loaded = load_script_config(
        _PresetModel,
        presets=catalog,
        args=[
            "--preset",
            "quality",
            "--title",
            "overridden",
            "--count",
            "8",
        ],
    )

    assert loaded == _PresetModel(
        preset="quality",
        title="overridden",
        count=8,
        path=tmp_path / "q",
    )


def test_load_script_config_supports_preset_overlays_before_cli(
    tmp_path: Path,
) -> None:
    preset_path = tmp_path / "base.json"
    overlay_path = tmp_path / "overlay.json"
    preset_path.write_text(
        '{"preset": "ignored", "title": "base", "count": 1, "path": "base"}'
    )
    overlay_path.write_text('{"title": "overlay", "count": 5}')
    catalog = ConfigPresetCatalog(
        model_cls=_PresetModel,
        presets={
            "base": ConfigPreset(
                name="base",
                path=preset_path,
                base_dir=tmp_path,
            ),
        },
        default="base",
    )

    loaded = load_script_config(
        _PresetModel,
        presets=catalog,
        args=[
            "--preset",
            "base",
            "--overlay",
            str(overlay_path),
            "--count",
            "8",
        ],
    )

    assert loaded.title == "overlay"
    assert loaded.count == 8


def test_config_preset_selector_updates_form_and_json(
    notebook_runtime: None,
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.json"
    quality_path = tmp_path / "quality.json"
    base_path.write_text(
        '{"preset": "ignored", "title": "base", "count": 1, "path": "base"}'
    )
    quality_path.write_text(
        '{"preset": "ignored", "title": "quality", "count": 2, "path": "q"}'
    )
    catalog = ConfigPresetCatalog(
        model_cls=_PresetModel,
        presets={
            "base": ConfigPreset(
                name="base",
                path=base_path,
                label="Base preset",
                base_dir=tmp_path,
            ),
            "quality": ConfigPreset(
                name="quality",
                path=quality_path,
                label="Quality preset",
                base_dir=tmp_path,
            ),
        },
        default="base",
    )
    form_gui_state, json_gui_state, bindings = create_config_state(
        _PresetModel,
        presets=catalog,
    )

    selector = config_preset_selector(
        bindings,
        presets=catalog,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    assert selector.value == "base"
    selector._on_change("quality")

    assert form_gui_state()["preset"] == "quality"
    assert form_gui_state()["title"] == "quality"
    assert '"preset": "quality"' in json_gui_state()


def test_form_generation_keeps_path_and_enum_widgets() -> None:
    path_gui = PydanticGui(_PathModel, include_json_editor=False)
    enum_gui = PydanticGui(_EnumModel, include_json_editor=False)

    assert type(path_gui.elements["source"]).__name__ == "file_browser"
    assert type(enum_gui.elements["mode"]).__name__ == "dropdown"
    assert type(enum_gui.elements["enum_mode"]).__name__ == "dropdown"


def test_dict_form_uses_json_code_editor() -> None:
    generated = PydanticGui(_DictModel, include_json_editor=False)

    assert type(generated.elements["payload"]).__name__ == "code_editor"
    assert generated.elements["payload"]._value_frontend == (
        '{\n  "alpha": 1,\n  "enabled": true\n}'
    )


def test_dict_form_parses_json_code_editor_changes() -> None:
    generated = PydanticGui(_DictModel, include_json_editor=False)

    value = generated._convert_value(
        {"payload": '{\n  "beta": 2,\n  "nested": {"ok": true}\n}'}
    )

    assert value == _DictModel(payload={"beta": 2, "nested": {"ok": True}})


def test_dict_form_rejects_non_object_json() -> None:
    generated = PydanticGui(_DictModel, include_json_editor=False)

    with pytest.raises(
        ValueError, match="top-level JSON value must be an object"
    ):
        generated._convert_value({"payload": "[1, 2, 3]"})


def test_short_primitive_tuple_form_uses_compact_widget() -> None:
    generated = PydanticGui(_PrimitiveSequenceModel, include_json_editor=False)

    assert type(generated.elements["opacity_range"]).__name__ == (
        "PrimitiveTupleGui"
    )
    assert type(generated.elements["crop_origin"]).__name__ == (
        "PrimitiveTupleGui"
    )
    assert type(generated.elements["weights"]).__name__ == ("PrimitiveTupleGui")
    assert generated.elements["opacity_range"]._value_frontend == {
        "0": 0.1,
        "1": 0.9,
    }
    assert generated.elements["weights"]._value_frontend == {
        "0": 0.1,
        "1": 0.2,
        "2": 0.3,
        "3": 0.4,
        "4": 0.5,
    }


def test_short_primitive_tuple_form_parses_changes() -> None:
    generated = PydanticGui(_PrimitiveSequenceModel, include_json_editor=False)

    value = generated._convert_value(
        {
            "opacity_range": {"0": 0.2, "1": 0.8},
            "crop_origin": {"0": 4, "1": 5, "2": 6},
            "weights": {"0": 0.5, "1": 0.4, "2": 0.3, "3": 0.2, "4": 0.1},
        }
    )

    assert value == _PrimitiveSequenceModel(
        opacity_range=(0.2, 0.8),
        crop_origin=(4, 5, 6),
        weights=(0.5, 0.4, 0.3, 0.2, 0.1),
    )


def test_config_gui_panel_syncs_primitive_sequences_to_json(
    notebook_runtime: None,
) -> None:
    form_gui_state, json_gui_state, bindings = _make_state(
        _PrimitiveSequenceModel
    )
    form = config_gui_panel(
        bindings,
        form_gui_state=form_gui_state,
    )

    _dispatch_form_change(
        form,
        {
            "opacity_range": {"0": 0.2, "1": 0.8},
            "labels": '[\n  "train",\n  "test"\n]',
        },
    )

    assert form_gui_state()["opacity_range"] == (0.2, 0.8)
    assert form_gui_state()["labels"] == ["train", "test"]
    assert '"opacity_range": [\n    0.2,\n    0.8\n  ]' in json_gui_state()
    assert '"labels": [\n    "train",\n    "test"\n  ]' in json_gui_state()


def test_primitive_list_form_uses_json_array_editor() -> None:
    generated = PydanticGui(_PrimitiveSequenceModel, include_json_editor=False)

    assert type(generated.elements["labels"]).__name__ == "code_editor"
    assert generated.elements["labels"]._value_frontend == (
        '[\n  "train",\n  "eval"\n]'
    )


def test_primitive_list_form_parses_variable_length_array() -> None:
    generated = PydanticGui(_PrimitiveSequenceModel, include_json_editor=False)

    value = generated._convert_value(
        {"labels": '[\n  "train",\n  "val",\n  "test"\n]'}
    )

    assert value == _PrimitiveSequenceModel(labels=["train", "val", "test"])


def test_primitive_list_form_rejects_non_array_json() -> None:
    generated = PydanticGui(_PrimitiveSequenceModel, include_json_editor=False)

    with pytest.raises(
        ValueError, match="top-level JSON value must be an array"
    ):
        generated._convert_value({"labels": '{"not": "an array"}'})


def test_variable_tuple_does_not_use_compact_widget() -> None:
    generated = PydanticGui(_VariableTupleModel, include_json_editor=False)

    assert type(generated.elements["values"]).__name__ != "PrimitiveTupleGui"


def test_long_fixed_tuple_uses_json_array_editor() -> None:
    generated = PydanticGui(_LongTupleModel, include_json_editor=False)

    assert type(generated.elements["values"]).__name__ == "code_editor"
    assert generated.elements["values"]._value_frontend == (
        "[\n  1,\n  2,\n  3,\n  4,\n  5,\n  6\n]"
    )


def test_enum_form_accepts_json_value_initial_payload() -> None:
    generated = PydanticGui(
        _EnumModel,
        value={"enum_mode": "quality"},
        include_json_editor=False,
    )

    assert generated.value == _EnumModel(enum_mode=_EnumMode.QUALITY)
    assert generated.elements["enum_mode"]._value_frontend == ["quality"]


def test_enum_form_uses_readable_display_labels() -> None:
    generated = PydanticGui(_EnumModel, include_json_editor=False)

    assert generated.elements["color_space"]._args.args["options"] == [
        "srgb",
        "linear rgb",
    ]
    assert generated.elements["color_space"]._value_frontend == ["srgb"]


def test_flag_form_uses_multiselect_and_serializes_names() -> None:
    generated = PydanticGui(_FlagModel, include_json_editor=False)

    assert type(generated.elements["features"]).__name__ == "multiselect"
    assert generated.elements["features"]._value_frontend == ["read", "write"]
    assert (
        widgets._payload_to_json({"features": generated.value.features})
        == '{\n  "features": [\n    "READ",\n    "WRITE"\n  ]\n}'
    )


def test_flag_form_parses_multiselect_changes() -> None:
    generated = PydanticGui(_FlagModel, include_json_editor=False)

    value = generated._convert_value(
        {
            "features": ["execute"],
            "permissions": ["read", "write"],
        }
    )

    assert value == _FlagModel(
        features=_FeatureFlag.EXECUTE,
        permissions=_PermissionFlag.READ | _PermissionFlag.WRITE,
    )


def test_optional_flag_form_parses_empty_selection() -> None:
    generated = PydanticGui(
        _OptionalFlagModel,
        value={"features": _FeatureFlag.READ},
        include_json_editor=False,
    )

    value = generated._convert_value(
        {
            "features": {
                "__enabled__": 1,
                "__value__": [],
            }
        }
    )

    assert value == _OptionalFlagModel(features=_FeatureFlag.NONE)


def test_form_generation_parses_enabled_nullable_field() -> None:
    generated = PydanticGui(_OptionalModel, include_json_editor=False)

    assert generated.value == _OptionalModel(source=Path("README.md"))


def test_form_generation_can_exclude_top_level_fields() -> None:
    generated = PydanticGui(
        _RequiredModel,
        value=_RequiredModel(title="kept", count=3),
        include_json_editor=False,
        exclude_fields=frozenset({"count"}),
    )

    assert "count" not in generated.elements
    assert generated.value == _RequiredModel(title="kept", count=3)


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

    unindented = widgets._indent_nested_layout(layout, current_level=0)
    indented = widgets._indent_nested_layout(layout, current_level=2)

    assert unindented is layout
    assert "border-left:" in indented.text
    assert "margin-left:" in indented.text


def test_union_json_serialization_keeps_kind(notebook_runtime: None) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_UnionRoot)
    json_view = config_json_editor(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )

    _dispatch_json_change(
        json_view,
        '{\n  "item": {\n    "__kind__": "_UnionB",\n    "title": "switched"\n  }\n}',
    )

    assert '"__kind__": "_UnionB"' in widgets._payload_to_json(form_gui_state())


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("TrainingConfig", "Training"),
        ("TrainingRunConfig", "Training Run"),
        ("HTTPServerConfig", "HTTP Server"),
        ("NDGSRendererConfig", "NDGS Renderer"),
        ("ResNet50Config", "ResNet50"),
        ("MobileNetV3Config", "MobileNetV3"),
        ("Stage2TrainingConfig", "Stage2 Training"),
        ("Gaussian2DTrainingConfig", "Gaussian2D Training"),
    ],
)
def test_model_name_humanization_is_alphanumeric_aware(
    identifier: str,
    expected: str,
) -> None:
    model_cls = type(identifier, (BaseModel,), {})

    assert labels.humanize_model_name(model_cls) == expected


def test_model_name_humanization_uses_pydantic_title() -> None:
    assert labels.humanize_model_name(_TitledResNetConfig) == "Custom ResNet-50"


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("titled_field", "Field Runtime"),
        ("titled_model", "Runtime Settings"),
        ("plain_model", "Plain Runtime"),
        ("plain_value", "Plain value"),
    ],
)
def test_field_label_precedence(
    field_name: str,
    expected: str,
) -> None:
    info = _LabelPrecedenceRoot.model_fields[field_name]

    assert labels.field_label(field_name, info, info.annotation) == expected


def test_config_gui_builds_model_union_field_default(
    notebook_runtime: None,
) -> None:
    config_gui = create_config_gui(_UnionFieldDefaultRoot, background=None)

    assert type(config_gui.gui_panel()).__name__ == "PydanticGui"


def test_union_branch_labels_use_model_titles() -> None:
    generated = PydanticGui(
        _UnionBranchLabelRoot,
        include_json_editor=False,
    )
    union_gui = generated.elements["subconfig"]

    assert union_gui._branch_labels == [
        "ResNet-50",
        "Context Network",
        "Custom Network",
    ]


def test_json_editor_uses_model_field_order(notebook_runtime: None) -> None:
    (
        form_gui_state,
        json_gui_state,
        bindings,
    ) = _make_state(_OrderModel)
    json_view = config_json_editor(
        bindings,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    assert json_view.value.index('"zeta"') < json_view.value.index('"alpha"')
    assert json_view.value.index('"alpha"') < json_view.value.index('"middle"')
