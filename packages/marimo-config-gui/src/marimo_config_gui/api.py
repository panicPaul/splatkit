"""Notebook-facing helpers for Pydantic config GUIs."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from enum import Flag
from typing import Annotated, Any, overload

import marimo as mo
import tyro
from marimo._plugins.ui._core.ui_element import UIElement
from pydantic import BaseModel
from tyro.constructors import ConstructorRegistry, PrimitiveConstructorSpec

from marimo_config_gui.elements import PydanticGui
from marimo_config_gui.presets import (
    ConfigPresetCatalog,
    load_json_config,
    load_preset_config,
    merge_preset_override,
    override_model_for_catalog,
    payload_for_config,
)
from marimo_config_gui.state import (
    ConfigBindings,
    JsonConfigSource,
    ModelT,
    ScriptConfigLoader,
)
from marimo_config_gui.widgets import (
    _order_payload_for_model,
    _payload_to_json,
    _resolve_cli_default,
    _resolve_initial_payload,
    _validate_payload_with_error,
)


def _flag_cli_registry() -> ConstructorRegistry:
    registry = ConstructorRegistry()

    @registry.primitive_rule
    def _flag_rule(type_info: Any) -> PrimitiveConstructorSpec[Any] | None:
        annotation = type_info.type
        if not (isinstance(annotation, type) and issubclass(annotation, Flag)):
            return None

        choices = tuple(
            name
            for name, member in annotation.__members__.items()
            if member.name == name
        )

        def _parse_flag(args: list[str]) -> Any:
            combined = annotation(0)
            for arg in args:
                member = annotation.__members__[arg]
                if member.value == 0:
                    combined = annotation(0)
                else:
                    combined |= member
            return combined

        def _flag_to_args(value: Any) -> list[str]:
            if value is None:
                return []
            return [member.name for member in value]

        return PrimitiveConstructorSpec(
            nargs="*",
            metavar="{" + ",".join(choices) + "}",
            instance_from_str=_parse_flag,
            is_instance=lambda value: isinstance(value, annotation),
            str_from_instance=_flag_to_args,
            choices=choices,
        )

    return registry


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
    parsed_payload, parse_error = _parse_json_editor_payload(
        model_cls, json_text
    )
    if parse_error is not None or parsed_payload is None:
        return parse_error
    _, validation_error = _validate_payload_with_error(
        model_cls, parsed_payload
    )
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
    presets: ConfigPresetCatalog[ModelT] | None = None,
) -> dict[str, Any]:
    if not mo.running_in_notebook():
        if script_loader is None:
            if presets is None:
                parsed = load_script_config(
                    model_cls,
                    value=value,
                    args=script_args,
                )
            else:
                parsed = load_script_config(
                    model_cls,
                    value=value,
                    args=script_args,
                    presets=presets,
                )
        else:
            parsed = script_loader(model_cls, value, script_args)
        resolved_value: ModelT | dict[str, Any] | None = parsed
    elif value is None and presets is not None:
        resolved_value = load_preset_config(presets)
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
    presets: ConfigPresetCatalog[ModelT] | None = None,
) -> ModelT:
    """Load a config in script mode via tyro CLI or JSON file.

    Args:
        model_cls: Pydantic model type to load.
        value: Optional default value for the `cli` subcommand.
        args: Optional CLI argument sequence. When omitted, uses `sys.argv`.
        presets: Optional named JSON preset catalog. When supplied, script mode
            also supports a `preset` subcommand with sparse dotted overrides.

    Returns:
        The validated model instance loaded from either the `cli` or `json`
        tyro subcommand.
    """
    default_value = (
        load_preset_config(presets) if value is None and presets else value
    )
    default = _resolve_cli_default(model_cls, default_value)
    if presets is None:
        script_input_type = (
            Annotated[model_cls, tyro.conf.subcommand("cli", default=default)]
            | Annotated[JsonConfigSource, tyro.conf.subcommand("json")]
        )
    else:
        override_model = override_model_for_catalog(presets)
        script_input_type = (
            Annotated[model_cls, tyro.conf.subcommand("cli", default=default)]
            | Annotated[JsonConfigSource, tyro.conf.subcommand("json")]
            | Annotated[
                override_model,
                tyro.conf.subcommand("preset", default=override_model()),
            ]
        )
    parsed = tyro.cli(
        script_input_type,
        args=args,
        registry=_flag_cli_registry(),
    )
    if isinstance(parsed, JsonConfigSource):
        return load_json_config(model_cls, parsed.path)
    if presets is not None and isinstance(parsed, BaseModel):
        if not isinstance(parsed, model_cls):
            return merge_preset_override(presets, parsed)
    return parsed


@overload
def create_config_state(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
    presets: ConfigPresetCatalog[ModelT] | None = None,
) -> tuple[Any, Any, ConfigBindings[ModelT]]: ...


def create_config_state(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
    presets: ConfigPresetCatalog[ModelT] | None = None,
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
        presets: Optional named JSON preset catalog used for notebook defaults
            and preset-aware script loading.

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
        presets=presets,
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


def config_gui_panel(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    form_gui_state: Any,
    set_form_gui_state: Callable[[dict[str, Any]], None] | None = None,
    set_json_gui_state: Callable[[str], None] | None = None,
    label: str = "",
    nested_models_multiple_open: bool = True,
    nested_models_flat_after_level: int | None = None,
    exclude_fields: set[str] | frozenset[str] = frozenset(),
) -> PydanticGui[ModelT]:
    """Build the structured GUI panel for a config model.

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
        exclude_fields: Top-level fields to keep in state but omit from this
            form view.

    Returns:
        A `PydanticGui` bound to the provided reactive state.
    """
    if isinstance(state_or_model_cls, ConfigBindings):
        model_cls = state_or_model_cls.model_cls
        set_form_gui_state = state_or_model_cls.set_form_gui_state
        set_json_gui_state = state_or_model_cls.set_json_gui_state
    else:
        model_cls = state_or_model_cls
        if set_form_gui_state is None or set_json_gui_state is None:
            raise TypeError(
                "config_gui_panel requires setters when not given ConfigBindings."
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
        exclude_fields=exclude_fields,
        on_change=_on_form_change,
    )
    form_ref["form"] = form
    return form


def config_preset_selector(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    presets: ConfigPresetCatalog[ModelT],
    form_gui_state: Any,
    json_gui_state: Any,
    set_form_gui_state: Callable[[dict[str, Any]], None] | None = None,
    set_json_gui_state: Callable[[str], None] | None = None,
    label: str = "Preset",
) -> Any:
    """Build a dropdown that replaces the draft config with a named preset."""
    del json_gui_state
    if isinstance(state_or_model_cls, ConfigBindings):
        model_cls = state_or_model_cls.model_cls
        set_form_gui_state = state_or_model_cls.set_form_gui_state
        set_json_gui_state = state_or_model_cls.set_json_gui_state
    else:
        model_cls = state_or_model_cls
        if set_form_gui_state is None or set_json_gui_state is None:
            raise TypeError(
                "config_preset_selector requires setters when not given "
                "ConfigBindings."
            )
    options = {
        preset.label or preset.name: name
        for name, preset in presets.presets.items()
    }
    option_label_by_name = {name: label for label, name in options.items()}
    current_payload = form_gui_state()
    current_value = current_payload.get(presets.preset_field or "preset")
    if current_value not in presets.presets:
        current_value = presets.default
    current_label = option_label_by_name[current_value]

    def _on_change(name: str) -> None:
        config = load_preset_config(presets, name)
        next_payload = _order_payload_for_model(
            model_cls,
            payload_for_config(config),
        )
        set_form_gui_state(next_payload)
        set_json_gui_state(_payload_to_json(next_payload))

    return mo.ui.dropdown(
        options=options,
        value=current_label,
        label=label,
        on_change=_on_change,
    )


def config_json_editor(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    form_gui_state: Any,
    set_form_gui_state: Callable[[dict[str, Any]], None] | None = None,
    json_gui_state: Any,
    set_json_gui_state: Callable[[str], None] | None = None,
    label: str = "",
) -> UIElement[Any, Any]:
    """Build the JSON editor panel for a config model.

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
        if set_form_gui_state is None or set_json_gui_state is None:
            raise TypeError(
                "config_json_editor requires setters when not given ConfigBindings."
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


def config_status_panel(
    state_or_model_cls: ConfigBindings[ModelT] | type[ModelT],
    *,
    form_gui_state: Any,
    json_gui_state: Any,
) -> Any:
    """Render the current config validation status.

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


def validated_config(
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

