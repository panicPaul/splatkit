"""Notebook-facing helpers for Pydantic config GUIs."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from enum import Flag
from pathlib import Path
from typing import Annotated, Any, cast

import marimo as mo
import tyro
from marimo._plugins.ui._core.ui_element import UIElement
from pydantic import BaseModel, ConfigDict, Field, create_model
from tyro.constructors import ConstructorRegistry, PrimitiveConstructorSpec

from marimo_config_gui.elements import ConfigBackground, ConfigGui, PydanticGui
from marimo_config_gui.presets import (
    ConfigFileEntry,
    ConfigPresetCatalog,
    load_json_config,
    load_preset_config,
    merge_config_override,
    override_model_for_config,
    payload_for_config,
    resolve_config_paths,
)
from marimo_config_gui.state import (
    ConfigBindings,
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
    overlays: Sequence[ConfigFileEntry] = (),
    path_defaults: Sequence[ConfigFileEntry] = (),
    path_defaults_source: str | Path | None = None,
) -> dict[str, Any]:
    resolved_path_defaults = _path_defaults_with_source(
        path_defaults,
        path_defaults_source=path_defaults_source,
    )
    if not mo.running_in_notebook():
        if script_loader is None:
            if presets is None:
                if overlays or resolved_path_defaults:
                    parsed = load_script_config(
                        model_cls,
                        value=value,
                        args=script_args,
                        overlays=overlays,
                        path_defaults=resolved_path_defaults,
                    )
                else:
                    parsed = load_script_config(
                        model_cls,
                        value=value,
                        args=script_args,
                    )
            else:
                if overlays or resolved_path_defaults:
                    parsed = load_script_config(
                        model_cls,
                        value=value,
                        args=script_args,
                        presets=presets,
                        overlays=overlays,
                        path_defaults=resolved_path_defaults,
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
        resolved_value: ModelT | dict[str, Any] | None = cast(ModelT, parsed)
    elif value is None and presets is not None:
        resolved_value = load_preset_config(presets)
    else:
        resolved_value = value
        if isinstance(resolved_value, BaseModel) and resolved_path_defaults:
            resolved_value = resolve_config_paths(
                resolved_value,
                path_defaults=resolved_path_defaults,
            )
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


def _path_defaults_with_source(
    path_defaults: Sequence[ConfigFileEntry],
    *,
    path_defaults_source: str | Path | None = None,
) -> tuple[ConfigFileEntry, ...]:
    if path_defaults_source is None:
        return tuple(path_defaults)
    source = Path(path_defaults_source).expanduser()
    source_dir = source if source.is_dir() else source.parent
    return (
        source_dir / ".path_defaults.json",
        *tuple(path_defaults),
    )


def _script_input_model_for(
    model_cls: type[BaseModel],
    *,
    presets: ConfigPresetCatalog[Any] | None = None,
) -> type[BaseModel]:
    base_override_model = override_model_for_config(model_cls)
    fields: dict[str, Any] = {
        "config": (Annotated[Path | None, tyro.conf.Positional], None),
        "overlay": (tuple[Path, ...], ()),
    }
    if presets is not None:
        fields["preset"] = (str | None, None)
    for field_name, field in base_override_model.model_fields.items():
        if field_name in fields:
            continue
        default = (
            Field(default_factory=field.default_factory)
            if field.default_factory is not None
            else field.default
        )
        fields[field_name] = (field.annotation, default)
    return create_model(
        f"{model_cls.__name__}ScriptInput",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _load_script_input_config(
    model_cls: type[ModelT],
    script_input: BaseModel,
    *,
    value: ModelT | dict[str, Any] | None = None,
    presets: ConfigPresetCatalog[ModelT] | None = None,
    overlays: Sequence[ConfigFileEntry] = (),
    path_defaults: Sequence[ConfigFileEntry] = (),
) -> ModelT:
    payload = script_input.model_dump(exclude_none=True)
    config_path = payload.pop("config", None)
    preset_name = payload.pop("preset", None)
    overlay_entries = payload.pop("overlay", ())
    resolved_overlays = tuple(overlays) + tuple(
        str(path.expanduser().resolve()) for path in overlay_entries
    )
    if config_path is not None and preset_name is not None:
        raise ValueError("Specify either a config path or --preset, not both.")
    if config_path is not None:
        base = load_json_config(
            model_cls,
            config_path,
            overlays=resolved_overlays,
            path_defaults=path_defaults,
        )
    elif presets is not None:
        selected_preset = preset_name or presets.default
        base = load_preset_config(presets, selected_preset)
        if resolved_overlays:
            base = load_json_config(
                model_cls,
                presets.presets[selected_preset].path,
                base_dir=presets.presets[selected_preset].base_dir,
                overlays=tuple(presets.overlays) + resolved_overlays,
                path_defaults=tuple(presets.path_defaults)
                + tuple(path_defaults),
            )
    else:
        base = _resolve_cli_default(model_cls, value)
        if not isinstance(base, model_cls):
            base = model_cls.model_validate(
                base if isinstance(base, dict) else {}
            )
        if resolved_overlays:
            payload_base = base.model_dump(mode="json")
            for layer, _layer_path in _load_script_overlay_layers(
                resolved_overlays
            ):
                payload_base = {**payload_base, **layer}
            base = model_cls.model_validate(payload_base)
        if path_defaults:
            base = resolve_config_paths(base, path_defaults=path_defaults)
    return merge_config_override(base, payload)


def _load_script_overlay_layers(
    overlays: Sequence[ConfigFileEntry],
) -> tuple[tuple[dict[str, Any], Path], ...]:
    from marimo_config_gui.presets import _load_json_layers

    return tuple(
        _load_json_layers(
            overlays,
            required_default=True,
        )
    )


def load_script_config(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    args: Sequence[str] | None = None,
    presets: ConfigPresetCatalog[ModelT] | None = None,
    overlays: Sequence[ConfigFileEntry] = (),
    path_defaults: Sequence[ConfigFileEntry] = (),
    path_defaults_source: str | Path | None = None,
) -> ModelT:
    """Load a config in script mode via tyro CLI or JSON file.

    Args:
        model_cls: Pydantic model type to load.
        value: Optional default value when no config path or preset is given.
        args: Optional CLI argument sequence. When omitted, uses `sys.argv`.
        presets: Optional named JSON preset catalog. When supplied, `--preset`
            selects a named preset.
        overlays: Sparse JSON config overlays applied before CLI field
            overrides.
        path_defaults: Local path-default files used to resolve typed `Path`
            values.
        path_defaults_source: Optional source file or directory whose sibling
            `.path_defaults.json` should be loaded.

    Returns:
        The validated model instance loaded from defaults, a JSON config path,
        or a named preset, with sparse CLI overrides applied last.
    """
    default_value = (
        load_preset_config(presets) if value is None and presets else value
    )
    resolved_path_defaults = _path_defaults_with_source(
        path_defaults,
        path_defaults_source=path_defaults_source,
    )
    script_input_model = _script_input_model_for(model_cls, presets=presets)
    parsed = tyro.cli(
        script_input_model,
        args=args,
        registry=_flag_cli_registry(),
    )
    return _load_script_input_config(
        model_cls,
        parsed,
        value=default_value,
        presets=presets,
        overlays=tuple(() if presets is None else presets.overlays)
        + tuple(overlays),
        path_defaults=tuple(() if presets is None else presets.path_defaults)
        + tuple(resolved_path_defaults),
    )


def create_config_state(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
    presets: ConfigPresetCatalog[ModelT] | None = None,
    overlays: Sequence[ConfigFileEntry] = (),
    path_defaults: Sequence[ConfigFileEntry] = (),
    path_defaults_source: str | Path | None = None,
) -> tuple[Any, Any, ConfigBindings[ModelT]]:
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
        overlays: Sparse JSON config overlays applied during initial loading.
        path_defaults: Local path-default files used to resolve typed `Path`
            values.
        path_defaults_source: Optional source file or directory whose sibling
            `.path_defaults.json` should be loaded.

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
        overlays=overlays,
        path_defaults=path_defaults,
        path_defaults_source=path_defaults_source,
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


def create_config_gui(
    model_cls: type[ModelT],
    *,
    value: ModelT | dict[str, Any] | None = None,
    script_loader: ScriptConfigLoader | None = None,
    script_args: Sequence[str] | None = None,
    presets: ConfigPresetCatalog[ModelT] | None = None,
    overlays: Sequence[ConfigFileEntry] = (),
    path_defaults: Sequence[ConfigFileEntry] = (),
    path_defaults_source: str | Path | None = None,
    background: ConfigBackground = "neutral",
    label: str = "",
    nested_models_multiple_open: bool = True,
    nested_models_flat_after_level: int | None = None,
    exclude_fields: set[str] | frozenset[str] = frozenset(),
) -> ConfigGui[ModelT]:
    """Create an owning synchronized config GUI for a Pydantic model.

    The returned object owns the draft payload, JSON text, validation status,
    and typed config value. Its `gui_panel()` and `json_editor()` views can be
    displayed independently or together, including inside marimo layout
    wrappers, while staying synchronized through the owner.
    """
    resolved_path_defaults = _path_defaults_with_source(
        path_defaults,
        path_defaults_source=path_defaults_source,
    )
    initial_payload = _initial_config_payload(
        model_cls,
        value=value,
        script_loader=script_loader,
        script_args=script_args,
        presets=presets,
        overlays=overlays,
        path_defaults=resolved_path_defaults,
    )
    return ConfigGui(
        model_cls,
        value=initial_payload,
        background=background,
        presets=presets,
        label=label,
        nested_models_multiple_open=nested_models_multiple_open,
        nested_models_flat_after_level=nested_models_flat_after_level,
        exclude_fields=exclude_fields,
        path_defaults=resolved_path_defaults,
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
) -> ModelT:
    """Validate and return the current config value.

    Args:
        state_or_model_cls: Either config bindings returned by
            `create_config_state(...)` or the raw model class.
        form_gui_state: Reactive structured form GUI state.
        json_gui_state: Reactive JSON draft state.

    Returns:
        The validated model instance. Invalid drafts stop the current notebook
        cell or raise `ValueError` outside notebook mode.
    """
    model_cls = _resolve_bound_model_cls(state_or_model_cls)
    current_error = _current_config_error(
        model_cls,
        form_gui_state=form_gui_state,
        json_gui_state=json_gui_state,
    )
    if current_error is not None:
        if mo.running_in_notebook():
            mo.stop(True, _build_error_view(current_error))
        raise ValueError(current_error)
    value, _ = _validate_payload_with_error(model_cls, form_gui_state())
    if value is None:
        raise ValueError("Config validation succeeded without a model value.")
    return cast(ModelT, value)
