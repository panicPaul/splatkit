"""Notebook-facing helpers for Pydantic config GUIs."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, overload

import marimo as mo
import tyro
from marimo._plugins.ui._core.ui_element import UIElement
from pydantic import BaseModel

from marimo_config_gui.elements import PydanticGui
from marimo_config_gui.state import (
    ConfigBindings,
    JsonConfigSource,
    ModelT,
    ScriptConfigLoader,
)
from marimo_config_gui.widgets import (
    _invalid_json_output,
    _json_output,
    _order_payload_for_model,
    _payload_to_json,
    _resolve_cli_default,
    _resolve_initial_payload,
    _validate_payload_with_error,
    _validation_output,
)


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
        | Annotated[JsonConfigSource, tyro.conf.subcommand("json")]
    )
    parsed = tyro.cli(script_input_type, args=args)
    if isinstance(parsed, JsonConfigSource):
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
        if set_form_gui_state is None or set_json_gui_state is None:
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
        if set_form_gui_state is None or set_json_gui_state is None:
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
    del (
        submit_label,
        nested_models_multiple_open,
        nested_models_flat_after_level,
    )
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
