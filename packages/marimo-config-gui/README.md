# marimo-config-gui

Mode-aware marimo configuration UIs for Pydantic models.

`marimo-config-gui` is for notebooks that need a real experiment config, not
just a few ad hoc widgets. Define the config once as a Pydantic model, edit it
in marimo as a form or JSON, validate it as a typed model, and run the same
notebook as a script with `tyro` CLI or JSON input.

## Install

This package is not currently published on PyPI. Install it from the Git archive
and package subdirectory:

```bash
uv add "marimo-config-gui @ git+https://github.com/panicPaul/ember.git@main#subdirectory=packages/marimo-config-gui"
```

## The Main Workflow

The common workflow is:

1. Create a Pydantic `BaseModel` for the notebook config.
2. Create shared GUI state with `create_config_state(...)`.
3. Render the config GUIs with `config_form(...)`, `config_json(...)`, and
   `config_error(...)`.
4. Extract the current typed value with `config_value(...)` or
   `config_require_valid(...)`.

```python
from pathlib import Path
from typing import Literal

import marimo as mo
from marimo_config_gui import (
    config_error,
    config_form,
    config_json,
    config_require_valid,
    config_value,
    create_config_state,
)
from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    device: Literal["cpu", "cuda"] = "cuda"
    seed: int = Field(0, ge=0, description="Random seed for the run.")


class Config(BaseModel):
    scene_path: Path = Field(
        Path("data/scene"),
        description="Path to a COLMAP-style scene directory.",
    )
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    max_steps: int = Field(30_000, ge=1)
```

In a marimo notebook, keep the state in one place and pass the returned
bindings into the GUI helpers:

```python
with app.setup:
    (
        config_form_state,
        config_json_state,
        config_bindings,
    ) = create_config_state(Config)
```

Render the form, JSON editor, and validation output in separate cells when that
makes the notebook layout easier to scan:

```python
config_form(
    config_bindings,
    form_gui_state=config_form_state,
)
```

```python
config_json(
    config_bindings,
    form_gui_state=config_form_state,
    json_gui_state=config_json_state,
)
```

```python
config_error(
    config_bindings,
    form_gui_state=config_form_state,
    json_gui_state=config_json_state,
)
```

Then extract the current typed config wherever the notebook needs it:

```python
config = config_value(
    config_bindings,
    form_gui_state=config_form_state,
    json_gui_state=config_json_state,
)

if config is None:
    mo.stop(True, mo.callout("Fix the config before running.", kind="warn"))
```

For cells that must not continue without a valid config, use
`config_require_valid(...)`:

```python
config = config_require_valid(
    config_bindings,
    form_gui_state=config_form_state,
    json_gui_state=config_json_state,
)
```

## Quick Single-Cell UI

For small notebooks, `config_gui(...)` builds state and returns renderable UI
elements in one call:

```python
from marimo_config_gui import config_gui

config_error_view, config_form_view = config_gui(Config)
```

Use the split state workflow for notebooks that need form and JSON editors in
different cells, need to expose the validated value to later cells, or need to
share one config between notebook and script mode.

## Script Mode

When the notebook is running as a script, `create_config_state(...)` loads the
initial config through `tyro`. The default loader supports two subcommands:

```bash
python notebook.py cli --scene-path data/bicycle --runtime.device cuda
python notebook.py json config.json
```

You can also call the loader directly:

```python
from marimo_config_gui import load_script_config

config = load_script_config(Config)
```

If a notebook should be usable with the `tyro`/JSON script loader, use one
loader-backed top-level config GUI for that notebook. Multiple independent
loader-backed config states all try to interpret the same script arguments,
which makes the command-line interface ambiguous. If you need several panels,
make them fields of one top-level `Config` model.

For custom script behavior, pass `script_loader=...` to
`create_config_state(...)`. The loader receives the model class, the optional
default value, and the optional argument sequence.

## Apply Before Expensive Work

Viewer reloads, scene loading, dataset preparation, and training runs are often
too expensive to trigger on every draft edit. Use committed config state when
the user should edit a draft first and explicitly apply it:

```python
from marimo_config_gui import (
    config_commit_button,
    config_committed_value,
    create_committed_config_state,
)

committed_state, set_committed_state = create_committed_config_state(Config)

apply_button = config_commit_button(
    config_bindings,
    form_gui_state=config_form_state,
    json_gui_state=config_json_state,
    committed_state=committed_state,
    set_committed_state=set_committed_state,
    label="Load scene",
)

active_config = config_committed_value(
    config_bindings,
    committed_state=committed_state,
)
```

Use the draft config from `config_value(...)` for cheap dependent controls. Use
the committed config from `config_committed_value(...)` for expensive work.

## Model Authoring Notes

- `Field(description=...)` becomes help text in the generated UI.
- `Literal[...]` and `Enum` fields render as dropdowns.
- `Path` fields render with a file browser.
- Nested Pydantic models render as nested sections.
- Optional fields render with a `None`/configure control.
- 1D and 2D NumPy or Torch arrays render as matrix widgets.
- Validation constraints from Pydantic are enforced before
  `config_value(...)` returns a model.

Field-level GUI hints can be provided through `json_schema_extra`:

```python
class OptimizerConfig(BaseModel):
    lr: float = Field(
        0.01,
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "marimo_config_gui": {"widget": "slider"},
        },
    )
```

For complex fields that are clearer as JSON than widgets, force JSON rendering:

```python
class Config(BaseModel):
    schedule: dict[str, float] = Field(
        default_factory=dict,
        json_schema_extra={
            "marimo_config_gui": {"render": "json"},
        },
    )
```

## marimo Patterns

- Return `config_form(...)`, `config_json(...)`, `config_error(...)`, and other
  reactive outputs directly from cells. Avoid wrapping them in extra layout
  containers when the wrapped object itself needs to stay reactive.
- Keep config state creation in `app.setup` when the config should behave the
  same in notebook mode and script mode.
- Treat `config_value(...)` as `Config | None`; invalid JSON or failed Pydantic
  validation returns `None`.

## License

This package is distributed under the Apache License 2.0.
