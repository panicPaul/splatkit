# marimo-config-gui

Mode-aware marimo configuration UIs for Pydantic models.

`marimo-config-gui` is for notebooks that need a real experiment config, not
just a few ad hoc widgets. Define the config once as a Pydantic model, edit it
in marimo as a form or JSON, validate it as a typed model, and run the same
notebook as a script with `tyro` CLI or JSON input.

For an interactive version, run from the repository root:

```bash
marimo run docs/marimo-config-gui.py
```

## Install

This package is not currently published on PyPI. Install it from the Git archive
and package subdirectory:

```bash
uv add "marimo-config-gui @ git+https://github.com/panicPaul/ember.git@main#subdirectory=packages/marimo-config-gui"
```

The same docs are available as an interactive marimo app:

```bash
marimo run docs/marimo-config-gui.py
```

## The Main Workflow

The common workflow is:

1. Create a Pydantic `BaseModel` for the notebook config.
2. Create an owning GUI with `create_config_gui(...)`.
3. Render any combination of `gui_panel()`, `json_editor()`, and
   `status_panel()`.
4. Extract the current typed value with `config_gui.validated_config()`.

```python
from pathlib import Path
from typing import Literal

import marimo as mo
from marimo_config_gui import (
    create_config_gui,
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

In a marimo notebook, keep the config GUI owner in one place:

```python
with app.setup:
    config_gui = create_config_gui(Config)
```

Render the form, JSON editor, and validation output in separate cells or wrapped
layouts when that makes the notebook easier to scan:

```python
config_gui.stacked()
```

`stacked()` renders the form and JSON editor side by side with the validation
status below. By default each visible panel uses the `neutral` background. Pass
`background=None` to `create_config_gui(...)`, `stacked(...)`, or an individual
view method for unstyled output. Named backgrounds mirror marimo callout kinds:
`neutral`, `warn`, `success`, `info`, and `danger`.

```python
config_gui.gui_panel()
```

```python
config_gui.json_editor()
```

```python
config_gui.status_panel()
```

The form and JSON editor synchronize both directions. For example, checking a
boolean flag in the form updates the JSON to `true`, and editing the JSON to
`true` updates the form checkbox.

Then extract the current typed config wherever the notebook needs it:

```python
config = config_gui.validated_config()
```

If the draft is invalid, `validated_config()` stops that consumer cell while
the GUI and status cells remain live so the config can be fixed.

## Script Mode

When the notebook is running as a script, `create_config_gui(...)` loads the
initial config through `tyro`. With no positional config path, CLI flags are
applied to the model defaults:

```bash
python notebook.py --scene-path data/bicycle --runtime.device cuda
```

A positional JSON config path can be used as the base. Sparse typed overrides
come after the JSON file, which is usually the most convenient workflow on a
server:

```bash
python notebook.py config.json --runtime.device cuda --max-steps 1000
```

When a notebook defines presets with `ConfigPresetCatalog`, select them with
`--preset`:

```bash
python notebook.py --preset garden --runtime.device cuda
```

Config overlay files can be declared by the notebook or passed on the command
line. They are sparse JSON objects merged left-to-right before explicit CLI
field overrides:

```bash
python notebook.py base.json --overlay server.json --max-steps 1000
```

In Python, overlay entries may be strings, `Path` objects, `(path, required)`
tuples, or `ConfigFile` objects. Bare overlay paths are required by default.

If a notebook should be usable with the `tyro`/JSON script loader, use one
loader-backed top-level config GUI for that notebook. Multiple independent
loader-backed config states all try to interpret the same script arguments,
which makes the command-line interface ambiguous. If you need several panels,
make them fields of one top-level `Config` model.

For custom script behavior, pass `script_loader=...` to
`create_config_gui(...)`. The loader receives the model class, the optional
default value, and the optional argument sequence.

## Local Path Defaults

Machine-specific dataset and artifact paths should not need to dirty tracked
presets. Put local path replacements in a `.path_defaults.json` file beside the
JSON config file:

```json
{
  "path_prefixes": {
    "dataset/mipnerf360": "/data/mipnerf360"
  },
  "fields": {
    "ply_path": "/home/user/scenes/example_scene.ply"
  }
}
```

Path defaults apply only to fields typed as `Path`. The `path_prefixes` section
uses longest-prefix matching for logical path values. The `fields` section
targets typed config fields by dotted field path:

```text
dataset/mipnerf360/garden -> /data/mipnerf360/garden
ply_path -> /home/user/scenes/example_scene.ply
```

Non-path strings such as `runtime.device = "mlx"` are never rewritten; use a
config overlay or CLI override for those.

For configs without JSON files or preset catalogs, pass a source path explicitly
so the helper can look for the source's sibling `.path_defaults.json`:

```python
create_config_gui(
    Config,
    value=Config(),
    path_defaults_source=NOTEBOOK_PATH,
)
```

Path-default entries are optional by default. Use `(path, True)` or
`ConfigFile(path=path, required=True)` when the file must exist.

## Apply Before Expensive Work

Viewer reloads, scene loading, dataset preparation, and training runs are often
too expensive to trigger on every draft edit. Use marimo gating when the user
should edit a draft first and explicitly apply it:

```python
load_button = mo.ui.run_button(
    label="Load scene",
    disabled=not config_gui.is_valid(),
)
```

Then gate the expensive cell:

```python
mo.stop(not load_button.value)
active_config = config_gui.validated_config()
```

Use `config_gui.validated_config()` directly for cheap dependent controls. Use
`mo.ui.run_button(...)` and `mo.stop(...)` for expensive work.

## Model Authoring Notes

- `Field(description=...)` becomes help text in the generated UI.
- `Literal[...]` and `Enum` fields render as dropdowns.
- `Path` fields render with a file browser.
- Nested Pydantic models render as nested sections.
- Optional fields render with a `None`/configure control.
- 1D and 2D NumPy or Torch arrays render as matrix widgets.
- Validation constraints from Pydantic are enforced before
  `config_gui.validated_config()` returns a model.

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

- Prefer `create_config_gui(...)` and render `config_gui.gui_panel()`,
  `config_gui.json_editor()`, and `config_gui.status_panel()`.
- The owner API is designed for wrapped layouts such as
  `mo.hstack([config_gui.gui_panel(), config_gui.json_editor()])` while keeping
  JSON and form controls synchronized.
- Use `config_gui.stacked()` for the default wrapped layout.
- Keep config state creation in `app.setup` when the config should behave the
  same in notebook mode and script mode.
- Treat `config_gui.validated_config()` as `Config`; invalid JSON or failed
  Pydantic validation stops the consumer cell in notebooks or raises
  `ValueError` in scripts.

## License

This package is distributed under the Apache License 2.0.
