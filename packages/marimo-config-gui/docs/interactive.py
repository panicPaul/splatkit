"""Interactive documentation for marimo-config-gui."""

# ruff: noqa: ANN001, ANN202, B018, D101

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="wide")

with app.setup:
    import inspect
    import json
    import tempfile
    import traceback
    from enum import Enum, Flag, auto
    from pathlib import Path
    from typing import Literal

    import marimo as mo
    from marimo_config_gui import (
        ConfigPreset,
        ConfigPresetCatalog,
        create_config_gui,
    )
    from marimo_config_gui.api import load_script_config
    from pydantic import BaseModel, ConfigDict, Field

    def docs_md(source: str) -> object:
        """Render markdown after removing Python source indentation."""
        return mo.md(inspect.cleandoc(source))

    def code_example(*lines: str) -> str:
        """Normalize source snippets before placing them in code editors."""
        return "\n".join(lines).strip() + "\n"

    def config_model_from_source(
        source: str,
    ) -> tuple[type[BaseModel] | None, str | None]:
        """Build a Pydantic Config model from a notebook code editor."""
        model_namespace = {
            "BaseModel": BaseModel,
            "ConfigDict": ConfigDict,
            "Enum": Enum,
            "Field": Field,
            "Flag": Flag,
            "Literal": Literal,
            "Path": Path,
            "auto": auto,
        }
        try:
            exec(source, model_namespace, model_namespace)
            config_model = model_namespace.get("Config")
            if not isinstance(config_model, type) or not issubclass(
                config_model,
                BaseModel,
            ):
                return None, "Define a Pydantic model class named Config."
            return config_model, None
        except Exception:
            return None, traceback.format_exc()


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        # marimo-config-gui

        Mode-aware marimo configuration UIs for Pydantic models.

        `marimo-config-gui` is for notebooks that need a real experiment config,
        not just a few ad hoc widgets. Define the config once as a Pydantic model,
        edit it in marimo as a form or JSON, validate it as a typed model, and
        run the same notebook as a script with `tyro` CLI or JSON input. It also
        supports Hydra-inspired presets, overlays, and typed CLI overrides for
        server-side experiments.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        ---

        ## Install

        This package is not currently published on PyPI. Install it from the Git
        archive and package subdirectory:

        ```bash
        uv add "marimo-config-gui @ git+https://github.com/panicPaul/ember.git@main#subdirectory=packages/marimo-config-gui"
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        ---

        ## The Main Workflow

        The common workflow is:

        1. Create a Pydantic `BaseModel` for the notebook config.
        2. Create an owning GUI with `create_config_gui(...)`.
        3. Render the form with `gui.gui_panel()`.
        4. Render the linked JSON editor and validation status with
           `gui.json_editor()` and `gui.status_panel()`.
        5. Extract the current typed value with `gui.validated_config()`.

        ### 1. Create a Pydantic BaseModel

        The first step is live in this notebook: edit the `Config` model below,
        and the generated GUI output in step 3 will update from that model.
        """
    )
    return


@app.cell
def _():
    main_workflow_source = code_example(
        "from enum import Flag, auto",
        "from typing import Literal",
        "",
        "from pydantic import BaseModel, Field",
        "",
        "",
        "class FeatureFlag(Flag):",
        "    READ = auto()",
        "    WRITE = auto()",
        "    EXECUTE = auto()",
        "",
        "",
        "class Config(BaseModel):",
        "    learning_rate: float = Field(",
        "        0.01,",
        "        ge=0.0,",
        "        le=1.0,",
        "        json_schema_extra={",
        '            "marimo_config_gui": {"widget": "slider"},',
        "        },",
        "    )",
        '    device: Literal["cpu", "cuda"] = "cuda"',
        "    features: FeatureFlag = FeatureFlag.READ | FeatureFlag.WRITE",
    )
    return (main_workflow_source,)


@app.cell
def _(main_workflow_source):
    main_workflow_editor = mo.ui.code_editor(
        value=main_workflow_source,
        language="python",
        min_height=360,
        debounce=500,
        label="Workflow Config model",
    )
    return (main_workflow_editor,)


@app.cell(hide_code=True)
def _(main_workflow_editor):
    main_workflow_editor
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""

        ### 2. Create shared GUI state

        ```python
        gui = create_config_gui(Config, value=Config())
        ```

        ### 3. Render the form

        ```python
        gui.gui_panel()
        ```

        The live form below is produced by that call.

        ### 4. Render the linked JSON editor and validation status

        ```python
        gui.json_editor()
        ```

        ```python
        gui.status_panel()
        ```

        ### 5. Extract the current typed value

        ```python
        config = gui.validated_config()
        ```

        If the draft is invalid, `gui.validated_config()` stops that consumer
        cell while the GUI and status cells remain live.

        Keep loader-backed config state creation in `app.setup` when notebook
        and script mode should share the same initialization path:

        ```python
        with app.setup:
            config_gui = create_config_gui(Config)
        ```

        The live editor below defines the model used by the generated form
        preview. It is separate from the authoring-pattern playground later in
        the docs.
        """
    )
    return


@app.cell
def _(main_workflow_editor):
    main_workflow_model, main_workflow_error = config_model_from_source(
        main_workflow_editor.value
    )
    return main_workflow_error, main_workflow_model


@app.cell(hide_code=True)
def _(main_workflow_error):
    mo.stop(main_workflow_error is None)
    mo.callout(
        f"Could not build `Config` from the workflow editor:\n\n```text\n{main_workflow_error}\n```",
        kind="danger",
    )
    return


@app.cell
def _(main_workflow_error, main_workflow_model):
    mo.stop(main_workflow_error is not None or main_workflow_model is None)
    main_workflow_gui = create_config_gui(
        main_workflow_model,
        value=main_workflow_model(),
    )
    return (main_workflow_gui,)


@app.cell(hide_code=True)
def _():
    docs_md("### 3. Generated GUI From the Workflow Model")
    return


@app.cell(hide_code=True)
def _(main_workflow_gui):
    main_workflow_gui.gui_panel()
    return


@app.cell
def _(main_workflow_gui):
    main_workflow_config = main_workflow_gui.validated_config()
    return (main_workflow_config,)


@app.cell(hide_code=True)
def _(main_workflow_config):
    docs_md("### Generated JSON Payload")
    return


@app.cell(hide_code=True)
def _(main_workflow_config):
    mo.json(main_workflow_config.model_dump(mode="json"))
    return


@app.cell
def _(authoring_examples):
    authoring_pattern = mo.ui.dropdown(
        options=list(authoring_examples),
        value="Minimal config model",
        label="Authoring pattern",
        full_width=True,
    )
    return (authoring_pattern,)


@app.cell(hide_code=True)
def _():
    docs_md(
        """
        ---

        ## Authoring Playground

        Use this section to try focused model-authoring patterns. The form,
        JSON editor, validation output, and typed payload below are all linked
        to the same draft config.

        The authoring-pattern selector contains live examples for:

        - `Field(description=...)` help text
        - `Literal[...]`, `Enum`, and `Flag` selectors
        - `Path` file-browser fields
        - nested Pydantic sections
        - generated labels from `Field(title=...)`, model titles, and class names
        - custom names for union branch tabs
        - optional `None`/configure controls
        - Pydantic validation constraints
        - field-level GUI hints for slider widgets and JSON-rendered fields

        Choose **Custom names and union labels** to see the naming precedence:
        `Field(title=...)` names the whole field, `ConfigDict(title=...)` names
        nested model sections and union branches, and class names are sanitized
        only when no explicit title is set.
        """
    )
    return


@app.cell
def _(authoring_examples, authoring_pattern):
    model_source_editor = mo.ui.code_editor(
        value=authoring_examples[authoring_pattern.value],
        language="python",
        min_height=360,
        debounce=500,
        label="Config model",
    )
    return (model_source_editor,)


@app.cell(hide_code=True)
def _():
    docs_md(
        """
        ### 1. Write a Typed Config Model

        Pick an authoring pattern or edit the code directly.
        """
    )
    return


@app.cell(hide_code=True)
def _(authoring_pattern):
    authoring_pattern
    return


@app.cell(hide_code=True)
def _(model_source_editor):
    model_source_editor
    return


@app.cell(hide_code=True)
def _(scratch_model_error):
    mo.stop(scratch_model_error is None)
    mo.callout(
        f"Could not build `Config` from the editor:\n\n```text\n{scratch_model_error}\n```",
        kind="danger",
    )
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        """
        ### 2. Generated Form View

        `gui.gui_panel()` renders a structured form from the same model.
        Edits here update the shared config draft, including the JSON editor
        below.
        """
    )
    return


@app.cell(hide_code=True)
def _(scratch_gui):
    scratch_gui.gui_panel()
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        """
        ### 3. JSON View of the Same Draft

        `gui.json_editor()` is not a separate config. It is linked to the
        form above. Editing either view updates the other, and validation still
        happens against the typed Pydantic model.
        """
    )
    return


@app.cell(hide_code=True)
def _(scratch_gui):
    scratch_gui.json_editor()
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        """
        ### 4. Validation Status

        `gui.status_panel()` itself is intentionally empty when the draft
        is valid. The docs render an explicit success message in that case so
        the empty-valid state is visible. Downstream cells should still treat
        `gui.validated_config()` as a strict `Config` value.
        """
    )
    return


@app.cell(hide_code=True)
def _(scratch_gui):
    status_panel = scratch_gui.status_panel()
    if status_panel.text == mo.md("").text:
        status_display = docs_md("No validation errors.")
    else:
        status_display = status_panel
    status_display
    return


@app.cell(hide_code=True)
def _(scratch_config):
    scratch_config
    return


@app.cell(hide_code=True)
def _(scratch_config):
    docs_md("### Validated JSON Payload")
    return


@app.cell(hide_code=True)
def _(scratch_config):
    mo.json(scratch_config.model_dump(mode="json"))
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        ### Wrapped Layouts

        The owner API can be wrapped in marimo layouts while keeping the form,
        JSON editor, and validation status synchronized.

        ```python
        gui.stacked()
        ```

        The default background is `neutral`. Pass `background=None` to
        `create_config_gui(...)` or to an individual view method for raw
        unstyled output.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        ---

        ## Script Mode

        When the notebook is running as a script, `create_config_gui(...)`
        loads the initial config through `tyro`. With no positional config path,
        CLI flags are applied to the model defaults:

        ```bash
        python notebook.py --scene-path data/bicycle --runtime.device cuda
        ```

        `--help` is generated from the typed Pydantic model. Field
        descriptions from `Field(description=...)` show up in the CLI help, so
        useful GUI help text also makes script mode easier to use:

        ```bash
        python notebook.py --help
        ```

        A positional JSON config path can be used as the starting config.
        Sparse typed overrides come after the JSON file, which is usually the
        most convenient workflow on a server:

        ```bash
        python notebook.py config.json --runtime.device cuda --max-steps 1000
        ```

        Relative `Path` field values loaded from JSON are resolved against the
        process current working directory, matching manual Pydantic validation
        and CLI path flags. Use an explicit `ConfigPreset(base_dir=...)` when a
        preset should anchor paths somewhere else, or
        `load_json_config(..., base_dir=...)` when loading JSON directly.

        When a notebook defines presets with `ConfigPresetCatalog`, select them
        with `--preset`:

        ```bash
        python notebook.py --preset garden --runtime.device cuda
        ```

        Config overlay files can be declared by the notebook or passed on the
        command line. They are sparse JSON objects merged left-to-right before
        explicit CLI field overrides:

        ```bash
        python notebook.py base.json --overlay server.json --max-steps 1000
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        ### Notebook Preset Selector

        Passing a `ConfigPresetCatalog` to `create_config_gui(...)` uses the
        catalog default in notebook mode and enables `--preset` in script mode.
        Render the selector from the same owner; dependent cells should read
        `gui.validated_config()`, not the selector value.

        ```python
        gui = create_config_gui(Config, presets=preset_catalog)
        preset_selector = gui.preset_selector(label="Preset")
        config = gui.validated_config()
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _(preset_gui):
    preset_gui.preset_selector(label="Example preset")
    return


@app.cell(hide_code=True)
def _(preset_gui):
    preset_gui.json_editor()
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        "The selected preset populates the form below and the linked JSON payload."
    )
    return


@app.cell(hide_code=True)
def _(preset_gui):
    preset_gui.gui_panel()
    return


@app.cell
def _(preset_gui):
    preset_config = preset_gui.validated_config()
    return (preset_config,)


@app.cell(hide_code=True)
def _(preset_config):
    mo.json(preset_config.model_dump(mode="json"))
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        ### Example: Base JSON, Overlay, Then CLI

        The example below resolves the same order a server command would use:

        ```bash
        python notebook.py base.json \
          --overlay server.json \
          --runtime.device cpu \
          --max-steps 1000
        ```

        `base.json` provides the starting config, `server.json` applies sparse
        machine or experiment defaults, and explicit CLI flags win last.
        """
    )
    return


@app.cell(hide_code=True)
def _(base_config_payload, overlay_config_payload):
    mo.hstack(
        [
            mo.json(base_config_payload, label="base.json"),
            mo.json(overlay_config_payload, label="server.json"),
        ],
        widths="equal",
        align="start",
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _():
    docs_md("Resolved config after applying base, overlay, and CLI overrides:")
    return


@app.cell(hide_code=True)
def _(script_config):
    mo.json(script_config.model_dump(mode="json"))
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        In Python, overlay entries may be strings, `Path` objects,
        `(path, required)` tuples, or `ConfigFile` objects. Bare overlay paths
        are required by default.

        If a notebook should be usable with the `tyro`/JSON script loader, use
        one loader-backed top-level config GUI for that notebook. Multiple
        independent loader-backed config states all try to interpret the same
        script arguments, which makes the command-line interface ambiguous. If
        you need several panels, make them fields of one top-level `Config`
        model.

        For custom script behavior, pass `script_loader=...` to
        `create_config_gui(...)`. The loader receives the model class, the
        optional default value, and the optional argument sequence.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        ---

        ## Local Path Defaults

        Machine-specific dataset and artifact paths should not need to dirty
        tracked presets. Put local path replacements in a `.path_defaults.json`
        file beside the JSON config file:

        ```text
        project/
          notebooks/
            train.py
          presets/
            garden.json
            bicycle.json
            .path_defaults.json
        ```

        In that layout, running `python notebooks/train.py presets/garden.json`
        loads `presets/.path_defaults.json` automatically because it is a
        sibling of the selected config file.

        A tracked preset can stay machine-independent:

        ```json
        {
          "scene_path": "dataset/mipnerf360/garden",
          "ply_path": "point_cloud.ply",
          "device": "cuda"
        }
        ```

        The local, ignored path-default file supplies machine paths:

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

        Path defaults apply only to fields typed as `Path`. The `path_prefixes`
        section uses longest-prefix matching for logical path values. The
        `fields` section targets typed config fields by dotted field path:

        ```text
        dataset/mipnerf360/garden -> /data/mipnerf360/garden
        ply_path -> /home/user/scenes/example_scene.ply
        ```

        Non-path strings such as `runtime.device = "mlx"` are never rewritten;
        use a config overlay or CLI override for those.
        """
    )
    return


@app.cell(hide_code=True)
def _(path_defaults_example):
    docs_md("### Path Defaults Example")
    return


@app.cell(hide_code=True)
def _(path_defaults_example):
    docs_md(path_defaults_example)
    return


@app.cell(hide_code=True)
def _(path_config):
    mo.json(path_config.model_dump(mode="json"))
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        For configs without JSON files or preset catalogs, pass a source path
        explicitly so the helper can look for the source's sibling
        `.path_defaults.json`:

        ```python
        create_config_gui(
            Config,
            value=Config(),
            path_defaults_source=NOTEBOOK_PATH,
        )
        ```

        Path-default entries are optional by default. Use `(path, True)` or
        `ConfigFile(path=path, required=True)` when the file must exist.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        ---

        ## Apply Before Expensive Work

        Viewer reloads, scene loading, dataset preparation, and training runs
        are often too expensive to trigger on every draft edit. Use marimo
        gating when the user should edit a draft first and explicitly apply it:

        ```python
        load_button = mo.ui.run_button(
            label="Load scene",
            disabled=not gui.is_valid(),
        )
        ```

        Then render an explicit status cell. In a real notebook, this is also
        where the expensive work would run:

        ```python
        if not load_button.value:
            mo.callout("Scene not loaded yet.", kind="warn")
        else:
            draft_config = gui.validated_config()
            scene_path = getattr(draft_config, "scene_path", None)
            if scene_path is None:
                mo.callout("Scene loaded.", kind="success")
            else:
                mo.callout(f"Scene loaded: `{scene_path}`.", kind="success")
        ```

        Use `gui.validated_config()` directly for cheap dependent controls. Use
        `mo.ui.run_button(...)` and `mo.stop(...)` for expensive work.
        """
    )
    return


@app.cell(hide_code=True)
def _(load_button):
    load_button
    return


@app.cell(hide_code=True)
def _(load_button, scratch_gui):
    if not load_button.value:
        load_status = mo.callout("Scene not loaded yet.", kind="warn")
    else:
        draft_config = scratch_gui.validated_config()
        scene_path = getattr(draft_config, "scene_path", None)
        if scene_path is None:
            load_status = mo.callout("Scene loaded.", kind="success")
        else:
            load_status = mo.callout(
                f"Scene loaded: `{scene_path}`.",
                kind="success",
            )
    load_status
    return


@app.cell(hide_code=True)
def _():
    docs_md(
        r"""
        ---

        ## License

        This package is distributed under the Apache License 2.0.
        """
    )
    return


@app.class_definition
class RuntimeConfig(BaseModel):
    device: Literal["cpu", "cuda"] = "cuda"
    seed: int = Field(0, ge=0, description="Random seed for the run.")


@app.class_definition
class Config(BaseModel):
    scene_path: Path = Field(
        Path("data/scene"),
        description="Path to a COLMAP-style scene directory.",
    )
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    max_steps: int = Field(30_000, ge=1)
    schedule: dict[str, float] = Field(
        default_factory=lambda: {"warmup": 0.1, "decay": 0.9},
        json_schema_extra={
            "marimo_config_gui": {"render": "json"},
        },
    )


@app.class_definition
class PathConfig(BaseModel):
    scene_path: Path = Path("dataset/mipnerf360/garden")
    ply_path: Path = Path("point_cloud.ply")
    device: Literal["cpu", "cuda", "mlx"] = "cuda"


@app.cell
def _():
    authoring_examples = {
        "Minimal config model": code_example(
            "from pydantic import BaseModel, Field",
            "",
            "",
            "class Config(BaseModel):",
            "    max_steps: int = Field(30_000, ge=1)",
        ),
        "Field descriptions and validation": code_example(
            "from pydantic import BaseModel, Field",
            "",
            "",
            "class Config(BaseModel):",
            "    max_steps: int = Field(",
            "        100,",
            "        ge=1,",
            "        le=100,",
            '        description="Try going over 1\'000",',
            "    )",
        ),
        "Literal, Enum, and Flag selectors": code_example(
            "from enum import Enum, Flag, auto",
            "from typing import Literal",
            "",
            "from pydantic import BaseModel, Field",
            "",
            "",
            "class RenderBackend(str, Enum):",
            '    CUDA = "cuda"',
            '    SLANGPY = "slangpy"',
            '    TORCH = "torch"',
            "",
            "",
            "class FeatureFlag(Flag):",
            "    READ = auto()",
            "    WRITE = auto()",
            "    EXECUTE = auto()",
            "",
            "",
            "class Config(BaseModel):",
            '    device: Literal["cpu", "cuda"] = "cuda"',
            "    backend: RenderBackend = RenderBackend.CUDA",
            "    features: FeatureFlag = FeatureFlag.READ | FeatureFlag.WRITE",
        ),
        "Path browser fields": code_example(
            "from pathlib import Path",
            "",
            "from pydantic import BaseModel, Field",
            "",
            "",
            "class Config(BaseModel):",
            "    scene_path: Path = Field(",
            '        Path("dataset/mipnerf360/garden"),',
            '        description="COLMAP-style scene directory.",',
            "    )",
        ),
        "Nested sections and optional field": code_example(
            "from pathlib import Path",
            "",
            "from pydantic import BaseModel, ConfigDict, Field",
            "",
            "",
            "class OptimizerConfig(BaseModel):",
            '    model_config = ConfigDict(title="Optimizer Settings")',
            "",
            "    position_lr: float = Field(1.6e-4, gt=0.0)",
            "",
            "",
            "class RenderConfig(BaseModel):",
            "    antialias: bool = True",
            "",
            "",
            "class Config(BaseModel):",
            "    optimizer: OptimizerConfig = Field(",
            "        default_factory=OptimizerConfig,",
            '        title="Training Optimizer",',
            "    )",
            "    render: RenderConfig = Field(default_factory=RenderConfig)",
            "    checkpoint_path: Path | None = Field(",
            "        None,",
            '        description="Leave unset for a fresh run.",',
            "    )",
        ),
        "Custom names and union labels": code_example(
            "from pydantic import BaseModel, ConfigDict, Field",
            "",
            "",
            "class ResNet50(BaseModel):",
            '    model_config = ConfigDict(title="ResNet-50")',
            "",
            "",
            "class ContextNet(BaseModel):",
            '    model_config = ConfigDict(title="Context Network")',
            "",
            "",
            "class CustomNetwork(BaseModel):",
            "    width: int = 128",
            "",
            "",
            "class Config(BaseModel):",
            "    subconfig: ResNet50 | ContextNet | CustomNetwork = Field(",
            "        default_factory=ResNet50,",
            '        title="Architecture",',
            "    )",
        ),
        "GUI hints and sequence fields": code_example(
            "from pydantic import BaseModel, Field",
            "",
            "",
            "class Config(BaseModel):",
            "    learning_rate: float = Field(",
            "        0.01,",
            "        ge=0.0,",
            "        le=1.0,",
            "        json_schema_extra={",
            '            "marimo_config_gui": {"widget": "slider"},',
            "        },",
            "        description=(",
            '            "Don\'t forget about adding useful information. "',
            '            "It\'ll also show up in the CLI."',
            "        ),",
            "    )",
            "    schedule: dict[str, float] = Field(",
            '        default_factory=lambda: {"warmup": 0.1, "decay": 0.9},',
            "        json_schema_extra={",
            '            "marimo_config_gui": {"render": "json"},',
            "        },",
            "    )",
            "    opacity_range: tuple[float, float] = Field(",
            "        (0.1, 0.9),",
            '        description="Fixed tuples of length 2-5 render as compact fields.",',
            "    )",
            "    color_correction: tuple[float, float, float, float, float, float] = Field(",
            "        (1.0, 0.0, 0.0, 1.0, 0.0, 1.0),",
            '        description="Longer tuples fall back to JSON arrays.",',
            "    )",
            "    milestones: list[int] = Field(",
            "        default_factory=lambda: [1000, 5000, 10000],",
            '        description="Lists stay variable-length and render as JSON arrays.",',
            "    )",
        ),
    }
    return (authoring_examples,)


@app.cell
def _(model_source_editor):
    scratch_model_namespace = {
        "BaseModel": BaseModel,
        "Enum": Enum,
        "Field": Field,
        "Flag": Flag,
        "Literal": Literal,
        "Path": Path,
        "auto": auto,
        "ConfigDict": ConfigDict,
    }
    scratch_model_error = None
    try:
        exec(
            model_source_editor.value,
            scratch_model_namespace,
            scratch_model_namespace,
        )
        scratch_config_model = scratch_model_namespace.get("Config")
        if not isinstance(scratch_config_model, type) or not issubclass(
            scratch_config_model,
            BaseModel,
        ):
            scratch_config_model = None
            scratch_model_error = "Define a Pydantic model class named Config."
    except Exception:
        scratch_config_model = None
        scratch_model_error = traceback.format_exc()
    return scratch_config_model, scratch_model_error


@app.cell
def _(scratch_config_model, scratch_model_error):
    mo.stop(scratch_model_error is not None or scratch_config_model is None)
    scratch_gui = create_config_gui(
        scratch_config_model,
        value=scratch_config_model(),
    )
    return (scratch_gui,)


@app.cell
def _(scratch_gui):
    scratch_config = scratch_gui.validated_config()
    return (scratch_config,)


@app.cell
def _(scratch_gui):
    load_button = mo.ui.run_button(
        label="Load scene",
        disabled=not scratch_gui.is_valid(),
    )
    return (load_button,)


@app.cell
def _():
    docs_tmp_dir = Path(tempfile.mkdtemp(prefix="marimo_config_gui_docs_"))
    return (docs_tmp_dir,)


@app.cell
def _(docs_tmp_dir):
    base_config_path = docs_tmp_dir / "base.json"
    bicycle_config_path = docs_tmp_dir / "bicycle.json"
    overlay_config_path = docs_tmp_dir / "server.json"
    base_config_payload = {
        "scene_path": "data/garden",
        "runtime": {"device": "cuda", "seed": 7},
        "max_steps": 30_000,
    }
    bicycle_config_payload = {
        "scene_path": "data/bicycle",
        "runtime": {"device": "cpu", "seed": 11},
        "max_steps": 10_000,
    }
    overlay_config_payload = {
        "runtime": {"seed": 42},
        "max_steps": 2_000,
    }
    base_config_path.write_text(json.dumps(base_config_payload, indent=2))
    bicycle_config_path.write_text(json.dumps(bicycle_config_payload, indent=2))
    overlay_config_path.write_text(json.dumps(overlay_config_payload, indent=2))
    return (
        base_config_path,
        base_config_payload,
        bicycle_config_path,
        bicycle_config_payload,
        overlay_config_path,
        overlay_config_payload,
    )


@app.cell
def _(Config, base_config_path, overlay_config_path):
    script_config = load_script_config(
        Config,
        args=[
            str(base_config_path),
            "--overlay",
            str(overlay_config_path),
            "--runtime.device",
            "cpu",
            "--max-steps",
            "1000",
        ],
    )
    return (script_config,)


@app.cell
def _(Config, base_config_path, bicycle_config_path):
    preset_catalog = ConfigPresetCatalog(
        model_cls=Config,
        presets={
            "garden": ConfigPreset(
                name="garden",
                path=base_config_path,
                label="Garden",
            ),
            "bicycle": ConfigPreset(
                name="bicycle",
                path=bicycle_config_path,
                label="Bicycle",
            ),
        },
        default="garden",
        preset_field=None,
    )
    return (preset_catalog,)


@app.cell
def _(Config, preset_catalog):
    preset_gui = create_config_gui(
        Config,
        presets=preset_catalog,
    )
    return (preset_gui,)


@app.cell
def _(PathConfig, docs_tmp_dir):
    path_config_path = docs_tmp_dir / "paths.json"
    path_defaults_path = docs_tmp_dir / ".path_defaults.json"
    path_config_path.write_text(
        json.dumps(
            {
                "scene_path": "dataset/mipnerf360/garden",
                "ply_path": "point_cloud.ply",
                "device": "cuda",
            },
            indent=2,
        )
    )
    path_defaults_path.write_text(
        json.dumps(
            {
                "path_prefixes": {"dataset/mipnerf360": "/data/mipnerf360"},
                "fields": {
                    "ply_path": "/data/point_clouds/garden/point_cloud.ply"
                },
            },
            indent=2,
        )
    )
    path_config = load_script_config(
        PathConfig,
        args=[str(path_config_path)],
    )
    path_defaults_example = (
        "The logical dataset path and `ply_path` field are rewritten from the "
        "sibling `.path_defaults.json`; the non-path `device` field is left "
        "unchanged."
    )
    return path_config, path_defaults_example


if __name__ == "__main__":
    app.run()
