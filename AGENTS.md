# Agent Guidelines

- Prefer declarative and functional code whenever appropriate.
- Use `jaxtyping` for tensor and NumPy array annotations.
- Before running `git commit`, inspect `git status --short`.
- If the worktree contains unrelated unstaged user changes, do not run a hook-enabled `git commit`; either use a scoped `git commit --no-verify` for the task's staged files or stop and ask the user. Do not rely on pre-commit's stash/restore cycle to preserve unrelated work.
- Before merging or pushing changes to `main`, make an explicit release version
  decision and bump/tag all package versions consistently. Most packages derive
  their versions from Git tags via `hatch-vcs`, so this usually means creating
  and pushing the intended shared `vX.Y.Z` tag after the merge; also update any
  packages that still carry static versions when they are part of the release.
- Before pushing packaging or dependency changes to `main`, run the local
  sandboxed packaging check, especially
  `sandboxed_notebooks/packaging_local.py`, so the current checkout and
  submodule pointers are validated before publishing.
- After pushing packaging or dependency changes to `main`, run the GitHub-source
  sandboxed checks, especially `sandboxed_notebooks/packaging_git_main.py` and
  `sandboxed_notebooks/splat_viewer_git_main.py`, so the published archive and
  Git dependencies are validated from a clean sandbox.
- When annotating a single dimension with `jaxtyping`, leave a single space in
  the dimension spec to avoid confusion with forward annotations.
- For mojo code: read https://docs.modular.com/llms-python.txt for MAX Python API documentation
- In this repo, `vendored` and `native` are distinct:
  - `third_party/*` submodules are pinned upstream references. Use them for
    parity checks, source reading, and tests, but do not make Ember runtime
    packages depend on importing them.
  - `native` means self-contained Ember-owned runtime code. Production code in
    `packages/ember-native-*` must not import upstream package names from
    `third_party/*`, and native package metadata must not depend on upstream
    submodule packages.
  - If upstream code is copied or ported into a native package, keep the
    license/provenance clear and rewrite imports to package-local
    `ember_native_*` modules. Tests may still compare against `third_party`
    reference implementations.
  - Native backend runtime code must be stage based. Follow the FasterGS and
    Stoch3DGS pattern: package-local typed runtime stage wrappers, thin
    `torch.library.custom_op` registration, explicit forward/backward stage
    boundaries, and separate C++/CUDA/Mojo/OptiX implementation files under the
    native package. Keep public render helpers as composition over those stages,
    not as monolithic upstream object wrappers.
  - Native C++/CUDA/Mojo/OptiX code should use full descriptive variable names.
    Prefer the same names across Python stage result types, custom-op schemas,
    pybind wrappers, and native kernels whenever the tensors represent the same
    concept. Match naming patterns already used by other native packages, such
    as FasterGS stage tensors, instead of introducing local abbreviations.
  - Native implementation files should include short comments around meaningful
    code blocks: validation, allocation/scratch setup, stage input packing,
    kernel launches, reductions/scans/sorts, backward-gradient assembly, and
    output packing. Avoid line-by-line narration, but make each block's role
    clear enough to compare against upstream and neighboring native backends.

Examples:

```python
from jaxtyping import Float
from numpy import ndarray
from torch import Tensor


def normalize(
    x: Float[Tensor, " batch channels"],
) -> Float[Tensor, " batch channels"]:
    ...


def project(
    points: Float[ndarray, " n 3"],
) -> Float[ndarray, " n 2"]:
    ...
```

- Read the NORTH_STAR.md, which is a rough sketch of what i want to achieve. Be careful it may be slightly outdated.
- For `marimo` notebooks, prefer `app = marimo.App(width="columns")` when a
  split layout is useful. Keep the main flow for the pure display cells that
  show the primary UI elements directly, especially cells that contain only a
  `load_form` or only a `viewer`, plus any short contextual text.
- For `marimo` notebooks with side columns, distinguish between producer cells
  and rendered output cells. Treat cells with no column annotation as the GUI
  column / first column / column0, equivalent to column 0. Do not write
  `column=0` explicitly; marimo filters it away and it should be treated as no
  annotation. Written column annotations therefore start after the GUI column:
  `column=1` is the second visual column, `column=2` is the third visual
  column, and so on. Keep rendered GUI/output cells in this unannotated GUI
  column. This includes cells whose final expression is a widget, form, error
  display, assembled controls layout, plot, viewer, or other displayed UI
  element.
- For `marimo` notebooks, put each function or class definition in its own
  cell. Do not batch multiple `def` or `class` definitions into one notebook
  cell unless there is a specific reason and the grouping materially improves
  readability. The notebook cell wrapper itself does not count here: the
  `def __(...):` introduced by `@app.cell` is just the cell definition, not a
  user-defined function for this rule.
- Explicitly annotate non-GUI cells with columns. This includes
  `@app.class_definition`, `@app.function`, helper logic, scene-loading
  helpers, pipeline wiring, `mo.ui.*` constructor cells, config state
  constructors, and other cells that produce values later rendered elsewhere.
  Leave config/model definitions and reusable functions unannotated when they
  should live with the GUI/config flow; otherwise put them in an explicit side
  column. Use `column=2` for helper/support code by default, and `column=3` for
  extra notebook-support cells when the notebook is large enough that the split
  keeps the main flow readable.
- Keep marimo notebook source sorted by column group: unannotated GUI/column0
  cells first, then `column=1`, then `column=2`, and so on. Prepend each column
  group with a markdown heading cell describing what that column contains. The
  column heading must be larger than any subheaders inside that same column, so
  the same semantic clusters remain clear and collapsible if the notebook is
  viewed without columns.
- When using `marimo-config-gui`, prefer creating config state in `app.setup`
  when the config model is available there. If the notebook is itself the
  primary artifact and defines the config model in notebook cells, it is fine
  to create config state in a producer cell near the config definition. Keep
  script mode aligned with the `tyro` CLI path when the notebook supports CLI
  execution.
- For `marimo` notebooks, do not wrap `config_form(...)`, `config_json(...)`,
  `config_error(...)`, or other reactive outputs in `mo.vstack(...)`,
  `mo.hstack(...)`, or similar containers when the wrapped object itself needs
  to remain reactive. Return the reactive element directly from the cell so
  marimo can register and update it correctly.
- For `marimo` notebooks, hide code by default for pure rendered display cells
  in the main flow. Do not hide code by default for producer cells such as
  `mo.ui.*` constructors, `config_form(...)`, `config_error(...)`, or other
  support cells unless there is a specific reason to do so.

## Interactive Package Docs

- For `marimo-config-gui` docs, the canonical interactive docs notebook lives at
  `packages/marimo-config-gui/docs/interactive.py`. The root docs entry
  `docs/marimo-config-gui.py` should remain a symlink to that file.
- Use `marimo.App(width="wide")` for linear docs notebooks unless columns are
  explicitly useful. Do not use columns for linear documentation pages.
- Keep major sections separated by markdown dividers and top-level section
  headers. Prefer short interactive examples over large all-in-one demos.
- For public package APIs, include at least one live example for each
  root-exported user feature when practical.
- Do not wrap reactive config GUI outputs such as `config_gui_panel(...)`,
  `config_json_editor(...)`, `config_preset_selector(...)`, or
  `config_status_panel(...)` in `mo.vstack(...)`, `mo.hstack(...)`, callouts,
  or other layout containers when they need to remain reactive. Return them
  directly from their own cells.
- Keep known bugs documented in the interactive docs when they affect
  user-facing behavior, especially marimo reactivity issues.
- Prefer live `mo.ui.code_editor(...)` examples for concepts users are expected
  to modify, and static markdown code blocks for supporting script-mode snippets.
