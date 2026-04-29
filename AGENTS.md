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
