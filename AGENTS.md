# Agent Guidelines

- Prefer declarative and functional code whenever appropriate.
- Use `jaxtyping` for tensor and NumPy array annotations.
- Before running `git commit`, inspect `git status --short`.
- If the worktree contains unrelated unstaged user changes, do not run a hook-enabled `git commit`; either use a scoped `git commit --no-verify` for the task's staged files or stop and ask the user. Do not rely on pre-commit's stash/restore cycle to preserve unrelated work.
- When annotating a single dimension with `jaxtyping`, leave a single space in
  the dimension spec to avoid confusion with forward annotations.

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
  `load_form` or only a `viewer`, plus any short contextual text. Put the
  supporting notebook machinery in `column=1`: section markdown, helper
  functions, state/config cells, scene loading, controls, and other plumbing.
  For very long notebooks, it is fine to add `column=2` for notebook-specific
  side material, usually GUI/control cells, when that keeps the main flow and
  primary support column cleaner.
- When using `marimo-config-gui`, create config state in `app.setup` rather
  than in a notebook cell. This keeps a single interactive GUI form and keeps
  script mode aligned with the `tyro` CLI path.
- For `marimo` notebooks, hide code by default for pure GUI/reactive display
  cells such as `config_form(...)`, `config_json(...)`, `config_error(...)`,
  buttons, and similar UI-only cells.
