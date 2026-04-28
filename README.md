# Ember

Extensible Modular Backend Ecosystem for Rendering

Contract-based tools to spark your inverse graphics research.

Research monorepo for fast, typed, reproducible experimentation around the 3D
Gaussian Splatting family and closely related explicit inverse-graphics
methods.

Ember is organized around explicit rendering and training contracts. Primitive
type, backend, initialization, densification, data preparation, training loop,
and pre/post-processing are meant to be recombinable pieces rather than one
hidden monolith. The workflow is notebook-first, backend packages are opt-in,
and the core stays focused on shared contracts for inverse graphics research.
See [NORTH_STAR.md](NORTH_STAR.md) for the longer design constitution.

## Status

Pre-alpha. APIs may change when a better boundary becomes clear.

This repo is currently useful for:

- shared `ember-core` / `ember_core` scene, camera, render, data, and training contracts
- adapter backends around existing third-party rasterizers
- first-party native backend families for hackable kernel/runtime work
- marimo-first paper demos, viewers, and training notebooks
- local CUDA/PyTorch environments managed primarily with `uv`, with Pixi as a
  fallback when system CUDA/toolchain management matters

## Install

For local development from a fresh checkout:

```bash
git submodule update --init --recursive
uv venv
uv sync --extra cu130-dev
```

`uv sync` installs the minimal development environment. Add one CUDA wheel
flavor and whichever backend families you need:

```bash
uv sync --extra adapter-backends --extra cu130
uv sync --extra native-faster-gs --extra cu130
uv sync --extra native-faster-gs-mojo --extra cu130
uv sync --extra native-3dgrt --extra cu130
uv sync --extra native-svraster --extra cu130
```

Install the first-party native backend families plus splatting training helpers:

```bash
uv sync --extra all-native --extra cu130
```

Install the full CUDA-specific developer environment:

```bash
uv sync --extra cu130-dev
```

Use `cu128` / `cu128-dev` instead of `cu130` / `cu130-dev` for the CUDA 12.8
PyTorch wheel set. The CUDA extras are mutually exclusive.

Native extension builds can use a lot of memory. To force serial builds:

```bash
UV_CONCURRENT_BUILDS=1 uv sync --extra cu130-dev
```

If Python-only environment management is not enough for a machine, use Pixi:

```bash
pixi install -e cu130
pixi shell -e cu130
```

## Using From Another Project

Use GitHub source archives for monorepo package subdirectories. They install the
selected package without initializing this repository's `third_party/`
submodules:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

```bash
uv add "ember-core[cu130] @ https://github.com/panicPaul/ember/archive/refs/heads/main.zip#subdirectory=packages/ember-core"
uv add "ember-native-faster-gs[cu130] @ https://github.com/panicPaul/ember/archive/refs/heads/main.zip#subdirectory=packages/ember-native-faster-gs"
uv add "ember-native-3dgrt[cu130] @ https://github.com/panicPaul/ember/archive/refs/heads/main.zip#subdirectory=packages/ember-native-3dgrt"
uv add "ember-native-svraster[cu130] @ https://github.com/panicPaul/ember/archive/refs/heads/main.zip#subdirectory=packages/ember-native-svraster"
```

The Mojo backend also needs Modular nightly packages:

```toml
[tool.uv]
prerelease = "allow"

[tool.uv.sources]
# merge these entries with the existing torch source table above
max = { index = "modular-nightly" }
mojo = { index = "modular-nightly" }

[[tool.uv.index]]
name = "modular-nightly"
url = "https://whl.modular.com/nightly/simple/"
```

```bash
uv add "ember-native-faster-gs-mojo[cu130] @ https://github.com/panicPaul/ember/archive/refs/heads/main.zip#subdirectory=packages/ember-native-faster-gs-mojo"
```

Prefer release tags over `main` once a release is cut:

```bash
uv add "ember-core[cu130] @ https://github.com/panicPaul/ember/archive/refs/tags/v0.0.1.zip#subdirectory=packages/ember-core"
```

For PEP 723 / `marimo edit --sandbox` notebooks, put the archive dependencies
directly in the notebook. Avoid forwarding the PyTorch CUDA index as a global
sandbox index: marimo exports script metadata to a plain requirements file
before the isolated install, so `tool.uv.sources` index affinity is not
preserved for that second resolver pass. Pin the PyTorch CUDA wheel as a direct
URL and only forward the Modular index when using the Mojo backend:

```python
# /// script
# dependencies = [
#     "marimo",
#     "torch @ https://download.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp314-cp314-manylinux_2_28_x86_64.whl",
#     "ember-core[cu130] @ https://github.com/panicPaul/ember/archive/refs/heads/main.zip#subdirectory=packages/ember-core",
#     "ember-native-faster-gs[cu130] @ https://github.com/panicPaul/ember/archive/refs/heads/main.zip#subdirectory=packages/ember-native-faster-gs",
#     "ember-native-faster-gs-mojo[cu130] @ https://github.com/panicPaul/ember/archive/refs/heads/main.zip#subdirectory=packages/ember-native-faster-gs-mojo",
# ]
# requires-python = ">=3.14"
#
# [tool.uv]
# prerelease = "allow"
#
# [tool.uv.sources]
# max = { index = "modular-nightly" }
# mojo = { index = "modular-nightly" }
#
# [[tool.uv.index]]
# name = "modular-nightly"
# url = "https://whl.modular.com/nightly/simple/"
# ///
```

This prevents the PyTorch CUDA index from shadowing PyPI packages such as
`cuda-bindings==13.2.0` during the final sandbox install.
Omit the Modular block and Mojo package if the notebook does not use the Mojo
backend.

See `sandboxed_notebooks/packaging_local.py`,
`sandboxed_notebooks/packaging_git_main.py`, and
`sandboxed_notebooks/splat_viewer_git_main.py` for working packaging probes.

## Backend Packages

There are two backend categories.

**Adapter backends** wrap external implementations behind the shared `ember-core`
contracts. They are primarily for comparison, interoperability, and reuse of
existing paper code.

```bash
uv sync --extra adapter-backends --extra cu130
```

Package: `packages/ember-adapter-backends`

Current adapter names include:

- `adapter.gsplat`
- `adapter.gsplat_2dgs`
- `adapter.inria`
- `adapter.fastgs`
- `adapter.fastergs`
- `adapter.stoch3dgs`

**Native backends** are first-party backend implementation packages maintained
in this repo. Some are vendored and modernized from upstream kernels, some are
true first-party implementations, and, except for the current Mojo path, they
expose reusable internal stages such as preprocess, sort, and blend. Those
stages are registered as Torch custom ops where useful. When an adapter-backed
reference exists, native paths are tested against it.

The point of exposing stages is local research change: for example, a scoring
method can replace only the blend forward pass while reusing the same
preprocess, sort, packing, and reference behavior. This keeps experiments small
and comparable instead of forcing a fork of an entire backend.

```bash
uv sync --extra all-native --extra cu130
```

Native packages:

- `packages/ember-native-faster-gs`
- `packages/ember-native-faster-gs-mojo`
- `packages/ember-native-3dgrt`
- `packages/ember-native-svraster`

Current native backend names include:

- `faster_gs.core`
- `faster_gs.depth`
- `faster_gs.gaussian_pop`
- `faster_gs_mojo.core`
- `3dgrt.stoch3dgs`
- `svraster.core`

Native backends JIT-build their native extension code on first use, so the
first render or viewer launch may spend time compiling.

## Package Map

```text
packages/
  ember-core/                    Core contracts, traits, registry, data helpers
  ember-adapter-backends/        Official wrappers around third-party backends
  ember-native-faster-gs/        First-party FasterGS-family native backends
  ember-native-faster-gs-mojo/   Mojo-backed FasterGS experiments
  ember-native-3dgrt/            First-party 3DGRT-family native backend
  ember-native-svraster/         First-party SVRaster-family native backend
  ember-splatting-training/      Optional splatting training utilities
  marimo-config-gui/             Pydantic-driven config UI helpers for marimo

papers/
  fastergs/                     Paper-specific notebook/config work

notebooks/                      Local demos, viewers, and research notebooks
tests/                          Cross-package tests and backend parity checks
third_party/                    Upstream projects and editable submodules
```

The root `pyproject.toml` is a development environment, not the main library
package. The library package is `packages/ember-core`.

## 3DGRT / OptiX

`3dgrt.stoch3dgs` uses OptiX. The header-only `optix-dev` subset needed to
compile the native package is vendored in this repository at:

```text
packages/ember-native-3dgrt/src/ember_native_3dgrt/core/native/stoch3dgs/dependencies/optix-dev/include/optix.h
```

You still need a system NVIDIA driver and OptiX-capable GPU. For local
development, install the normal NVIDIA CUDA/driver stack for your machine; the
vendored headers are not a replacement for the driver/runtime components. If
you need the full OptiX SDK, download it from NVIDIA:
<https://developer.nvidia.com/designworks/optix/download>.

The adapter path `adapter.stoch3dgs` wraps the upstream `third_party/Stoch3DGS`
checkout. If you use that path, initialize nested submodules too:

```bash
git submodule update --init --recursive
```

The upstream adapter header path is:

```text
third_party/Stoch3DGS/threedgrt_tracer/dependencies/optix-dev/include/optix.h
```

If that file is missing, inspect nested submodule state:

```bash
git -C third_party/Stoch3DGS submodule status --recursive
```

## Smoke Tests And Images

Fresh-clone CUDA install smoke test:

```bash
python scripts/smoke_fresh_clone.py --cuda cu130
```

Use the current checkout instead of a fresh clone:

```bash
python scripts/smoke_fresh_clone.py --source worktree --cuda cu130
```

Build an Apptainer/SIF artifact for cluster use:

```bash
python scripts/build_sif.py --cuda cu130
apptainer exec --nv dist/ember-cu130.sif python -c "import torch"
```

These flows require an NVIDIA GPU and the relevant container tooling.

## Versioning

Most packages derive versions from Git tags via `hatch-vcs`. A tag like
`v0.0.1` builds package version `0.0.1`; commits after a tag build development
versions tied to that tag.

Before merging or pushing to `main`, make an explicit release/version decision.
If the change is part of a release, tag the final `main` commit:

```bash
git tag v0.0.1
git push origin main v0.0.1
```

GitHub source-archive pinning examples:

```bash
pip install "https://github.com/panicPaul/ember/archive/refs/tags/v0.0.1.zip#subdirectory=packages/ember-core"
pip install "https://github.com/panicPaul/ember/archive/refs/tags/v0.0.1.zip#subdirectory=packages/ember-adapter-backends"
pip install "https://github.com/panicPaul/ember/archive/refs/tags/v0.0.1.zip#subdirectory=packages/ember-native-faster-gs"
```

Use source archives instead of `git+https` URLs for package subdirectories.
Archive installs do not initialize monorepo submodules.

## Development Notes

- Prefer `uv` first; use Pixi when CUDA/toolchain packaging needs Conda-level
  control.
- Package-local dependencies should be edited from the package directory.
  `packages/*` are intentionally not true `uv` workspace members because the
  package-local CUDA/PyTorch index settings conflict during workspace locking.
- For native backend rebuilds after CUDA/C++ edits, reinstall the target
  package or upstream adapter path explicitly.
- If a killed marimo notebook leaves GPU worker processes alive:

```bash
uv run python scripts/cleanup_stale_gpu_workers.py
uv run python scripts/cleanup_stale_gpu_workers.py --execute
```

The first cleanup command is a dry run; the second sends `SIGTERM` to stale
repository-owned workers.
