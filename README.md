# splatkit

Monorepo for a modular Gaussian splatting stack centered around a minimal core,
typed backend interfaces, and fast research iteration.

The default workflow is `uv`. If `uv` is not enough for your machine or native
backend setup, fall back to Pixi.

## Install

For local development from the monorepo root, initialize submodules and create
the virtual environment:

```bash
git submodule update --init --recursive
uv venv
```

Then install dependencies with the default `torch` build:

```bash
uv sync
```

To try a CUDA-specific PyTorch wheel through `uv`, select an extra:

```bash
uv sync --extra cu128
```

or:

```bash
uv sync --extra cu130
```

Notes:
- plain `uv sync` uses the default `torch` dependency
- `cu128` and `cu130` are mutually exclusive
- the CUDA extras only affect Python package resolution, especially the PyTorch
  wheel index
- prefer `uv` first because it is the simplest path
- fall back to Pixi when `uv` is not enough due to CUDA or toolchain issues

Pixi uses Conda packages under the hood, so it can ship CUDA libraries,
compiler toolchains like `gcc`, and related native dependencies explicitly.
That makes it the better fallback when Python package resolution alone is not
enough.

For a Pixi-managed environment:

```bash
pixi install
pixi shell
```

or select a CUDA-specific environment explicitly:

```bash
pixi install -e cu128
pixi shell -e cu128
```

```bash
pixi install -e cu130
pixi shell -e cu130
```

## What This Repository Is For

This repository exists to make experimentation with Gaussian splatting faster,
more reproducible, and easier to hack on.

It is not meant to be:
- another monolithic backend
- a direct competitor to polished specialized tools like `gsplat` as a pure
  rasterization backend
- a direct competitor to end-user-focused products like Lichtfeld Studio

It is meant to:
- provide a very small `splatkit` core with minimal dependencies
- make it easy to switch between different backends behind a shared interface
- keep code strongly typed, declarative, and functional where appropriate
- use a trait/capability system so backend outputs stay precisely type-hinted
- grow toward optional modules for training loops, dataloading, densification,
  and related research workflows
- integrate deeply with `marimo` for rapid prototyping without giving up normal
  Python script execution
- make everything as pip-installable as possible, with `uv` as the default
  workflow and optional Pixi support for easier full-environment setup

A major motivation is that it is still surprisingly hard to compare different
3DGS implementations fairly. Differences often come not just from the
rasterizer, but from densification, dataloading, evaluation, scene
normalization, and other surrounding decisions. This repository aims to make
those pieces easier to share and compare without forcing them into one monolith.

## Design Philosophy

- Minimal core: `splatkit` should stay small, dependency-light, and focused on
  contracts, traits, registries, and shared abstractions.
- Backends as plugins: backend implementations should live outside the core and
  be swappable without changing user code.
- Strong typing first: data contracts, backend options, and output capabilities
  should be explicit and well-typed.
- Declarative APIs: scene state, camera state, render options, and future
  training components should be represented as clean data objects rather than
  implicit mutable state.
- Functional where appropriate: operations should prefer explicit inputs and
  outputs over hidden side effects.
- Reproducible and hackable: installation and execution should be easy to
  reproduce locally, while still being simple to modify during experiments.
- Notebook and script parity: research code should feel great to prototype in a
  notebook and then run as a normal Python script with the CLI tooling you are
  used to.
- Opt-in utilities: training loops, dataloading, densification, evaluation,
  scene normalization, and related helpers should be available when useful, but
  they should stay optional rather than becoming mandatory framework baggage.
- Config-driven ergonomics: `pydantic`, `tyro`, and related tools should make
  serializable configs, CLIs, and generated UI controls line up cleanly.

## Roadmap Direction

- `splatkit`: the stripped-down core package for contracts, traits,
  registration, and shared optional modules over time.
- `splatkit-backends`: a separate package containing wrappers for commonly used
  backend implementations.
- `marimo-3dv`: today a separate third-party package in this repo, eventually
  intended to become part of the broader splatkit ecosystem as the notebook/UI
  layer.
- `faster-gaussian-splatting`: included as a third-party reference
  implementation and planned wrapper target.
- `sv_raster`: included as a third-party reference implementation from the
  wider splatting ecosystem; currently tracked for its `backends/new_cuda`
  path rather than as a clean `splatkit` contract match.
- FasterGS backend: planned both as a wrapper around the third-party reference
  implementation and as a more stripped-down modular variant aimed at rapid
  prototyping and interchangeable parts.

The long-term direction is not just rendering. The same design should extend to
training loops, dataloading, densification, evaluation, and other research
infrastructure without collapsing into a single rigid framework.

If you only want to try a shiny new rasterizer backend and do not care about
the rest of the stack, that should be fine too. The optional pieces are meant
to help comparison and experimentation, not to force adoption of everything at
once.

## Repository Layout

```text
packages/
  splatkit/                Minimal core contracts and shared abstractions
  splatkit-backends/       Official backend adapters and wrappers
third_party/
  marimo-3dv/              Viewer and marimo utility layer
  diff-gaussian-rasterization/  Forked Inria rasterizer
  faster-gaussian-splatting/    FasterGS reference implementation
  sv_raster/               SV Raster reference repo; see backends/new_cuda
tests/                     Cross-package tests
notebooks/                 Monorepo-level examples
```

## Package Roles

- `splatkit`: backend-agnostic contracts, capabilities, type-driven traits,
  registration, and shared runtime helpers, with opt-in extras like
  `viewer`, `training`, `eval`, and `all`; the viewer stack remains separate
  from `all` for now
- `splatkit-backends`: wrappers around commonly used implementations such as
  `gsplat`, the local Inria path, and the local `Stoch3DGS` path, with
  per-backend extras plus `all`
- `marimo-3dv`: utilities for `marimo`, desktop viewers, and auto-generated GUI
  options that work well with serializable configs

Typical package install paths:

```bash
pip install splatkit
pip install "splatkit[viewer]"
pip install "splatkit[all]"
pip install "splatkit-backends[gsplat]"
pip install "splatkit-backends[inria]"
pip install "splatkit-backends[stoch3dgs]"
pip install "splatkit-backends[all]"
```

## Supported Backends

Currently registered backends in `splatkit-backends`:

| Backend name | Scene type | Alpha | Depth | Normals | 2D projections | Projective intersection transforms | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gsplat_2dgs` | `GaussianScene2D` | ✅ | ✅ | ❌ | ❌ | ✅ | 2D Gaussian backend via `gsplat.rasterization_2dgs` |
| `fastergs` | `GaussianScene3D` | ❌ | ❌ | ❌ | ❌ | ❌ | RGB-only FasterGS adapter |
| `gsplat` | `GaussianScene3D` | ✅ | ✅ | ❌ | ✅ | ❌ | 3D Gaussian backend via `gsplat.rasterization` |
| `inria` | `GaussianScene3D` | ❌ | ✅ | ❌ | ❌ | ❌ | GraphDeco/Inria rasterizer adapter |
| `stoch3dgs` | `GaussianScene3D` | ✅ | ✅ | ❌ | ❌ | ❌ | Stochastic 3DGRT adapter |
| `svraster` | `SparseVoxelScene` | ❌ | ✅ | ❌ | ❌ | ❌ | Sparse voxel rasterization backend |

Capability notes:
- `alpha`: per-pixel accumulated opacity/transmittance output.
- `depth`: per-pixel depth output in the backend's native shared render surface.
- `normals`: per-pixel surface or rendered normal output. This capability exists in the shared API, but no official backend exposes it yet.
- `2d_projections`: projected Gaussian centers plus compact conic coefficients via `projected_means` and `projected_conics`.
- `projective_intersection_transforms`: projected Gaussian centers plus 2DGS projective intersection geometry via `projected_means` and `projective_intersection_transforms`.

## Stoch3DGS / OptiX

The `stoch3dgs` backend requires NVIDIA OptiX to build the upstream 3DGRT
runtime.

What is required:
- an NVIDIA driver and GPU that support OptiX
- a working CUDA toolkit/toolchain that matches your PyTorch install
- the `optix-dev` headers, which are tracked in this repo as a nested submodule
  of `third_party/Stoch3DGS`

The header-only `optix-dev` checkout should appear at:

```text
third_party/Stoch3DGS/threedgrt_tracer/dependencies/optix-dev/include/optix.h
```

If that file is missing, initialize nested submodules again:

```bash
git submodule update --init --recursive
```

If you already had the repo cloned before `Stoch3DGS` was added, rerun the same
command after pulling the latest changes. In particular, `stoch3dgs` will not
compile unless the nested `optix-dev` submodule inside `third_party/Stoch3DGS`
is present.

Recommended install flow for the backend:

```bash
git submodule update --init --recursive
uv pip install -e './packages/splatkit-backends[stoch3dgs,cu130]'
```

If OptiX headers are still missing after a recursive submodule update, inspect:

```bash
git -C third_party/Stoch3DGS submodule status --recursive
```

You should see an entry for:

```text
threedgrt_tracer/dependencies/optix-dev
```

## Versioning

`packages/splatkit` and `packages/splatkit-backends` now derive their published
versions from Git tags via `hatch-vcs`.

- tagged commits build stable versions like `0.1.0`
- commits after a tag build unique dev versions tied to that tagged line
- before the first tag, builds fall back to `0.1.0a0`

Recommended release flow:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Recommended pinning flow for colleagues:

```bash
pip install "git+<repo-url>@v0.1.0#subdirectory=packages/splatkit"
pip install "git+<repo-url>@v0.1.0#subdirectory=packages/splatkit-backends"
```

Both packages also expose their installed version at runtime via
`splatkit.__version__` and `splatkit_backends.__version__`.

For local monorepo development, `uv` keeps these sources editable:

- `packages/splatkit`
- `packages/splatkit-backends`
- `third_party/marimo-3dv`
- `third_party/diff-gaussian-rasterization`
- `third_party/faster-gaussian-splatting/FasterGSCudaBackend`

The workspace root is a development environment package. It depends on
`splatkit[all]`, `splatkit-backends[all]`, `marimo`, `marimo-3dv`, `torch`,
`ruff`, and `ty`, rather than depending on individual backend wheels directly.

Development note: we tried making `packages/*` true `uv` workspace members, but
`uv` currently rejects that setup because the package-local `cu128` / `cu130`
PyTorch source configuration is resolved as conflicting indexes during
workspace-wide locking. For now, prefer changing into the target package
directory and running `uv add ...` there when adding package-specific
dependencies.

For native backend rebuilds after changing CUDA / C++ sources, reinstall the
backend package explicitly:

```bash
uv pip install --reinstall third_party/sv_raster/backends/new_cuda
uv pip install --reinstall third_party/faster-gaussian-splatting/FasterGSCudaBackend
uv pip install --reinstall third_party/diff-gaussian-rasterization
```

## Container Smoke Tests

For a fresh-clone install smoke test in Docker, build the smoke image from the
remote recursively cloned repository:

```bash
python scripts/smoke_fresh_clone.py
```

This infers the supported CUDA flavors from `pyproject.toml`, orders them with
the Pixi default first, and currently tests `cu128` and `cu130`.

To test one flavor only:

```bash
python scripts/smoke_fresh_clone.py --cuda cu130
```

For CI or local iteration against the current checkout instead of a fresh
clone:

```bash
python scripts/smoke_fresh_clone.py --source worktree
```

These smoke tests require an NVIDIA GPU. The Docker image just captures the
repo and toolchain; the actual validation runs inside `docker run --gpus all`.

The smoke container installs the frozen Pixi environment from `pixi.lock`,
runs `uv sync --locked --extra <flavor>`, and verifies that these imports
succeed:

- `torch`
- `diff_gaussian_rasterization`
- `FasterGSCudaBackend`
- `new_svraster_cuda`

## Cluster SIF Builds

For cluster deployment, build a `.sif` artifact through Docker and Apptainer:

```bash
python scripts/build_sif.py --cuda cu128
```

That creates `dist/splatkit-cu128.sif`, then runs a GPU-backed import smoke test
against the resulting artifact with `apptainer exec --nv`.

To build the other CUDA flavor or choose a different output path:

```bash
python scripts/build_sif.py --cuda cu130
python scripts/build_sif.py --cuda cu130 --output dist/custom-cu130.sif
```

On the cluster, run the SIF with NVIDIA passthrough:

```bash
apptainer exec --nv dist/splatkit-cu128.sif python -c "import torch"
```

For Tyro-based CLI script completion, install shell completion files for all
detected executable Tyro scripts with:

```bash
./scripts/install_tab_completion.py --shell zsh
./scripts/install_tab_completion.py --shell bash
```

To install completion for one specific script instead:

```bash
./scripts/install_tab_completion.py --shell zsh --script scripts/reinstall.py
```

For zsh, ensure `~/.zfunc` is on `fpath` and run
`autoload -Uz compinit && compinit`. Bash completion support for local scripts is
more limited and expects the script to be invoked directly as a command.

## Troubleshooting

### Orphaned Marimo Viewer Processes

If a `marimo` notebook is killed improperly, it may leave behind orphaned
Python worker processes that still hold GPU memory. This is a known issue.

For now, if you notice that this happened, use the cleanup script from the
repository root:

```bash
uv run python scripts/cleanup_stale_gpu_workers.py
uv run python scripts/cleanup_stale_gpu_workers.py --execute
```

The first command is a dry run. The second sends `SIGTERM` to stale
repository-owned multiprocessing workers.

## Goals

- Build a minimal, stable core for splatting experiments rather than a
  monolithic end-to-end framework.
- Make backend switching easy and explicit.
- Keep backend-specific capabilities type-safe through shared traits.
- Make research code pleasant to prototype in a notebook and then run as a
  normal Python script with the CLI usage you are used to.
- Make installation simple enough that local experiments are easy to reproduce.
- Keep the system open to future training and data pipeline modules without
  bloating the core package.

## Progress

| Goal | Subgoals | Status |
| --- | --- | --- |
| Core contracts and traits | Typed scene/camera contracts, capability traits, backend registry, shared render surface across `inria`, `gsplat`, and other registered backends | Done |
| Viewer workflow | Viewer works well in `marimo`, plus an experimental desktop viewer that reuses marimo-defined GUI elements | In progress |
| Notebook to script workflow | Define custom GUI elements in a notebook, then run the notebook as a Python file for a more interactive local viewer flow | In progress |
| Backend wrappers | Current wrappers for `inria`, `gsplat`, `svraster`, `fastergs`, and `stoch3dgs`, with broader training-oriented wrappers still planned | In progress |
| Hackable backend implementations | Stripped-down, modular variant of `FasterGS` aimed at rapid experimentation and easier modification while minimizing perfomance cost | Planned |
| Training and data pipeline | Dataloading, training scripts, training utilities, and evaluation scripts | Not started |
| Densification framework | Hackable densification design that stays flexible without becoming overly verbose | Not started |

## Non-Goals

- Being the best standalone rasterizer implementation. Check out `gsplat` or `Lichtfeld Studio` or the awesome `FasterGS` for that.
- Replacing polished backend-specific projects
- Optimizing primarily for end-user product UX over research flexibility
- Hiding backend differences so aggressively that important semantics disappear

The goal is a nicer interface over heterogeneous tools, not pretending those
tools are all the same.

## Project Hygiene

- Reproducible installs are already a baseline requirement here rather than a
  roadmap item.
- The main workflow is `uv`, with optional CUDA wheel selection and optional
  Pixi fallback when Python-only environment management is not enough.
- Local development is intended to stay editable, scriptable, and easy to
  reproduce across machines.
