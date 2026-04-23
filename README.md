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

Then install just the core dev environment:

```bash
uv sync
```

Opt into specific backend families as needed:

```bash
uv sync --extra adapter-backends --extra cu130
uv sync --extra native-faster-gs --extra cu130
uv sync --extra native-3dgrt --extra cu130
uv sync --extra native-svraster --extra cu130
uv sync --extra native-svraster --extra svraster-adapter --extra cu130
```

Or install the full CUDA-specific developer environment in one step, including
all adapter backends, all native families, and the restricted SVRaster family:

```bash
uv sync --extra cu130-dev
```

If you want the same install with lower peak memory usage during native builds,
you can do that with `uv` directly by forcing serial builds:

```bash
UV_CONCURRENT_BUILDS=1 uv sync --extra cu128-dev
UV_CONCURRENT_BUILDS=1 uv sync --extra cu130-dev
```

Notes:
- plain `uv sync` keeps the root environment minimal
- `adapter-backends`, `native-faster-gs`, `native-3dgrt`, and
  `native-svraster` are opt-in family/category extras
- `svraster-adapter` is a separate opt-in for the upstream
  `new-svraster-cuda` comparison path used by the SVRaster notebook
- `cu128` and `cu130` are mutually exclusive, as are `cu128-dev` and
  `cu130-dev`
- `cu128-dev` and `cu130-dev` install everything in this monorepo’s Python
  package graph, including `splatkit-adapter-backends[all]` and
  `splatkit-native-svraster`
- `UV_CONCURRENT_BUILDS=1` forces `uv` to build packages serially, which is
  slower but avoids the very high RAM spikes from parallel native extension
  builds
- all first-party native backends are JIT-compiled on first use, so expect some
  compile time the first time you launch a viewer or render through a native
  family backend
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

If you prefer not to remember those `uv` commands, there are also optional Pixi
task wrappers for the same serial `uv` installs:

```bash
pixi run sync_cu128_dev_serial
pixi run sync_cu130_dev_serial
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
- `splatkit-adapter-backends`: a separate package containing wrappers for commonly used
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
  splatkit-adapter-backends/ Official backend adapters and wrappers
  splatkit-native-faster-gs/ FasterGS-family native backends
  splatkit-native-3dgrt/    3DGRT-family native backends
  splatkit-native-svraster/ SVRaster-family native backends
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
- `splatkit-adapter-backends`: wrappers around third-party implementations such
  as `adapter.gsplat`, `adapter.inria`, and `adapter.stoch3dgs`, with
  per-backend extras plus `all`
- `splatkit-native-faster-gs`: the FasterGS native family with
  `faster_gs.core`, `faster_gs.depth`, and `faster_gs.gaussian_pop`
- `splatkit-native-3dgrt`: the 3DGRT native family with the reusable `core`
  module and the `3dgrt.stoch3dgs` backend
- `splatkit-native-svraster`: the SVRaster native family with the reusable
  `core` module and the `svraster.core` backend; the upstream
  `new-svraster-cuda` adapter path remains a separate optional install
- `marimo-3dv`: utilities for `marimo`, desktop viewers, and auto-generated GUI
  options that work well with serializable configs

All first-party native family packages use JIT compilation for their native
extensions. The first render or viewer launch for a given native backend will
usually spend some time compiling kernels before the runtime is warm.

## Upstream Inspirations

The backend families in this repository started from existing upstream
implementations and papers. The main references are:

- GraphDeco / Inria 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- gsplat: https://docs.gsplat.studio/main/
- FasterGS: https://fhahlbohm.github.io/faster-gaussian-splatting/
- FastGS: https://fastgs.github.io/
- 3DGRT: https://gaussiantracer.github.io/
- Stoch3DGS: https://xupaya.github.io/stoch3DGS/
- SVRaster: https://svraster.github.io/
- GaussianPOP paper: https://arxiv.org/pdf/2602.06830

Typical package install paths:

```bash
pip install splatkit
pip install "splatkit[viewer]"
pip install "splatkit[all]"
pip install "splatkit-adapter-backends[gsplat]"
pip install "splatkit-adapter-backends[inria]"
pip install "splatkit-adapter-backends[stoch3dgs]"
pip install "splatkit-adapter-backends[all]"
pip install splatkit-native-faster-gs
pip install splatkit-native-3dgrt
pip install splatkit-native-svraster
```

## Supported Backends

Official adapter backends:

| Backend name | Scene type | Alpha | Depth | Normals | 2D projections | Projective intersection transforms | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `adapter.fastgs` | `GaussianScene3D` | ❌ | ❌ | ❌ | ❌ | ❌ | RGB plus backend-specific FastGS refinement signals |
| `adapter.fastergs` | `GaussianScene3D` | ❌ | ❌ | ❌ | ❌ | ❌ | RGB-only FasterGS adapter |
| `adapter.gsplat` | `GaussianScene3D` | ✅ | ✅ | ❌ | ✅ | ❌ | 3D Gaussian backend via `gsplat.rasterization` |
| `adapter.gsplat_2dgs` | `GaussianScene2D` | ✅ | ✅ | ❌ | ❌ | ✅ | 2D Gaussian backend via `gsplat.rasterization_2dgs` |
| `adapter.inria` | `GaussianScene3D` | ❌ | ✅ | ❌ | ❌ | ❌ | GraphDeco/Inria rasterizer adapter |
| `adapter.stoch3dgs` | `GaussianScene3D` | ✅ | ✅ | ❌ | ❌ | ❌ | Stochastic 3DGRT adapter |

Official native family backends:

| Backend name | Scene type | Alpha | Depth | Normals | 2D projections | Projective intersection transforms | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `faster_gs.core` | `GaussianScene3D` | ❌ | ❌ | ❌ | ❌ | ❌ | FasterGS-family native root backend |
| `faster_gs.depth` | `GaussianScene3D` | ❌ | ✅ | ❌ | ❌ | ❌ | FasterGS-family native depth backend |
| `faster_gs.gaussian_pop` | `GaussianScene3D` | ❌ | ✅ | ❌ | ❌ | ❌ | FasterGS-family native scoring backend |
| `3dgrt.stoch3dgs` | `GaussianScene3D` | ✅ | ✅ | ✅ | ❌ | ❌ | 3DGRT-family native traced backend |
| `svraster.core` | `SparseVoxelScene` | ❌ | ✅ | ❌ | ❌ | ❌ | SVRaster-family native backend |

Capability notes:
- `alpha`: per-pixel accumulated opacity/transmittance output.
- `depth`: per-pixel depth output in the backend's native shared render surface.
- `normals`: per-pixel surface or rendered normal output. This capability exists in the shared API, but no official backend exposes it yet.
- `2d_projections`: projected Gaussian centers plus compact conic coefficients via `projected_means` and `projected_conics`.
- `projective_intersection_transforms`: projected Gaussian centers plus 2DGS projective intersection geometry via `projected_means` and `projective_intersection_transforms`.

## 3DGRT / OptiX

The `adapter.stoch3dgs` and `3dgrt.stoch3dgs` backends require NVIDIA OptiX to
build the upstream 3DGRT runtime.

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
command after pulling the latest changes. In particular, the 3DGRT backends
will not compile unless the nested `optix-dev` submodule inside
`third_party/Stoch3DGS` is present.

Recommended install flow for the backend:

```bash
git submodule update --init --recursive
uv pip install -e './packages/splatkit-adapter-backends[stoch3dgs,cu130]'
uv pip install -e './packages/splatkit-native-3dgrt[cu130]'
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

`packages/splatkit`, `packages/splatkit-adapter-backends`, and the native family
packages now derive their published versions from Git tags via `hatch-vcs`.

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
pip install "git+<repo-url>@v0.1.0#subdirectory=packages/splatkit-adapter-backends"
pip install "git+<repo-url>@v0.1.0#subdirectory=packages/splatkit-native-faster-gs"
pip install "git+<repo-url>@v0.1.0#subdirectory=packages/splatkit-native-3dgrt"
pip install "git+<repo-url>@v0.1.0#subdirectory=packages/splatkit-native-svraster"
```

Both packages also expose their installed version at runtime via
`splatkit.__version__` and `splatkit_adapter_backends.__version__`.

For local monorepo development, `uv` keeps these sources editable:

- `packages/splatkit`
- `packages/splatkit-adapter-backends`
- `packages/splatkit-native-faster-gs`
- `packages/splatkit-native-3dgrt`
- `packages/splatkit-native-svraster`
- `third_party/marimo-3dv`
- `third_party/diff-gaussian-rasterization`
- `third_party/faster-gaussian-splatting/FasterGSCudaBackend`
- `third_party/sv_raster/backends/new_cuda`

The workspace root is a development environment package. Its default install is
minimal, and backend families are pulled in through opt-in extras such as
`adapter-backends`, `native-faster-gs`, `native-3dgrt`, `native-svraster`,
`all-unrestricted`, and `cu130-dev`.

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
| Core contracts and traits | Typed scene/camera contracts, capability traits, backend registry, shared render surface across `adapter.inria`, `adapter.gsplat`, and other registered backends | Done |
| Viewer workflow | Viewer works well in `marimo`, plus an experimental desktop viewer that reuses marimo-defined GUI elements | In progress |
| Notebook to script workflow | Define custom GUI elements in a notebook, then run the notebook as a Python file for a more interactive local viewer flow | In progress |
| Backend wrappers | Current adapter backends for `adapter.inria`, `adapter.gsplat`, `adapter.fastergs`, and `adapter.stoch3dgs`, with broader training-oriented wrappers still planned | In progress |
| Hackable backend implementations | Modular native families for `faster_gs.*`, `3dgrt.*`, and `svraster.*`, aimed at rapid experimentation and easier modification while minimizing performance cost | In progress |
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
