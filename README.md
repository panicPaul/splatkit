# splatkit

Research monorepo for fast, typed, reproducible experimentation around the 3D
Gaussian Splatting family and closely related explicit inverse-graphics
methods.

The core idea is to make orthogonal research choices easy to recombine:
primitive type, backend, initialization, densification, data preparation,
training loop, and pre/post-processing should be explicit pieces rather than
one hidden monolith. See [NORTH_STAR.md](NORTH_STAR.md) for the longer design
constitution.

## Status

Pre-alpha. APIs may change when a better boundary becomes clear.

This repo is currently useful for:

- shared `splatkit` scene, camera, render, data, and training contracts
- adapter backends around existing third-party rasterizers
- first-party native backend families for hackable kernel/runtime work
- marimo-first paper demos, viewers, and training notebooks
- local CUDA/PyTorch environments managed primarily with `uv`, with Pixi as a
  fallback when system CUDA/toolchain management matters

## Install

From a fresh checkout:

```bash
git submodule update --init --recursive
uv venv
uv sync
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

Install the first-party native backend families plus Gaussian training helpers:

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

## Backend Packages

There are two backend categories.

**Adapter backends** wrap external implementations behind the shared `splatkit`
contracts. They are primarily for comparison, interoperability, and reuse of
existing paper code.

```bash
uv sync --extra adapter-backends --extra cu130
```

Package: `packages/splatkit-adapter-backends`

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

- `packages/splatkit-native-faster-gs`
- `packages/splatkit-native-faster-gs-mojo`
- `packages/splatkit-native-3dgrt`
- `packages/splatkit-native-svraster`

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
  splatkit/                     Core contracts, traits, registry, data helpers
  splatkit-adapter-backends/    Official wrappers around third-party backends
  splatkit-native-faster-gs/    First-party FasterGS-family native backends
  splatkit-native-faster-gs-mojo/ Mojo-backed FasterGS experiments
  splatkit-native-3dgrt/        First-party 3DGRT-family native backend
  splatkit-native-svraster/     First-party SVRaster-family native backend
  splatkit-gaussian-training/   Optional Gaussian-specific training utilities
  marimo-config-gui/            Pydantic-driven config UI helpers for marimo

papers/
  fastergs/                     Paper-specific notebook/config work

notebooks/                      Local demos, viewers, and research notebooks
tests/                          Cross-package tests and backend parity checks
third_party/                    Upstream projects and editable submodules
```

The root `pyproject.toml` is a development environment, not the main library
package. The library package is `packages/splatkit`.

## 3DGRT / OptiX

`3dgrt.stoch3dgs` uses OptiX. The header-only `optix-dev` subset needed to
compile the native package is vendored in this repository at:

```text
packages/splatkit-native-3dgrt/src/splatkit_native_3dgrt/core/native/stoch3dgs/dependencies/optix-dev/include/optix.h
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
apptainer exec --nv dist/splatkit-cu130.sif python -c "import torch"
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

Git pinning examples:

```bash
pip install "git+<repo-url>@v0.0.1#subdirectory=packages/splatkit"
pip install "git+<repo-url>@v0.0.1#subdirectory=packages/splatkit-adapter-backends"
pip install "git+<repo-url>@v0.0.1#subdirectory=packages/splatkit-native-faster-gs"
```

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
