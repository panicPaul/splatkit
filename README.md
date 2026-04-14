# splatkit Monorepo

Monorepo for `splatkit`, the core contracts and official backend adapters for
modular Gaussian splatting.

## Layout

```text
packages/
  splatkit/                Core package and shared contracts
  splatkit-backends/       Official backend adapters
third_party/
  marimo-3dv/              Viewer dependency
  diff-gaussian-rasterization/  Upstream Inria rasterizer
patches/
  diff-gaussian-rasterization/  Local third-party patches
tests/                     Cross-package tests
notebooks/                 Monorepo-level examples
scripts/                   Bootstrap and verification helpers
```

## Install

For local development from the monorepo root:

```bash
uv sync
./scripts/bootstrap.sh
```

This workspace keeps local editable sources pointed at:

- `packages/splatkit`
- `packages/splatkit-backends`
- `third_party/marimo-3dv`
- `third_party/diff-gaussian-rasterization`

## Packages

- `splatkit`: backend-agnostic contracts, capabilities, registry, and generic
  render wrapper
- `splatkit-backends`: official adapters such as `gsplat` and `inria`

Typical install paths:

```bash
pip install splatkit
pip install "splatkit-backends[gsplat]"
pip install "splatkit-backends[inria]"
```
