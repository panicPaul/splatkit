# `nht_rasterizer`

This directory contains the vendored native implementation used by
`nht.3dgut`.

The split mirrors the staged native backends elsewhere in the repo:

- `upstream/` contains vendored CUDA/C++ source files.
- `bindings.cpp` exposes only the stage functions used by Ember.
- generated build directories are written under `build/generated/`.

Runtime Python code must not import a copied upstream Python package from this
directory. It should go through the staged runtime and this local extension.
