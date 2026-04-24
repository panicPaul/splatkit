# `native`

This directory contains the native extension for `faster_gs`.

It is the lowest layer of the backend and is responsible for calling the actual FasterGS CUDA implementation.

## What lives here

- [`bindings.cpp`](./bindings.cpp)
  Pybind module definition that exposes the native stage wrappers to Python.
- [`rasterization/`](./rasterization)
  Stage wrappers plus vendored FasterGS rasterization code and headers.
- [`utils/`](./utils)
  Small helper headers copied from the FasterGS codebase.

## Relationship to the Python runtime

The Python runtime does not call CUDA kernels directly.

Instead:

1. `runtime/_extension.py` JIT-builds this extension.
2. `runtime/ops/*.py` calls the exposed pybind functions.
3. The pybind functions call the stage wrappers declared in `rasterization/src/stages.h`.
4. The stage wrappers launch the vendored FasterGS kernels.

## Design constraints

- The public backend API should not expose native buffer management.
- The native layer can use reusable scratch buffers internally.
- The vendored FasterGS kernel code is kept close to upstream where possible.

## Where to start

If you are trying to understand the native path, read:

1. [`bindings.cpp`](./bindings.cpp)
2. [`rasterization/README.md`](./rasterization/README.md)

