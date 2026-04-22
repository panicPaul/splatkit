# `faster_gs_native`

This package is the native FasterGS backend inside the `splatkit` ecosystem.

It has two jobs:
- expose a normal splatkit backend entrypoint via [`renderer.py`](./renderer.py)
- expose a staged native runtime via [`runtime/`](./runtime) built on torch custom ops and vendored CUDA/C++ code in [`native/`](./native)

## Mental model

If you are new to the codebase, read the package in this order:

1. [`renderer.py`](./renderer.py)
   This is the splatkit-facing adapter. It validates inputs, loops over cameras, and calls the runtime.
2. [`runtime/README.md`](./runtime/README.md)
   This explains the Python runtime layer: staged functions, custom ops, result types, and extension loading.
3. [`native/README.md`](./native/README.md)
   This explains the C++/CUDA layer and how the vendored FasterGS code is wrapped.

## Layer boundaries

- `renderer.py`
  Thin backend adapter. This is where splatkit contracts like `GaussianScene3D` and `CameraState` are handled.
- `runtime/`
  Python runtime layer. This is where staged APIs such as `preprocess`, `sort`, `blend`, and `render` live.
- `native/`
  Native extension layer. This is where the actual CUDA work happens.

## Public API

From Python, the main entrypoints are:

- `register()`
- `render_faster_gs_native(...)`
- `runtime.preprocess(...)`
- `runtime.sort(...)`
- `runtime.blend(...)`
- `runtime.render(...)`

The raw torch custom ops are registered under the `faster_gs_native::` namespace and live under [`runtime/ops`](./runtime/ops).

## Design goals

- Keep the backend API splatkit-native.
- Keep buffers and scratch allocation hidden behind the custom-op boundary.
- Keep preprocess, sort, and blend modular so future native backends can copy this structure.
- Preserve parity with the FasterGS CUDA backend while keeping the code readable enough to modify.

## Reference backend note

The external `FasterGSCudaBackend` is still used as the image/gradient reference in tests.
Its backward path is not safe to interleave with unrelated CUDA autograd work inside one
long-lived Python process: after one reference backward, the next CUDA backward in that
process can fail even if it does not use `splatkit` at all.

Because of that, reference comparisons are limited to sequences that keep the external
backend at the end of the CUDA work they perform. The native backends in this package
should not rely on the external backend for process-local coexistence guarantees.
