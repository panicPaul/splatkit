# `native/rasterization`

This directory contains the native rasterization wrappers and the vendored
FasterGS rasterization headers.

## Layout

- [`include/`](./include)
  Vendored FasterGS headers and CUDA helper files.
- [`src/`](./src)
  Thin splatkit-owned wrapper code that validates torch tensors, manages
  scratch buffers, and launches the vendored kernels.

## Important distinction

This directory mixes two kinds of code:

- vendored FasterGS implementation code in `include/`
- local wrapper/orchestration code in `src/`

The goal is to keep local changes concentrated in `src/` so kernel logic stays easy to compare against upstream FasterGS.

## Stage wrappers

The main native entrypoints are declared in [`src/stages.h`](./src/stages.h):

- `preprocess_fwd_wrapper`
- `preprocess_bwd_wrapper`
- `sort_fwd_wrapper`
- `blend_fwd_wrapper`
- `blend_bwd_wrapper`

Their implementations are split by stage:

- [`src/preprocess.cu`](./src/preprocess.cu)
- [`src/sort.cu`](./src/sort.cu)
- [`src/blend.cu`](./src/blend.cu)

## Why there is still a single `stages.cu`

[`src/stages.cu`](./src/stages.cu) is an aggregator translation unit that
includes the stage `.cu` files.

That exists because the vendored FasterGS `.cuh` files contain concrete kernel
definitions, not only declarations. Compiling each wrapper as its own CUDA
translation unit would therefore create duplicate symbols at link time.

So the code is split for readability, but compiled as one CUDA translation
unit for correctness.

## Scratch buffers

Reusable temporary buffers are managed in [`src/common.h`](./src/common.h).

Those scratch tensors are:

- private to the native layer
- keyed by label/device/dtype/stream
- used to reduce repeated large temporary allocations during rendering

They are intentionally not part of the public Python or splatkit API.

