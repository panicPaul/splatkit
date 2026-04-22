# `runtime/ops`

This directory contains the torch custom-op registration layer for
`faster_gs_native`.

## Purpose

Each file here does one thing:

- declare raw custom ops such as `preprocess_fwd` or `blend_bwd`
- register fake implementations for FakeTensor and `torch.compile`
- register autograd only on the combined user-facing ops

This layer should stay thin. It should not contain backend adapter logic or CUDA implementation details.

## Files

- [`preprocess.py`](./preprocess.py)
  Registers `preprocess_fwd`, `preprocess_bwd`, and `preprocess`.
- [`sort.py`](./sort.py)
  Registers `sort_fwd` and public non-differentiable `sort`.
- [`blend.py`](./blend.py)
  Registers `blend_fwd`, `blend_bwd`, and `blend`.
- [`render.py`](./render.py)
  Registers `render_fwd`, `render_bwd`, and `render` by composing the stage ops.
- [`_common.py`](./_common.py)
  Small shared helpers such as extension access and stage constants.

## Op pattern

The naming pattern is:

- `*_fwd`
  Raw forward op with no autograd registration.
- `*_bwd`
  Raw backward op with no autograd registration.
- stage name without suffix
  Public op. Only these combined ops carry autograd where it makes sense.

Examples:

- `preprocess_fwd`
- `preprocess_bwd`
- `preprocess`

- `blend_fwd`
- `blend_bwd`
- `blend`

- `render_fwd`
- `render_bwd`
- `render`

`sort` is intentionally public but non-differentiable.

## What to change here vs elsewhere

- Change this directory if you need to alter op schemas, fake implementations, or autograd wiring.
- Change [`../packing.py`](../packing.py) if you need to alter tuple layout handling.
- Change [`../../native`](../../native) if you need to alter the actual CUDA/C++ implementation.

