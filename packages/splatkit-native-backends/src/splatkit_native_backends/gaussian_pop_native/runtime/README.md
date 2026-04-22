# `runtime`

This directory contains the Python runtime layer for `gaussian_pop_native`.

This backend adds a small native helper for the GaussianPOP score while
reusing the existing FasterGS native stages for preprocess, sort, RGB blend,
and expected depth.

1. reuses the root-owned staged custom ops through the `reuse` surfaces
2. owns the GaussianPOP blend forward stage in Python
3. computes `gaussian_impact_score` inside that blend stage via a backend-owned
   CUDA helper

## What lives here

- [`blend.py`](./blend.py)
  Backend-owned blend stage. It reuses the existing native blend ops and adds
  the GaussianPOP score in forward.
- [`render.py`](./render.py)
  Composes preprocess, sort, and the backend-owned blend stage.
- [`_extension.py`](./_extension.py)
  JIT loader for the GaussianPOP native helper extension.
- [`types.py`](./types.py)
  Structured runtime result types.

## Design intent

- Preserve FasterGS RGB and expected-depth gradients by reusing the existing
  autograd-enabled staged ops unchanged.
- Keep `gaussian_impact_score` out of autograd.
- Reuse root-owned stages directly instead of copying or re-exporting them.
- Keep the GaussianPOP-specific native code limited to the score computation.
