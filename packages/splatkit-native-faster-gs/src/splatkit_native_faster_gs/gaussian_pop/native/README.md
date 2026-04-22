# `gaussian_pop.native`

This directory contains the native helper code specific to the GaussianPOP
backend.

## What is native here

- [`pop_blend.cu`](./pop_blend.cu)
  CUDA wrapper code for the GaussianPOP score accumulation kernel.

## What is intentionally reused

This backend does not vendor another full FasterGS rasterizer. It reuses the
root FasterGS native core for preprocess, sort, and RGB blend, reuses the
depth-native helper for expected depth, and adds only the native blend-side
forward helper it needs for `gaussian_impact_score`.
