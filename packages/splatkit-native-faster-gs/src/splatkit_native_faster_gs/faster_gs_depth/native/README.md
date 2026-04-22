# `faster_gs_depth.native`

This directory contains the native extension code that is specific to the depth backend.

## What is native here

- [`depth_blend.cu`](./depth_blend.cu)
  CUDA wrapper code for the depth-aware blend forward and backward paths.

## What is intentionally reused

This backend does not vendor another full FasterGS rasterizer. It reuses the root FasterGS
native core for preprocess and sort, and only adds the blend-side native code it needs for
depth output and depth gradients.

That keeps the override backend small and makes the stage replacement boundary easy to
follow.
