# `core/native`

Vendored SVRaster native sources used by the staged `core.runtime`.

This is intentionally a render-focused subset of the upstream `new_cuda`
backend:

- preprocess
- raster forward/backward
- spherical-harmonic evaluation
- trilinear gather helpers
- sparse-voxel utility kernels

Training-only kernels such as optimizer and TV helpers are intentionally not
included here.
