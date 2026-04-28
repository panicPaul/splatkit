# `core/runtime`

Staged SVRaster runtime for the native `svraster.core` backend.

The public stages are:

- `preprocess`
- `sh_eval`
- `rasterize`
- `render`

The runtime also exposes trilinear gather helpers and sparse-voxel utility
wrappers used by the broader `ember-core` sparse-voxel family.
