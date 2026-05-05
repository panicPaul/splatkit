# SVRaster Backend Gaps

The notebook uses the native Ember `svraster.core` render backend and the
`ember-svraster-training` package for training-only CUDA utilities.

## Covered

- Sparse Adam is exposed through `ember_svraster_training.SVRasterSparseAdam`.
- Total-variation density gradient accumulation is exposed through
  `ember_svraster_training.SVRasterTVDensityHook`.
- These utilities are not registered as render backends.

## Still Paper-Local

- Adaptive prune/subdivide scheduling is kept in the notebook to stress-test
  the training API.
- The notebook initializer provides a simple sparse-voxel grid bootstrap instead
  of a full upstream scene-specific outside-voxel heuristic.

## Open Native Gaps

- The upstream training rasterizer exposes additional loss/priority outputs
  such as concentration, ascending, distortion, max-weight tracking, and
  subdivision priority. The current render backend wrapper does not expose that
  full training surface yet.
- Faithful subdivision priority and pruning by tracked training statistics need
  the training rasterizer outputs before they should be moved out of the
  notebook.
- No Torch fallbacks should be added for kernels that exist in the vendored
  CUDA backend.
