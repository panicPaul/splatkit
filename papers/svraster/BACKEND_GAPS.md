# SVRaster Backend Gaps

The notebook uses the native Ember `svraster.core` render backend and the
`ember-svraster-training` package for training-only CUDA utilities.

## Covered

- Sparse Adam is exposed through `ember_svraster_training.SVRasterSparseAdam`.
- Total-variation density gradient accumulation is exposed through
  `ember_svraster_training.SVRasterTVDensityHook`.
- The native `svraster.core` render path exposes the training-time outputs and
  gradients used by the current paper notebook, including transmittance,
  max-weight tracking, distortion, ascending, color concentration, and
  subdivision-priority gradients.
- These utilities are not registered as render backends.

## Still Paper-Local

- Adaptive prune/subdivide scheduling remains a paper-training utility in
  `ember-svraster-training` instead of the core render backend.
- Paper initialization, loss assembly, optimizer grouping, and regularization
  live in `ember-svraster-training` so the notebook can stay declarative while
  retaining paper-specific behavior.

## Open Native Gaps

- The current integration targets the `new_cuda` SVRaster backend path. Other
  upstream variants are not wrapped as Ember backends.
- SVRaster training helpers are intentionally packaged outside `ember-core`
  until more sparse-voxel papers prove which abstractions should be shared.
- No Torch fallbacks should be added for kernels that exist in the vendored
  CUDA backend.
