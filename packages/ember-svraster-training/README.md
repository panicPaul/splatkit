# `ember-svraster-training`

Optional SVRaster training utilities for `ember-core`.

This package contains reusable sparse-voxel training add-ons, including:

- the SVRaster sparse Adam optimizer wrapper
- native total-variation density gradient helpers
- SVRaster optimization recipes

The CUDA kernels are provided by `ember-native-svraster.training`. They are
separate from the registered `svraster.core` render backend so training-only
behavior does not become a render backend concern.

## Install

```bash
pip install ember-svraster-training
```

CUDA-specific torch wheels should be selected through the same environment
strategy used for the rest of `ember-core`.

## License

This package carries the upstream SVRaster licensing surface.
