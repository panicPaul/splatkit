# SVRaster

Paper notebook for:

- SVRaster: Efficient 3D Gaussian Splatting with Sparse Voxels

Paper links:

- Project page: https://svraster.github.io/
- arXiv: https://arxiv.org/abs/2405.15307

## Abstract

SVRaster replaces the unstructured Gaussian primitive set with an adaptive
sparse-voxel representation. The method trains voxel geometry and color
features directly, prunes weak voxels, subdivides high-priority voxels, and uses
custom CUDA training utilities such as sparse Adam and total-variation density
regularization.

This notebook targets the standard sparse-voxel reconstruction path through the
native Ember `svraster.core` render backend. Paper-specific scheduling,
initialization, pruning, subdivision, and experiment defaults live in the
notebook so the implementation can stress-test the training API.

## Notebook

Primary artifact:

- `papers/svraster/notebook.py`

Run interactively:

```bash
uv run marimo run papers/svraster/notebook.py
```

The notebook is also the Python config/module surface used by script mode:

- `papers/svraster/notebook.py`

Default JSON configs:

- `garden_svraster`
- `garden_fast_train`
- `garden_debug_val`

Stored at:

- `papers/svraster/defaults/garden_svraster.json`
- `papers/svraster/defaults/garden_fast_train.json`
- `papers/svraster/defaults/garden_debug_val.json`

Backend choice:

- `svraster.core`

Training utilities:

- `ember-svraster-training`
- `ember_native_svraster.training`

Each successful run writes the resolved training artifact directory configured by
`checkpoint.output_dir`, including:

- `config.json`
- `metadata.json`
- `model.ckpt`
- `scene.ply` when `checkpoint.export_ply=true`

The native backend requires the vendored SVRaster CUDA extension stack. The
notebook intentionally does not provide Torch fallbacks for kernels that are
implemented by the native training/runtime packages.
