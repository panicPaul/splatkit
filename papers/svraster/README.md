# SVRaster

Paper notebook for:

- Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering

Paper links:

- Project page: https://svraster.github.io/
- arXiv: https://arxiv.org/abs/2412.04459

## Abstract

SVRaster replaces the unstructured Gaussian primitive set with an adaptive
sparse-voxel representation. The method trains voxel geometry and color
features directly, prunes weak voxels, subdivides high-priority voxels, and uses
custom CUDA training utilities such as sparse Adam and total-variation density
regularization.

This notebook targets the standard sparse-voxel reconstruction path through the
native Ember `svraster.core` render backend. Paper-specific scheduling,
initialization, pruning, subdivision, optimization, and loss helpers live in
`ember-svraster-training` so the notebook stays aligned with the current
FastGS/FasterGS/Stoch3DGS paper-notebook flow.

## Notebook

Primary artifact:

- `papers/svraster/notebook.py`

Run interactively:

```bash
uv run marimo run papers/svraster/notebook.py
```

The notebook includes the current paper training controls: preset selection,
config editing, prepare/train/stop buttons, live viewer output, and compact
training status with throughput and ETA. It is also the Python config/module
surface used by script mode:

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

Data loading follows the sibling paper notebooks: resized training images can
be cached under `<scene.path>/ember_cache/resized_images/...`, and prepared
frame materialization defaults to eager prepared samples with worker threads.

Each successful run writes the resolved training artifact directory configured by
`checkpoint.output_dir`, including:

- `config.json`
- `metadata.json`
- `model.ckpt`
- `scene.ply` when `checkpoint.export_ply=true`

The native backend requires the vendored SVRaster CUDA extension stack. The
notebook intentionally does not provide Torch fallbacks for kernels that are
implemented by the native training/runtime packages.
