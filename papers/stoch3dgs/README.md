# Stoch3DGS

Paper notebook for:

- Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting

Paper links:

- Project page: https://xupaya.github.io/stoch3DGS/
- arXiv: https://arxiv.org/abs/2603.23637

## Abstract

Stoch3DGS accelerates 3D Gaussian Splatting reconstruction by replacing the
deterministic sorted ray-tracing pass used by 3DGRT with stochastic ray tracing
for training. The method uses stochastic sampling in the forward and backward
passes and reports faster reconstruction on MipNeRF360 with modest quality
tradeoffs.

This notebook targets the standard reconstruction path from the paper using the
native Ember `3dgrt.stoch3dgs` backend. Paper-specific initialization,
progressive SH scheduling, densification, pruning, and cleanup logic live in the
notebook so the implementation can stress-test the current training API.

## Notebook

Primary artifact:

- `papers/stoch3dgs/notebook.py`

Run interactively:

```bash
uv run marimo run papers/stoch3dgs/notebook.py
```

The notebook is also the Python config/module surface used by script mode:

- `papers/stoch3dgs/notebook.py`

Default JSON configs:

- `garden_stoch`
- `garden_debug_val`

Stored at:

- `papers/stoch3dgs/defaults/garden_stoch.json`
- `papers/stoch3dgs/defaults/garden_debug_val.json`

Backend choice:

- `3dgrt.stoch3dgs`

Each successful run writes the resolved training artifact directory configured by
`checkpoint.output_dir`, including:

- `config.json`
- `metadata.json`
- `model.ckpt`
- `scene.ply` when `checkpoint.export_ply=true`

The native backend requires the vendored Stoch3DGS/OptiX extension stack. The
notebook intentionally does not provide Torch fallbacks for kernels that are
implemented by the native backend.
