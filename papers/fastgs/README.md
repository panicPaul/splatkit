# FastGS

Paper notebook for:

- FastGS: Training 3D Gaussian Splatting in 100 Seconds

Paper links:

- Project page: https://fastgs.github.io/
- arXiv: https://arxiv.org/abs/2511.04283

## Abstract

FastGS is a general acceleration framework for 3D Gaussian Splatting training.
It uses multi-view consistent densification and targeted pruning to control
primitive growth, plus rasterization-oriented changes such as compact boxes, to
reduce redundant training cost while preserving reconstruction quality.

This notebook targets the FastGS reconstruction path and keeps paper-specific
densification and pruning logic in the notebook so it can stress-test the
current training API.

## Notebook

Primary artifact:

- `papers/fastgs/notebook.py`

Run interactively:

```bash
uv run marimo run papers/fastgs/notebook.py
```

The notebook is also the Python config/module surface used by script mode:

- `papers/fastgs/notebook.py`

Default JSON configs:

- `garden_base`
- `garden_base_native`
- `garden_big`
- `garden_debug_val`

Stored at:

- `papers/fastgs/defaults/garden_base.json`
- `papers/fastgs/defaults/garden_base_native.json`
- `papers/fastgs/defaults/garden_big.json`
- `papers/fastgs/defaults/garden_debug_val.json`

Backend choices:

- `adapter.fastgs`
- `faster_gs.fastgs`

Each successful run writes the resolved training artifact directory configured by
`checkpoint.output_dir`, including:

- `config.json`
- `metadata.json`
- `model.ckpt`
- `scene.ply` when `checkpoint.export_ply=true`
