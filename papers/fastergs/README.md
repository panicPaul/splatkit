# FasterGS

Paper notebook for:

- Faster-GS: Analyzing and Improving Gaussian Splatting Optimization

Paper links:

- Project page: https://fhahlbohm.github.io/faster-gaussian-splatting/
- arXiv: https://arxiv.org/abs/2602.09999
- PDF: https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf

## Abstract

Recent advances in 3D Gaussian Splatting (3DGS) have focused on accelerating optimization while preserving reconstruction quality. However, many proposed methods entangle implementation-level improvements with fundamental algorithmic modifications or trade performance for fidelity, leading to a fragmented research landscape that complicates fair comparison.

In this work, we consolidate and evaluate the most effective and broadly applicable strategies from prior 3DGS research and augment them with several novel optimizations. We further investigate underexplored aspects of the framework, including numerical stability, Gaussian truncation, and gradient approximation. The resulting system, Faster-GS, provides a rigorously optimized algorithm that we evaluate across a comprehensive suite of benchmarks.

Our experiments demonstrate that Faster-GS achieves up to 5× faster training while maintaining visual quality, establishing a new cost-effective and resource efficient baseline for 3DGS optimization. Furthermore, we demonstrate that optimizations can be applied to 4D Gaussian reconstruction, leading to efficient non-rigid scene optimization.

## Notebook

Primary artifact:

- `papers/fastergs/notebook.py`

Run interactively:

```bash
uv run marimo run papers/fastergs/notebook.py
```

Run in script mode with a named default JSON config plus CLI overrides:

```bash
uv run papers/fastergs/notebook.py cli \
  --preset garden_baseline \
  --scene.path /path/to/mipnerf360/garden \
  --execution.max-steps 100 \
  --backend faster_gs.core
```

Replay an exported resolved config:

```bash
uv run papers/fastergs/notebook.py json checkpoints/papers/fastergs/garden_baseline/adapter.fastergs/config.json
```

Python config module:

- `papers/fastergs/config.py`

Default JSON configs:

- `garden_baseline`
- `garden_mcmc`

Stored at:

- `papers/fastergs/defaults/garden_baseline.json`
- `papers/fastergs/defaults/garden_mcmc.json`

Backend choices:

- `adapter.fastergs`
- `faster_gs.core`

Each successful run writes the resolved training artifact directory configured by
`checkpoint.output_dir`, including:

- `config.json`
- `metadata.json`
- `model.ckpt`
- `scene.ply` when `checkpoint.export_ply=true`
