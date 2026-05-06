# Scaffold-GS

Paper notebook for **Scaffold-GS: Structured 3D Gaussians for View-Adaptive
Rendering**.

The implementation keeps the paper-specific model in
[`notebook.py`](notebook.py). It expands anchor features into view-adaptive
direct-RGB Gaussians with notebook-local MLP heads, then renders the generated
Gaussians through the native `faster_gs.core` backend using its
`color_source="direct_rgb"` mode.

## Presets

- `garden_scaffold_gs`: paper-faithful Mip-NeRF 360 garden training preset.
- `garden_debug_val`: smaller validation preset for notebook iteration.

Run the notebook as a script with a preset:

```bash
marimo run papers/scaffold_gs/notebook.py -- --preset garden_scaffold_gs
```

Or override fields from the command line:

```bash
marimo run papers/scaffold_gs/notebook.py -- \
  --preset garden_scaffold_gs \
  --training.optimization.iterations 5000
```
