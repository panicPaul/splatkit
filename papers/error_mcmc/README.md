# Error-MCMC

Experimental paper notebook for error-guided MCMC relocation in 3D Gaussian
Splatting.

The method keeps the useful MCMC behavior of recycling dead primitives and
growing under a primitive cap, but changes the source distribution. Instead of
teleporting from opacity alone, it samples source primitives from FastGS-style
high-error attribution. The default score uses a top-k mean across probe views
so background regions that are only visible in a few cameras can still drive
relocation instead of being averaged away.

## Notebook

Primary artifact:

- `papers/error_mcmc/notebook.py`

Run interactively:

```bash
uv run marimo run papers/error_mcmc/notebook.py
```

Default JSON configs:

- `garden_error_mcmc`
- `garden_debug_val`

Stored at:

- `papers/error_mcmc/defaults/garden_error_mcmc.json`
- `papers/error_mcmc/defaults/garden_debug_val.json`

Backend choice:

- `faster_gs.fastgs`

Each successful run writes the resolved training artifact directory configured
by `checkpoint.output_dir`, including:

- `config.json`
- `metadata.json`
- `model.ckpt`
- `scene.ply` when `checkpoint.export_ply=true`
