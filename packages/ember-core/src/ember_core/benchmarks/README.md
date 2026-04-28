# `ember_core.benchmarks`

Small benchmark helpers for smoke-level performance checks.

Current scope:
- COLMAP dataloader timing
- Gaussian render timing from deterministic point-cloud initialization
- Gaussian render forward/backward timing for backend regression checks

Current non-goals:
- representative paper-quality performance claims
- large-scene benchmark orchestration

Defaults:
- use the bundled `bicycle_smoke` COLMAP sample scene
- allow overriding the scene with `--colmap-root` or `EMBER_COLMAP_ROOT`

CLI entry points:

```bash
python -m ember_core.benchmarks.dataloader
python -m ember_core.benchmarks.render
```

Examples:

```bash
python -m ember_core.benchmarks.dataloader --measured-steps 100
python -m ember_core.benchmarks.render --backend adapter.gsplat --device cuda
python -m ember_core.benchmarks.render --colmap-root /path/to/scene
python -m ember_core.benchmarks.ply_render --include-backward --device cuda
```

Notes:
- render benchmarks report first-frame timing separately from warm timings
- `--include-backward` reports forward and backward timings separately
- CUDA render timings synchronize before and after each measured render
- the bundled sample is meant for regression smoke tests and notebook defaults,
  not for realistic end-to-end performance comparison
