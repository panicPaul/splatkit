# `gaussian_pop`

This package is a GaussianPOP-style FasterGS backend that reuses the existing
native FasterGS stages, overrides the blend forward stage, and adds a
forward-only `gaussian_impact_score`.

It keeps RGB and expected-depth behavior aligned with the existing native
backends and computes the GaussianPOP score as a per-camera, per-Gaussian
leave-one-out-style squared render change estimate.

## Mental model

If you are new to this backend, read it in this order:

1. [`renderer.py`](./renderer.py)
   This is the splatkit-facing adapter. It handles output selection and
   delegates to the existing FasterGS backends when no score is requested.
2. [`runtime/README.md`](./runtime/README.md)
   This explains the Python runtime layer that reuses root-owned staged ops and
   owns the GaussianPOP blend forward stage.
3. [`reuse/README.md`](./reuse/README.md)
   This explains the stage ownership rule for this backend.

## Reuse Table

| Surface | Owned here? | Reused from | Notes |
| --- | --- | --- | --- |
| Backend adapter | Yes | `gaussian_pop.renderer` | Public `render(...)` entrypoint and output packaging |
| Preprocess stage | No | `faster_gs.reuse` | Reuse the root FasterGS preprocess op directly |
| Sort stage | No | `faster_gs.reuse` | Reuse the root FasterGS sort op directly |
| Blend forward stage | Yes | `gaussian_pop.runtime.blend` | Backend-owned blend stage that reuses root/depth blend ops and adds the GaussianPOP score |
| RGB blend kernels | No | `faster_gs.reuse` | Root-owned RGB blend op reused inside the GaussianPOP blend stage |
| Expected-depth blend kernels | No | `faster_gs_depth.reuse` | Depth-native blend op reused inside the GaussianPOP blend stage |
| Gaussian impact score | Yes | `gaussian_pop.runtime.blend` | Forward-only score computed inside the backend-owned blend stage |
| Native score helper | Yes | `gaussian_pop.native` | Backend-owned CUDA helper for GaussianPOP score accumulation |

## Public API

From Python, the main entrypoints are:

- `register()`
- `render_gaussian_pop(...)`
- `runtime.blend(...)`
- `runtime.render(...)`

## Score semantics

`gaussian_impact_score` is backend-defined in the shared contract. In this
backend it is the per-camera, per-Gaussian accumulated squared RGB change
predicted by removing that Gaussian from the render order, using the
GaussianPOP analytic criterion.
