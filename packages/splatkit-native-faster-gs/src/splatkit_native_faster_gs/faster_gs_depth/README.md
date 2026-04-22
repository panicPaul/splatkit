# `faster_gs_depth`

This package is a depth-aware FasterGS-derived backend built on top of
[`faster_gs`](../faster_gs).

## Upstream Reference

- FasterGS: https://fhahlbohm.github.io/faster-gaussian-splatting/

It exists as the first proof that a native backend can reuse raw stages from another
backend, override only the stages it changes, and then re-glue a full `render` op.

## Mental model

If you are new to this backend, read it in this order:

1. [`renderer.py`](./renderer.py)
   This is the splatkit-facing adapter. It handles backend options and packages RGB plus
   depth outputs.
2. [`runtime/README.md`](./runtime/README.md)
   This explains the Python runtime layer and how the depth backend reuses FasterGS
   preprocess/sort while overriding only the blend stage.
3. [`reuse/README.md`](./reuse/README.md)
   This explains which raw ops this backend owns and which ones it intentionally does not
   re-export.
4. [`native/README.md`](./native/README.md)
   This explains the native extension that implements the depth-specific blend kernels.

## Design intent

- Reuse `faster_gs` preprocess and sort without copying them.
- Override only the blend pair because this backend needs extra depth outputs and depth
  gradients.
- Keep the public backend API splatkit-native while keeping the native stage boundaries
  explicit.

## Public API

From Python, the main entrypoints are:

- `register()`
- `render_faster_gs_depth(...)`
- `runtime.blend(...)`
- `runtime.render(...)`

This package only exports the raw stage ops it actually owns plus the fully glued render
surface. Reused raw ops stay owned by `faster_gs`.
