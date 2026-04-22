# `faster_gs_depth_native.runtime`

This runtime layer exposes the depth-aware staged API for the backend.

## What lives here

- [`__init__.py`](./__init__.py)
  Public runtime entrypoints.
- [`blend.py`](./blend.py)
  Depth-aware blend custom-op registration and autograd glue.
- [`render.py`](./render.py)
  Full render custom-op registration built from reused FasterGS stages plus this
  backend's blend pair.
- [`packing.py`](./packing.py)
  Small helpers that turn raw custom-op outputs into typed Python result objects.
- [`types.py`](./types.py)
  Result containers for the depth backend.
- [`_extension.py`](./_extension.py)
  Lazy JIT loader for the depth-specific native extension.

## Reuse boundary

This backend does not implement preprocess or sort again. Its render glue imports those raw
ops from `faster_gs_native.reuse` and combines them with the depth backend's own
`blend_fwd` and `blend_bwd`.

That is the main architectural point of this package: a new backend can stay small when it
only changes one stage.
