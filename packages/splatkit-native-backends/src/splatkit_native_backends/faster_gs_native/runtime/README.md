# `runtime`

This directory contains the Python runtime layer for `faster_gs_native`.

It sits between the splatkit adapter and the native C++/CUDA extension.

## What lives here

- [`__init__.py`](./__init__.py)
  Public staged Python API.
- [`types.py`](./types.py)
  Structured stage result objects returned by the public runtime wrappers.
- [`packing.py`](./packing.py)
  Helpers that convert between raw custom-op tensor tuples and structured Python results.
- [`_extension.py`](./_extension.py)
  JIT loading for the native extension.
- [`ops/`](./ops)
  Torch custom-op registration and autograd wiring.

## How the runtime is organized

There are three stage groups plus a combined render path:

- `preprocess`
  Projects primitives into screen-space and produces the tensors needed by sort and backward.
- `sort`
  Expands visible primitives into tile instances and sorts them into blend order.
- `blend`
  Produces the final image and the tensors needed by blend backward.
- `render`
  Composes `preprocess -> sort -> blend` in forward and `blend_bwd -> preprocess_bwd` in backward.

## Public surface vs internal surface

The public runtime functions are:

- `preprocess(...)`
- `sort(...)`
- `blend(...)`
- `render(...)`

These return typed Python objects from [`types.py`](./types.py).

The raw torch custom ops live in [`ops/`](./ops) and operate on flattened tensor tuples. Those are the lower-level building blocks used by the public runtime wrappers and by autograd registration.

## Why both wrappers and raw ops exist

- The raw ops are the stable native building blocks.
- The Python wrappers make the stage boundaries readable and easier to use from tests and experiments.
- The wrapper layer is where tuple packing/unpacking is kept out of the rest of the code.

## Where backward state lives

Backward state is carried explicitly through the combined custom-op outputs and `setup_context`.

Important consequence:
- there is no public buffer object
- there is no global render-context cache
- the combined op still has access to all tensors needed for backward

