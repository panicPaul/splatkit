# splatkit

Minimal, backend-agnostic Gaussian splatting contracts and runtime helpers.

## What It Contains
- `splatkit.core`: canonical scene/camera contracts, output capability protocols, backend registry, and the generic `render(...)` wrapper
- optional extras over time for reusable tooling that should remain separate from the minimal core dependency set

Current extra:
- `viewer`: installs the viewer dependency stack via `marimo-3dv`

## Install
Base package:

```bash
pip install splatkit
```

With viewer dependencies:

```bash
pip install "splatkit[viewer]"
```

## Usage
Construct shared contract values and render through a registered backend:

```python
import splatkit as sk

scene = sk.GaussianScene(...)
camera = sk.CameraState(...)

output = sk.render(scene, camera, backend="gsplat")
image = output.render
```

Backends are registered separately. `splatkit` does not import backend packages automatically.

## Registering Backends
`splatkit` owns the shared registry, but backend packages are responsible for registering themselves.

We provide some official backend adapters, but they are not special from the point of view of `splatkit`.
The intended design is that researchers can write their own backend packages, local experiment modules, or lab-specific adapters against the same contracts and registry.
Using `splatkit_backends.*` is a convenience, not a requirement.

Typical usage:

```python
import splatkit as sk
import splatkit_backends.gsplat as sk_gsplat

sk_gsplat.register()

output = sk.render(scene, camera, backend="gsplat")
```

Backend packages should:
- implement their own render function and backend-specific option/output types
- expose a `register()` function that calls `splatkit.register_backend(...)`
- declare which shared outputs they support via `supported_outputs`

You are free, and encouraged, to register your own backends the same way:

```python
import splatkit as sk
from splatkit import RenderOptions


class MyBackendOptions(RenderOptions):
    ...


def render_my_backend(scene, camera, *, options=None, **kwargs):
    ...


def register() -> None:
    sk.register_backend(
        name="my-backend",
        default_options=MyBackendOptions(),
        supported_outputs=frozenset({"alpha"}),
    )(render_my_backend)
```

Only promote new output capabilities into `splatkit` when they become broadly useful across backends or tools.

TODO: add a worked example showing how to wrap an Inria-style rasterizer as an external backend package or local research adapter.

## Package Layout
Source lives under:

```text
src/splatkit/
  __init__.py
  core/
```

The public API is currently re-exported from `splatkit` for convenience, with the implementation living in `splatkit.core`.

In the monorepo, this package lives under `packages/splatkit`.
