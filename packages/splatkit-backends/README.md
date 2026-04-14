# splatkit-backends

Official backend adapters for `splatkit`.

## What It Contains
- `splatkit_backends.gsplat`: gsplat adapter and backend-specific options/output types
- `splatkit_backends.inria`: adapter shim for the patched GraphDeco/Inria rasterizer
- additional official backends can be added as subpackages over time

These official adapters are examples and conveniences, not privileged integrations.
`splatkit` is designed so you can write and register your own backend packages without depending on this repository.

Current extra:
- `gsplat`: installs the `gsplat` runtime dependency
- `inria`: installs the patched `diff-gaussian-rasterization` runtime dependency

## Install
Base package:

```bash
pip install splatkit-backends
```

With the gsplat backend dependency:

```bash
pip install "splatkit-backends[gsplat]"
```

With the Inria backend dependency:

```bash
pip install "splatkit-backends[inria]"
```

`splatkit` is a required dependency of this package.

## Registration
Backends are activated explicitly:

```python
import splatkit as sk
import splatkit_backends.gsplat as sk_gsplat

sk_gsplat.register()

output = sk.render(scene, camera, backend="gsplat")
```

`splatkit_backends.gsplat` currently also auto-registers on import for compatibility, but `register()` is the intended public pattern.

## Adding A New Backend
New backends should register themselves into `splatkit` explicitly.

Typical shape:

```python
from splatkit import RenderOptions
from splatkit.core.registry import register_backend


class MyBackendOptions(RenderOptions):
    ...


def render_my_backend(scene, camera, *, options=None, **kwargs):
    ...


def register() -> None:
    register_backend(
        name="my-backend",
        default_options=MyBackendOptions(),
        supported_outputs=frozenset({"alpha", "depth"}),
    )(render_my_backend)
```

Usage:

```python
import splatkit as sk
import splatkit_backends.my_backend as sk_my_backend

sk_my_backend.register()
output = sk.render(scene, camera, backend="my-backend")
```

Guidelines:
- keep backend-specific options and output types inside the backend package
- only add new shared traits/capabilities to `splatkit` when they become broadly useful
- prefer explicit `register()` over relying only on import side effects

TODO: add an Inria-style backend example showing how to wrap a rasterizer with awkward packaging or licensing constraints.

## Package Layout
Source lives under:

```text
src/splatkit_backends/
  __init__.py
  gsplat/
  inria/
```

In the monorepo, this package lives under `packages/splatkit-backends`.
