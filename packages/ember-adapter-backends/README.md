# ember-adapter-backends

Official backend adapters for `ember-core`.

## What It Contains
- `ember_adapter_backends.gsplat`: gsplat adapter and backend-specific options/output types
- `ember_adapter_backends.inria`: adapter shim for the forked GraphDeco/Inria rasterizer
- `ember_adapter_backends.stoch3dgs`: adapter shim for the stochastic 3DGRT backend from `Stoch3DGS`
- additional official backends can be added as subpackages over time

These official adapters are examples and conveniences, not privileged integrations.
`ember-core` is designed so you can write and register your own backend packages without depending on this repository.

SVRaster is no longer part of this package. It now lives in the
`ember-native-svraster` family package.

## Upstream References

- GraphDeco / Inria 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- gsplat: https://docs.gsplat.studio/main/
- FasterGS: https://fhahlbohm.github.io/faster-gaussian-splatting/
- FastGS: https://fastgs.github.io/
- Stoch3DGS: https://xupaya.github.io/stoch3DGS/

The backend subpackages under `src/ember_adapter_backends/` include short
local READMEs pointing back to the upstream backend each adapter wraps.

## Supported Backends

Currently registered backends in this package:

| Backend name | Scene type | Alpha | Depth | Normals | 2D projections | Projective intersection transforms | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `adapter.fastgs` | `GaussianScene3D` | ❌ | ❌ | ❌ | ❌ | ❌ | RGB plus backend-specific FastGS refinement signals |
| `adapter.fastergs` | `GaussianScene3D` | ❌ | ❌ | ❌ | ❌ | ❌ | RGB-only FasterGS adapter |
| `adapter.gsplat` | `GaussianScene3D` | ✅ | ✅ | ❌ | ✅ | ❌ | 3D Gaussian backend via `gsplat.rasterization` |
| `adapter.gsplat_2dgs` | `GaussianScene2D` | ✅ | ✅ | ❌ | ❌ | ✅ | 2D Gaussian backend via `gsplat.rasterization_2dgs` |
| `adapter.inria` | `GaussianScene3D` | ❌ | ✅ | ❌ | ❌ | ❌ | GraphDeco/Inria rasterizer adapter |
| `adapter.stoch3dgs` | `GaussianScene3D` | ✅ | ✅ | ❌ | ❌ | ❌ | Stochastic 3DGRT adapter |

Capability notes:
- `alpha`: per-pixel accumulated opacity/transmittance output.
- `depth`: per-pixel depth output in the backend's native shared render surface.
- `normals`: per-pixel surface or rendered normal output. This capability exists in the shared API, but no official backend exposes it yet.
- `2d_projections`: projected Gaussian centers plus compact conic coefficients via `projected_means` and `projected_conics`.
- `projective_intersection_transforms`: projected Gaussian centers plus 2DGS projective intersection geometry via `projected_means` and `projective_intersection_transforms`.

Current extras:
- `gsplat`: installs the `gsplat` runtime dependency
- `fastgs`: installs the local `FastGS` rasterizer runtime dependency
- `fastergs`: installs the local `faster-gaussian-splatting` runtime dependency
- `inria`: installs the local `diff-gaussian-rasterization` runtime dependency
- `stoch3dgs`: installs the local `Stoch3DGS` runtime dependency
- `all`: installs all officially supported backend runtime dependencies

## Install
Base package:

```bash
pip install ember-adapter-backends
```

With the gsplat backend dependency:

```bash
pip install "ember-adapter-backends[gsplat]"
```

With the FasterGS backend dependency:

```bash
pip install "ember-adapter-backends[fastergs]"
```

With the FastGS backend dependency:

```bash
pip install "ember-adapter-backends[fastgs]"
```

With the Inria backend dependency:

```bash
pip install "ember-adapter-backends[inria]"
```

With the Stoch3DGS backend dependency:

```bash
pip install "ember-adapter-backends[stoch3dgs]"
```

With all current backend dependencies:

```bash
pip install "ember-adapter-backends[all]"
```

`ember-core` is a required dependency of this package.

`stoch3dgs` also requires OptiX headers through the nested
`third_party/Stoch3DGS/threedgrt_tracer/dependencies/optix-dev` submodule in
the monorepo checkout. Before installing it locally, run:

```bash
git submodule update --init --recursive
```

`fastgs` also depends on the nested CUDA rasterizer submodule under
`third_party/FastGS/submodules/diff-gaussian-rasterization_fastgs`, so the same
recursive submodule initialization is required in the monorepo checkout.

## Registration
Backends are activated explicitly:

```python
import ember_core as sk
import ember_adapter_backends.gsplat as sk_gsplat

sk_gsplat.register()

output = sk.render(scene, camera, backend="adapter.gsplat")
```

`ember_adapter_backends.gsplat` currently also auto-registers on import for compatibility, but `register()` is the intended public pattern.

## Adding A New Backend
New backends should register themselves into `ember-core` explicitly.

Typical shape:

```python
from ember_core import RenderOptions
from ember_core.core.registry import register_backend


class MyBackendOptions(RenderOptions):
    ...


def render_my_backend(scene, camera, *, options=None, **kwargs):
    ...


def register() -> None:
    register_backend(
        name="my-backend",
        default_options=MyBackendOptions(),
        accepted_scene_types=(sk.GaussianScene3D,),
        supported_outputs=frozenset({"alpha", "depth"}),
    )(render_my_backend)
```

Usage:

```python
import ember_core as sk
import ember_adapter_backends.my_backend as sk_my_backend

sk_my_backend.register()
output = sk.render(scene, camera, backend="my-backend")
```

Guidelines:
- keep backend-specific options and output types inside the backend package
- only add new shared traits/capabilities to `ember-core` when they become broadly useful
- prefer explicit `register()` over relying only on import side effects

TODO: add an Inria-style backend example showing how to wrap a rasterizer with awkward packaging or licensing constraints.

## Package Layout
Source lives under:

```text
src/ember_adapter_backends/
  __init__.py
  fastgs/
  gsplat/
  inria/
  stoch3dgs/
```

In the monorepo, this package lives under `packages/ember-adapter-backends`.
