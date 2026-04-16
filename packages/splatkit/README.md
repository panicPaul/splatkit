# splatkit

Minimal, backend-agnostic scene contracts, scene I/O, initialization, and declarative training helpers.

## What It Contains
- `splatkit.core`: canonical scene/camera contracts, output capability protocols, backend registry, and the generic `render(...)` wrapper
- `splatkit.data`: dataset ingestion plus lazy camera-batched image preparation
- `splatkit.initialization`: reusable scene/model initialization helpers for training
- `splatkit.io`: scene-only load/save helpers for 3DGS PLY and SV Raster checkpoints
- `splatkit.training`: declarative training configs, render/loss builders, and reproducible checkpoint-directory export
- optional extras over time for reusable tooling that should remain separate from the minimal core dependency set

Current extras:
- `viewer`: installs the viewer dependency stack via `marimo-3dv`
- `eval`: reserved for future evaluation utilities
- `all`: installs the non-viewer optional utility stacks; `viewer` remains separate for now

## Install
Base package:

```bash
pip install splatkit
```

With viewer dependencies:

```bash
pip install "splatkit[viewer]"
```

With all current optional utilities:

```bash
pip install "splatkit[all]"
```

## Usage
Construct shared contract values and render through a registered backend:

```python
import splatkit as sk

scene = sk.GaussianScene3D(...)
camera = sk.CameraState(...)

output = sk.render(scene, camera, backend="gsplat")
image = output.render
```

Backends are registered separately. `splatkit` does not import backend packages automatically.

## Training
Training is organized around declarative configs plus importable pipeline stages.

Typical flow:

```python
import splatkit as sk

dataset = sk.load_dataset(
    sk.ColmapDatasetConfig(
        path="scene_dir",
        source_pipes=(sk.HorizonAlignPipeConfig(),),
        cache_pipes=(sk.ResizePipeConfig(width_target=1980),),
        prepare_pipes=(sk.NormalizePipeConfig(),),
    )
)
config = sk.TrainingConfig(
    render=sk.RenderPipelineSpec(backend="gsplat"),
    loss=sk.LossConfig(
        target=sk.CallableSpec(target="my_project.losses.rgb_l2_loss")
    ),
    optimization=sk.OptimizationConfig(
        parameter_groups=[
            sk.ParameterGroupConfig(selector="scene.feature", lr=1e-2),
        ]
    ),
)
result = sk.run_training(dataset, config)
checkpoint = sk.load_checkpoint_dir(result.checkpoint_dir)
```

`ColmapDatasetConfig` is the default user path. For extensions, subclass the
concrete dataset config, register any new pipe spec classes plus runtime
implementations, and override the ordered `source_pipes`, `cache_pipes`, and
`prepare_pipes` tuples on that subclass.

Built-in presets:
- `ColmapDatasetConfig`: plain COLMAP defaults
- `MipNerf360IndoorDatasetConfig`: default cache resize with `width_scale=0.25`
- `MipNerf360OutdoorDatasetConfig`: default cache resize with `width_scale=0.5`

Checkpoints are saved as directories containing `config.json`, `metadata.json`, `model.ckpt`, and optionally `scene.ply`.

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
        accepted_scene_types=(sk.GaussianScene3D,),
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
