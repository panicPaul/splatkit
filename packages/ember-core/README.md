# ember-core

Minimal, backend-agnostic scene contracts, scene I/O, initialization, and declarative training helpers.

## What It Contains
- `ember_core.core`: canonical scene/camera contracts, output capability protocols, backend registry, and the generic `render(...)` wrapper
- `ember_core.data`: scene-record ingestion plus prepared-frame dataset construction
- `ember_core.initialization`: reusable scene/model initialization helpers for training
- `ember_core.io`: scene-only load/save helpers for 3DGS PLY and SV Raster checkpoints
- `ember_core.training`: declarative training configs, render/loss builders, and reproducible checkpoint-directory export
- optional extras over time for reusable tooling that should remain separate from the minimal core dependency set

Current extras:
- `eval`: reserved for future evaluation utilities
- `all`: installs the current optional utility stacks

Install `ember-core[viewer]` to include the marimo 3D viewer integration.

## Install
Base package:

```bash
pip install ember-core
```

With all current optional utilities:

```bash
pip install "ember-core[all]"
```

## Usage
Construct shared contract values and render through a registered backend:

```python
import ember_core as sk

scene = sk.GaussianScene3D(...)
camera = sk.CameraState(...)

output = sk.render(scene, camera, backend="adapter.gsplat")
image = output.render
```

Backends are registered separately. `ember-core` does not import backend packages automatically.

## Core Surface

`ember-core` is intentionally the package that other projects should be able to
adopt without inheriting the whole research monorepo. It owns the reusable
contracts and orchestration points:

- `ember_core.core`: scene families, camera contracts, render outputs, output
  capabilities, backend registration, and the generic `render(...)` wrapper.
- `ember_core.data`: scene-record loaders, source pipes, frame splitting,
  materialization, image preparation, and prepared-frame datasets.
- `ember_core.io`: scene-only load/save helpers for Gaussian PLY and SVRaster
  checkpoints.
- `ember_core.initialization`: helpers that turn loaded scene records into
  trainable scene/model state.
- `ember_core.densification`: typed densification families, passes, collectors,
  and composition helpers.
- `ember_core.training`: declarative runtime, render, loss, optimization,
  hook, checkpoint, and training-run helpers.
- `ember_core.viewer`: optional bridge utilities for viewer payloads,
  prep/cache keys, mode selection, scene filtering, and stats preparation.

Heavy renderer implementations, upstream adapters, and native kernels live in
separate packages. Import those packages explicitly and call their `register()`
functions before rendering through them.

## Training
Training is organized around declarative configs plus importable pipeline stages.

Typical flow:

```python
import ember_core as sk

scene_record = sk.load_scene_record(
    sk.ColmapSceneConfig(
        path="scene_dir",
        source_pipes=(sk.HorizonAlignPipeConfig(),),
    )
)
dataset = sk.prepare_frame_dataset(
    scene_record,
    sk.PreparedFrameDatasetConfig(
        image_preparation=sk.ImagePreparationConfig(
            resize_width_target=1980,
            normalize=True,
        ),
    ),
)
config = sk.TrainingConfig(
    render=sk.RenderPipelineSpec(backend="adapter.gsplat"),
    loss=sk.LossConfig(
        target=sk.CallableSpec(target="my_project.losses.rgb_l2_loss")
    ),
    optimization=sk.OptimizationConfig(
        parameter_groups=[
            sk.ParameterGroupConfig(
                target=sk.ParameterTargetSpec(
                    scope="scene",
                    name="feature",
                ),
                lr=1e-2,
            ),
        ]
    ),
)
result = sk.run_training(dataset, config)
checkpoint = sk.load_checkpoint_dir(result.checkpoint_dir)
```

## Typed Authoring And Private Extensions
Python code should prefer typed refs and exported constants, while JSON,
checkpoints, and plugin discovery keep stable string identifiers.

```python
from ember_native_faster_gs.backends import FASTER_GS_FASTGS

render = sk.RenderPipelineSpec(
    backend=FASTER_GS_FASTGS,
    options=FASTER_GS_FASTGS.options(compact_box_scale=0.5),
)
```

External packages can define their own `SceneFamilyKey`, `ProductKey`,
`StageKey`, `BufferKey`, and `BackendRef` constants without editing
`ember-core`. Private primitive families should register family ops with
`register_family_ops(...)` when they want to reuse Ember densification
composition, or implement `bind_context(...)` on their densification method
when they need direct access to optimizer bindings and private topology logic.
Private training loops can reuse Ember setup through `prepare_training_loop(...)`
or pass a `trainer=` object to `run_training(...)`.

Research notebooks can keep their public surface cleaner by exposing one typed
Pydantic config and materializing it into `TrainingConfig` only at runtime:

```python
from pydantic import BaseModel, Field


class ExperimentTrainingConfig(BaseModel):
    runtime: sk.RuntimeConfig = Field(default_factory=sk.RuntimeConfig)
    render_backend: str = "adapter.gsplat"
    position_lr: float = 1.6e-4

    def to_training_config(
        self,
        frame_dataset: sk.PreparedFrameDataset | None = None,
    ) -> sk.TrainingConfig:
        camera_extent = (
            sk.compute_frame_camera_extent(frame_dataset)
            if frame_dataset is not None
            else 1.0
        )
        return sk.TrainingConfig(
            runtime=self.runtime,
            render=sk.RenderPipelineSpec(backend=self.render_backend),
            optimization=sk.OptimizationConfig(
                parameter_groups=[
                    sk.ParameterGroupConfig(
                        target=sk.ParameterTargetSpec(
                            scope="scene",
                            name="center_position",
                        ),
                        lr=self.position_lr * camera_extent,
                    )
                ]
            ),
        )


result = sk.run_training(dataset, ExperimentTrainingConfig())
```

Use typed config methods and properties for user-facing computed fields. Keep
`CallableSpec.kwargs` as a low-level runtime representation for importable
pipeline stages, preset export, and checkpoint replay.

`ColmapSceneConfig` is the default scene-loading entrypoint. Prepared-frame
policies such as camera selection, split, materialization, and image resizing
live in `PreparedFrameDatasetConfig`.

Built-in presets:
- `ColmapSceneConfig`: plain COLMAP scene-record defaults
- `MipNerf360IndoorPreparedFrameDatasetConfig`: default image resize with `width_scale=0.25`
- `MipNerf360OutdoorPreparedFrameDatasetConfig`: default image resize with `width_scale=0.5`

Checkpoints are saved as directories containing `config.json`, `metadata.json`, `model.ckpt`, and optionally `scene.ply`.

## Viewer Bridge

Install `ember-core[viewer]` when you want the optional `marimo-3dv`
integration:

```bash
pip install "ember-core[viewer]"
```

The viewer bridge keeps rendering and browser state separate. `marimo-3dv`
owns the live widget, controls, linked state, and click handling. `ember-core`
owns conversion and preparation helpers that know about Ember scene and camera
contracts:

- `camera_from_viewer_payload(...)` and `camera_to_viewer_payload(...)`
- `select_viewer_camera(...)`
- `filter_gaussian_scene(...)` and `replace_gaussian_features(...)`
- `viewer_prep_key(...)` and `config_cache_key(...)`
- `prepare_viewer_stats_series(...)`
- `launch_viewer(...)`

For package-level viewer behavior, read the `marimo-3dv` docs. For an Ember
workflow tutorial, run:

```bash
marimo run docs/interactive/ember/viewer.py
```

from the repository root.

## Registering Backends
`ember-core` owns the shared registry, but backend packages are responsible for registering themselves.

We provide some official backend adapters, but they are not special from the point of view of `ember-core`.
The intended design is that researchers can write their own backend packages, local experiment modules, or lab-specific adapters against the same contracts and registry.
Using `ember_adapter_backends.*` is a convenience, not a requirement.

Typical usage:

```python
import ember_core as sk
import ember_adapter_backends.gsplat as sk_gsplat

sk_gsplat.register()

output = sk.render(scene, camera, backend="adapter.gsplat")
```

Backend packages should:
- implement their own render function and backend-specific option/output types
- expose a `register()` function that calls `ember_core.register_backend(...)`
- declare which shared outputs they support via `supported_outputs`

You are free, and encouraged, to register your own backends the same way:

```python
import ember_core as sk
from ember_core import RenderOptions


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

Only promote new output capabilities into `ember-core` when they become broadly useful across backends or tools.

TODO: add a worked example showing how to wrap an Inria-style rasterizer as an external backend package or local research adapter.

## Package Layout
Source lives under:

```text
src/ember_core/
  __init__.py
  core/
```

The public API is currently re-exported from `ember-core` for convenience, with the implementation living in `ember_core.core`.

In the monorepo, this package lives under `packages/ember-core`.

## License

This package is distributed under the Apache License 2.0.
