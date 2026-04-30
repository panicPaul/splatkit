# Ember Docs

This directory is a crosswalk for the package-local documentation in this
repository. Package READMEs are the canonical reference docs; this directory
keeps the reading order and interactive marimo entrypoints easy to find.

## Start Here

- [Root README](../README.md): install, package map, backend families, packaging
  and development notes.
- [North Star](../NORTH_STAR.md): design goals and architectural direction.
- [ember-core](../packages/ember-core/README.md): contracts, data, rendering,
  training, and viewer bridge.
- [marimo-config-gui](marimo-config-gui.md): typed config UIs for notebooks.
- [marimo-3dv](marimo-3dv.md): native marimo viewer and splat notebook helpers.

## Interactive Docs

Run the interactive docs from the repository root:

```bash
marimo run docs/marimo-config-gui.py
marimo run docs/interactive/marimo-3dv.py
marimo run docs/interactive/ember/contracts.py
marimo run docs/interactive/ember/data.py
marimo run docs/interactive/ember/viewer.py
marimo run docs/interactive/ember/training.py
marimo run docs/interactive/ember/extension.py
```

The interactive docs are tutorial-style. They favor live controls, toy data,
and small examples over exhaustive API coverage. Use package READMEs for the
full reference surface.

## Package Reference Map

Core and workflow packages:

- [ember-core](../packages/ember-core/README.md): minimal shared contracts,
  registry, data loading, IO, initialization, densification, training, and
  optional viewer bridge.
- [ember-splatting-training](../packages/ember-splatting-training/README.md):
  reusable splatting-specific training utilities.
- [marimo-config-gui](marimo-config-gui.md): Pydantic-backed marimo config
  panels and script-mode config loading.
- [marimo-3dv](marimo-3dv.md): marimo-native viewer, viewer controls, linked
  state, splat ops, and setup helpers.

Backend packages:

- [ember-adapter-backends](../packages/ember-adapter-backends/README.md):
  official wrappers around external rasterizers.
- [ember-native-faster-gs](../packages/ember-native-faster-gs/README.md):
  first-party FasterGS-family native backends.
- [ember-native-faster-gs-mojo](../packages/ember-native-faster-gs-mojo/README.md):
  Mojo-backed FasterGS experiments.
- [ember-native-3dgrt](../packages/ember-native-3dgrt/README.md):
  first-party 3DGRT-family native backend.
- [ember-native-svraster](../packages/ember-native-svraster/README.md):
  first-party SVRaster-family native backend with restricted upstream license
  terms.

## Suggested Reading Order

1. Read the root README and North Star to understand the repository shape.
2. Read `ember-core` for the contracts and recomposition model.
3. Read one backend package README to understand registration and capabilities.
4. Run the config and viewer interactive docs.
5. Run the Ember workflow notebooks in order: contracts, data, viewer,
   training, extension.

## Docs Convention

- Package-local `README.md` files are the exhaustive reference docs.
- Interactive marimo docs live under `docs/interactive/`.
- This directory may contain compatibility symlinks so older short commands and
  package-local references continue to work.
