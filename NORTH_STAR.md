# North Star

This repository is a research-focused system for rapid prototyping,
reimplementation, and recombination of new and existing research papers in
explicit inverse graphics, centered on the 3D Gaussian Splatting family and
closely related methods.

The primary goal is fast feedback loops without giving up reproducibility,
testability, strong typing, and clear contracts.

This document is a working constitution for both new and existing code. The
repository is pre-alpha, so improving the architecture is more important than
preserving weak or accidental APIs.

## Scope

- The repository is specialized on novel-view synthesis and the 3DGS family.
- Current and near-term primitive families include 3DGS, NDGS, 2DGS, sparse
  voxels, meshes, and related hybrids.
- Methods adjacent to this family, such as RadiantFoam-style approaches,
  should be possible to include.
- Purely implicit methods such as NeRFs are not a priority requirement.
- Evaluation is comparatively stable.
- Viewer code is not a core architectural driver and can live in notebooks.

## Core Goal

Code in this repository should make orthogonal research ideas testable in any
combination that does not violate explicit contracts.

Examples include:

- new densification strategies
- alternative training regimes and regularizers
- post-processing methods such as meshification
- pre-render and post-render methods such as anchor-based GS and PPISP
- alternative primitive kernels
- alternative initializations, including dense Gaussian initialization from SV
  Raster instead of relying only on COLMAP point clouds
- methods that need backend changes, without forcing wholesale rewrites

## First-Class Axes Of Modularity

The repository should treat these as first-class:

- primitive type
- backend
- densification
- training regime
- preprocessing and postprocessing
- pre-render and post-render methods

Primitive type is the highest-level organizing axis. Primitive families help,
but composable traits are required.

## Architectural Principles

- Prefer additive extension over mutation.
- Avoid changing existing behavior in place when a new paper or method can be
  introduced as a new backend, trait, pass, notebook, or config.
- Avoid monolithic code and entanglement.
- Prefer declarative and mostly functional code over imperative pipelines.
- Prefer explicit contracts over hidden assumptions.
- Prefer typed, serializable configuration over ad hoc script state.
- Prefer small reusable abstractions with defaults over large rigid frameworks.
- Avoid boilerplate, but do not hide important boundaries behind abstraction.
- Paper-faithful reimplementation is valuable, even if it temporarily duplicates
  code, when that duplication helps reveal the right long-term abstraction.

## What The Core Should Own

The core package should own:

- contracts
- traits and capabilities
- registries
- typed config models
- shared training abstractions
- shared data abstractions
- shared composition machinery

The core package should not own:

- heavy backend-specific logic
- viewer logic
- paper-specific implementation code unless it has proven reusable

## Dependency And Packaging Goals

The repository should keep the core dependency surface low enough that other
people can adopt `ember-core` contracts and swap backends in their own codebases
without opting into the more opinionated research stack.

- The core should stay lightweight and broadly installable.
- More opinionated training, notebook, viewer, and research utilities should be
  opt-in.
- Third-party backend installation pain should be abstracted away as much as
  possible by this repository.
- A design goal is to make supported backends feel `uv`-installable behind a
  consistent interface, even when upstream projects are difficult to install.
- CUDA packaging should support explicit dependency groups for CUDA 12.8 and
  13.0, while keeping heavier toolchain requirements out of the minimal core
  path.

## Notebook-First Workflow

`marimo` notebooks are first-class research artifacts.

- A paper implementation should usually start as a notebook.
- A notebook should also be runnable as a normal Python script.
- Some implementations may remain notebook-only.
- Code should be promoted from notebooks into package code only when the
  abstraction proves reusable.

A typical paper implementation in this repository should be:

- one marimo notebook
- JSON configs for experiment hyperparameters
- an optional new backend when required

## Backend Strategy

There are two backend categories:

- third-party backends: wrappers around external implementations
- native backends: first-party modular kernel stacks developed in this repo

Third-party backends exist to make comparison and recombination easier, not to
force the rest of the repository to inherit their code structure.

Wrapping an existing paper implementation into the shared backend abstraction is
a valid and encouraged integration strategy when code already exists. This
allows the repository to preserve a common input path, validation protocol, and
experiment surface while still exploring other orthogonal axes around that
paper's implementation.

Native backends should trend toward modular forward and backward stages, such as:

- preprocessing
- sorting
- blending
- utilities

These stages should be exposed as proper Torch custom ops where useful, along
with combined default ops or factories so experimentation stays low-friction.

When a method needs backend-specific support, the default preference is to add a
new backend instead of mutating an existing backend and risking breakage or
entanglement.

## Contracts And Defaults

Contracts should be strong, typed, and explicit, but they should still come
with good defaults for rapid prototyping.

- Defaults should make common experiments easy.
- Contracts should remain inspectable and overrideable.
- Orthogonal ideas should compose through contracts rather than through copy
  pasted training code.
- Traits should become the long-term way to express reusable behavior across
  primitive families.
- Families are still useful as an organizing tool, especially while the system
  is evolving.

## Reproducibility And Quality

Non-negotiables:

- strong typing
- serializable configs compatible with the GUI and Pydantic-based tooling
- mostly declarative, mostly functional structure
- reproducibility
- testability

The repository should optimize first for:

- rapid personal prototyping
- rapid paper implementation with or without released code
- fair evaluation

Over time, it should accumulate a toolbox of proven ideas that can be
recombined cleanly.

## Pre-Alpha Policy

This repository is still early.

- Breaking changes are acceptable when they materially improve the architecture.
- Existing inconsistencies should be filed away and reduced over time.
- Weak contracts should not be preserved just because they exist already.
- The goal is not early API stability.
- The goal is to converge on the right abstraction boundaries.

## Litmus Test For New Code

Before adding new code, ask:

- Does this introduce a new idea additively, or does it mutate existing
  behavior unnecessarily?
- Is the code organized around primitive type, backend, densification,
  training regime, or pre/post processing in a way that keeps boundaries clear?
- Could this start as a notebook first?
- If this code becomes reusable, is there an obvious contract or trait boundary
  to promote?
- Are defaults easy without making the abstraction opaque?
- Does this reduce entanglement, or just move it around?

If the answer is unclear, prefer the simpler, more local, more explicit design.
