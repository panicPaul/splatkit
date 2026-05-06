# Ember Quality And Consistency Audit

Date: 2026-05-06

This audit evaluates Ember as a research bed for standalone paper
implementations. The main question is whether a paper author can implement a
paper cleanly in one notebook, without touching Ember internals, except when a
paper genuinely needs novel CUDA or Mojo kernels.

Standalone `marimo-config-gui` and `marimo-3dv` package internals are excluded
for now. They are mentioned only where Ember paper notebooks depend on their
public behavior.

## Scope

Included:

- `packages/ember-core`
- `packages/ember-adapter-backends`
- `packages/ember-native-*`
- `packages/ember-splatting-training`
- `packages/ember-svraster-training`
- `papers/*/notebook.py`
- Native CUDA, Mojo, JIT extension, registration, packaging, and test seams

Excluded for this pass:

- `packages/marimo-config-gui`
- `packages/marimo-3dv`
- standalone marimo config/viewer docs and viewer-only notebooks

## Executive Summary

Ember is directionally aligned with the north star: paper notebooks are already
usable proof-of-concept artifacts, paper configs are serializable, backend
packages are opt-in, and native packages expose reusable stages instead of
forcing whole-backend forks.

The biggest quality gap is not notebook duplication. Duplication is often
correct here because the notebooks are paper implementations. The real issues
are places where a paper author must understand Ember runtime internals:
`CallableSpec`, `context_kwargs`, backend option dictionaries, optimizer target
binding, render requirement merging, backend registration, and stale data config
shapes.

The highest-value work is to strengthen notebook-facing extension seams, not to
move paper logic wholesale into packages. Ember should own the boring, typed,
reproducible machinery. Paper notebooks should own paper-specific config,
initialization, loss, densification, scheduling, and small orchestration logic.

## Evidence

Commands and observations from this audit:

- `uv run pytest tests/test_paper_fastgs.py tests/test_paper_fastergs.py tests/test_paper_stoch3dgs.py tests/test_paper_svraster.py -q`
  passed: `33 passed`.
- Scoped Ruff over included packages and `papers` reported `2` fixable issues.
- Scoped ty over included packages and `papers` reported `178` diagnostics.
- Core/user-facing subset reported `125 passed, 1 skipped, 1 failed`; the
  failure compares nondeterministic timing fields in reproducibility history.
- Native backend/runtime collection found `62` tests.
- Native non-CUDA checks passed: `13 passed, 19 deselected`.
- Pixi quality commands did not reach Ruff or ty because environment solving
  failed while building `fused-ssim`; the build backend could not import
  `torch` during build isolation.
- Worktree already contained unrelated user edits under `papers/fastgs`; this
  audit did not modify them.

## Priority Findings

### P0: Stale Public Config APIs Leak Into Paper Notebooks

Some paper notebooks still call old Ember data config shapes. For example,
Stoch3DGS and SVRaster construct `ember.SceneLoadConfig(...)` with fields such
as `source`, `path`, `image_root`, `postprocess`, and `cache`, while current
core exposes concrete configs such as `ColmapSceneConfig`. They also pass
`mode=...` to `SplitConfig`, but current `SplitConfig` derives mode from
`target`, `every_n`, and `train_ratio`.

Impact:

- Authors copying these notebooks learn obsolete APIs.
- Type checking correctly flags them.
- Paper workflow tests still pass because they focus on resolved training
  config paths, not every scene/data helper.

Recommended fix:

- Add a small, current, notebook-facing data builder layer in `ember-core`:
  `colmap_scene_config(...)`, `split_config(...)`, and
  `prepared_frame_config(...)` helper functions or similarly named factories.
- Update paper notebooks to use current helpers while keeping paper-local
  fields and logic in the notebook.
- Add tests that call each paper notebook's `build_scene_load_config(...)` and
  `build_prepared_frame_dataset_config(...)`.

### P1: Runtime Plumbing Is Too Visible In Notebook Authoring

Paper notebooks and package recipes frequently expose low-level training
plumbing:

- `CallableSpec(target=...)`
- `context_kwargs={"device": "device"}`
- backend option dictionaries
- string targets for notebook-local builders
- optimizer target scopes and tensor views
- render requirement flags

This is acceptable as a runtime representation, but it is too prominent as an
authoring interface.

Impact:

- A paper author has to know Ember's internal materialization context.
- Configs remain serializable, but notebook code becomes less declarative than
  the goal.
- Small mistakes become stale import-string or context-key bugs.

Recommended fix:

- Keep `CallableSpec` as the low-level serialized contract.
- Add notebook-facing builders that accept local callables and typed context
  requirements without spelling internal context keys:
  `initializer(...)`, `loss(...)`, `hook(...)`, `densification(...)`,
  `backend_options_builder(...)`.
- Add typed context aliases for common runtime values: `device`,
  `frame_dataset`, `camera_extent`, `max_steps`, `backend`.
- Prefer user code such as:
  `ember.initializer(initialize_scene, needs=("device", "frame_dataset"))`
  over direct `CallableSpec(..., context_kwargs=...)`.

### P1: Reproducibility History Mixes Science Metrics With Timing Metrics

`run_training(...)` appends timing fields into the same per-step history as
loss and quality metrics:

- `step_seconds`
- `elapsed_seconds`
- `iterations_per_second`

The same-seed reproducibility test fails because those timing values differ
between runs, even when loss and model state match.

Impact:

- Reproducibility checks are noisy.
- Checkpoint history combines deterministic experiment results with
  nondeterministic runtime observations.
- Paper authors may compare histories and see false differences.

Recommended fix:

- Split deterministic metrics from runtime timing metrics.
- Options:
  - Add `TrainingResult.history` for science metrics and
    `TrainingResult.runtime_history` for timing/profiling.
  - Or keep one history but put timings under a nested `runtime` key and make
    reproducibility helpers ignore it.
- Update `test_training_is_reproducible_for_same_seed` to compare deterministic
  metrics explicitly.

### P1: Backend Registration Is Explicit But Not Ergonomic Enough

Explicit registration is the right package boundary, but notebooks currently
need to import backend packages and call `.register()` in paper-specific setup.
This is visible boilerplate and can fail late when backend names are selected
from config.

Impact:

- Authors must remember which package owns which backend name.
- Configs use backend strings, but package activation is imperative.
- Missing registration errors are runtime-only.

Recommended fix:

- Keep explicit registration as the core policy.
- Add an optional `ensure_backend_registered(name: str)` helper in package or
  notebook support code, backed by a small registry of official backend-name to
  package-register function mappings.
- Use it at config materialization or training launch boundaries.
- Error messages should say exactly which extra/package/register call is needed.

### P1: Type Health Is Weakest At Public Extension Seams

Scoped ty reported `178` diagnostics after excluding marimo config/viewer
packages. The highest concentrations are:

- `ember_core.data.loaders.colmap`
- paper notebooks
- `ember_core.data.__init__`
- `ember_core.data.contracts`
- native renderer/runtime wrappers
- lazy exports in training packages

Impact:

- The public surface appears less stable than the passing behavior tests imply.
- Paper authors get poor editor feedback for notebook-local extension code.
- Lazy root exports often appear as `object` to ty, causing call/type-form
  diagnostics in notebooks.

Recommended fix:

- Add a scoped ty gate for included Ember packages and paper notebooks.
- Fix stale API diagnostics first.
- For lazy exports, use `if TYPE_CHECKING` imports or `.pyi` stubs so public
  symbols such as `TrainingViewerConfig`, `GaussianMCMC`, and optimizers are
  visible to static analysis.
- Avoid using lazy-exported package attributes in type annotations inside paper
  notebooks; import types from concrete modules when needed.

### P2: Native JIT Build Paths Need Cleaner Operational Boundaries

Native packages use `torch.utils.cpp_extension.load(...)` extensively, which is
reasonable for research iteration. The rough edges are:

- 3DGRT stages generated sources under
  `src/ember_native_3dgrt/core/native/stoch3dgs/build/...`.
- The global `.gitignore` ignores these build trees, but the old ignore comment
  still references `packages/splatkit-native-backends`.
- JIT build products live close to package source, which makes local trees noisy
  and can confuse audits, tooling, and source packaging reasoning.

Impact:

- Researchers may confuse generated staged code with source of truth.
- Package source directories accumulate build state.
- Native build failures are harder to reason about.

Recommended fix:

- Move 3DGRT staged/generated build roots into a cache directory outside
  package source by default, e.g. `.cache/ember/native/3dgrt/...` or Torch's
  extension build root.
- Keep an environment override for debugging staged sources.
- Update `.gitignore` comments to match current package names.
- Document native build cache locations in native package READMEs.

### P2: SVRaster Native Kernels Still Contain Debug Print Sanity Checks

SVRaster native CUDA sources contain active `printf` or `std::printf` sanity
checks in render/preprocess/backward code paths.

Impact:

- Kernel output can pollute notebook/server logs.
- Debug checks are not structured diagnostics.
- Failures are harder for paper authors to interpret.

Recommended fix:

- Replace active prints with explicit guarded validation where possible before
  launching kernels.
- For device-only checks, use debug-build-only diagnostics or return structured
  error/status buffers that Python can interpret.
- Add a native smoke test ensuring normal render paths do not print warnings.

### P2: Packaging And Dependency Checks Need A Narrower Reliable Gate

Pixi task invocation failed before running checks because `fused-ssim` did not
have `torch` available during build isolation. Root `pyproject.toml` already
has an extra-build-dependency entry for `fused_ssim`, but the failing package
name appears as `fused-ssim`.

Impact:

- Developers cannot rely on `pixi run ruff` or `pixi run ty` as a clean quality
  entrypoint.
- Environment failures mask real code quality signals.

Recommended fix:

- Align the extra-build-dependencies key with the package name that uv reports.
- Add a lightweight local quality command that does not force full CUDA/native
  dependency solving when only Python static checks are needed.
- Keep full packaging checks for release/dependency changes.

### P2: Native Test Coverage Is Strong But CUDA-Gated

Native tests cover useful behaviors: dispatch, stage composition, gradients,
reference parity, fake tensor mode, and `torch.compile` on some paths. The
coverage is meaningful, but most confidence depends on CUDA availability.

Impact:

- Local non-CUDA checks only validate import and rejection paths.
- Native regressions may be missed unless CUDA tests run regularly.

Recommended fix:

- Keep CUDA marks, but define a named CUDA test profile for native changes.
- Add small CPU/static native checks where possible:
  - package import without CUDA build
  - registration metadata
  - options validation
  - source packaging includes
  - no generated build output in source tree
- Preserve reference parity tests as the high-confidence CUDA gate.

## What Not To Fix

Do not remove notebook-local duplication just because similar code appears in
multiple paper notebooks. Duplication is acceptable when it keeps a paper
implementation self-contained, readable, and faithful.

Good duplication:

- paper-specific Pydantic config models
- paper-specific schedule names and defaults
- paper-specific initialization, loss, densification, pruning, cleanup, and
  training hooks
- explicit notebook-local functions that make a paper readable without jumping
  into package internals

Duplication to challenge:

- repeated stale Ember config translation
- repeated cache materialization helpers that should be a stable data utility
- repeated backend registration boilerplate
- repeated low-level `CallableSpec` and context-key construction
- repeated native build/package setup patterns that paper authors should not
  need to understand

## Recommended Implementation Order

1. Fix stale data config helpers in paper notebooks and add tests that call the
   scene/dataset config builders.
2. Split deterministic training history from runtime timing metrics.
3. Add notebook-friendly wrappers around `CallableSpec` and context injection.
4. Improve lazy export typing for notebook-facing training utilities.
5. Add `ensure_backend_registered(...)` for official backends with clear error
   messages.
6. Clean native build staging and SVRaster debug prints.
7. Repair static-check/package-check entrypoints so the quality gates are
   dependable.

## Acceptance Criteria

The audit remediation should be considered successful when:

- paper workflow tests still pass
- paper scene/dataset config builder tests pass
- scoped Ruff passes with no diagnostics
- scoped ty has no stale public API diagnostics in paper notebooks
- same-seed training reproducibility compares deterministic metrics cleanly
- native non-CUDA tests pass locally
- CUDA native parity tests pass in the CUDA environment
- package/static quality commands do not trigger unrelated native dependency
  solving unless explicitly requested

