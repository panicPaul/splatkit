"""Run controlled FastGS training profiles for base/big bottleneck diagnosis."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from marimo_config_gui.presets import load_preset_config

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "papers" / "fastgs" / "notebook.py"


@dataclass(frozen=True)
class ProfileCase:
    """One controlled profile run."""

    name: str
    preset: str
    disable_densification: bool = False
    fixed_batch: bool = False


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile FastGS garden_base/garden_big with densification and "
            "dataloader controls."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/ember_fastgs_training_matrix"),
        help="Directory for per-case profile JSONL and manifest files.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1200,
        help="Training steps per case.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Profiler JSONL logging frequency.",
    )
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=None,
        help="Optional override for the preset scene path.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[
            "base",
            "big",
            "base_no_densification",
            "big_no_densification",
            "base_fixed_batch",
            "big_fixed_batch",
            "base_no_densification_fixed_batch",
            "big_no_densification_fixed_batch",
        ],
        help=(
            "Case to run. Repeat to run several. Defaults to the six-case "
            "diagnostic matrix."
        ),
    )
    parser.add_argument(
        "--include-fixed-no-densification",
        action="store_true",
        help="Also run fixed-batch no-densification cases.",
    )
    parser.add_argument(
        "--sync-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize CUDA around profiled phases.",
    )
    parser.add_argument(
        "--cuda-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record CUDA memory counters in profiler output.",
    )
    parser.add_argument(
        "--disable-checkpoint-export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable PLY export and write minimal checkpoints under output-dir.",
    )
    return parser.parse_args()


def load_fastgs_module() -> Any:
    """Import the FastGS notebook as a normal Python module."""
    spec = importlib.util.spec_from_file_location(
        "papers.fastgs.notebook",
        NOTEBOOK_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {NOTEBOOK_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def default_cases(*, include_fixed_no_densification: bool) -> list[ProfileCase]:
    """Return the default diagnostic case matrix."""
    cases = [
        ProfileCase("base", "garden_base"),
        ProfileCase("big", "garden_big"),
        ProfileCase(
            "base_no_densification",
            "garden_base",
            disable_densification=True,
        ),
        ProfileCase(
            "big_no_densification",
            "garden_big",
            disable_densification=True,
        ),
        ProfileCase("base_fixed_batch", "garden_base", fixed_batch=True),
        ProfileCase("big_fixed_batch", "garden_big", fixed_batch=True),
    ]
    if include_fixed_no_densification:
        cases.extend(
            [
                ProfileCase(
                    "base_no_densification_fixed_batch",
                    "garden_base",
                    disable_densification=True,
                    fixed_batch=True,
                ),
                ProfileCase(
                    "big_no_densification_fixed_batch",
                    "garden_big",
                    disable_densification=True,
                    fixed_batch=True,
                ),
            ]
        )
    return cases


def selected_cases(args: argparse.Namespace) -> list[ProfileCase]:
    """Resolve requested profile cases."""
    cases_by_name = {
        case.name: case
        for case in default_cases(include_fixed_no_densification=True)
    }
    if args.case:
        return [cases_by_name[name] for name in args.case]
    return default_cases(
        include_fixed_no_densification=args.include_fixed_no_densification
    )


def load_experiment_config(
    module: Any,
    case: ProfileCase,
    *,
    scene_path: Path | None,
) -> Any:
    """Load a FastGS preset and apply run-local scene overrides."""
    config = load_preset_config(module.fastgs_preset_catalog(), case.preset)
    if scene_path is None:
        return config
    return config.model_copy(
        update={"scene": config.scene.model_copy(update={"path": scene_path})},
        deep=True,
    )


def build_training_config(
    module: Any,
    experiment_config: Any,
    frame_dataset: Any,
    case: ProfileCase,
    args: argparse.Namespace,
) -> Any:
    """Build a per-case training config with profiler/checkpoint overrides."""
    training_config = module.resolve_training_config(
        experiment_config,
        frame_dataset,
    )
    updates: dict[str, Any] = {
        "runtime": training_config.runtime.model_copy(
            update={"max_steps": args.steps}
        ),
        "logging": training_config.logging.model_copy(
            update={"enabled": False}
        ),
        "profiler": training_config.profiler.model_copy(
            update={
                "enabled": True,
                "log_every": args.log_every,
                "sync_timing": args.sync_timing,
                "cuda_memory": args.cuda_memory,
                "output_path": args.output_dir / f"{case.name}.jsonl",
            }
        ),
    }
    if case.disable_densification:
        updates["densification"] = None
    if args.disable_checkpoint_export:
        updates["checkpoint"] = training_config.checkpoint.model_copy(
            update={
                "output_dir": args.output_dir / case.name / "checkpoint",
                "export_ply": False,
                "overwrite": True,
            }
        )
    return training_config.model_copy(update=updates)


def prepare_case(
    module: Any,
    case: ProfileCase,
    args: argparse.Namespace,
) -> tuple[Any, Any, Any]:
    """Load scene, frame dataset, and training config for a case."""
    experiment_config = load_experiment_config(
        module,
        case,
        scene_path=args.scene_path,
    )
    scene_record = module.ember.load_scene_record(
        module.build_scene_load_config(experiment_config)
    )
    frame_dataset = module.ember.prepare_frame_dataset(
        scene_record,
        module.build_prepared_frame_dataset_config(experiment_config),
    )
    training_config = build_training_config(
        module,
        experiment_config,
        frame_dataset,
        case,
        args,
    )
    return experiment_config, frame_dataset, training_config


def run_standard_case(
    module: Any,
    frame_dataset: Any,
    experiment_config: Any,
    training_config: Any,
) -> Any:
    """Run the normal training loop with the configured dataloader."""
    return module.run_fastgs_training(
        frame_dataset,
        experiment_config,
        training_config,
    )


def run_fixed_batch_case(
    module: Any,
    frame_dataset: Any,
    training_config: Any,
) -> list[dict[str, float]]:
    """Run train_step repeatedly on one already-loaded batch."""
    from ember_core.densification.runtime import (
        bind_densification,
        call_densification_hook,
        make_lifecycle_context,
    )
    from ember_core.training import (
        TrainState,
        build_densification_for_context,
        build_loss_fn,
        build_optimizer_set,
        build_raw_render_fn,
        build_render_fn_with_requirements,
        build_training_render_fn,
        build_training_run_context,
        initialize_model,
        set_torch_seed,
        train_step,
    )
    from ember_core.training.profiling import build_training_profiler
    from ember_core.training.runtime import (
        _TrainingDensificationRuntime,
        build_dataloader,
    )

    set_torch_seed(training_config.runtime.seed)
    device = torch.device(training_config.runtime.device)
    run_context = build_training_run_context(
        frame_dataset,
        training_config,
        device=device,
    )
    model = initialize_model(
        frame_dataset.scene_record,
        training_config,
        context=run_context,
    ).to(device)
    state = TrainState(
        model=model,
        step=0,
        seed=training_config.runtime.seed,
        device=device,
    )
    dataloader = build_dataloader(frame_dataset, training_config)
    fixed_batch = next(iter(dataloader))
    fixed_batch = fixed_batch.to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    raw_render_fn = build_raw_render_fn(training_config)
    render_fn = build_training_render_fn(
        training_config,
        state,
        context=run_context,
    )
    render_fn_with_requirements = build_render_fn_with_requirements(
        training_config,
        state=state,
        context=run_context,
    )
    loss_fn = build_loss_fn(training_config)
    optimizers = build_optimizer_set(
        state,
        training_config,
        context=run_context,
    )
    densification = build_densification_for_context(
        training_config,
        context=run_context,
    )
    densification = bind_densification(densification, state, optimizers)
    probe_runtime = None
    if densification is not None:
        probe_runtime = _TrainingDensificationRuntime(
            backend_name=training_config.render.backend,
            render_options=module.ember.resolve_backend_options(
                training_config
            ),
            frame_dataset=frame_dataset,
            raw_render_fn=raw_render_fn,
            device=device,
        )
        call_densification_hook(
            densification.before_training,
            make_lifecycle_context(
                state=state,
                optimizers=optimizers,
                runtime=probe_runtime,
            ),
        )

    profiler = build_training_profiler(training_config.profiler)
    history: list[dict[str, float]] = []
    training_started_at = time.perf_counter()
    for _ in range(training_config.runtime.max_steps):
        step_started_at = time.perf_counter()
        profile = None if profiler is None else profiler.start_step(state)
        metrics = train_step(
            state,
            fixed_batch,
            render_fn=render_fn,
            render_fn_with_requirements=render_fn_with_requirements,
            loss_fn=loss_fn,
            optimizers=optimizers,
            densification=densification,
            probe_runtime=probe_runtime,
            profile=profile,
        )
        step_duration_seconds = max(
            time.perf_counter() - step_started_at,
            1e-12,
        )
        metrics["step_seconds"] = step_duration_seconds
        metrics["elapsed_seconds"] = time.perf_counter() - training_started_at
        metrics["iterations_per_second"] = 1.0 / step_duration_seconds
        if profiler is not None:
            profiler.finish_step(state, metrics, profile)
        history.append(metrics)

    if densification is not None:
        call_densification_hook(
            densification.after_training,
            make_lifecycle_context(
                state=state,
                optimizers=optimizers,
                runtime=probe_runtime,
            ),
        )
    return history


def write_manifest(
    output_dir: Path,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Write run metadata for the matrix."""
    manifest = {
        "steps": args.steps,
        "log_every": args.log_every,
        "sync_timing": args.sync_timing,
        "cuda_memory": args.cuda_memory,
        "scene_path": str(args.scene_path) if args.scene_path else None,
        "cases": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    """Run the requested profiling matrix."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    module = load_fastgs_module()
    manifest_records: list[dict[str, Any]] = []

    for case in selected_cases(args):
        profile_path = args.output_dir / f"{case.name}.jsonl"
        if profile_path.exists():
            profile_path.unlink()
        print(f"==> Running {case.name}", flush=True)
        started_at = time.perf_counter()
        experiment_config, frame_dataset, training_config = prepare_case(
            module,
            case,
            args,
        )
        if case.fixed_batch:
            history = run_fixed_batch_case(module, frame_dataset, training_config)
            checkpoint_dir = None
        else:
            result = run_standard_case(
                module,
                frame_dataset,
                experiment_config,
                training_config,
            )
            history = result.history
            checkpoint_dir = result.checkpoint_dir
        elapsed_seconds = time.perf_counter() - started_at
        manifest_records.append(
            {
                "name": case.name,
                "preset": case.preset,
                "disable_densification": case.disable_densification,
                "fixed_batch": case.fixed_batch,
                "profile_path": str(profile_path),
                "checkpoint_dir": checkpoint_dir,
                "elapsed_seconds": elapsed_seconds,
                "history_steps": len(history),
                "final_primitives": (
                    history[-1].get("primitives") if history else None
                ),
            }
        )
        write_manifest(args.output_dir, manifest_records, args)
        print(
            f"<== Finished {case.name} in {elapsed_seconds:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
