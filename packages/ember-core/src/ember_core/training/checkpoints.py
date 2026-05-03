"""Checkpoint directory helpers for declarative training."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import (
    PackageNotFoundError,
    packages_distributions,
    version,
)
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ember_core.core.contracts import GaussianScene3D, Scene
from ember_core.core.registry import BACKEND_REGISTRY
from ember_core.data.adapters import PreparedFrameDataset
from ember_core.data.contracts import SceneRecord
from ember_core.initialization import InitializedModel
from ember_core.io import load_gaussian_ply, save_gaussian_ply
from ember_core.training.config import CheckpointMetadata, TrainingConfig
from ember_core.training.protocols import (
    LoadedCheckpoint,
    TrainingRunContext,
    TrainState,
)
from ember_core.training.runtime import (
    build_modules,
    build_parameters,
    build_render_fn,
)


def _json_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")


def _safe_git_output(args: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_commit() -> str | None:
    return _safe_git_output(["git", "rev-parse", "HEAD"])


def _git_dirty() -> bool | None:
    output = _safe_git_output(["git", "status", "--porcelain"])
    if output is None:
        return None
    return bool(output)


def _git_commit_for_path(path: Path) -> str | None:
    return _safe_git_output(["git", "rev-parse", "HEAD"], cwd=path.parent)


def _git_dirty_for_path(path: Path) -> bool | None:
    output = _safe_git_output(["git", "status", "--porcelain"], cwd=path.parent)
    if output is None:
        return None
    return bool(output)


def _package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    for package in (
        "ember-core",
        "ember-splatting-training",
        "ember-adapter-backends",
        "ember-native-faster-gs",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def _import_paths(config: TrainingConfig) -> list[str]:
    import_paths = [
        config.initialization.initializer.target,
        config.loss.target.target,
    ]
    import_paths.extend(
        builder.target for builder in config.model.modules.values()
    )
    if config.render.feature_fn is not None:
        import_paths.append(config.render.feature_fn.target)
    if config.render.postprocess_fn is not None:
        import_paths.append(config.render.postprocess_fn.target)
    if config.render.training_backend_options_builder is not None:
        import_paths.append(config.render.training_backend_options_builder.target)
    if config.densification is not None:
        import_paths.extend(
            builder.target for builder in config.densification.builders
        )
    if config.optimization.builder is not None:
        import_paths.append(config.optimization.builder.target)
    import_paths.extend(builder.target for builder in config.hooks.builders)
    import_paths.extend(
        group.scheduler.target
        for group in config.optimization.parameter_groups
        if group.scheduler is not None
    )
    import_paths.extend(
        group.optimizer
        for group in config.optimization.parameter_groups
        if "." in group.optimizer
    )
    return sorted(set(import_paths))


def _module_name_for_target(target: str) -> str | None:
    module_name, _, attr_path = target.rpartition(".")
    while module_name:
        try:
            import_module(module_name)
            return module_name
        except Exception:
            module_name, _, tail = module_name.rpartition(".")
            if not tail:
                return None
            attr_path = f"{tail}.{attr_path}"
    return None


def _package_names_for_module(module_name: str) -> list[str]:
    root_name = module_name.split(".", maxsplit=1)[0]
    distribution_names = packages_distributions().get(root_name, [])
    return sorted(distribution_names)


def _provenance_for_module(module_name: str) -> dict[str, Any]:
    module = import_module(module_name)
    module_file = getattr(module, "__file__", None)
    package_names = _package_names_for_module(module_name)
    package_versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            continue
    path = None if module_file is None else Path(module_file).resolve()
    return {
        "module": module_name,
        "file": None if path is None else str(path),
        "packages": package_versions,
        "git_commit": None if path is None else _git_commit_for_path(path),
        "git_dirty": None if path is None else _git_dirty_for_path(path),
    }


def _provenance_for_target(target: str) -> dict[str, Any]:
    module_name = _module_name_for_target(target)
    if module_name is None:
        return {"target": target, "module": None}
    provenance = _provenance_for_module(module_name)
    provenance["target"] = target
    return provenance


def _provenance(config: TrainingConfig) -> dict[str, dict[str, Any]]:
    entries = {
        "ember_core": _provenance_for_module("ember_core"),
    }
    for optional_module in (
        "ember_splatting_training",
        "ember_adapter_backends",
        "ember_native_faster_gs",
    ):
        try:
            entries[optional_module] = _provenance_for_module(optional_module)
        except Exception:
            continue
    backend = BACKEND_REGISTRY.get(config.render.backend)
    if backend is not None:
        entries[f"backend:{config.render.backend}"] = _provenance_for_target(
            f"{backend.render_fn.__module__}.{backend.render_fn.__qualname__}"
        )
    for import_path in _import_paths(config):
        entries[f"target:{import_path}"] = _provenance_for_target(import_path)
    return entries


def _run_summary(
    config: TrainingConfig,
    run_context: TrainingRunContext | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "backend": config.render.backend,
        "checkpoint_output_dir": str(config.checkpoint.output_dir),
        "max_steps": config.runtime.max_steps,
    }
    if run_context is not None:
        summary.update(
            {
                "camera_extent": run_context.camera_extent,
                "context_max_steps": run_context.max_steps,
                "context_backend": run_context.backend,
            }
        )
    return summary


def ensure_checkpoint_output_writable(
    output_dir: str | Path,
    *,
    overwrite: bool,
    artifacts: tuple[str, ...] = (
        "config.json",
        "metadata.json",
        "model.ckpt",
        "scene.ply",
    ),
) -> None:
    """Fail before overwriting an existing checkpoint artifact."""
    if overwrite:
        return
    checkpoint_dir = Path(output_dir)
    if any((checkpoint_dir / artifact).exists() for artifact in artifacts):
        raise FileExistsError(
            "Checkpoint output directory already contains training artifacts: "
            f"{checkpoint_dir}. Set checkpoint.overwrite=true to replace them."
        )


def checkpoint_run_dir(
    output_dir: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Resolve an experiment output prefix to a concrete run directory."""
    output_path = Path(output_dir)
    if overwrite:
        return _checkpoint_run_path(output_path, 1)
    run_index = 1
    while True:
        candidate = _checkpoint_run_path(output_path, run_index)
        if not candidate.exists():
            return candidate
        run_index += 1


def _checkpoint_run_path(output_dir: Path, run_index: int) -> Path:
    return output_dir.with_name(f"{output_dir.name}_run_{run_index}")


def _scene_record_summary(scene_record: SceneRecord) -> dict[str, Any]:
    return {
        "num_frames": scene_record.num_frames,
        "source_format": scene_record.source_format,
        "source_uris": None
        if scene_record.source_uris is None
        else list(scene_record.source_uris),
        "available_camera_sensor_ids": list(
            scene_record.available_camera_sensor_ids
        ),
        "default_camera_sensor_id": scene_record.default_camera_sensor_id,
        "has_point_cloud": scene_record.point_cloud is not None,
    }


def build_checkpoint_metadata(
    state: TrainState,
    config: TrainingConfig,
    *,
    frame_dataset: PreparedFrameDataset | None = None,
    run_context: TrainingRunContext | None = None,
) -> CheckpointMetadata:
    """Build reproducibility metadata for a checkpoint directory."""
    dataset_summary = (
        {}
        if frame_dataset is None
        else _scene_record_summary(frame_dataset.scene_record)
    )
    return CheckpointMetadata(
        timestamp_utc=datetime.now(UTC).isoformat(),
        seed=state.seed,
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        scene_type=type(state.model.scene).__name__,
        backend_name=config.render.backend,
        export_ply=config.checkpoint.export_ply,
        import_paths=_import_paths(config),
        package_versions=_package_versions(),
        provenance=_provenance(config),
        run_summary=_run_summary(config, run_context),
        dataset_summary=dataset_summary,
    )


def _cpu_state_dict(module: nn.Module) -> dict[str, Any]:
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
    }


def _cpu_parameter(parameter: nn.Parameter) -> nn.Parameter:
    return nn.Parameter(
        parameter.detach().cpu(),
        requires_grad=parameter.requires_grad,
    )


def _cpu_scene(scene: Scene) -> Scene:
    return scene.to(torch.device("cpu"))


def _load_scene_from_payload(
    checkpoint_dir: Path,
    payload: dict[str, Any],
) -> Scene:
    scene = payload.get("scene")
    if scene is not None:
        return scene
    ply_path = checkpoint_dir / "scene.ply"
    if ply_path.exists():
        return load_gaussian_ply(ply_path)
    raise ValueError(
        f"Checkpoint at {checkpoint_dir} does not contain a serialized scene."
    )


def save_checkpoint_dir(
    path: str | Path,
    state: TrainState,
    config: TrainingConfig,
    *,
    frame_dataset: PreparedFrameDataset | None = None,
    run_context: TrainingRunContext | None = None,
) -> Path:
    """Save a reproducible checkpoint directory."""
    checkpoint_dir = Path(path)
    ensure_checkpoint_output_writable(
        checkpoint_dir,
        overwrite=config.checkpoint.overwrite,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _json_write(
        checkpoint_dir / "config.json",
        config.model_dump_json(indent=2),
    )
    metadata = build_checkpoint_metadata(
        state,
        config,
        frame_dataset=frame_dataset,
        run_context=run_context,
    )
    _json_write(
        checkpoint_dir / "metadata.json",
        metadata.model_dump_json(indent=2),
    )

    scene_payload: Scene | None = _cpu_scene(state.model.scene)
    if config.checkpoint.export_ply:
        if not isinstance(state.model.scene, GaussianScene3D):
            raise ValueError(
                "PLY export is only supported for GaussianScene3D checkpoints."
            )
        save_gaussian_ply(
            state.model.scene.to(torch.device("cpu")),
            checkpoint_dir / "scene.ply",
        )
        scene_payload = None

    model_payload = {
        "scene": scene_payload,
        "modules": {
            name: _cpu_state_dict(module)
            for name, module in state.model.modules.items()
        },
        "parameters": {
            name: _cpu_parameter(parameter)
            for name, parameter in state.model.parameters.items()
        },
        "buffers": {
            name: buffer.detach().cpu()
            for name, buffer in state.model.buffers.items()
        },
        "step": state.step,
    }
    torch.save(
        model_payload,
        checkpoint_dir / "model.ckpt",
    )
    return checkpoint_dir


def load_checkpoint_dir(path: str | Path) -> LoadedCheckpoint:
    """Load a checkpoint directory into an inference-ready payload."""
    checkpoint_dir = Path(path)
    config = TrainingConfig.model_validate_json(
        (checkpoint_dir / "config.json").read_text()
    )
    metadata = CheckpointMetadata.model_validate_json(
        (checkpoint_dir / "metadata.json").read_text()
    )
    payload = torch.load(
        checkpoint_dir / "model.ckpt",
        weights_only=False,
    )
    scene = _load_scene_from_payload(checkpoint_dir, payload)
    modules = build_modules(config)
    for name, state_dict in payload["modules"].items():
        modules[name].load_state_dict(state_dict)
    parameters = build_parameters(config)
    for name, parameter in payload["parameters"].items():
        parameters[name] = nn.Parameter(
            parameter.detach().clone(),
            requires_grad=parameter.requires_grad,
        )
    model = InitializedModel(
        scene=scene,
        modules=modules,
        parameters=parameters,
        buffers={
            name: buffer.detach().clone()
            for name, buffer in payload.get("buffers", {}).items()
        },
        metadata={"checkpoint_step": int(payload.get("step", 0))},
    )
    return LoadedCheckpoint(
        model=model,
        render_fn=build_render_fn(config),
        config=config,
        metadata=metadata,
    )


def build_inference_pipeline(path: str | Path) -> LoadedCheckpoint:
    """Load a checkpoint directory and rebuild its render pipeline."""
    return load_checkpoint_dir(path)


__all__ = [
    "build_checkpoint_metadata",
    "build_inference_pipeline",
    "checkpoint_run_dir",
    "load_checkpoint_dir",
    "save_checkpoint_dir",
]
