"""Sparse-voxel checkpoint import/export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from splatkit.core.contracts import SparseVoxelScene
from splatkit.core.sparse_voxel import SUPPORTED_SVRASTER_BACKENDS


def _dequantize_entry(entry: Any) -> Any:
    if isinstance(entry, dict) and set(entry) == {"index", "codebook"}:
        return entry["codebook"][entry["index"].long()]
    if isinstance(entry, list):
        return torch.cat([_dequantize_entry(value) for value in entry], dim=1)
    return entry


def _dequantize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict.get("quantized", False):
        return state_dict
    return {
        **state_dict,
        "_geo_grid_pts": _dequantize_entry(state_dict["_geo_grid_pts"]),
        "_sh0": _dequantize_entry(state_dict["_sh0"]),
        "_shs": _dequantize_entry(state_dict["_shs"]),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}.")
    return data


def _resolve_svraster_checkpoint_path(
    path: str | Path,
    iteration: int | None,
) -> tuple[Path, Path | None]:
    candidate = Path(path)
    if candidate.is_dir():
        config_path = candidate / "config.yaml"
        checkpoints_dir = candidate / "checkpoints"
        if not checkpoints_dir.exists():
            raise FileNotFoundError(
                f"SV Raster run directory is missing checkpoints/: {candidate}"
            )
        checkpoints = sorted(checkpoints_dir.glob("iter*_model.pt"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No SV Raster checkpoints found in {checkpoints_dir}."
            )
        if iteration is None:
            return checkpoints[-1], config_path if config_path.exists() else None
        return (
            checkpoints_dir / f"iter{iteration:06d}_model.pt",
            config_path if config_path.exists() else None,
        )
    config_path = None
    if candidate.parent.name == "checkpoints":
        sibling_config = candidate.parent.parent / "config.yaml"
        if sibling_config.exists():
            config_path = sibling_config
    return candidate, config_path


def load_svraster_checkpoint(
    path: str | Path,
    iteration: int | None = None,
) -> SparseVoxelScene:
    """Load a scene-only SV Raster checkpoint."""
    checkpoint_path, config_path = _resolve_svraster_checkpoint_path(
        path,
        iteration,
    )
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected checkpoint mapping in {checkpoint_path}, got "
            f"{type(state_dict).__name__}."
        )
    state_dict = _dequantize_state_dict(state_dict)
    backend_name = state_dict.get("backend_name", "new_cuda")
    color_layout = state_dict.get("color_layout", "voxel")
    geo_layout = state_dict.get("geo_layout", "trilinear")
    if backend_name not in SUPPORTED_SVRASTER_BACKENDS:
        raise RuntimeError(
            f"Unsupported SV Raster backend {backend_name!r}. "
            f"Supported backends: {sorted(SUPPORTED_SVRASTER_BACKENDS)}."
        )
    if color_layout != "voxel":
        raise RuntimeError(
            "Only voxel-color SV Raster checkpoints are supported; got "
            f"{color_layout!r}."
        )
    if geo_layout != "trilinear":
        raise RuntimeError(
            "Only trilinear-geometry SV Raster checkpoints are supported; got "
            f"{geo_layout!r}."
        )
    max_num_levels = 16
    if config_path is not None:
        config = _load_yaml(config_path)
        model_config = config.get("model", {})
        if isinstance(model_config, dict):
            max_num_levels = int(model_config.get("max_num_levels", 16))

    return SparseVoxelScene(
        backend_name=backend_name,
        active_sh_degree=int(state_dict["active_sh_degree"]),
        max_num_levels=max_num_levels,
        scene_center=state_dict["scene_center"].reshape(3).to(torch.float32),
        scene_extent=state_dict["scene_extent"].reshape(1).to(torch.float32),
        inside_extent=state_dict["inside_extent"].reshape(1).to(torch.float32),
        octpath=state_dict["octpath"].reshape(-1, 1).to(torch.int64),
        octlevel=state_dict["octlevel"].reshape(-1, 1).to(torch.int8),
        geo_grid_pts=state_dict["_geo_grid_pts"].reshape(-1, 1).to(
            torch.float32
        ),
        sh0=state_dict["_sh0"].to(torch.float32),
        shs=state_dict["_shs"].to(torch.float32),
    )


def save_svraster_checkpoint(
    scene: SparseVoxelScene,
    path: str | Path,
    iteration: int | None = None,
) -> None:
    """Save a SparseVoxelScene in the scene-only SV Raster checkpoint schema."""
    output_path = Path(path)
    if output_path.suffix != ".pt":
        iteration_value = 0 if iteration is None else iteration
        output_path = (
            output_path
            / "checkpoints"
            / f"iter{iteration_value:06d}_model.pt"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        "backend_name": scene.backend_name,
        "color_layout": "voxel",
        "geo_layout": "trilinear",
        "active_sh_degree": scene.active_sh_degree,
        "scene_center": scene.scene_center.detach().cpu().contiguous(),
        "inside_extent": scene.inside_extent.detach().cpu().contiguous(),
        "scene_extent": scene.scene_extent.detach().cpu().contiguous(),
        "octpath": scene.octpath.detach().cpu().contiguous(),
        "octlevel": scene.octlevel.detach().cpu().contiguous(),
        "_geo_grid_pts": scene.geo_grid_pts.detach().cpu().contiguous(),
        "_sh0": scene.sh0.detach().cpu().contiguous(),
        "_shs": scene.shs.detach().cpu().contiguous(),
        "quantized": False,
    }
    torch.save(state_dict, output_path)
