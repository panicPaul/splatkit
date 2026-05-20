"""Shared notebook utilities for foam-family paper implementations."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Protocol

import ember_core as ember

InterpolationMode = Literal["nearest", "bilinear", "bicubic"]
SplitTarget = Literal["train", "val", "all"]
MaterializationStage = Literal["none", "decoded", "prepared"]
MaterializationMode = Literal["lazy", "eager"]


class FoamSceneConfig(Protocol):
    """Scene fields shared by foam paper notebooks."""

    path: Path
    image_root: Path | None
    undistort_output_dir: Path | None
    align_horizon: bool


class FoamDataConfig(Protocol):
    """Dataset fields shared by foam paper notebooks."""

    camera_sensor_id: str | None
    image_scale_factor: float
    cache_resized_images: bool
    resized_image_cache_root: Path | None
    max_resized_image_caches: int
    split_target: SplitTarget
    split_every_n: int | None
    materialization_stage: MaterializationStage
    materialization_mode: MaterializationMode
    materialization_num_workers: int | None
    normalize_images: bool
    interpolation: InterpolationMode


class FoamExperimentConfig(Protocol):
    """Experiment fields shared by foam paper notebooks."""

    scene: FoamSceneConfig
    data: FoamDataConfig


def foam_resized_cache_enabled(config: FoamExperimentConfig) -> bool:
    """Return whether a derived resized image cache should be used."""
    return (
        config.data.cache_resized_images
        and config.data.image_scale_factor != 1.0
    )


def foam_resized_cache_parent(config: FoamExperimentConfig) -> Path:
    """Return the reusable derived image cache parent for the scene."""
    if config.data.resized_image_cache_root is not None:
        return config.data.resized_image_cache_root.expanduser()
    return config.scene.path.expanduser() / "ember_cache" / "resized_images"


def foam_source_image_root(config: FoamExperimentConfig) -> Path:
    """Return the full-resolution source image root."""
    if config.scene.image_root is not None:
        return config.scene.image_root.expanduser()
    return config.scene.path.expanduser() / "images"


def foam_resized_cache_root(config: FoamExperimentConfig) -> Path:
    """Return the derived resized image cache root for this config."""
    scale_name = (
        f"{config.data.image_scale_factor:.6f}".rstrip("0").rstrip(".")
    )
    scale_name = scale_name.replace(".", "p")
    return foam_resized_cache_parent(config) / (
        f"scale_{scale_name}_{config.data.interpolation}"
    )


def foam_pillow_resampling(interpolation: InterpolationMode) -> object:
    """Translate notebook interpolation names to Pillow resampling filters."""
    from PIL import Image

    if interpolation == "nearest":
        return Image.Resampling.NEAREST
    if interpolation == "bilinear":
        return Image.Resampling.BILINEAR
    if interpolation == "bicubic":
        return Image.Resampling.BICUBIC
    raise ValueError(f"Unsupported interpolation mode {interpolation!r}.")


def _image_paths(source_root: Path) -> Iterable[Path]:
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return (
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )


def enforce_foam_resized_cache_limit(
    *,
    cache_root: Path,
    max_caches: int,
) -> None:
    """Keep only a bounded number of reusable resized image caches."""
    parent = cache_root.parent
    if not parent.exists():
        return
    cache_dirs = [
        path
        for path in parent.iterdir()
        if path.is_dir() and path.name.startswith("scale_")
    ]
    overflow = len(cache_dirs) - max_caches
    if overflow <= 0:
        return
    evictable = sorted(
        (path for path in cache_dirs if path != cache_root),
        key=lambda path: path.stat().st_mtime,
    )
    for stale_cache in evictable[:overflow]:
        shutil.rmtree(stale_cache)


def materialize_foam_resized_image_cache(
    *,
    source_root: Path,
    cache_root: Path,
    scale: float,
    interpolation: InterpolationMode,
    max_caches: int,
) -> Path:
    """Create or update a derived resized image cache from full-res images."""
    from PIL import Image
    from tqdm.auto import tqdm

    source_paths = sorted(_image_paths(source_root))
    if not source_paths:
        raise ValueError(f"No source images found under {source_root}.")
    resampling = foam_pillow_resampling(interpolation)
    enforce_foam_resized_cache_limit(
        cache_root=cache_root,
        max_caches=max_caches,
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    def resize_one(source_path: Path) -> None:
        relative_path = source_path.relative_to(source_root)
        target_path = cache_root / relative_path
        if (
            target_path.exists()
            and target_path.stat().st_mtime >= source_path.stat().st_mtime
        ):
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            resized_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            resized = rgb.resize(resized_size, resampling)
            save_kwargs = (
                {"quality": 95}
                if target_path.suffix.lower() in {".jpg", ".jpeg"}
                else {}
            )
            resized.save(target_path, **save_kwargs)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(resize_one, path) for path in source_paths]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preparing resized image cache",
        ):
            future.result()
    (cache_root / "cache_metadata.json").write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "scale": scale,
                "interpolation": interpolation,
                "num_images": len(source_paths),
            },
            indent=2,
            sort_keys=True,
        )
    )
    cache_root.touch()
    enforce_foam_resized_cache_limit(
        cache_root=cache_root,
        max_caches=max_caches,
    )
    return cache_root


def build_foam_scene_load_config(
    config: FoamExperimentConfig,
) -> ember.ColmapSceneConfig:
    """Translate a foam paper config into an Ember scene loader config."""
    source_pipes = (
        (ember.HorizonAlignPipeConfig(),) if config.scene.align_horizon else ()
    )
    image_root = (
        materialize_foam_resized_image_cache(
            source_root=foam_source_image_root(config),
            cache_root=foam_resized_cache_root(config),
            scale=config.data.image_scale_factor,
            interpolation=config.data.interpolation,
            max_caches=config.data.max_resized_image_caches,
        )
        if foam_resized_cache_enabled(config)
        else (
            config.scene.image_root.expanduser()
            if config.scene.image_root is not None
            else None
        )
    )
    return ember.ColmapSceneConfig(
        path=config.scene.path.expanduser(),
        image_root=image_root,
        undistort_output_dir=(
            config.scene.undistort_output_dir.expanduser()
            if config.scene.undistort_output_dir is not None
            else None
        ),
        source_pipes=source_pipes,
    )


def build_foam_prepared_frame_dataset_config(
    config: FoamExperimentConfig,
) -> ember.PreparedFrameDatasetConfig:
    """Translate a foam paper config into an Ember frame dataset config."""
    split = (
        ember.SplitConfig(target="all", every_n=None, train_ratio=None)
        if config.data.split_target == "all"
        else ember.SplitConfig(
            target=config.data.split_target,
            every_n=config.data.split_every_n,
            train_ratio=None,
        )
    )
    return ember.PreparedFrameDatasetConfig(
        camera_sensor_id=config.data.camera_sensor_id,
        split=split,
        materialization=ember.MaterializationConfig(
            stage=config.data.materialization_stage,
            mode=config.data.materialization_mode,
            num_workers=config.data.materialization_num_workers,
        ),
        image_preparation=ember.ImagePreparationConfig(
            normalize=config.data.normalize_images,
            resize_width_scale=(
                None
                if foam_resized_cache_enabled(config)
                else config.data.image_scale_factor
            ),
            resize_width_target=None,
            interpolation=config.data.interpolation,
        ),
    )
