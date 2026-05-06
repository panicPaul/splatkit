"""CLI entrypoint for COLMAP dataloader benchmarks."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ember_core.benchmarks import benchmark_dataloader
from ember_core.data import (
    ImagePreparationConfig,
    MaterializationConfig,
    PreparedFrameDataset,
    PreparedFrameDatasetConfig,
    collate_frame_samples,
    load_colmap_scene_record,
    resolve_colmap_scene_path,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark ember-core COLMAP dataloader iteration speed."
    )
    parser.add_argument("--colmap-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-iters-per-sec", type=float, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--materialization-stage", default="prepared")
    parser.add_argument("--materialization-mode", default="eager")
    parser.add_argument("--materialization-num-workers", type=int, default=8)
    parser.add_argument("--resize-width-target", type=int, default=None)
    parser.add_argument("--resize-width-scale", type=float, default=None)
    parser.add_argument("--interpolation", default="bicubic")
    parser.add_argument(
        "--cache-resized-images",
        action="store_true",
        help=(
            "Materialize a derived resized image cache and load from it. "
            "Requires --resize-width-scale and keeps full-res images as source."
        ),
    )
    parser.add_argument(
        "--resized-image-cache-root",
        type=Path,
        default=None,
        help=(
            "Parent directory for resized image caches. Defaults to "
            "<colmap-root>/ember_cache/resized_images."
        ),
    )
    parser.add_argument("--max-resized-image-caches", type=int, default=4)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable image normalization in prepared samples.",
    )
    return parser


def _pillow_resampling(interpolation: str) -> Any:
    from PIL import Image

    if interpolation == "nearest":
        return Image.Resampling.NEAREST
    if interpolation == "bilinear":
        return Image.Resampling.BILINEAR
    if interpolation == "bicubic":
        return Image.Resampling.BICUBIC
    raise ValueError(f"Unsupported interpolation mode {interpolation!r}.")


def _resized_cache_root(args: argparse.Namespace, dataset_root: Path) -> Path:
    parent = (
        args.resized_image_cache_root
        if args.resized_image_cache_root is not None
        else dataset_root / "ember_cache" / "resized_images"
    )
    scale_name = f"{args.resize_width_scale:.6f}".rstrip("0").rstrip(".")
    scale_name = scale_name.replace(".", "p")
    return parent / f"scale_{scale_name}_{args.interpolation}"


def _enforce_resized_cache_limit(cache_root: Path, max_caches: int) -> None:
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


def _materialize_resized_image_cache(
    args: argparse.Namespace,
    dataset_root: Path,
) -> Path:
    if args.resize_width_scale is None:
        raise ValueError(
            "--cache-resized-images requires --resize-width-scale."
        )
    source_root = dataset_root / "images"
    if not source_root.exists():
        raise ValueError(f"No source image directory found at {source_root}.")
    cache_root = _resized_cache_root(args, dataset_root)
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    source_paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    if not source_paths:
        raise ValueError(f"No source images found under {source_root}.")

    from tqdm.auto import tqdm

    resampling = _pillow_resampling(args.interpolation)
    _enforce_resized_cache_limit(cache_root, args.max_resized_image_caches)
    cache_root.mkdir(parents=True, exist_ok=True)

    def resize_one(source_path: Path) -> None:
        from PIL import Image

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
            resized = rgb.resize(
                (
                    max(1, round(width * args.resize_width_scale)),
                    max(1, round(height * args.resize_width_scale)),
                ),
                resampling,
            )
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
            disable=not args.progress,
        ):
            future.result()

    (cache_root / "cache_metadata.json").write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "scale": args.resize_width_scale,
                "interpolation": args.interpolation,
                "num_images": len(source_paths),
            },
            indent=2,
            sort_keys=True,
        )
    )
    cache_root.touch()
    _enforce_resized_cache_limit(cache_root, args.max_resized_image_caches)
    return cache_root


def _build_dataloader(args: argparse.Namespace) -> DataLoader[object]:
    dataset_root = resolve_colmap_scene_path(args.colmap_root)
    cached_image_root = (
        _materialize_resized_image_cache(args, dataset_root)
        if args.cache_resized_images
        else None
    )
    scene_record = load_colmap_scene_record(
        dataset_root,
        image_root=cached_image_root,
    )
    frame_dataset = PreparedFrameDataset(
        scene_record,
        config=PreparedFrameDatasetConfig(
            split=None,
            materialization=MaterializationConfig(
                stage=args.materialization_stage,
                mode=args.materialization_mode,
                num_workers=args.materialization_num_workers,
            ),
            image_preparation=ImagePreparationConfig(
                normalize=not args.no_normalize,
                resize_width_scale=(
                    None if cached_image_root is not None else args.resize_width_scale
                ),
                resize_width_target=args.resize_width_target,
                interpolation=args.interpolation,
            ),
        ),
    )
    return DataLoader(
        frame_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        persistent_workers=(
            args.persistent_workers and args.dataloader_num_workers > 0
        ),
        pin_memory=args.pin_memory,
        collate_fn=collate_frame_samples,
    )


def _move_batch_to_device(batch: object, device: torch.device) -> object:
    moved = batch.to(device, non_blocking=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return moved


def main() -> None:
    """Run the dataloader benchmark CLI and print JSON metrics."""
    args = _build_parser().parse_args()
    device = torch.device(args.device) if args.device is not None else None
    result = benchmark_dataloader(
        _build_dataloader(args),
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        prepare_batch=(
            None
            if device is None
            else lambda batch: _move_batch_to_device(batch, device)
        ),
        show_progress=args.progress,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if (
        args.min_iters_per_sec is not None
        and result.iters_per_sec < args.min_iters_per_sec
    ):
        raise SystemExit(
            "Dataloader benchmark below threshold: "
            f"{result.iters_per_sec:.2f} it/s < "
            f"{args.min_iters_per_sec:.2f} it/s"
        )


if __name__ == "__main__":
    main()
