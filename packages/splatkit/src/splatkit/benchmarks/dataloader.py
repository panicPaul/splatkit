"""CLI entrypoint for COLMAP dataloader benchmarks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

from splatkit.benchmarks import benchmark_dataloader
from splatkit.data import (
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
        description="Benchmark splatkit COLMAP dataloader iteration speed."
    )
    parser.add_argument("--colmap-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--materialization-stage", default="prepared")
    parser.add_argument("--materialization-mode", default="eager")
    parser.add_argument("--materialization-num-workers", type=int, default=0)
    parser.add_argument("--resize-width-target", type=int, default=None)
    parser.add_argument("--resize-width-scale", type=float, default=None)
    parser.add_argument("--interpolation", default="bicubic")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable image normalization in prepared samples.",
    )
    return parser


def _build_dataloader(args: argparse.Namespace) -> DataLoader[object]:
    dataset_root = resolve_colmap_scene_path(args.colmap_root)
    scene_record = load_colmap_scene_record(dataset_root)
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
                resize_width_scale=args.resize_width_scale,
                resize_width_target=args.resize_width_target,
                interpolation=args.interpolation,
            ),
        ),
    )
    return DataLoader(
        frame_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_frame_samples,
    )


def main() -> None:
    """Run the dataloader benchmark CLI and print JSON metrics."""
    args = _build_parser().parse_args()
    result = benchmark_dataloader(
        _build_dataloader(args),
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
