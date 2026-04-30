"""Interactive COLMAP dataset loading notebook."""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="columns")

with app.setup:
    import json
    import os
    import shutil
    from dataclasses import dataclass
    from pathlib import Path
    from time import perf_counter
    from typing import Any, Literal

    import ember_core as ember
    import marimo as mo
    import numpy as np
    import torch
    from marimo_3dv import CameraState, Viewer, ViewerState
    from marimo_config_gui import create_config_gui
    from pydantic import BaseModel, ConfigDict, Field, create_model
    from torch.utils.data import DataLoader

    DEFAULT_COLMAP_PATH = Path(
        os.environ.get(
            "EMBER_COLMAP_ROOT",
            "/home/schlack/Documents/3DGS_scenes/360/garden",
        )
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Test Dataloading

    Load a COLMAP dataset, inspect its sparse point cloud, preview frames, and
    benchmark dataloader throughput with the same resized-image cache path used
    by training.
    """)
    return


@app.cell(hide_code=True)
def _(dataset_gui):
    dataset_gui.gui_panel()
    return


@app.cell(hide_code=True)
def _(point_cloud_gui):
    point_cloud_gui.gui_panel()
    return


@app.cell(hide_code=True)
def _(benchmark_gui):
    benchmark_gui.gui_panel()
    return


@app.cell(hide_code=True)
def _(benchmark_button):
    benchmark_button
    return


@app.cell
def _(benchmark_config, dataset_config, sample_button):
    _ = sample_button.value
    dataloader = load_dataloader(dataset_config, benchmark_config, shuffle=True)
    dataloader_sample = next(iter(dataloader))
    return (dataloader_sample,)


@app.cell
def _(benchmark_button, benchmark_config, dataset_config):
    benchmark_report = (
        run_dataloader_benchmark(dataset_config, benchmark_config)
        if bool(benchmark_button.value)
        else None
    )
    return (benchmark_report,)


@app.cell(hide_code=True)
def _(benchmark_report):
    if benchmark_report is None:
        benchmark_view = mo.callout(
            "Submit the dataloader benchmark form to measure worker throughput.",
            kind="info",
        )
    else:
        normal_result = benchmark_report.worker_results[
            benchmark_report.normal_workers
        ]
        worker_summary = ", ".join(
            f"`{num_workers}` workers: `{result.iters_per_sec:.1f}` it/s"
            for num_workers, result in sorted(
                benchmark_report.worker_results.items()
            )
        )
        benchmark_view = mo.md(
            "Dataloader: "
            f"`{normal_result.iters_per_sec:.1f}` it/s "
            f"(`{normal_result.mean_ms_per_batch:.3f}` ms/batch, "
            f"p90 `{normal_result.p90_ms_per_batch:.3f}` ms) with "
            f"`{benchmark_report.normal_workers}` workers.\n\n"
            "Prepared image size: "
            f"`{benchmark_report.image_width} x "
            f"{benchmark_report.image_height}` px.\n\n"
            "Single-process stage breakdown: "
            f"decode `{benchmark_report.breakdown.decode_ms:.3f}` ms, "
            f"prepare `{benchmark_report.breakdown.prepare_ms:.3f}` ms, "
            f"collate `{benchmark_report.breakdown.collate_ms:.3f}` ms, "
            f"transfer `{benchmark_report.breakdown.transfer_ms:.3f}` ms.\n\n"
            f"Worker sweep: {worker_summary}."
        )
    benchmark_view
    return


@app.cell(hide_code=True)
def _(viewer):
    viewer
    return


@app.cell(hide_code=True)
def _(sample_button):
    sample_button
    return


@app.cell
def _(dataloader_sample):
    dataloader_sample.images.shape
    return


@app.cell(hide_code=True)
def _(dataloader_sample):
    mo.image(
        image_tensor_to_uint8(dataloader_sample.images[0]),
        width="100%",
        caption="Dataloader sample tensor",
    )
    return


@app.cell(hide_code=True)
def _(dataset):
    num_points = (
        0
        if dataset.point_cloud is None
        else dataset.point_cloud.points.shape[0]
    )
    summary = mo.callout(
        (
            f"Loaded `{dataset.root_path}` with {dataset.num_frames} frames "
            f"and {num_points} sparse points."
        ),
        kind="success",
    )
    summary
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md("""
    # Configuration
    """)
    return


@app.class_definition
class DatasetConfig(BaseModel):
    path: Path = DEFAULT_COLMAP_PATH
    write_undistorted_cache: bool = False
    undistort_output_dir: Path = Path(
        os.environ.get(
            "EMBER_COLMAP_UNDISTORTED",
            str(DEFAULT_COLMAP_PATH / "undistorted"),
        )
    )
    apply_horizon_adjustment: bool = True


@app.class_definition
class PointCloudRenderConfig(BaseModel):
    max_points: int = Field(default=50000, ge=1000, le=500000)
    point_radius: int = Field(default=2, ge=1, le=6)
    background_brightness: int = Field(default=18, ge=0, le=255)


@app.class_definition
class DataloaderBenchmarkConfig(BaseModel):
    image_scale_factor: float = Field(default=0.25, gt=0.0)
    cache_resized_images: bool = True
    resized_image_cache_root: Path | None = None
    max_resized_image_caches: int = Field(default=4, ge=1)
    split_target: Literal["train", "val", "all"] = "all"
    split_every_n: int | None = Field(default=8, ge=1)
    warmup_steps: int = Field(default=50, ge=0)
    measured_steps: int = Field(default=300, ge=1)
    breakdown_steps: int = Field(default=16, ge=1)
    batch_size: int = Field(default=1, ge=1)
    normal_num_workers: int = Field(default=4, ge=0)
    worker_counts: str = "0,2,4,8"
    persistent_workers: bool = True
    pin_memory: bool = True
    device: str = "cuda"
    normalize_images: bool = True
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic"


@app.cell
def _():
    dataset_gui = create_config_gui(
        DatasetConfig,
        value=DatasetConfig(),
        label="COLMAP Dataset",
    )
    return (dataset_gui,)


@app.cell
def _():
    point_cloud_gui = create_config_gui(
        PointCloudRenderConfig,
        value=PointCloudRenderConfig(),
        label="Point Cloud Render",
    )
    return (point_cloud_gui,)


@app.cell
def _():
    benchmark_gui = create_config_gui(
        DataloaderBenchmarkConfig,
        value=DataloaderBenchmarkConfig(),
        label="Dataloader Benchmark",
    )
    return (benchmark_gui,)


@app.cell
def _():
    benchmark_button = mo.ui.run_button(label="Benchmark dataloader")
    return (benchmark_button,)


@app.cell
def _():
    sample_button = mo.ui.run_button(label="Sample dataloader image")
    return (sample_button,)


@app.cell
def _():
    viewer_state = ViewerState(camera_convention="opencv")
    return (viewer_state,)


@app.cell
def _(dataset_gui):
    dataset_config = dataset_gui.validated_config()
    return (dataset_config,)


@app.cell
def _(benchmark_gui):
    benchmark_config = benchmark_gui.validated_config()
    return (benchmark_config,)


@app.cell
def _(point_cloud_gui):
    point_cloud_config = point_cloud_gui.validated_config()
    return (point_cloud_config,)


@app.cell
def _(dataset_config):
    dataset = load_colmap_scene_record_from_form(dataset_config)
    return (dataset,)


@app.cell
def _(dataset):
    frame_count = dataset.num_frames
    FrameSelectionConfig = create_model(
        "FrameSelectionConfig",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        frame_index=(
            int,
            Field(default=0, ge=0, le=max(0, frame_count - 1)),
        ),
    )
    frame_gui = create_config_gui(
        FrameSelectionConfig,
        value=FrameSelectionConfig(frame_index=0),
        label="Frame",
    )
    return (frame_gui,)


@app.cell
def _(frame_gui):
    selected_frame_index = frame_gui.validated_config().frame_index
    return (selected_frame_index,)


@app.cell
def _(dataset, selected_frame_index):
    selected_camera = dataset_frame_to_viewer_camera(
        dataset,
        selected_frame_index,
    )
    return (selected_camera,)


@app.cell
def _(selected_camera, viewer_state):
    viewer_state.set_camera(selected_camera)
    return


@app.cell
def _(dataset, point_cloud_config, viewer_state):
    def render_frame(camera_state):
        return point_cloud_render(
            camera_state,
            dataset,
            point_cloud_config,
        )

    viewer = Viewer(render_frame, state=viewer_state)
    return (viewer,)


@app.cell(column=2, hide_code=True)
def _():
    mo.md("""
    # Benchmark Helpers
    """)
    return


@app.function
def load_colmap_scene_record_from_form(load_config):
    """Load a COLMAP scene record from a form config."""
    undistort_output_dir = (
        load_config.undistort_output_dir
        if load_config.write_undistorted_cache
        else None
    )
    return ember.load_scene_record(
        ember.ColmapSceneConfig(
            path=load_config.path,
            undistort_output_dir=undistort_output_dir,
            source_pipes=(
                ember.HorizonAlignPipeConfig(
                    enabled=load_config.apply_horizon_adjustment
                ),
            ),
        )
    )


@app.class_definition
@dataclass(frozen=True)
class DataloaderBreakdown:
    """Single-process timing breakdown for one prepared dataset."""

    decode_ms: float
    prepare_ms: float
    collate_ms: float
    transfer_ms: float


@app.class_definition
@dataclass(frozen=True)
class DataloaderBenchmarkReport:
    """Dataloader worker sweep plus local stage timings."""

    normal_workers: int
    worker_results: dict[int, Any]
    breakdown: DataloaderBreakdown
    image_width: int
    image_height: int


@app.function
def parse_worker_counts(
    worker_counts: str,
    normal_num_workers: int,
) -> tuple[int, ...]:
    """Parse a comma-separated worker sweep string."""
    parsed = {
        int(token.strip())
        for token in worker_counts.split(",")
        if token.strip()
    }
    parsed.add(int(normal_num_workers))
    if any(worker < 0 for worker in parsed):
        raise ValueError("Worker counts must be non-negative.")
    return tuple(sorted(parsed))


@app.function
def benchmark_cache_enabled(config) -> bool:
    """Return whether benchmark loading should use a resized image cache."""
    return config.cache_resized_images and config.image_scale_factor != 1.0


@app.function
def benchmark_source_image_root(dataset_config) -> Path:
    """Return the full-resolution source image root."""
    return dataset_config.path.expanduser() / "images"


@app.function
def benchmark_resized_cache_parent(dataset_config, benchmark_config) -> Path:
    """Return the reusable resized image cache parent."""
    if benchmark_config.resized_image_cache_root is not None:
        return benchmark_config.resized_image_cache_root.expanduser()
    return (
        dataset_config.path.expanduser()
        / "ember_cache"
        / "resized_images"
    )


@app.function
def benchmark_resized_cache_root(dataset_config, benchmark_config) -> Path:
    """Return the derived resized image cache root."""
    scale_name = (
        f"{benchmark_config.image_scale_factor:.6f}".rstrip("0").rstrip(".")
    )
    scale_name = scale_name.replace(".", "p")
    return benchmark_resized_cache_parent(dataset_config, benchmark_config) / (
        f"scale_{scale_name}_{benchmark_config.interpolation}"
    )


@app.function
def benchmark_pillow_resampling(interpolation: str) -> Any:
    """Translate interpolation names to Pillow resampling filters."""
    from PIL import Image

    if interpolation == "nearest":
        return Image.Resampling.NEAREST
    if interpolation == "bilinear":
        return Image.Resampling.BILINEAR
    if interpolation == "bicubic":
        return Image.Resampling.BICUBIC
    raise ValueError(f"Unsupported interpolation mode {interpolation!r}.")


@app.function
def enforce_resized_cache_limit(cache_root: Path, max_caches: int) -> None:
    """Keep only a bounded number of resized image caches."""
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


@app.function
def materialize_resized_image_cache(
    *,
    source_root: Path,
    cache_root: Path,
    scale: float,
    interpolation: str,
    max_caches: int,
) -> Path:
    """Create/update a derived resized image cache from full-res images."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from PIL import Image
    from tqdm.auto import tqdm

    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    source_paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    if not source_paths:
        raise ValueError(f"No source images found under {source_root}.")
    resampling = benchmark_pillow_resampling(interpolation)
    enforce_resized_cache_limit(cache_root, max_caches)
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
            resized = rgb.resize(
                (
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
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
    enforce_resized_cache_limit(cache_root, max_caches)
    return cache_root


@app.function
def load_benchmark_scene_record(dataset_config, benchmark_config):
    """Load scene record for the benchmark, optionally through a resized cache."""
    image_root = (
        materialize_resized_image_cache(
            source_root=benchmark_source_image_root(dataset_config),
            cache_root=benchmark_resized_cache_root(
                dataset_config, benchmark_config
            ),
            scale=benchmark_config.image_scale_factor,
            interpolation=benchmark_config.interpolation,
            max_caches=benchmark_config.max_resized_image_caches,
        )
        if benchmark_cache_enabled(benchmark_config)
        else None
    )
    return ember.load_scene_record(
        ember.ColmapSceneConfig(
            path=dataset_config.path,
            image_root=image_root,
            source_pipes=(
                ember.HorizonAlignPipeConfig(
                    enabled=dataset_config.apply_horizon_adjustment
                ),
            ),
        )
    )


@app.function
def build_benchmark_frame_dataset(scene_record, benchmark_config):
    """Build the prepared dataset used by the dataloader benchmark."""
    split = (
        ember.SplitConfig(target="all", every_n=None, train_ratio=None)
        if benchmark_config.split_target == "all"
        else ember.SplitConfig(
            target=benchmark_config.split_target,
            every_n=benchmark_config.split_every_n,
            train_ratio=None,
        )
    )
    return ember.PreparedFrameDataset(
        scene_record,
        config=ember.PreparedFrameDatasetConfig(
            split=split,
            materialization=ember.MaterializationConfig(
                stage="none",
                mode="lazy",
                num_workers=0,
            ),
            image_preparation=ember.ImagePreparationConfig(
                normalize=benchmark_config.normalize_images,
                resize_width_scale=(
                    None
                    if benchmark_cache_enabled(benchmark_config)
                    else benchmark_config.image_scale_factor
                ),
                interpolation=benchmark_config.interpolation,
            ),
        ),
    )


@app.function
def benchmark_sample_path(
    frame_dataset,
    *,
    device: torch.device,
    measured_steps: int,
) -> DataloaderBreakdown:
    """Time direct single-process decode/prepare/collate/transfer stages."""
    from ember_core.data.preprocess import prepare_decoded_image_and_camera

    decode_samples_ms = []
    prepare_samples_ms = []
    collate_samples_ms = []
    transfer_samples_ms = []
    for index in range(measured_steps):
        sample_index = index % len(frame_dataset)
        decode_start = perf_counter()
        decoded_sample = frame_dataset._decode_sample(sample_index)
        decode_samples_ms.append((perf_counter() - decode_start) * 1000.0)

        prepare_start = perf_counter()
        image, camera = prepare_decoded_image_and_camera(
            decoded_sample.image,
            decoded_sample.camera,
            frame_dataset.preparation,
        )
        prepared_sample = ember.PreparedFrameSample(
            frame=decoded_sample.frame,
            image=image,
            camera=camera,
        )
        prepare_samples_ms.append((perf_counter() - prepare_start) * 1000.0)

        collate_start = perf_counter()
        batch = ember.collate_frame_samples([prepared_sample])
        collate_samples_ms.append((perf_counter() - collate_start) * 1000.0)

        transfer_start = perf_counter()
        batch.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_samples_ms.append((perf_counter() - transfer_start) * 1000.0)

    def mean(samples: list[float]) -> float:
        return sum(samples) / len(samples) if samples else 0.0

    return DataloaderBreakdown(
        decode_ms=mean(decode_samples_ms),
        prepare_ms=mean(prepare_samples_ms),
        collate_ms=mean(collate_samples_ms),
        transfer_ms=mean(transfer_samples_ms),
    )


@app.function
def load_dataloader(
    dataset_config,
    benchmark_config,
    *,
    shuffle: bool = False,
) -> DataLoader:
    """Load a prepared-frame dataloader for the benchmark config."""
    scene_record = load_benchmark_scene_record(
        dataset_config,
        benchmark_config,
    )
    frame_dataset = build_benchmark_frame_dataset(
        scene_record,
        benchmark_config,
    )
    dataloader = torch.utils.data.DataLoader(
        frame_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=ember.collate_frame_samples,
    )
    return dataloader


@app.function
def image_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a dataloader image tensor to displayable uint8 RGB."""
    image = image.detach().cpu()
    if torch.is_floating_point(image):
        image = image / 255.0 if float(image.max()) > 1.5 else image
        image = torch.clamp(image, 0.0, 1.0) * 255.0
    else:
        image = torch.clamp(image, 0, 255)
    return image.to(torch.uint8).numpy()


@app.function
def benchmark_prepared_image_size(frame_dataset) -> tuple[int, int]:
    """Return the prepared sample image size as width, height."""
    sample = frame_dataset[0]
    height, width = sample.image.shape[:2]
    return int(width), int(height)


@app.function
def run_dataloader_benchmark(dataset_config, benchmark_config):
    """Run the worker sweep and stage breakdown for the selected dataset."""
    from ember_core.benchmarks import benchmark_dataloader

    scene_record = load_benchmark_scene_record(
        dataset_config, benchmark_config
    )
    frame_dataset = build_benchmark_frame_dataset(
        scene_record, benchmark_config
    )
    image_width, image_height = benchmark_prepared_image_size(frame_dataset)
    device = torch.device(benchmark_config.device)

    def move_batch_to_device(batch):
        moved = batch.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return moved

    worker_results = {}
    for num_workers in parse_worker_counts(
        benchmark_config.worker_counts,
        benchmark_config.normal_num_workers,
    ):
        dataloader = torch.utils.data.DataLoader(
            frame_dataset,
            batch_size=benchmark_config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(
                benchmark_config.persistent_workers and num_workers > 0
            ),
            pin_memory=benchmark_config.pin_memory,
            collate_fn=ember.collate_frame_samples,
        )
        worker_results[num_workers] = benchmark_dataloader(
            dataloader,
            warmup_steps=benchmark_config.warmup_steps,
            measured_steps=benchmark_config.measured_steps,
            prepare_batch=move_batch_to_device,
            show_progress=True,
        )
    breakdown = benchmark_sample_path(
        frame_dataset,
        device=device,
        measured_steps=min(benchmark_config.breakdown_steps, len(frame_dataset)),
    )
    return DataloaderBenchmarkReport(
        normal_workers=benchmark_config.normal_num_workers,
        worker_results=worker_results,
        breakdown=breakdown,
        image_width=image_width,
        image_height=image_height,
    )


@app.cell(column=3, hide_code=True)
def _():
    mo.md("""
    # Viewer Helpers
    """)
    return


@app.function
def dataset_frame_to_viewer_camera(dataset, frame_index):
    """Convert one dataset frame camera into a marimo-3dv camera state."""
    backend_camera = ember.CameraState(
        width=dataset.camera.width[frame_index : frame_index + 1],
        height=dataset.camera.height[frame_index : frame_index + 1],
        fov_degrees=dataset.camera.fov_degrees[frame_index : frame_index + 1],
        cam_to_world=dataset.camera.cam_to_world[frame_index : frame_index + 1],
        intrinsics=(
            None
            if dataset.camera.intrinsics is None
            else dataset.camera.intrinsics[frame_index : frame_index + 1]
        ),
        camera_convention=dataset.camera.camera_convention,
        up_direction=dataset.camera.up_direction,
    )
    return CameraState(
        fov_degrees=float(backend_camera.fov_degrees[0].item()),
        width=int(backend_camera.width[0].item()),
        height=int(backend_camera.height[0].item()),
        cam_to_world=backend_camera.cam_to_world[0].cpu().numpy(),
        camera_convention="opencv",
    )


@app.function
def dataset_frame_image_path(dataset, frame) -> Path:
    """Return the backing image path for a dataset frame."""
    image_source = dataset.resolve_camera_sensor().image_source
    return image_source.path_for_frame(frame)


@app.function
def dataset_frame_image(dataset, frame) -> np.ndarray:
    """Load a frame image through the dataset image source."""
    image_source = dataset.resolve_camera_sensor().image_source
    return image_source.load_rgb(frame)


@app.function
def point_cloud_buffers(dataset, max_points):
    """Move point cloud tensors to the active render device with caching."""
    if dataset is None or dataset.point_cloud is None:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (
        str(dataset.root_path),
        int(dataset.point_cloud.points.shape[0]),
        int(max_points),
        device.type,
    )
    cache = getattr(point_cloud_buffers, "_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    points = dataset.point_cloud.points
    if dataset.point_cloud.colors is None:
        colors = torch.full_like(points, 255.0)
    else:
        colors = torch.clamp(dataset.point_cloud.colors * 255.0, 0.0, 255.0)
    if points.shape[0] > max_points:
        selection = torch.linspace(
            0,
            points.shape[0] - 1,
            steps=max_points,
            dtype=torch.int64,
        )
        points = points[selection]
        colors = colors[selection]
    cache[cache_key] = (
        points.to(device=device, dtype=torch.float32),
        colors.to(device=device, dtype=torch.float32),
    )
    point_cloud_buffers._cache = cache
    return cache[cache_key]


@app.function
def point_cloud_render(
    camera_state,
    dataset,
    render_config,
):
    """Render a sparse point cloud from the current viewer camera."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.full(
        (camera_state.height, camera_state.width, 3),
        render_config.background_brightness,
        dtype=torch.uint8,
        device=device,
    )
    point_cloud = point_cloud_buffers(dataset, render_config.max_points)
    if point_cloud is None:
        return image.cpu().numpy()

    points, colors = point_cloud
    if points.shape[0] == 0:
        return image.cpu().numpy()
    camera_opencv = camera_state.with_convention("opencv")
    world_to_camera = torch.linalg.inv(
        torch.as_tensor(
            camera_opencv.cam_to_world,
            dtype=torch.float32,
            device=device,
        )
    )
    homogeneous_points = torch.cat(
        [
            points,
            torch.ones(
                (points.shape[0], 1), dtype=torch.float32, device=device
            ),
        ],
        dim=1,
    )
    camera_points = (world_to_camera @ homogeneous_points.T).T[:, :3]
    depth = camera_points[:, 2]
    valid = depth > 1e-4
    if not bool(valid.any()):
        return image.cpu().numpy()

    fx = (camera_opencv.width / 2.0) / np.tan(
        np.deg2rad(camera_opencv.fov_degrees) / 2.0
    )
    fy = fx
    cx = camera_opencv.width / 2.0
    cy = camera_opencv.height / 2.0

    x = camera_points[valid, 0]
    y = camera_points[valid, 1]
    z = depth[valid]
    u = torch.round(fx * (x / z) + cx).to(torch.int64)
    v = torch.round(fy * (y / z) + cy).to(torch.int64)
    inside = (
        (u >= 0)
        & (u < camera_opencv.width)
        & (v >= 0)
        & (v < camera_opencv.height)
    )
    if not bool(inside.any()):
        return image.cpu().numpy()

    u_valid = u[inside]
    v_valid = v[inside]
    z_valid = z[inside]
    color_valid = colors[valid][inside].to(torch.uint8)
    order = torch.argsort(z_valid, descending=True)
    radius = render_config.point_radius
    for delta_y in range(-radius + 1, radius):
        for delta_x in range(-radius + 1, radius):
            if delta_x * delta_x + delta_y * delta_y >= radius * radius:
                continue
            draw_u = u_valid[order] + delta_x
            draw_v = v_valid[order] + delta_y
            draw_inside = (
                (draw_u >= 0)
                & (draw_u < camera_opencv.width)
                & (draw_v >= 0)
                & (draw_v < camera_opencv.height)
            )
            if not bool(draw_inside.any()):
                continue
            image[draw_v[draw_inside], draw_u[draw_inside]] = color_valid[
                order
            ][draw_inside]
    return image.cpu().numpy()


if __name__ == "__main__":
    app.run()
