"""Dataset adapters for downstream training and batching."""

from __future__ import annotations

import multiprocessing
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import marimo as mo
import numpy as np
import torch
from beartype import beartype
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ember_core.core.contracts import CameraState
from ember_core.data.config_contracts import (
    ImagePreparationConfig,
    PreparedFrameDatasetConfig,
    SplitConfig,
)
from ember_core.data.contracts import (
    DecodedFrameSample,
    ImagePreparationSpec,
    MaterializationMode,
    MaterializationStage,
    PreparedFrameBatch,
    PreparedFrameSample,
    ResizeSpec,
    SceneRecord,
    horizontal_fov_degrees,
)
from ember_core.data.preprocess import (
    prepare_decoded_image_and_camera,
    resize_intrinsics,
    resolve_resize_shape,
)


@dataclass(frozen=True)
class MaterializationProgress:
    """Progress update for eager prepared-frame materialization."""

    label: str
    current: int
    total: int


MaterializationProgressCallback = Callable[[MaterializationProgress], None]
_MATERIALIZATION_PROGRESS_CALLBACK: ContextVar[
    MaterializationProgressCallback | None
] = ContextVar("ember_materialization_progress_callback", default=None)


def _select_camera(camera: CameraState, index: int) -> CameraState:
    return replace(
        camera,
        width=camera.width[index : index + 1],
        height=camera.height[index : index + 1],
        fov_degrees=camera.fov_degrees[index : index + 1],
        cam_to_world=camera.cam_to_world[index : index + 1],
        intrinsics=(
            camera.intrinsics[index : index + 1]
            if camera.intrinsics is not None
            else None
        ),
    )


def _resolve_materialization_num_workers(
    num_workers: int | None,
) -> int:
    if num_workers is None:
        return max(1, min(8, multiprocessing.cpu_count() or 1))
    resolved = int(num_workers)
    if resolved == 1:
        raise ValueError(
            "materialization_num_workers must be 0, None, or >= 2."
        )
    return resolved


def _disable_materialization_progress() -> bool:
    """Return whether eager data materialization should suppress tqdm output."""
    return mo.running_in_notebook()


@contextmanager
def materialization_progress_callback(
    callback: MaterializationProgressCallback,
) -> Iterator[None]:
    """Install a callback for eager materialization progress in this context."""
    token = _MATERIALIZATION_PROGRESS_CALLBACK.set(callback)
    try:
        yield
    finally:
        _MATERIALIZATION_PROGRESS_CALLBACK.reset(token)


def _report_materialization_progress(
    *,
    label: str,
    current: int,
    total: int,
) -> None:
    callback = _MATERIALIZATION_PROGRESS_CALLBACK.get()
    if callback is None:
        return
    callback(
        MaterializationProgress(
            label=label,
            current=current,
            total=total,
        )
    )


def _resolve_split_indices(
    num_frames: int,
    split: SplitConfig | None,
) -> tuple[int, ...]:
    if split is None or split.target == "all" or split.mode == "none":
        return tuple(range(num_frames))
    if split.mode == "every_n":
        if split.every_n is None:
            raise ValueError("every_n split mode requires split.every_n.")
        every_n = split.every_n
        return tuple(
            index
            for index in range(num_frames)
            if (index % every_n == 0) == (split.target == "val")
        )
    if split.train_ratio is None:
        raise ValueError("ratio split mode requires split.train_ratio.")
    split_index = round(num_frames * split.train_ratio)
    if split.target == "train":
        return tuple(range(split_index))
    return tuple(range(split_index, num_frames))


def _build_preparation(
    config: ImagePreparationConfig | None,
) -> ImagePreparationSpec:
    if config is None:
        return ImagePreparationSpec()
    resize = None
    if (
        config.resize_width_scale is not None
        or config.resize_width_target is not None
    ):
        resize = ResizeSpec(
            width_scale=config.resize_width_scale,
            width_target=config.resize_width_target,
            interpolation=config.interpolation,
        )
    return ImagePreparationSpec(
        resize=resize,
        normalize=config.normalize,
    )


@beartype
class PreparedFrameCache:
    """Shared source-index cache for prepared frame samples."""

    def __init__(
        self,
        scene_record: SceneRecord,
        *,
        camera_sensor_id: str | None = None,
        preparation: ImagePreparationSpec | None = None,
        config: PreparedFrameDatasetConfig | None = None,
    ) -> None:
        self.scene_record = scene_record
        resolved_camera_sensor_id = camera_sensor_id
        if config is not None:
            resolved_camera_sensor_id = (
                resolved_camera_sensor_id or config.camera_sensor_id
            )
            self.preparation = _build_preparation(config.image_preparation)
        else:
            self.preparation = preparation or ImagePreparationSpec()
        self.camera_stream = scene_record.resolve_camera_sensor(
            resolved_camera_sensor_id
        )
        self._decoded_samples: dict[int, DecodedFrameSample] = {}
        self._prepared_samples: dict[int, PreparedFrameSample] = {}
        self._prepared_cameras: dict[int, CameraState] = {}
        self._lock = threading.RLock()

    def __getstate__(self) -> dict[str, object]:
        """Drop the process-local lock before dataloader worker pickling."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Recreate the process-local lock after dataloader worker unpickling."""
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def _decode_source_sample(
        self,
        source_index: int,
    ) -> DecodedFrameSample:
        frame = self.camera_stream.frames[source_index]
        camera = _select_camera(self.camera_stream.camera, frame.camera_index)
        image = torch.from_numpy(
            np.array(self.camera_stream.image_source.load_rgb(frame), copy=True)
        )
        return DecodedFrameSample(
            frame=frame,
            image=image,
            camera=camera,
        )

    def decoded_sample(self, source_index: int) -> DecodedFrameSample:
        """Return a decoded sample by source-frame index."""
        with self._lock:
            cached = self._decoded_samples.get(source_index)
        if cached is not None:
            return cached

        decoded_sample = self._decode_source_sample(source_index)
        with self._lock:
            return self._decoded_samples.setdefault(
                source_index,
                decoded_sample,
            )

    def _prepare_source_sample(
        self,
        source_index: int,
    ) -> PreparedFrameSample:
        decoded_sample = self.decoded_sample(source_index)
        image, prepared_camera = prepare_decoded_image_and_camera(
            decoded_sample.image,
            decoded_sample.camera,
            self.preparation,
        )
        prepared_frame = replace(
            decoded_sample.frame,
            width=int(prepared_camera.width[0].item()),
            height=int(prepared_camera.height[0].item()),
        )
        return PreparedFrameSample(
            frame=prepared_frame,
            image=image,
            camera=prepared_camera,
        )

    def prepared_sample(self, source_index: int) -> PreparedFrameSample:
        """Return a prepared sample by source-frame index."""
        with self._lock:
            cached = self._prepared_samples.get(source_index)
        if cached is not None:
            return cached

        prepared_sample = self._prepare_source_sample(source_index)
        with self._lock:
            return self._prepared_samples.setdefault(
                source_index,
                prepared_sample,
            )

    def prepared_camera(self, source_index: int) -> CameraState:
        """Return a prepared camera by source-frame index without loading RGB."""
        with self._lock:
            cached = self._prepared_cameras.get(source_index)
        if cached is not None:
            return cached

        frame = self.camera_stream.frames[source_index]
        camera = _select_camera(self.camera_stream.camera, frame.camera_index)
        original_width = frame.width
        original_height = frame.height
        resized_width, resized_height = resolve_resize_shape(
            original_width,
            original_height,
            self.preparation.resize,
        )
        intrinsics = camera.get_intrinsics()[0]
        resized_intrinsics = resize_intrinsics(
            intrinsics,
            original_width=original_width,
            original_height=original_height,
            resized_width=resized_width,
            resized_height=resized_height,
        )
        prepared_camera = replace(
            camera,
            width=torch.tensor([resized_width], dtype=torch.int64),
            height=torch.tensor([resized_height], dtype=torch.int64),
            fov_degrees=torch.tensor(
                [horizontal_fov_degrees(resized_width, resized_intrinsics)],
                dtype=torch.float32,
            ),
            intrinsics=resized_intrinsics[None],
        )
        with self._lock:
            return self._prepared_cameras.setdefault(
                source_index,
                prepared_camera,
            )


@beartype
class PreparedFrameDataset(Dataset[PreparedFrameSample]):
    """Prepared frame dataset view over a scene record."""

    def __init__(
        self,
        scene_record: SceneRecord,
        *,
        config: PreparedFrameDatasetConfig | None = None,
        camera_sensor_id: str | None = None,
        preparation: ImagePreparationSpec | None = None,
        materialization_stage: MaterializationStage = "prepared",
        materialization_mode: MaterializationMode = "eager",
        materialization_num_workers: int | None = 8,
        sample_cache: PreparedFrameCache | None = None,
    ) -> None:
        self.scene_record = scene_record
        self.config = config
        resolved_camera_sensor_id = camera_sensor_id
        if config is not None:
            resolved_camera_sensor_id = (
                resolved_camera_sensor_id or config.camera_sensor_id
            )
            self.preparation = _build_preparation(config.image_preparation)
            resolved_materialization = config.materialization
            if resolved_materialization is None:
                self.materialization_stage = materialization_stage
                self.materialization_mode = materialization_mode
                self.materialization_num_workers = materialization_num_workers
            else:
                self.materialization_stage = resolved_materialization.stage
                self.materialization_mode = resolved_materialization.mode
                self.materialization_num_workers = (
                    resolved_materialization.num_workers
                )
            self.camera_stream = scene_record.resolve_camera_sensor(
                resolved_camera_sensor_id
            )
            self.indices = _resolve_split_indices(
                len(self.camera_stream.frames),
                config.split,
            )
        else:
            self.preparation = preparation or ImagePreparationSpec()
            self.materialization_stage = materialization_stage
            self.materialization_mode = materialization_mode
            self.materialization_num_workers = materialization_num_workers
            self.camera_stream = scene_record.resolve_camera_sensor(
                resolved_camera_sensor_id
            )
            self.indices = tuple(range(len(self.camera_stream.frames)))
        self.sample_cache = sample_cache
        if self.sample_cache is not None:
            if self.sample_cache.scene_record is not scene_record:
                raise ValueError(
                    "PreparedFrameDataset sample_cache must use the same "
                    "SceneRecord instance."
                )
            if (
                self.sample_cache.camera_stream.sensor_id
                != self.camera_stream.sensor_id
            ):
                raise ValueError(
                    "PreparedFrameDataset sample_cache camera sensor does not "
                    "match this dataset."
                )
            if self.sample_cache.preparation != self.preparation:
                raise ValueError(
                    "PreparedFrameDataset sample_cache preparation does not "
                    "match this dataset."
                )
        self._decoded_samples: list[DecodedFrameSample | None] | None = None
        self._prepared_samples: list[PreparedFrameSample | None] | None = None

        if self.materialization_stage == "decoded":
            self._decoded_samples = [None] * len(self.indices)
        elif self.materialization_stage == "prepared":
            self._prepared_samples = [None] * len(self.indices)

        if (
            self.materialization_stage != "none"
            and self.materialization_mode == "eager"
        ):
            if self.materialization_stage == "decoded":
                self._materialize_decoded_samples_eager()
            else:
                assert self._prepared_samples is not None
                label = f"Materializing {self.materialization_stage} dataset"
                total = len(self.indices)
                _report_materialization_progress(
                    label=label,
                    current=0,
                    total=total,
                )
                for index in tqdm(
                    range(len(self.indices)),
                    desc=label,
                    total=total,
                    disable=_disable_materialization_progress(),
                ):
                    self._prepared_samples[index] = self._prepare_sample(index)
                    _report_materialization_progress(
                        label=label,
                        current=index + 1,
                        total=total,
                    )
            self._detach_sample_cache_for_workers()

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict[str, object]:
        """Drop redundant shared cache state from worker pickles."""
        state = self.__dict__.copy()
        if self._materialized_for_worker_pickle():
            state["sample_cache"] = None
        return state

    def _materialized_for_worker_pickle(self) -> bool:
        """Return whether local materialized samples make the cache redundant."""
        if self._prepared_samples is not None:
            return all(sample is not None for sample in self._prepared_samples)
        if self._decoded_samples is not None:
            return all(sample is not None for sample in self._decoded_samples)
        return False

    def _detach_sample_cache_for_workers(self) -> None:
        """Avoid sending the shared catalog cache to dataloader workers."""
        if self._materialized_for_worker_pickle():
            self.sample_cache = None

    def _decode_sample(self, index: int) -> DecodedFrameSample:
        dataset_index = self.indices[index]
        if self.sample_cache is not None:
            return self.sample_cache.decoded_sample(dataset_index)
        frame = self.camera_stream.frames[dataset_index]
        camera = _select_camera(self.camera_stream.camera, frame.camera_index)
        image = torch.from_numpy(
            np.array(self.camera_stream.image_source.load_rgb(frame), copy=True)
        )
        return DecodedFrameSample(
            frame=frame,
            image=image,
            camera=camera,
        )

    def _materialize_decoded_samples_eager(self) -> None:
        assert self._decoded_samples is not None
        resolved_num_workers = _resolve_materialization_num_workers(
            self.materialization_num_workers
        )
        label = "Materializing decoded dataset"
        total = len(self.indices)
        _report_materialization_progress(
            label=label,
            current=0,
            total=total,
        )
        if resolved_num_workers <= 0:
            for index in tqdm(
                range(len(self.indices)),
                desc=label,
                total=total,
                disable=_disable_materialization_progress(),
            ):
                self._decoded_samples[index] = self._decode_sample(index)
                _report_materialization_progress(
                    label=label,
                    current=index + 1,
                    total=total,
                )
            return

        with ThreadPoolExecutor(max_workers=resolved_num_workers) as executor:
            futures = {
                executor.submit(self._decode_sample, index): index
                for index in range(len(self.indices))
            }
            progress = tqdm(
                total=total,
                desc=label,
                disable=_disable_materialization_progress(),
            )
            try:
                for completed, future in enumerate(
                    as_completed(futures),
                    start=1,
                ):
                    index = futures[future]
                    self._decoded_samples[index] = future.result()
                    progress.update(1)
                    _report_materialization_progress(
                        label=label,
                        current=completed,
                        total=total,
                    )
            except Exception:
                for future in futures:
                    future.cancel()
                raise
            finally:
                progress.close()

    def _prepare_sample(self, index: int) -> PreparedFrameSample:
        if self.sample_cache is not None:
            return self.sample_cache.prepared_sample(self.indices[index])
        if self._decoded_samples is not None:
            decoded_sample = self._decoded_samples[index]
            if decoded_sample is None:
                decoded_sample = self._decode_sample(index)
                if self.materialization_stage == "decoded":
                    self._decoded_samples[index] = decoded_sample
            image, prepared_camera = prepare_decoded_image_and_camera(
                decoded_sample.image,
                decoded_sample.camera,
                self.preparation,
            )
            frame = decoded_sample.frame
        else:
            dataset_index = self.indices[index]
            frame = self.camera_stream.frames[dataset_index]
            camera = _select_camera(
                self.camera_stream.camera, frame.camera_index
            )
            image = torch.from_numpy(
                np.array(
                    self.camera_stream.image_source.load_rgb(frame), copy=True
                )
            )
            image, prepared_camera = prepare_decoded_image_and_camera(
                image,
                camera,
                self.preparation,
            )
        prepared_frame = replace(
            frame,
            width=int(prepared_camera.width[0].item()),
            height=int(prepared_camera.height[0].item()),
        )
        return PreparedFrameSample(
            frame=prepared_frame,
            image=image,
            camera=prepared_camera,
        )

    def __getitem__(self, index: int) -> PreparedFrameSample:
        if self._prepared_samples is not None:
            prepared_sample = self._prepared_samples[index]
            if prepared_sample is None:
                prepared_sample = self._prepare_sample(index)
                self._prepared_samples[index] = prepared_sample
            return prepared_sample
        return self._prepare_sample(index)

    def prepared_camera(self, index: int) -> CameraState:
        """Return the prepared camera for an indexed frame without loading RGB."""
        source_index = self.indices[index]
        if self.sample_cache is not None:
            return self.sample_cache.prepared_camera(source_index)
        frame = self.camera_stream.frames[source_index]
        camera = _select_camera(self.camera_stream.camera, frame.camera_index)
        original_width = frame.width
        original_height = frame.height
        resized_width, resized_height = resolve_resize_shape(
            original_width,
            original_height,
            self.preparation.resize,
        )
        intrinsics = camera.get_intrinsics()[0]
        resized_intrinsics = resize_intrinsics(
            intrinsics,
            original_width=original_width,
            original_height=original_height,
            resized_width=resized_width,
            resized_height=resized_height,
        )
        return replace(
            camera,
            width=torch.tensor([resized_width], dtype=torch.int64),
            height=torch.tensor([resized_height], dtype=torch.int64),
            fov_degrees=torch.tensor(
                [horizontal_fov_degrees(resized_width, resized_intrinsics)],
                dtype=torch.float32,
            ),
            intrinsics=resized_intrinsics[None],
        )

    def prepared_cameras(self) -> tuple[CameraState, ...]:
        """Return all prepared cameras without loading RGB images."""
        return tuple(self.prepared_camera(index) for index in range(len(self)))


def collate_frame_samples(
    samples: list[PreparedFrameSample],
) -> PreparedFrameBatch:
    """Collate prepared frame samples into a batch."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")
    images = torch.stack([sample.image for sample in samples], dim=0)
    first_camera = samples[0].camera
    batched_camera = replace(
        first_camera,
        width=torch.cat([sample.camera.width for sample in samples], dim=0),
        height=torch.cat([sample.camera.height for sample in samples], dim=0),
        fov_degrees=torch.cat(
            [sample.camera.fov_degrees for sample in samples],
            dim=0,
        ),
        cam_to_world=torch.cat(
            [sample.camera.cam_to_world for sample in samples],
            dim=0,
        ),
        intrinsics=torch.cat(
            [sample.camera.get_intrinsics() for sample in samples],
            dim=0,
        ),
    )
    return PreparedFrameBatch(
        frames=tuple(sample.frame for sample in samples),
        images=images,
        camera=batched_camera,
    )
