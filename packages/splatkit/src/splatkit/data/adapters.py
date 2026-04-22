"""Dataset adapters for downstream training and batching."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import torch
from beartype import beartype
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from splatkit.core.contracts import CameraState
from splatkit.data.config_contracts import (
    FrameDatasetConfig,
    ImagePreparationConfig,
    SplitConfig,
)
from splatkit.data.contracts import (
    DecodedFrameSample,
    ImagePreparationSpec,
    MaterializationMode,
    MaterializationStage,
    PreparedFrameBatch,
    PreparedFrameSample,
    ResizeSpec,
    SceneDataset,
)
from splatkit.data.preprocess import (
    prepare_decoded_image_and_camera,
)


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


def _resolve_split_indices(
    num_frames: int,
    split: SplitConfig | None,
) -> tuple[int, ...]:
    if split is None or split.target == "all" or split.mode == "none":
        return tuple(range(num_frames))
    if split.mode == "every_n":
        return tuple(
            index
            for index in range(num_frames)
            if (index % split.every_n == 0) == (split.target == "val")
        )
    split_denominator = split.train_ratio + split.val_ratio
    split_index = round(num_frames * split.train_ratio / split_denominator)
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
class FrameDataset(Dataset[PreparedFrameSample]):
    """Dataset wrapper over a SceneDataset with staged caching."""

    def __init__(
        self,
        dataset: SceneDataset,
        *,
        config: FrameDatasetConfig | None = None,
        camera_sensor_id: str | None = None,
        preparation: ImagePreparationSpec | None = None,
        materialization_stage: MaterializationStage = "decoded",
        materialization_mode: MaterializationMode = "eager",
        materialization_num_workers: int | None = 0,
    ) -> None:
        self.dataset = dataset
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
            self.camera_stream = dataset.resolve_camera_sensor(
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
            self.camera_stream = dataset.resolve_camera_sensor(
                resolved_camera_sensor_id
            )
            self.indices = tuple(range(len(self.camera_stream.frames)))
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
                for index in tqdm(
                    range(len(self.indices)),
                    desc=(
                        f"Materializing {self.materialization_stage} dataset"
                    ),
                    total=len(self.indices),
                ):
                    self._prepared_samples[index] = self._prepare_sample(index)

    def __len__(self) -> int:
        return len(self.indices)

    def _decode_sample(self, index: int) -> DecodedFrameSample:
        dataset_index = self.indices[index]
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
        if resolved_num_workers <= 0:
            for index in tqdm(
                range(len(self.indices)),
                desc="Materializing decoded dataset",
                total=len(self.indices),
            ):
                self._decoded_samples[index] = self._decode_sample(index)
            return

        with ThreadPoolExecutor(max_workers=resolved_num_workers) as executor:
            futures = {
                executor.submit(self._decode_sample, index): index
                for index in range(len(self.indices))
            }
            progress = tqdm(
                total=len(self.indices),
                desc="Materializing decoded dataset",
            )
            try:
                for future in as_completed(futures):
                    index = futures[future]
                    self._decoded_samples[index] = future.result()
                    progress.update(1)
            except Exception:
                for future in futures:
                    future.cancel()
                raise
            finally:
                progress.close()

    def _prepare_sample(self, index: int) -> PreparedFrameSample:
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
            camera = _select_camera(self.camera_stream.camera, frame.camera_index)
            image = torch.from_numpy(
                np.array(self.camera_stream.image_source.load_rgb(frame), copy=True)
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
