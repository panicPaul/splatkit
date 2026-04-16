"""Dataset adapters for downstream training and batching."""

from __future__ import annotations

from dataclasses import replace

import torch
from beartype import beartype
from torch.utils.data import Dataset

from splatkit.core.contracts import CameraState
from splatkit.data.contracts import (
    ImagePreparationSpec,
    PreparedFrameBatch,
    PreparedFrameSample,
    SceneDataset,
)
from splatkit.data.preprocess import prepare_image_and_camera


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


@beartype
class FrameDataset(Dataset[PreparedFrameSample]):
    """Lazy dataset wrapper over a SceneDataset."""

    def __init__(
        self,
        dataset: SceneDataset,
        *,
        preparation: ImagePreparationSpec | None = None,
    ) -> None:
        self.dataset = dataset
        self.preparation = preparation or ImagePreparationSpec()

    def __len__(self) -> int:
        return self.dataset.num_frames

    def __getitem__(self, index: int) -> PreparedFrameSample:
        frame = self.dataset.frames[index]
        camera = _select_camera(self.dataset.camera, frame.camera_index)
        image, prepared_camera = prepare_image_and_camera(
            frame.image_path,
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
