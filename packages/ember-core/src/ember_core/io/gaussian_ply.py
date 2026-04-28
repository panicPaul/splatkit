"""Gaussian-scene PLY import/export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from ember_core.core.contracts import GaussianScene3D


def infer_sh_degree(num_bases: int) -> int:
    """Infer the SH degree from the number of basis functions."""
    degree = int(np.sqrt(num_bases)) - 1
    if (degree + 1) ** 2 != num_bases:
        raise ValueError(f"Invalid SH basis count: {num_bases}.")
    if degree < 0:
        raise ValueError(f"Invalid SH degree inferred from {num_bases}.")
    return degree


def load_gaussian_ply(path: str | Path) -> GaussianScene3D:
    """Load a 3DGS-style PLY file into a GaussianScene3D."""
    ply_data = PlyData.read(path)
    vertices = ply_data["vertex"]
    property_names = list(vertices.data.dtype.names)

    centers = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]],
        axis=1,
    ).astype(np.float32)
    dc_coefficients = np.stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]],
        axis=1,
    ).astype(np.float32)
    rest_feature_names = sorted(
        [name for name in property_names if name.startswith("f_rest_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    num_rest_coefficients = len(rest_feature_names)
    if num_rest_coefficients % 3 != 0:
        raise ValueError(
            "Expected the number of `f_rest_*` attributes to be divisible by 3."
        )

    num_bases = 1 + num_rest_coefficients // 3
    sh_degree = infer_sh_degree(num_bases)
    sh_coefficients = np.zeros(
        (centers.shape[0], num_bases, 3),
        dtype=np.float32,
    )
    sh_coefficients[:, 0, :] = dc_coefficients
    if rest_feature_names:
        rest_coefficients = np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rest_feature_names
            ],
            axis=1,
        )
        rest_coefficients = rest_coefficients.reshape(
            centers.shape[0],
            3,
            num_bases - 1,
        )
        sh_coefficients[:, 1:num_bases, :] = np.transpose(
            rest_coefficients,
            (0, 2, 1),
        )

    scale_feature_names = sorted(
        [name for name in property_names if name.startswith("scale_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    rotation_feature_names = sorted(
        [name for name in property_names if name.startswith("rot")],
        key=lambda name: int(name.split("_")[-1]),
    )
    log_scales = (
        np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in scale_feature_names
            ],
            axis=1,
        )
        if scale_feature_names
        else np.full((centers.shape[0], 3), np.log(0.01), dtype=np.float32)
    )
    rotations = (
        np.stack(
            [
                np.asarray(vertices[name], dtype=np.float32)
                for name in rotation_feature_names
            ],
            axis=1,
        )
        if rotation_feature_names
        else np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (centers.shape[0], 1),
        )
    )
    opacity_logits = np.asarray(vertices["opacity"], dtype=np.float32)

    return GaussianScene3D(
        center_position=torch.from_numpy(centers),
        log_scales=torch.from_numpy(log_scales),
        quaternion_orientation=torch.from_numpy(rotations),
        logit_opacity=torch.from_numpy(opacity_logits),
        feature=torch.from_numpy(sh_coefficients),
        sh_degree=sh_degree,
    )


def save_gaussian_ply(
    scene: GaussianScene3D,
    path: str | Path,
) -> None:
    """Save a GaussianScene3D as a 3DGS-style PLY file."""
    feature = scene.feature
    if feature.ndim == 2:
        if feature.shape[1] != 3:
            raise ValueError(
                "2D Gaussian feature export expects plain RGB features with "
                f"shape (num_splats, 3); got {tuple(feature.shape)}."
            )
        sh_coefficients = feature[:, None, :]
    else:
        sh_coefficients = feature

    num_splats = scene.center_position.shape[0]
    num_bases = int(sh_coefficients.shape[1])
    dc_coefficients = sh_coefficients[:, 0, :].detach().cpu().numpy()
    rest_coefficients = (
        sh_coefficients[:, 1:, :]
        .permute(0, 2, 1)
        .reshape(num_splats, -1)
        .detach()
        .cpu()
        .numpy()
    )
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    dtype.extend(
        (f"f_rest_{index}", "f4") for index in range(3 * max(0, num_bases - 1))
    )
    dtype.extend(
        [
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )
    vertex = np.empty(num_splats, dtype=dtype)
    centers = scene.center_position.detach().cpu().numpy()
    scales = scene.log_scales.detach().cpu().numpy()
    rotations = scene.quaternion_orientation.detach().cpu().numpy()
    opacities = scene.logit_opacity.detach().cpu().numpy()
    vertex["x"] = centers[:, 0]
    vertex["y"] = centers[:, 1]
    vertex["z"] = centers[:, 2]
    vertex["nx"] = 0.0
    vertex["ny"] = 0.0
    vertex["nz"] = 0.0
    vertex["f_dc_0"] = dc_coefficients[:, 0]
    vertex["f_dc_1"] = dc_coefficients[:, 1]
    vertex["f_dc_2"] = dc_coefficients[:, 2]
    for index in range(rest_coefficients.shape[1]):
        vertex[f"f_rest_{index}"] = rest_coefficients[:, index]
    vertex["opacity"] = opacities
    vertex["scale_0"] = scales[:, 0]
    vertex["scale_1"] = scales[:, 1]
    vertex["scale_2"] = scales[:, 2]
    vertex["rot_0"] = rotations[:, 0]
    vertex["rot_1"] = rotations[:, 1]
    vertex["rot_2"] = rotations[:, 2]
    vertex["rot_3"] = rotations[:, 3]
    ply = PlyData([PlyElement.describe(vertex, "vertex")])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ply.write(path)
