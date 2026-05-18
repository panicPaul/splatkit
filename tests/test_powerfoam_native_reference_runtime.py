from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
POWERFOAM_PACKAGE_SRC = REPO_ROOT / "packages" / "ember-native-powerfoam" / "src"
POWERFOAM_REFERENCE_SRC = REPO_ROOT / "third_party" / "powerfoam"
sys.path.insert(0, str(POWERFOAM_PACKAGE_SRC))
sys.path.insert(0, str(POWERFOAM_REFERENCE_SRC))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PowerFoam native/reference parity requires CUDA.",
)


def _powerfoam_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260513)
    points = torch.rand((16, 3), device="cuda", dtype=torch.float32)
    radii = torch.rand((16,), device="cuda", dtype=torch.float32)
    return (points * 2.0 - 1.0).contiguous(), (radii * 0.4 + 0.3).contiguous()


def _reference_topology(
    points: torch.Tensor,
    radii: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from powerfoam.bvh import AABBTree

    wp.init()
    tree = AABBTree(points.device)
    tree.update(points.detach(), radii.detach())
    adjacency, adjacency_offsets = tree.build_cech_complex()
    torch.cuda.synchronize()
    return adjacency, adjacency_offsets


def test_powerfoam_build_topology_kernel_matches_reference() -> None:
    from ember_native_powerfoam.powerfoam.runtime.ops import build_topology_op

    points, radii = _powerfoam_inputs()
    reference_adjacency, reference_offsets = _reference_topology(points, radii)
    native_adjacency, native_offsets = build_topology_op(points, radii)
    torch.cuda.synchronize()

    torch.testing.assert_close(native_adjacency, reference_adjacency)
    torch.testing.assert_close(native_offsets, reference_offsets)


def test_powerfoam_interpenetration_forward_backward_matches_reference() -> None:
    from ember_native_powerfoam.powerfoam.runtime.ops import interpenetration_op
    from powerfoam.geometry import InterpenetrationFunction

    points, radii = _powerfoam_inputs()
    adjacency, adjacency_offsets = _reference_topology(points, radii)
    reference_points = points.detach().clone().requires_grad_()
    reference_radii = radii.detach().clone().requires_grad_()
    native_points = points.detach().clone().requires_grad_()
    native_radii = radii.detach().clone().requires_grad_()

    reference_areas = InterpenetrationFunction.apply(
        reference_points,
        reference_radii,
        adjacency,
        adjacency_offsets,
    )
    native_areas = interpenetration_op(
        native_points,
        native_radii,
        adjacency,
        adjacency_offsets,
    )

    torch.testing.assert_close(native_areas, reference_areas)
    grad_areas = torch.linspace(
        0.1,
        1.0,
        steps=reference_areas.shape[0],
        device=reference_areas.device,
        dtype=reference_areas.dtype,
    )
    reference_areas.backward(grad_areas)
    native_areas.backward(grad_areas)
    torch.cuda.synchronize()

    torch.testing.assert_close(native_points.grad, reference_points.grad)
    torch.testing.assert_close(native_radii.grad, reference_radii.grad)


def test_powerfoam_spherical_voronoi_forward_backward_matches_reference() -> None:
    from ember_native_powerfoam.powerfoam.native.warp.camera import TorchCamera
    from ember_native_powerfoam.powerfoam.runtime.ops import (
        spherical_voronoi_colors,
    )
    from powerfoam.camera import TorchCamera as ReferenceTorchCamera
    from powerfoam.color_fn import SphericalVoronoi

    wp.init()
    torch.manual_seed(20260514)
    num_points = 8
    sv_dof = 3
    points = torch.randn(
        (num_points, 3),
        device="cuda",
        dtype=torch.float32,
    )
    eye = torch.tensor([0.0, 0.0, -2.0], device="cuda")
    right = torch.tensor([1.0, 0.0, 0.0], device="cuda")
    up = torch.tensor([0.0, 1.0, 0.0], device="cuda")
    native_camera = TorchCamera(
        eye=eye,
        right=right,
        up=up,
        width=16,
        height=16,
    )
    reference_camera = ReferenceTorchCamera(
        eye=eye,
        right=right,
        up=up,
        width=16,
        height=16,
    )
    att_sites = torch.randn(
        (sv_dof, num_points, 3),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    att_sites.data = att_sites.data / att_sites.data.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-6)
    att_values = torch.randn(
        (sv_dof, num_points, 3),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    att_temps = (
        torch.rand(
            (sv_dof, num_points),
            device="cuda",
            dtype=torch.float32,
        )
        + 0.5
    ).requires_grad_()
    reference_sites = att_sites.detach().clone().requires_grad_()
    reference_values = att_values.detach().clone().requires_grad_()
    reference_temps = att_temps.detach().clone().requires_grad_()
    native_sites = att_sites.detach().clone().requires_grad_()
    native_values = att_values.detach().clone().requires_grad_()
    native_temps = att_temps.detach().clone().requires_grad_()

    reference_stage = SphericalVoronoi(
        SimpleNamespace(sv_dof=sv_dof),
        torch.device("cuda"),
        "float",
    )
    reference_stage.fov_cos_cutoff = SphericalVoronoi.compute_fov_cos_cutoff(
        reference_camera
    )
    reference_values_out = reference_stage.forward(
        points.detach(),
        reference_camera,
        reference_sites,
        reference_values,
        reference_temps,
    )
    native_values_out = spherical_voronoi_colors(
        points.detach(),
        native_camera,
        native_sites,
        native_values,
        native_temps,
        sv_dof=sv_dof,
        attr_dtype="float",
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(native_values_out, reference_values_out)

    torch.manual_seed(20260515)
    grad_values = torch.randn_like(reference_values_out)
    reference_values_out.backward(grad_values)
    native_values_out.backward(grad_values)
    torch.cuda.synchronize()

    torch.testing.assert_close(native_sites.grad, reference_sites.grad)
    torch.testing.assert_close(native_values.grad, reference_values.grad)
    torch.testing.assert_close(native_temps.grad, reference_temps.grad)


def test_powerfoam_rasterize_forward_backward_matches_reference() -> None:
    from ember_native_powerfoam.powerfoam.native.warp.camera import TorchCamera
    from ember_native_powerfoam.powerfoam.runtime.ops import rasterize_powerfoam
    from powerfoam.camera import TorchCamera as ReferenceTorchCamera
    from powerfoam.rasterize import Rasterizer as ReferenceRasterizer

    wp.init()
    torch.manual_seed(20260516)
    device = torch.device("cuda")
    height = 16
    width = 16
    points = torch.tensor(
        [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.0],
            [0.0, 0.0, 2.3],
            [0.0, 0.3, 2.2],
            [0.3, 0.0, 2.2],
            [-0.3, 0.0, 2.2],
        ],
        dtype=torch.float32,
        device=device,
    )
    radii = torch.full((points.shape[0],), 0.35, dtype=torch.float32, device=device)
    adjacency, adjacency_offsets = _reference_topology(points, radii)
    eye = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32, device=device)
    right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    base_camera = TorchCamera(
        eye=eye,
        right=right,
        up=up,
        width=width,
        height=height,
    )
    ray_maps = base_camera._build_pinhole_ray_maps().contiguous()
    native_camera = TorchCamera(
        eye=eye,
        right=right,
        up=up,
        width=width,
        height=height,
        ray_maps=ray_maps,
    )
    reference_camera = ReferenceTorchCamera(
        eye=eye,
        right=right,
        up=up,
        width=width,
        height=height,
        ray_maps=ray_maps,
    )
    num_texel_sites = 8
    texel_offsets = 0.02 * torch.randn(
        (points.shape[0], num_texel_sites, 3),
        dtype=torch.float32,
        device=device,
    )
    texel_sites = points[:, None, :] + texel_offsets
    texel_rgb = torch.rand(
        (points.shape[0], num_texel_sites, 3),
        dtype=torch.float32,
        device=device,
    )
    texel_height = torch.zeros(
        (points.shape[0], num_texel_sites),
        dtype=torch.float32,
        device=device,
    )
    normals = torch.nn.functional.normalize(
        -points,
        dim=-1,
    )
    density = torch.full(
        (points.shape[0],),
        0.1,
        dtype=torch.float32,
        device=device,
    )
    depth_quantiles = torch.full(
        (height, width, 1),
        0.5,
        dtype=torch.float32,
        device=device,
    )
    reference_points = points.detach().clone().requires_grad_()
    reference_radii = radii.detach().clone().requires_grad_()
    reference_density = density.detach().clone().requires_grad_()
    reference_normals = normals.detach().clone().requires_grad_()
    reference_texel_sites = texel_sites.detach().clone().requires_grad_()
    reference_texel_rgb = texel_rgb.detach().clone().requires_grad_()
    reference_texel_height = texel_height.detach().clone().requires_grad_()
    native_points = points.detach().clone().requires_grad_()
    native_radii = radii.detach().clone().requires_grad_()
    native_density = density.detach().clone().requires_grad_()
    native_normals = normals.detach().clone().requires_grad_()
    native_texel_sites = texel_sites.detach().clone().requires_grad_()
    native_texel_rgb = texel_rgb.detach().clone().requires_grad_()
    native_texel_height = texel_height.detach().clone().requires_grad_()
    args = SimpleNamespace(
        render_objective="volume",
        num_texel_sites=num_texel_sites,
        sv_dof=8,
        disable_coop_prim_load=False,
        disable_coop_adj_load=False,
        is_pinhole=True,
    )
    reference_rasterizer = ReferenceRasterizer(args, device, "float")
    reference_outputs = reference_rasterizer.forward(
        reference_camera,
        depth_quantiles,
        reference_points,
        reference_radii,
        reference_density,
        reference_normals,
        reference_texel_sites,
        reference_texel_rgb,
        reference_texel_height,
        adjacency,
        adjacency_offsets,
        None,
        False,
    )
    native_outputs = rasterize_powerfoam(
        native_camera,
        depth_quantiles,
        native_points,
        native_radii,
        native_density,
        native_normals,
        native_texel_sites,
        native_texel_rgb,
        native_texel_height,
        adjacency,
        adjacency_offsets,
        None,
        False,
        render_objective="volume",
        num_texel_sites=num_texel_sites,
        sv_dof=8,
        disable_coop_prim_load=False,
        disable_coop_adj_load=False,
        is_pinhole=True,
        attr_dtype="float",
    )
    torch.cuda.synchronize()

    for native_output, reference_output in zip(
        native_outputs[:7],
        reference_outputs[:7],
        strict=True,
    ):
        torch.testing.assert_close(native_output, reference_output)
    torch.testing.assert_close(native_outputs[8], reference_outputs[8])

    reference_loss = (
        reference_outputs[0].square().mean()
        + reference_outputs[1].mean()
        + 0.01 * reference_outputs[2].mean()
        + 0.01 * reference_outputs[3].square().mean()
        + 0.001 * reference_outputs[4].mean()
        + 0.001 * reference_outputs[6].mean()
    )
    native_loss = (
        native_outputs[0].square().mean()
        + native_outputs[1].mean()
        + 0.01 * native_outputs[2].mean()
        + 0.01 * native_outputs[3].square().mean()
        + 0.001 * native_outputs[4].mean()
        + 0.001 * native_outputs[6].mean()
    )
    reference_loss.backward()
    native_loss.backward()
    torch.cuda.synchronize()

    torch.testing.assert_close(native_points.grad, reference_points.grad)
    torch.testing.assert_close(native_radii.grad, reference_radii.grad)
    torch.testing.assert_close(native_density.grad, reference_density.grad)
    torch.testing.assert_close(native_normals.grad, reference_normals.grad)
    torch.testing.assert_close(native_texel_sites.grad, reference_texel_sites.grad)
    torch.testing.assert_close(native_texel_rgb.grad, reference_texel_rgb.grad)
    torch.testing.assert_close(native_texel_height.grad, reference_texel_height.grad)
