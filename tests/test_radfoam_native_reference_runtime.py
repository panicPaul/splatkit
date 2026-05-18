from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RADFOAM_REFERENCE_EXTENSION = (
    REPO_ROOT / "third_party" / "radfoam" / "radfoam"
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="RADFOAM native/reference parity requires CUDA.",
)


KERNEL_SCRIPT = r"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

backend_name = sys.argv[1]
kernel_name = sys.argv[2]
output_path = Path(sys.argv[3])
repo_root = Path.cwd()

if backend_name == "reference":
    sys.path.insert(0, str(repo_root / "third_party" / "radfoam"))
    import radfoam as reference_radfoam
else:
    sys.path.insert(0, str(repo_root / "packages" / "ember-native-radfoam" / "src"))
    from ember_native_radfoam.radfoam.runtime.ops.topology import (
        build_aabb_tree_op,
        farthest_neighbor_op,
        nearest_neighbor_op,
        triangulate_op,
    )
    from ember_native_radfoam.radfoam.runtime.ops.trace import (
        trace_bwd_op,
        trace_fwd_op,
    )


def make_points() -> torch.Tensor:
    torch.manual_seed(20260513)
    points = torch.rand((32, 3), device="cuda", dtype=torch.float32)
    return (points * 2.0 - 1.0).contiguous()


def make_queries() -> torch.Tensor:
    torch.manual_seed(20260514)
    points = torch.rand((11, 3), device="cuda", dtype=torch.float32)
    return (points * 2.0 - 1.0).contiguous()


def triangulate(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if backend_name == "reference":
        triangulation = reference_radfoam.Triangulation(points.contiguous())
        return (
            triangulation.permutation().to(torch.long).clone(),
            triangulation.point_adjacency().clone(),
            triangulation.point_adjacency_offsets().clone(),
        )
    return triangulate_op(points.contiguous())


def build_aabb_tree(points: torch.Tensor) -> torch.Tensor:
    if backend_name == "reference":
        return reference_radfoam.build_aabb_tree(points.contiguous())
    return build_aabb_tree_op(points.contiguous())


def nearest_neighbor(
    points: torch.Tensor,
    aabb_tree: torch.Tensor,
    queries: torch.Tensor,
) -> torch.Tensor:
    if backend_name == "reference":
        return reference_radfoam.nn(
            points.contiguous(),
            aabb_tree.contiguous(),
            queries.contiguous(),
        )
    return nearest_neighbor_op(
        points.contiguous(),
        aabb_tree.contiguous(),
        queries.contiguous(),
    )


def farthest_neighbor(
    points: torch.Tensor,
    point_adjacency: torch.Tensor,
    point_adjacency_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend_name == "reference":
        return reference_radfoam.farthest_neighbor(
            points.contiguous(),
            point_adjacency.contiguous(),
            point_adjacency_offsets.contiguous(),
        )
    return farthest_neighbor_op(
        points.contiguous(),
        point_adjacency.contiguous(),
        point_adjacency_offsets.contiguous(),
    )


def trace_forward(
    points: torch.Tensor,
    attributes: torch.Tensor,
    point_adjacency: torch.Tensor,
    point_adjacency_offsets: torch.Tensor,
    rays: torch.Tensor,
    start_point: torch.Tensor,
    depth_quantiles: torch.Tensor,
    sh_degree: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if backend_name == "reference":
        pipeline = reference_radfoam.create_pipeline(sh_degree, attributes.dtype)
        output = pipeline.trace_forward(
            points.contiguous(),
            attributes.contiguous(),
            point_adjacency.contiguous(),
            point_adjacency_offsets.contiguous(),
            rays.contiguous(),
            start_point.contiguous(),
            depth_quantiles=depth_quantiles.contiguous(),
            weight_threshold=0.001,
            max_intersections=128,
            return_contribution=True,
        )
        return (
            output["rgba"],
            output["depth"],
            output["depth_indices"],
            output["contribution"],
            output["num_intersections"],
        )
    return trace_fwd_op(
        points.contiguous(),
        attributes.contiguous(),
        point_adjacency.contiguous(),
        point_adjacency_offsets.contiguous(),
        rays.contiguous(),
        start_point.contiguous(),
        depth_quantiles.contiguous(),
        True,
        sh_degree,
        0.001,
        128,
    )


def trace_backward(
    points: torch.Tensor,
    attributes: torch.Tensor,
    point_adjacency: torch.Tensor,
    point_adjacency_offsets: torch.Tensor,
    rays: torch.Tensor,
    start_point: torch.Tensor,
    rgba: torch.Tensor,
    grad_rgba: torch.Tensor,
    depth_quantiles: torch.Tensor,
    depth_indices: torch.Tensor,
    grad_depth: torch.Tensor,
    sh_degree: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if backend_name == "reference":
        pipeline = reference_radfoam.create_pipeline(sh_degree, attributes.dtype)
        output = pipeline.trace_backward(
            points.contiguous(),
            attributes.contiguous(),
            point_adjacency.contiguous(),
            point_adjacency_offsets.contiguous(),
            rays.contiguous(),
            start_point.contiguous(),
            rgba.contiguous(),
            grad_rgba.contiguous(),
            depth_quantiles.contiguous(),
            depth_indices.contiguous(),
            grad_depth.contiguous(),
            None,
            0.001,
            128,
        )
        return output["points_grad"], output["attr_grad"], output["ray_grad"]
    points_grad, attributes_grad, ray_grad, _point_error = trace_bwd_op(
        points.contiguous(),
        attributes.contiguous(),
        point_adjacency.contiguous(),
        point_adjacency_offsets.contiguous(),
        rays.contiguous(),
        start_point.contiguous(),
        rgba.contiguous(),
        grad_rgba.contiguous(),
        depth_quantiles.contiguous(),
        depth_indices.contiguous(),
        grad_depth.contiguous(),
        torch.empty((0,), device=points.device, dtype=attributes.dtype),
        True,
        False,
        sh_degree,
        0.001,
        128,
    )
    return points_grad, attributes_grad, ray_grad


def make_trace_inputs() -> tuple[torch.Tensor, ...]:
    points = make_points()
    permutation, point_adjacency, point_adjacency_offsets = triangulate(points)
    ordered_points = points[permutation].contiguous()
    aabb_tree = build_aabb_tree(ordered_points)

    sh_degree = 0
    torch.manual_seed(20260515)
    attributes = torch.rand(
        (ordered_points.shape[0], 1 + 3 * (1 + sh_degree) * (1 + sh_degree)),
        device=ordered_points.device,
        dtype=torch.float32,
    )
    attributes[:, -1] = torch.rand(
        (ordered_points.shape[0],),
        device=ordered_points.device,
        dtype=torch.float32,
    ) * 4.0 + 0.2

    torch.manual_seed(20260516)
    ray_directions = torch.randn((9, 3), device=ordered_points.device)
    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    ray_origins = torch.tensor(
        [0.0, 0.0, -2.0],
        device=ordered_points.device,
        dtype=torch.float32,
    ).expand_as(ray_directions)
    rays = torch.cat([ray_origins, ray_directions], dim=-1).contiguous()
    start_point = nearest_neighbor(ordered_points, aabb_tree, ray_origins[:1])
    start_point = start_point.to(torch.uint32).expand((rays.shape[0],)).contiguous()
    depth_quantiles = torch.full(
        (rays.shape[0], 1),
        0.5,
        device=ordered_points.device,
        dtype=torch.float32,
    )
    return (
        ordered_points,
        attributes,
        point_adjacency,
        point_adjacency_offsets,
        rays,
        start_point,
        depth_quantiles,
        torch.tensor(sh_degree, device=ordered_points.device),
    )


def save_tensors(*tensors: torch.Tensor) -> None:
    torch.cuda.synchronize()
    torch.save({str(index): tensor.detach().cpu() for index, tensor in enumerate(tensors)}, output_path)


points = make_points()

if kernel_name == "triangulate":
    save_tensors(*triangulate(points))
elif kernel_name == "build_aabb_tree":
    save_tensors(build_aabb_tree(points))
elif kernel_name == "nearest_neighbor":
    tree = build_aabb_tree(points)
    save_tensors(nearest_neighbor(points, tree, make_queries()))
elif kernel_name == "farthest_neighbor":
    num_points = points.shape[0]
    point_indices = torch.arange(num_points, device=points.device, dtype=torch.long)
    left = ((point_indices - 1) % num_points).to(torch.uint32)
    right = ((point_indices + 1) % num_points).to(torch.uint32)
    adjacency = torch.stack([left, right], dim=1).reshape(-1).contiguous()
    offsets = torch.arange(
        0,
        2 * num_points + 1,
        2,
        device=points.device,
        dtype=torch.long,
    ).to(torch.uint32)
    save_tensors(*farthest_neighbor(points, adjacency, offsets))
elif kernel_name == "trace_fwd":
    trace_inputs = make_trace_inputs()
    save_tensors(*trace_forward(*trace_inputs[:-1], int(trace_inputs[-1].item())))
elif kernel_name == "trace_bwd":
    trace_inputs = make_trace_inputs()
    (
        trace_points,
        trace_attributes,
        trace_point_adjacency,
        trace_point_adjacency_offsets,
        trace_rays,
        trace_start_point,
        trace_depth_quantiles,
        trace_sh_degree,
    ) = trace_inputs
    sh_degree = int(trace_inputs[-1].item())
    rgba, depth, depth_indices, _contribution, _num_intersections = trace_forward(
        *trace_inputs[:-1],
        sh_degree,
    )
    torch.manual_seed(20260517)
    grad_rgba = torch.randn_like(rgba)
    grad_depth = torch.linspace(
        -0.25,
        0.25,
        steps=depth.numel(),
        device=depth.device,
        dtype=depth.dtype,
    ).reshape_as(depth)
    save_tensors(
        *trace_backward(
            trace_points,
            trace_attributes,
            trace_point_adjacency,
            trace_point_adjacency_offsets,
            trace_rays,
            trace_start_point,
            rgba,
            grad_rgba,
            trace_depth_quantiles,
            depth_indices,
            grad_depth,
            int(trace_sh_degree.item()),
        )
    )
else:
    raise AssertionError(f"unknown kernel: {kernel_name}")
"""


def _reference_extension_available() -> bool:
    return any(RADFOAM_REFERENCE_EXTENSION.glob("torch_bindings*.so"))


def _run_kernel(
    backend_name: str,
    kernel_name: str,
    tmp_path: Path,
) -> dict[str, torch.Tensor]:
    output_path = tmp_path / f"{backend_name}_{kernel_name}.pt"
    subprocess.run(
        [sys.executable, "-c", KERNEL_SCRIPT, backend_name, kernel_name, str(output_path)],
        cwd=REPO_ROOT,
        check=True,
    )
    return torch.load(output_path, map_location="cpu")


def _assert_kernel_matches_reference(
    kernel_name: str,
    tmp_path: Path,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    if not _reference_extension_available():
        pytest.skip("third_party/radfoam reference extension is not built")
    reference_output = _run_kernel("reference", kernel_name, tmp_path)
    native_output = _run_kernel("native", kernel_name, tmp_path)

    assert reference_output.keys() == native_output.keys()
    for key in reference_output:
        torch.testing.assert_close(
            native_output[key],
            reference_output[key],
            atol=atol,
            rtol=rtol,
        )


def test_radfoam_triangulate_kernel_matches_reference(tmp_path: Path) -> None:
    _assert_kernel_matches_reference("triangulate", tmp_path)


def test_radfoam_build_aabb_tree_kernel_matches_reference(tmp_path: Path) -> None:
    _assert_kernel_matches_reference("build_aabb_tree", tmp_path)


def test_radfoam_nearest_neighbor_kernel_matches_reference(tmp_path: Path) -> None:
    _assert_kernel_matches_reference("nearest_neighbor", tmp_path)


def test_radfoam_farthest_neighbor_kernel_matches_reference(tmp_path: Path) -> None:
    _assert_kernel_matches_reference("farthest_neighbor", tmp_path)


def test_radfoam_trace_forward_kernel_matches_reference(tmp_path: Path) -> None:
    _assert_kernel_matches_reference("trace_fwd", tmp_path, atol=1e-6, rtol=1e-6)


def test_radfoam_trace_backward_kernel_matches_reference(tmp_path: Path) -> None:
    _assert_kernel_matches_reference("trace_bwd", tmp_path, atol=1e-5, rtol=1e-5)
