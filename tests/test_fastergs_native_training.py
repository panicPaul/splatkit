from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.cuda]


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FasterGS native training tests.")


def test_relocation_adjustment_matches_reference_cuda_backend() -> None:
    _require_cuda()
    reference_backend = pytest.importorskip("FasterGSCudaBackend")
    from ember_native_faster_gs.faster_gs.training import (
        relocation_adjustment,
    )

    old_opacities = torch.tensor(
        [0.9, 0.5, 0.1],
        dtype=torch.float32,
        device="cuda",
    )
    old_scales = torch.tensor(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    counts = torch.tensor([1, 4, 12], dtype=torch.int64, device="cuda")

    actual_opacities, actual_scales = relocation_adjustment(
        old_opacities,
        old_scales,
        counts,
    )
    expected_opacities, expected_scales = (
        reference_backend.relocation_adjustment(
            old_opacities,
            old_scales,
            counts,
        )
    )

    torch.testing.assert_close(actual_opacities, expected_opacities)
    torch.testing.assert_close(actual_scales, expected_scales)


def test_add_noise_matches_reference_cuda_backend() -> None:
    _require_cuda()
    reference_backend = pytest.importorskip("FasterGSCudaBackend")
    from ember_native_faster_gs.faster_gs.training import add_noise

    raw_scales = torch.full((3, 3), -1.0, dtype=torch.float32, device="cuda")
    raw_rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.2, 0.3], [0.7, 0.2, 0.3, 0.4]],
        dtype=torch.float32,
        device="cuda",
    )
    raw_opacities = torch.tensor(
        [[2.0], [1.0], [0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    actual_means = torch.zeros((3, 3), dtype=torch.float32, device="cuda")
    expected_means = actual_means.clone()

    torch.manual_seed(123)
    add_noise(raw_scales, raw_rotations, raw_opacities, actual_means, 0.01)
    torch.manual_seed(123)
    reference_backend.add_noise(
        raw_scales,
        raw_rotations,
        raw_opacities,
        expected_means,
        0.01,
    )

    torch.testing.assert_close(actual_means, expected_means)


def test_fused_adam_matches_reference_cuda_backend() -> None:
    _require_cuda()
    reference_backend = pytest.importorskip("FasterGSCudaBackend")
    from ember_native_faster_gs.faster_gs.training import FusedAdam

    actual_parameter = torch.tensor(
        [[1.0, 2.0, 3.0]],
        dtype=torch.float32,
        device="cuda",
    )
    expected_parameter = actual_parameter.clone()
    gradient = torch.tensor(
        [[0.1, -0.2, 0.3]],
        dtype=torch.float32,
        device="cuda",
    )
    actual_parameter.grad = gradient.clone()
    expected_parameter.grad = gradient.clone()

    actual_optimizer = FusedAdam([actual_parameter], lr=0.01, eps=1e-15)
    expected_optimizer = reference_backend.FusedAdam(
        [expected_parameter],
        lr=0.01,
        eps=1e-15,
    )

    actual_optimizer.step()
    expected_optimizer.step()

    torch.testing.assert_close(actual_parameter, expected_parameter)
    actual_state = actual_optimizer.state[actual_parameter]
    expected_state = expected_optimizer.state[expected_parameter]
    torch.testing.assert_close(
        actual_state["exp_avg"], expected_state["exp_avg"]
    )
    torch.testing.assert_close(
        actual_state["exp_avg_sq"],
        expected_state["exp_avg_sq"],
    )
