"""Scene normalization utilities.

Two-stage pipeline for centering, orienting, and scaling a 3-D scene:

  1. ``similarity_from_cameras`` — derive a similarity transform from camera
     poses so that the scene is recentered and the average scene up direction
     aligns with +Z while the cameras keep OpenCV forward (+Z).
  2. ``pca_transform_from_points`` — refine orientation via PCA so that the
     principal axes of a point set align with the coordinate axes.

The transforms are plain 4x4 float64 NumPy arrays (homogeneous / SE(3) with
optional scale baked into the rotation block).  Separate ``apply_to_points``
and ``apply_to_cameras`` helpers apply a transform without knowing anything
about what the points represent, so the same utilities work for Gaussian splat
centers, raw point clouds, mesh vertices, etc.

Auto-converted from gsplat.
"""

from typing import Literal

import numpy as np
from jaxtyping import Float
from numpy import ndarray

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def apply_to_points(
    transform: Float[ndarray, "4 4"],
    points: Float[ndarray, "N 3"],
) -> Float[ndarray, "N 3"]:
    """Apply a 4x4 homogeneous transform to an (N, 3) point array.

    The rotation block may carry a scale factor (as produced by
    ``similarity_from_cameras``); that scale is applied to the points.

    Args:
        transform: 4x4 similarity / SE(3) transform (scale may be baked into
            the rotation block).
        points: (N, 3) point positions.

    Returns:
        (N, 3) transformed point positions.
    """
    return points @ transform[:3, :3].T + transform[:3, 3]


def apply_to_cameras(
    transform: Float[ndarray, "4 4"],
    camera_to_world: Float[ndarray, "N 4 4"],
) -> Float[ndarray, "N 4 4"]:
    """Apply a 4x4 similarity transform to an array of camera-to-world matrices.

    After multiplication the rotation blocks are renormalised to unit scale so
    that downstream code (renderers, etc.) receives proper rotation matrices.

    Args:
        transform: 4x4 similarity / SE(3) transform (scale may be baked into
            the rotation block).
        camera_to_world: (N, 4, 4) camera-to-world matrices in OpenCV
            convention.

    Returns:
        (N, 4, 4) transformed camera-to-world matrices with unit-scale
        rotations.
    """
    transformed = np.einsum("nij,ki->nkj", camera_to_world, transform)
    scale = np.linalg.norm(transformed[:, 0, :3], axis=1)
    transformed[:, :3, :3] /= scale[:, None, None]
    return transformed


# ---------------------------------------------------------------------------
# Stage 1 - camera-based similarity transform
# ---------------------------------------------------------------------------


def similarity_from_cameras(
    camera_to_world: Float[ndarray, "N 4 4"],
    *,
    center_method: Literal["focus", "poses"] = "focus",
    strict_scaling: bool = False,
) -> Float[ndarray, "4 4"]:
    """Compute a similarity transform that normalises a set of camera poses.

    The returned 4x4 matrix (scale baked into the rotation block) can be
    passed directly to ``apply_to_points`` and ``apply_to_cameras``.

    Three sequential operations are composed into a single matrix:

    1. **Up-axis alignment** - rotate the world so that the mean camera up
       vector aligns with +Z (using Rodrigues' formula, no gimbal lock).
    2. **Recentering** - translate so that the scene focus / median camera
       position moves to the origin.
    3. **Rescaling** - scale so that the typical camera-to-origin distance
       equals 1.

    Args:
        camera_to_world: (N, 4, 4) camera-to-world matrices in OpenCV
            convention (camera -Y is up, +Z is forward).
        center_method: ``"focus"`` (default) finds the median nearest point
            along each camera's forward ray — good for inward-facing captures.
            ``"poses"`` simply uses the median camera position.
        strict_scaling: If ``True`` scale by the *maximum* camera distance
            (scene fits inside the unit sphere); if ``False`` (default) scale
            by the *median* distance (more robust to outlier cameras).

    Returns:
        4x4 similarity transform as a float64 NumPy array.
    """
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)

    positions = camera_to_world[:, :3, 3]  # (N, 3) camera positions
    rotations = camera_to_world[:, :3, :3]  # (N, 3, 3) rotation matrices

    # ------------------------------------------------------------------
    # Step 1: align world up to +Z
    # ------------------------------------------------------------------
    # In OpenCV convention the camera's local up is -Y, so the world up
    # vectors are the negated Y columns of the rotation matrices.
    up_vectors = np.sum(
        rotations * np.array([0.0, -1.0, 0.0]), axis=-1
    )  # (N, 3)
    world_up = np.mean(up_vectors, axis=0)
    world_up /= np.linalg.norm(world_up)

    # Rotate the mean scene up vector onto +Z via Rodrigues' formula.
    target_up = np.array([0.0, 0.0, 1.0])
    cos_angle = float(np.dot(target_up, world_up))
    cross = np.cross(world_up, target_up)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if cos_angle > -1.0 + 1e-6:
        rotation_align = (
            np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + cos_angle))
        )
    else:
        # ~180° rotation: the mean up points at -Z, flip about X.
        rotation_align = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        )

    # Apply the up-alignment rotation to positions and rotations.
    rotations_aligned = np.einsum("ij,njk->nik", rotation_align, rotations)
    positions_aligned = (rotation_align @ positions.T).T  # (N, 3)

    # ------------------------------------------------------------------
    # Step 2: recenter
    # ------------------------------------------------------------------
    if center_method == "focus":
        # Forward direction of each camera in the aligned frame.
        forward_vectors = np.sum(
            rotations_aligned * np.array([0.0, 0.0, 1.0]), axis=-1
        )  # (N, 3)
        # Nearest point on each forward ray to the origin.
        nearest = positions_aligned + (
            (forward_vectors * -positions_aligned).sum(-1)[:, None]
            * forward_vectors
        )
        translation = -np.median(nearest, axis=0)
    else:
        translation = -np.median(positions_aligned, axis=0)

    # ------------------------------------------------------------------
    # Step 3: rescale
    # ------------------------------------------------------------------
    distances = np.linalg.norm(positions_aligned + translation, axis=-1)
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / float(scale_fn(distances))

    # ------------------------------------------------------------------
    # Compose: T_final = Scale @ Translate @ R_align
    # ------------------------------------------------------------------
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_align
    transform[:3, 3] = translation
    transform[:3, :] *= scale

    return transform


# ---------------------------------------------------------------------------
# Stage 2 - PCA-based orientation refinement
# ---------------------------------------------------------------------------


def pca_transform_from_points(
    points: Float[ndarray, "N 3"],
) -> Float[ndarray, "4 4"]:
    """Compute a rigid transform that aligns a point set to its principal axes.

    The eigenvector with the *largest* eigenvalue maps to +X, the next to +Y,
    and the smallest to +Z. The determinant is checked and corrected so the
    result is always a proper rotation (det = +1), then a final sign fix
    removes the 180-degree ambiguity by choosing the polarity that matches
    viser's +Z-up scene convention for grounded captures.

    Args:
        points: (N, 3) point positions (e.g. Gaussian centres, SfM sparse
            cloud).

    Returns:
        4x4 rigid transform (no scale) as a float64 NumPy array.
    """
    points = np.asarray(points, dtype=np.float64)

    centroid = np.median(points, axis=0)
    centered = points - centroid

    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort descending so the principal axis goes to X.
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]

    # Ensure a right-handed coordinate system.
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    rotation = eigenvectors.T  # (3, 3): rows are the new basis vectors

    # PCA leaves each axis sign ambiguous. Empirically, grounded captures in
    # our viewer line up with viser's +Z-up convention when the broader
    # vertical extent lies on the negative-Z side before camera placement.
    transformed_points = centered @ rotation.T
    max_height = float(np.max(transformed_points[:, 2]))
    min_height = float(np.min(transformed_points[:, 2]))
    if max_height > abs(min_height):
        rotation = np.diag([1.0, -1.0, -1.0]) @ rotation

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -(rotation @ centroid)

    return transform


# ---------------------------------------------------------------------------
# Per-primitive attribute updates
# ---------------------------------------------------------------------------


def apply_scale_to_log_scales(
    scene_scale: float,
    log_scales: Float[ndarray, "N D"],
) -> Float[ndarray, "N D"]:
    """Add the scene scale into stored log-scale attributes.

    Gaussian splats store scales as log(s). Multiplying the primitive by a
    scene scale factor is equivalent to adding log(scene_scale) to each
    stored value.

    Args:
        scene_scale: Scalar scene scale factor (>0).
        log_scales: (N, D) array of log-scale values, where D is typically
            3 for anisotropic Gaussians.

    Returns:
        (N, D) updated log-scale array.
    """
    return log_scales + np.log(scene_scale)


def apply_rotation_to_quaternions(
    rotation: Float[ndarray, "3 3"],
    quaternions: Float[ndarray, "N 4"],
) -> Float[ndarray, "N 4"]:
    """Compose a scene rotation into per-primitive quaternions.

    Quaternions are expected in (w, x, y, z) order and do not need to be
    normalised on input; normalisation is preserved on output.

    Args:
        rotation: (3, 3) rotation matrix extracted from the scene transform.
        quaternions: (N, 4) per-primitive quaternions in wxyz order.

    Returns:
        (N, 4) rotated quaternions in wxyz order, with the same norm as the
        inputs.
    """
    # Convert the scene rotation matrix to a quaternion (w, x, y, z).
    trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        rw = 0.25 / s
        rx = (rotation[2, 1] - rotation[1, 2]) * s
        ry = (rotation[0, 2] - rotation[2, 0]) * s
        rz = (rotation[1, 0] - rotation[0, 1]) * s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * np.sqrt(
            1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
        )
        rw = (rotation[2, 1] - rotation[1, 2]) / s
        rx = 0.25 * s
        ry = (rotation[0, 1] + rotation[1, 0]) / s
        rz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * np.sqrt(
            1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
        )
        rw = (rotation[0, 2] - rotation[2, 0]) / s
        rx = (rotation[0, 1] + rotation[1, 0]) / s
        ry = 0.25 * s
        rz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(
            1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
        )
        rw = (rotation[1, 0] - rotation[0, 1]) / s
        rx = (rotation[0, 2] + rotation[2, 0]) / s
        ry = (rotation[1, 2] + rotation[2, 1]) / s
        rz = 0.25 * s

    scene_q = np.array([rw, rx, ry, rz], dtype=np.float64)

    # Hamilton product: scene_q * primitive_q for each primitive.
    pw, px, py, pz = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )
    rw, rx, ry, rz = scene_q

    out = np.stack(
        [
            rw * pw - rx * px - ry * py - rz * pz,
            rw * px + rx * pw + ry * pz - rz * py,
            rw * py - rx * pz + ry * pw + rz * px,
            rw * pz + rx * py - ry * px + rz * pw,
        ],
        axis=-1,
    )
    return out.astype(quaternions.dtype)


def _wigner_d_matrices(
    rotation: Float[ndarray, "3 3"],
    max_degree: int,
) -> list[Float[ndarray, "..."]]:
    """Compute Wigner D-matrices for real SH bands up to max_degree.

    Uses the Ivanic & Ruedenberg recurrence (doi:10.1021/jp953350u) to build
    each band from the previous one, seeded by the D1 matrix which equals the
    rotation matrix itself (in the real-SH yzx ordering used by 3DGS).

    Args:
        rotation: (3, 3) rotation matrix.
        max_degree: Highest SH degree to compute (0..4).

    Returns:
        List of (2l+1, 2l+1) D-matrices, one per degree l=0..max_degree.
    """
    r = rotation.astype(np.float64)

    # D0 is trivially 1x1 identity.
    d_matrices = [np.ones((1, 1))]
    if max_degree == 0:
        return d_matrices

    # D1: real SH degree-1 ordering is (y, z, x) <-> rows/cols of R.
    # Permutation from world (x,y,z) to SH-l1 (y,z,x): p = [1, 2, 0].
    p = [1, 2, 0]
    d1 = r[np.ix_(p, p)]
    d_matrices.append(d1)

    # Recurrence for l >= 2.
    for degree in range(2, max_degree + 1):
        size = 2 * degree + 1
        d_prev = d_matrices[degree - 1]  # (2l-1, 2l-1)
        d_prev2 = d_matrices[degree - 2]  # (2l-3, 2l-3)
        d_curr = np.zeros((size, size))

        # Use the Ivanic & Ruedenberg u/v/w recurrence.
        # For each output row m and column n in -l..l:
        for m_idx in range(size):
            m = m_idx - degree
            for n_idx in range(size):
                n = n_idx - degree
                d_curr[m_idx, n_idx] = _ivanic_entry(
                    degree, m, n, d1, d_prev, d_prev2
                )

        d_matrices.append(d_curr)

    return d_matrices


def _ivanic_entry(
    l: int,
    m: int,
    n: int,
    d1: Float[ndarray, "3 3"],
    d_prev: Float[ndarray, "..."],
    d_prev2: Float[ndarray, "..."],
) -> float:
    """Compute one entry of the Wigner D-matrix at degree l via recurrence.

    Implements the Ivanic & Ruedenberg (1996) real-SH recurrence relation.
    """

    # Helper: index into (2k+1) matrix for value m at degree k.
    def idx(k: int, v: int) -> int:
        return v + k

    result = 0.0

    # d1 entries needed by the recurrence (row = m', col = n' in degree-1).
    # d1 is indexed in (-1,0,1) x (-1,0,1).
    def r1(mp: int, np_: int) -> float:
        i, j = idx(1, mp), idx(1, np_)
        if 0 <= i < 3 and 0 <= j < 3:
            return d1[i, j]
        return 0.0

    def rp(mp: int, np_: int) -> float:
        i, j = idx(l - 1, mp), idx(l - 1, np_)
        s = 2 * (l - 1) + 1
        if 0 <= i < s and 0 <= j < s:
            return d_prev[i, j]
        return 0.0

    def rpp(mp: int, np_: int) -> float:
        s = 2 * (l - 2) + 1
        i, j = idx(l - 2, mp), idx(l - 2, np_)
        if 0 <= i < s and 0 <= j < s:
            return d_prev2[i, j]
        return 0.0

    def delta(a: int, b: int) -> float:
        return 1.0 if a == b else 0.0

    def P(i: int, l_: int, a: int, b: int) -> float:
        if abs(b) < l_:
            return r1(i, 0) * rp(a, b)
        elif b == l_:
            return r1(i, 1) * rp(a, l_ - 1) - r1(i, -1) * rp(a, -(l_ - 1))
        else:  # b == -l_
            return r1(i, 1) * rp(a, -(l_ - 1)) + r1(i, -1) * rp(a, l_ - 1)

    def u_coeff(l_: int, m_: int, n_: int) -> float:
        denom = (
            (l_ + n_) * (l_ - n_) if abs(n_) < l_ else (2 * l_) * (2 * l_ - 1)
        )
        if denom == 0:
            return 0.0
        num = np.sqrt(((l_ + m_) * (l_ - m_)) / denom)
        return float(num)

    def v_coeff(l_: int, m_: int, n_: int) -> float:
        denom = (
            (l_ + n_) * (l_ - n_) if abs(n_) < l_ else (2 * l_) * (2 * l_ - 1)
        )
        if denom == 0:
            return 0.0
        sign = 1.0 if m_ >= 0 else -1.0
        num = 0.5 * np.sqrt(
            (1.0 + delta(m_, 0)) * (l_ + abs(m_) - 1.0) * (l_ + abs(m_)) / denom
        )
        return float(sign * num)

    def w_coeff(l_: int, m_: int, n_: int) -> float:
        if m_ == 0:
            return 0.0
        denom = (
            (l_ + n_) * (l_ - n_) if abs(n_) < l_ else (2 * l_) * (2 * l_ - 1)
        )
        if denom == 0:
            return 0.0
        sign = 1.0 if m_ > 0 else -1.0
        num = -0.5 * np.sqrt((l_ - abs(m_) - 1.0) * (l_ - abs(m_)) / denom)
        return float(sign * num)

    u = u_coeff(l, m, n)
    v = v_coeff(l, m, n)
    w = w_coeff(l, m, n)

    if u != 0.0:
        result += u * P(0, l, m, n)
    if v != 0.0:
        am = abs(m)
        if m >= 0:
            result += v * P(1, l, am - 1, n) if am > 0 else v * P(1, l, 1, n)
        else:
            result += v * (
                P(-1, l, -(am - 1), n) if am > 0 else P(-1, l, -1, n)
            )
    if w != 0.0:
        am = abs(m)
        if m > 0:
            result += w * P(1, l, am + 1, n)
        else:
            result += w * P(-1, l, -(am + 1), n)

    return result


def apply_rotation_to_sh_coefficients(
    rotation: Float[ndarray, "3 3"],
    sh_coefficients: Float[ndarray, "N num_bases 3"],
) -> Float[ndarray, "N num_bases 3"]:
    """Rotate per-primitive SH coefficients by a scene rotation.

    Band l=0 is invariant under rotation and is unchanged. Bands l=1..4 are
    rotated using Wigner D-matrices computed via the Ivanic & Ruedenberg
    recurrence.

    The number of bases must be a perfect square: 1, 4, 9, 16, or 25
    (corresponding to max degrees 0, 1, 2, 3, 4).

    Args:
        rotation: (3, 3) rotation matrix extracted from the scene transform.
        sh_coefficients: (N, num_bases, 3) SH coefficient array in the 3DGS
            real-SH ordering (degree-major, m = -l..+l within each degree).

    Returns:
        (N, num_bases, 3) rotated SH coefficient array.
    """
    num_bases = sh_coefficients.shape[1]
    max_degree = int(np.sqrt(num_bases)) - 1
    if (max_degree + 1) ** 2 != num_bases:
        raise ValueError(
            f"num_bases must be a perfect square (1/4/9/16/25), got {num_bases}"
        )

    d_matrices = _wigner_d_matrices(rotation, max_degree)
    result = sh_coefficients.copy()

    for degree in range(1, max_degree + 1):
        start = degree**2
        end = (degree + 1) ** 2
        d = d_matrices[degree]
        result[:, start:end, :] = np.einsum(
            "ij,njc->nic", d, sh_coefficients[:, start:end, :]
        )

    return result


# ---------------------------------------------------------------------------
# Convenience: compose two transforms
# ---------------------------------------------------------------------------


def compose_transforms(
    first: Float[ndarray, "4 4"],
    second: Float[ndarray, "4 4"],
) -> Float[ndarray, "4 4"]:
    """Return ``second @ first`` — apply *first* then *second*.

    Useful when chaining ``similarity_from_cameras`` and
    ``pca_transform_from_points`` into a single matrix.

    Args:
        first: The transform applied first.
        second: The transform applied second.

    Returns:
        Composed 4x4 transform.
    """
    return second @ first
