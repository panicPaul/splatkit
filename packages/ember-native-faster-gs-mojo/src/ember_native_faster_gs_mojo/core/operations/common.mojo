from std.math import copysign, exp, log, max, min, rsqrt, sqrt
from std.utils.numerics import FPUtils

from layout import Layout, LayoutTensor
from tensor import InputTensor


@fieldwise_init
struct Float2(ImplicitlyCopyable):
    var x: Float32
    var y: Float32


@fieldwise_init
struct Float3(ImplicitlyCopyable):
    var x: Float32
    var y: Float32
    var z: Float32


@fieldwise_init
struct Float4(ImplicitlyCopyable):
    var x: Float32
    var y: Float32
    var z: Float32
    var w: Float32


comptime DILATION = Float32(0.3)
comptime DILATION_PROPER_ANTIALIASING = Float32(0.1)
comptime MIN_COV2D_DETERMINANT = Float32(1.0e-6)
comptime MIN_ALPHA_THRESHOLD_RCP = Float32(255.0)
comptime MIN_ALPHA_THRESHOLD = Float32(1.0) / MIN_ALPHA_THRESHOLD_RCP
comptime MAX_POWER_THRESHOLD = Float32(5.54126354516)
comptime ORIGINAL_OPACITY_INTERPRETATION = True
comptime TILE_WIDTH = 16
comptime TILE_HEIGHT = 16
comptime BLOCK_SIZE_BLEND = TILE_WIDTH * TILE_HEIGHT
comptime BUCKET_SIZE = 32

comptime C0 = Float32(0.28209479177387814)
comptime C1 = Float32(0.48860251190291987)
comptime C2A = Float32(1.0925484305920792)
comptime C2B = Float32(0.94617469575755997)
comptime C2C = Float32(0.31539156525251999)
comptime C2D = Float32(0.54627421529603959)
comptime C2E = Float32(1.8923493915151202)
comptime C3A = Float32(0.59004358992664352)
comptime C3B = Float32(1.7701307697799304)
comptime C3C = Float32(2.8906114426405538)
comptime C3D = Float32(0.45704579946446572)
comptime C3E = Float32(2.2852289973223288)
comptime C3F = Float32(1.865881662950577)
comptime C3G = Float32(1.1195289977703462)
comptime C3H = Float32(1.4453057213202769)
comptime C3I = Float32(3.5402615395598609)
comptime C3J = Float32(4.5704579946446566)
comptime C3K = Float32(5.597644988851731)


@always_inline
def div_round_up(n: Int, d: Int) -> Int:
    """Return the ceiling of `n / d` for positive integers."""
    return (n + d - 1) // d


@always_inline
def sigmoid(x: Float32) -> Float32:
    """Apply the logistic sigmoid used by FasterGS opacity decoding."""
    return Float32(1.0) / (Float32(1.0) + exp(-x))


@always_inline
def saturate(x: Float32) -> Float32:
    """Clamp a scalar into the closed unit interval."""
    return min(max(x, Float32(0.0)), Float32(1.0))


@always_inline
def normalize3(x: Float32, y: Float32, z: Float32) -> Float3:
    """Return the normalized 3D vector, or zero if the norm degenerates."""
    norm_sq = x * x + y * y + z * z
    if norm_sq <= Float32(0.0):
        return Float3(Float32(0.0), Float32(0.0), Float32(0.0))
    inv_norm = rsqrt(norm_sq)
    return Float3(x * inv_norm, y * inv_norm, z * inv_norm)


@always_inline
def float_bitcast_to_int32(x: Float32) -> Int32:
    """Bitcast a float depth value into its sortable 32-bit integer key."""
    return Int32(FPUtils[DType.float32].bitcast_to_uint(x))


@always_inline
def will_primitive_contribute(
    mean_x: Float32,
    mean_y: Float32,
    conic_x: Float32,
    conic_y: Float32,
    conic_z: Float32,
    tile_x: Int,
    tile_y: Int,
    power_threshold: Float32,
) -> Bool:
    """Return whether a primitive can contribute to at least one pixel in a tile."""
    # This mirrors the FasterGS tile-overlap predicate: first do the cheap
    # rectangle containment test, then only solve the quadratic corner case
    # when the mean lies outside the tile extent.
    rect_min_x = Float32(tile_x * TILE_WIDTH)
    rect_min_y = Float32(tile_y * TILE_HEIGHT)
    rect_max_x = Float32((tile_x + 1) * TILE_WIDTH - 1)
    rect_max_y = Float32((tile_y + 1) * TILE_HEIGHT - 1)

    x_min_diff = rect_min_x - mean_x
    x_left = Float32(1.0 if x_min_diff > Float32(0.0) else 0.0)
    not_in_x_range = x_left + Float32(1.0 if mean_x > rect_max_x else 0.0)

    y_min_diff = rect_min_y - mean_y
    y_above = Float32(1.0 if y_min_diff > Float32(0.0) else 0.0)
    not_in_y_range = y_above + Float32(1.0 if mean_y > rect_max_y else 0.0)

    if not_in_x_range + not_in_y_range == Float32(0.0):
        return True

    closest_corner_x = rect_max_x if x_left == Float32(0.0) else rect_min_x
    closest_corner_y = rect_max_y if y_above == Float32(0.0) else rect_min_y
    diff_x = mean_x - closest_corner_x
    diff_y = mean_y - closest_corner_y

    d_x = copysign(Float32(TILE_WIDTH - 1), x_min_diff)
    d_y = copysign(Float32(TILE_HEIGHT - 1), y_min_diff)
    t_x = not_in_y_range * saturate((d_x * conic_x * diff_x + d_x * conic_y * diff_y) / (d_x * conic_x * d_x))
    t_y = not_in_x_range * saturate((d_y * conic_y * diff_x + d_y * conic_z * diff_y) / (d_y * conic_z * d_y))

    max_point_x = closest_corner_x + t_x * d_x
    max_point_y = closest_corner_y + t_y * d_y
    delta_x = mean_x - max_point_x
    delta_y = mean_y - max_point_y
    max_power = Float32(0.5) * (
        conic_x * delta_x * delta_x
        + conic_z * delta_y * delta_y
    ) + conic_y * delta_x * delta_y
    return max_power <= power_threshold


@always_inline
def compute_exact_n_touched_tiles(
    mean_x: Float32,
    mean_y: Float32,
    conic_x: Float32,
    conic_y: Float32,
    conic_z: Float32,
    screen_min_x: Int,
    screen_max_x: Int,
    screen_min_y: Int,
    screen_max_y: Int,
    power_threshold: Float32,
) -> Int32:
    """Count exactly how many tiles survive the contribution predicate."""
    # The preprocess stage uses a conservative screen-space box first, then
    # refines it here so sort/blend only see real tile instances.
    shifted_mean_x = mean_x - Float32(0.5)
    shifted_mean_y = mean_y - Float32(0.5)
    touched = Int32(0)
    for tile_y in range(screen_min_y, screen_max_y):
        for tile_x in range(screen_min_x, screen_max_x):
            if will_primitive_contribute(
                shifted_mean_x,
                shifted_mean_y,
                conic_x,
                conic_y,
                conic_z,
                tile_x,
                tile_y,
                power_threshold,
            ):
                touched += 1
    return touched


@always_inline
def load_sh0[
    sh0_layout: Layout,
](
    sh_coefficients_0: LayoutTensor[DType.float32, sh0_layout, MutAnyOrigin],
    primitive_idx: Int,
) -> Float3:
    """Load the DC spherical-harmonics coefficient triplet."""
    return Float3(
        rebind[Float32](sh_coefficients_0[primitive_idx, 0, 0]),
        rebind[Float32](sh_coefficients_0[primitive_idx, 0, 1]),
        rebind[Float32](sh_coefficients_0[primitive_idx, 0, 2]),
    )


@always_inline
def load_shrest[
    shrest_layout: Layout,
](
    sh_coefficients_rest: LayoutTensor[
        DType.float32,
        shrest_layout,
        MutAnyOrigin,
    ],
    primitive_idx: Int,
    base_idx: Int,
) -> Float3:
    """Load one higher-order spherical-harmonics coefficient triplet."""
    return Float3(
        rebind[Float32](sh_coefficients_rest[primitive_idx, base_idx, 0]),
        rebind[Float32](sh_coefficients_rest[primitive_idx, base_idx, 1]),
        rebind[Float32](sh_coefficients_rest[primitive_idx, base_idx, 2]),
    )


@always_inline
def convert_sh_to_color[
    active_sh_bases: Int,
    sh0_layout: Layout,
    shrest_layout: Layout,
](
    sh_coefficients_0: LayoutTensor[DType.float32, sh0_layout, MutAnyOrigin],
    sh_coefficients_rest: LayoutTensor[
        DType.float32,
        shrest_layout,
        MutAnyOrigin,
    ],
    position_x: Float32,
    position_y: Float32,
    position_z: Float32,
    cam_position_x: Float32,
    cam_position_y: Float32,
    cam_position_z: Float32,
    primitive_idx: Int,
) -> Float3:
    """Evaluate the compile-time SH basis-count case into a view-dependent RGB color."""
    comptime assert (
        active_sh_bases == 1 or active_sh_bases == 4
        or active_sh_bases == 9 or active_sh_bases == 16
    ), "Unsupported FasterGS SH basis-count specialization."

    # The basis expansion follows the FasterGS core thresholds exactly:
    # `1`, `4`, `9`, and `16` total bases including the DC term.
    color0 = load_sh0(sh_coefficients_0, primitive_idx)
    var result = Float3(
        Float32(0.5) + C0 * color0.x,
        Float32(0.5) + C0 * color0.y,
        Float32(0.5) + C0 * color0.z,
    )
    comptime if active_sh_bases > 1:
        var direction = normalize3(
            position_x - cam_position_x,
            position_y - cam_position_y,
            position_z - cam_position_z,
        )
        x = direction.x
        y = direction.y
        z = direction.z

        coeff0 = load_shrest(sh_coefficients_rest, primitive_idx, 0)
        coeff1 = load_shrest(sh_coefficients_rest, primitive_idx, 1)
        coeff2 = load_shrest(sh_coefficients_rest, primitive_idx, 2)
        result = Float3(
            result.x - C1 * y * coeff0.x + C1 * z * coeff1.x - C1 * x * coeff2.x,
            result.y - C1 * y * coeff0.y + C1 * z * coeff1.y - C1 * x * coeff2.y,
            result.z - C1 * y * coeff0.z + C1 * z * coeff1.z - C1 * x * coeff2.z,
        )

        comptime if active_sh_bases > 4:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            xz = x * z
            yz = y * z
            coeff3 = load_shrest(sh_coefficients_rest, primitive_idx, 3)
            coeff4 = load_shrest(sh_coefficients_rest, primitive_idx, 4)
            coeff5 = load_shrest(sh_coefficients_rest, primitive_idx, 5)
            coeff6 = load_shrest(sh_coefficients_rest, primitive_idx, 6)
            coeff7 = load_shrest(sh_coefficients_rest, primitive_idx, 7)
            result = Float3(
                result.x
                + C2A * xy * coeff3.x
                - C2A * yz * coeff4.x
                + (C2B * zz - C2C) * coeff5.x
                - C2A * xz * coeff6.x
                + C2D * (xx - yy) * coeff7.x,
                result.y
                + C2A * xy * coeff3.y
                - C2A * yz * coeff4.y
                + (C2B * zz - C2C) * coeff5.y
                - C2A * xz * coeff6.y
                + C2D * (xx - yy) * coeff7.y,
                result.z
                + C2A * xy * coeff3.z
                - C2A * yz * coeff4.z
                + (C2B * zz - C2C) * coeff5.z
                - C2A * xz * coeff6.z
                + C2D * (xx - yy) * coeff7.z,
            )

            comptime if active_sh_bases > 9:
                coeff8 = load_shrest(sh_coefficients_rest, primitive_idx, 8)
                coeff9 = load_shrest(sh_coefficients_rest, primitive_idx, 9)
                coeff10 = load_shrest(sh_coefficients_rest, primitive_idx, 10)
                coeff11 = load_shrest(sh_coefficients_rest, primitive_idx, 11)
                coeff12 = load_shrest(sh_coefficients_rest, primitive_idx, 12)
                coeff13 = load_shrest(sh_coefficients_rest, primitive_idx, 13)
                coeff14 = load_shrest(sh_coefficients_rest, primitive_idx, 14)
                result = Float3(
                    result.x
                    + y * (C3A * yy - C3B * xx) * coeff8.x
                    + C3C * xy * z * coeff9.x
                    + y * (C3D - C3E * zz) * coeff10.x
                    + z * (C3F * zz - C3G) * coeff11.x
                    + x * (C3D - C3E * zz) * coeff12.x
                    + C3H * z * (xx - yy) * coeff13.x
                    + x * (C3B * yy - C3A * xx) * coeff14.x,
                    result.y
                    + y * (C3A * yy - C3B * xx) * coeff8.y
                    + C3C * xy * z * coeff9.y
                    + y * (C3D - C3E * zz) * coeff10.y
                    + z * (C3F * zz - C3G) * coeff11.y
                    + x * (C3D - C3E * zz) * coeff12.y
                    + C3H * z * (xx - yy) * coeff13.y
                    + x * (C3B * yy - C3A * xx) * coeff14.y,
                    result.z
                    + y * (C3A * yy - C3B * xx) * coeff8.z
                    + C3C * xy * z * coeff9.z
                    + y * (C3D - C3E * zz) * coeff10.z
                    + z * (C3F * zz - C3G) * coeff11.z
                    + x * (C3D - C3E * zz) * coeff12.z
                    + C3H * z * (xx - yy) * coeff13.z
                    + x * (C3B * yy - C3A * xx) * coeff14.z,
                )
    return result
