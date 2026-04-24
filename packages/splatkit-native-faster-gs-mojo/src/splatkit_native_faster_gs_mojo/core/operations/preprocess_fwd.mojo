import compiler

from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives.block import sum as block_sum
from std.math import exp, log, max, min, sqrt
from std.runtime.asyncrt import DeviceContextPtr
from std.utils import Index
from tensor import InputTensor, OutputTensor

from .common import (
    DILATION,
    DILATION_PROPER_ANTIALIASING,
    MAX_POWER_THRESHOLD,
    MIN_ALPHA_THRESHOLD,
    MIN_ALPHA_THRESHOLD_RCP,
    MIN_COV2D_DETERMINANT,
    ORIGINAL_OPACITY_INTERPRETATION,
    TILE_HEIGHT,
    TILE_WIDTH,
    compute_exact_n_touched_tiles,
    convert_sh_to_color,
    div_round_up,
    float_bitcast_to_int32,
    sigmoid,
)

comptime ROW_MAJOR_1D = Layout.row_major(Int())
comptime ROW_MAJOR_2D = Layout.row_major(Int(), Int())
comptime ROW_MAJOR_3D = Layout.row_major(Int(), Int(), Int())
comptime COUNT_BLOCK_SIZE = 256


@always_inline
def zero_preprocess_counters_kernel[
    visible_count_layout: Layout,
    instance_count_layout: Layout,
](
    visible_count: LayoutTensor[DType.int32, visible_count_layout, MutAnyOrigin],
    instance_count: LayoutTensor[DType.int32, instance_count_layout, MutAnyOrigin],
):
    """Reset the visible and instance counters before preprocess launches."""
    if Int(block_idx.x * block_dim.x + thread_idx.x) == 0:
        visible_count[0] = Int32(0)
        instance_count[0] = Int32(0)


@always_inline
def reduce_preprocess_counts_blocks_kernel[
    num_touched_tiles_layout: Layout,
    block_visible_counts_layout: Layout,
    block_instance_counts_layout: Layout,
](
    num_touched_tiles: LayoutTensor[
        DType.int32,
        num_touched_tiles_layout,
        MutAnyOrigin,
    ],
    block_visible_counts: LayoutTensor[
        DType.int32,
        block_visible_counts_layout,
        MutAnyOrigin,
    ],
    block_instance_counts: LayoutTensor[
        DType.int32,
        block_instance_counts_layout,
        MutAnyOrigin,
    ],
    n_primitives: Int,
):
    """Reduce per-primitive touched-tile counts into one count pair per block."""
    primitive_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var visible = Int32(0)
    var instances = Int32(0)
    if primitive_idx < n_primitives:
        touched = rebind[Int32](num_touched_tiles[primitive_idx])
        if touched > 0:
            visible = Int32(1)
            instances = touched

    visible_sum = block_sum[block_dim_x=COUNT_BLOCK_SIZE, block_dim_y=1](
        SIMD[DType.int32, 1](visible)
    )
    instance_sum = block_sum[block_dim_x=COUNT_BLOCK_SIZE, block_dim_y=1](
        SIMD[DType.int32, 1](instances)
    )
    if Int(thread_idx.x) == 0:
        block_visible_counts[Int(block_idx.x)] = Int32(Int(visible_sum))
        block_instance_counts[Int(block_idx.x)] = Int32(Int(instance_sum))


@always_inline
def finalize_preprocess_counts_kernel[
    block_visible_counts_layout: Layout,
    block_instance_counts_layout: Layout,
    visible_count_layout: Layout,
    instance_count_layout: Layout,
](
    block_visible_counts: LayoutTensor[DType.int32, block_visible_counts_layout, MutAnyOrigin],
    block_instance_counts: LayoutTensor[DType.int32, block_instance_counts_layout, MutAnyOrigin],
    visible_count: LayoutTensor[DType.int32, visible_count_layout, MutAnyOrigin],
    instance_count: LayoutTensor[DType.int32, instance_count_layout, MutAnyOrigin],
    block_count: Int,
):
    """Finish block-count reduction into the scalar preprocess outputs."""
    if Int(block_idx.x * block_dim.x + thread_idx.x) == 0:
        var visible = Int32(0)
        var instances = Int32(0)
        for block in range(block_count):
            visible += rebind[Int32](block_visible_counts[block])
            instances += rebind[Int32](block_instance_counts[block])
        visible_count[0] = visible
        instance_count[0] = instances


def preprocess_fwd_kernel[
    proper_antialiasing: Bool,
    active_sh_bases: Int,
    center_positions_layout: Layout,
    log_scales_layout: Layout,
    unnormalized_rotations_layout: Layout,
    opacities_layout: Layout,
    sh_coefficients_0_layout: Layout,
    sh_coefficients_rest_layout: Layout,
    world_2_camera_layout: Layout,
    camera_position_layout: Layout,
    projected_means_layout: Layout,
    conic_opacity_layout: Layout,
    colors_rgb_layout: Layout,
    primitive_depth_layout: Layout,
    depth_keys_layout: Layout,
    primitive_indices_layout: Layout,
    num_touched_tiles_layout: Layout,
    screen_bounds_layout: Layout,
](
    center_positions: LayoutTensor[DType.float32, center_positions_layout, MutAnyOrigin],
    log_scales: LayoutTensor[DType.float32, log_scales_layout, MutAnyOrigin],
    unnormalized_rotations: LayoutTensor[
        DType.float32,
        unnormalized_rotations_layout,
        MutAnyOrigin,
    ],
    opacities: LayoutTensor[DType.float32, opacities_layout, MutAnyOrigin],
    sh_coefficients_0: LayoutTensor[
        DType.float32,
        sh_coefficients_0_layout,
        MutAnyOrigin,
    ],
    sh_coefficients_rest: LayoutTensor[
        DType.float32,
        sh_coefficients_rest_layout,
        MutAnyOrigin,
    ],
    world_2_camera: LayoutTensor[
        DType.float32,
        world_2_camera_layout,
        MutAnyOrigin,
    ],
    camera_position: LayoutTensor[
        DType.float32,
        camera_position_layout,
        MutAnyOrigin,
    ],
    projected_means: LayoutTensor[
        DType.float32,
        projected_means_layout,
        MutAnyOrigin,
    ],
    conic_opacity: LayoutTensor[
        DType.float32,
        conic_opacity_layout,
        MutAnyOrigin,
    ],
    colors_rgb: LayoutTensor[DType.float32, colors_rgb_layout, MutAnyOrigin],
    primitive_depth: LayoutTensor[
        DType.float32,
        primitive_depth_layout,
        MutAnyOrigin,
    ],
    depth_keys: LayoutTensor[DType.int32, depth_keys_layout, MutAnyOrigin],
    primitive_indices: LayoutTensor[
        DType.int32,
        primitive_indices_layout,
        MutAnyOrigin,
    ],
    num_touched_tiles: LayoutTensor[
        DType.int32,
        num_touched_tiles_layout,
        MutAnyOrigin,
    ],
    screen_bounds: LayoutTensor[
        DType.uint16,
        screen_bounds_layout,
        MutAnyOrigin,
    ],
    near_plane: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    far_plane: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    image_width: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    image_height: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    focal_x_tensor: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    focal_y_tensor: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    center_x_tensor: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    center_y_tensor: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    n_primitives: Int,
):
    """Project primitives and write per-primitive forward state."""
    primitive_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    width_value = Int(rebind[Int32](image_width[0]))
    height_value = Int(rebind[Int32](image_height[0]))
    grid_width = div_round_up(width_value, TILE_WIDTH)
    grid_height = div_round_up(height_value, TILE_HEIGHT)
    width = Float32(width_value)
    height = Float32(height_value)
    focal_x = rebind[Float32](focal_x_tensor[0])
    focal_y = rebind[Float32](focal_y_tensor[0])
    center_x = rebind[Float32](center_x_tensor[0])
    center_y = rebind[Float32](center_y_tensor[0])
    near_plane_value = rebind[Float32](near_plane[0])
    far_plane_value = rebind[Float32](far_plane[0])
    var valid = primitive_idx < n_primitives
    var contributes = valid
    var local_depth_key = Int32(0)
    var local_touched_tiles = Int32(0)

    if valid:
        primitive_indices[primitive_idx] = Int32(primitive_idx)
        depth_keys[primitive_idx] = Int32(0)
        num_touched_tiles[primitive_idx] = Int32(0)

        # Stage 1 matches the FasterGS core forward pass: depth cull first so we
        # avoid all downstream work for primitives that cannot contribute.
        mean3d_x = rebind[Float32](center_positions[primitive_idx, 0])
        mean3d_y = rebind[Float32](center_positions[primitive_idx, 1])
        mean3d_z = rebind[Float32](center_positions[primitive_idx, 2])

        w2c_r3_x = rebind[Float32](world_2_camera[2, 0])
        w2c_r3_y = rebind[Float32](world_2_camera[2, 1])
        w2c_r3_z = rebind[Float32](world_2_camera[2, 2])
        w2c_r3_w = rebind[Float32](world_2_camera[2, 3])
        depth = (
            w2c_r3_x * mean3d_x
            + w2c_r3_y * mean3d_y
            + w2c_r3_z * mean3d_z
            + w2c_r3_w
        )
        primitive_depth[primitive_idx] = depth
        if depth < near_plane_value or depth > far_plane_value:
            contributes = False

        var opacity = Float32(0.0)
        var conic_x = Float32(0.0)
        var conic_y = Float32(0.0)
        var conic_z = Float32(0.0)
        var mean2d_x = Float32(0.0)
        var mean2d_y = Float32(0.0)
        var screen_min_x = 0
        var screen_max_x = 0
        var screen_min_y = 0
        var screen_max_y = 0
        var power_threshold = MAX_POWER_THRESHOLD
        if contributes:
            raw_opacity = rebind[Float32](opacities[primitive_idx, 0])
            opacity = sigmoid(raw_opacity)
            comptime if ORIGINAL_OPACITY_INTERPRETATION:
                if opacity < MIN_ALPHA_THRESHOLD:
                    contributes = False

        if contributes:
            # Reconstruct the full 3D covariance from log-scales and the
            # unnormalized quaternion, then project it into screen space
            # exactly like the FasterGS core reference.
            raw_scale_x = rebind[Float32](log_scales[primitive_idx, 0])
            raw_scale_y = rebind[Float32](log_scales[primitive_idx, 1])
            raw_scale_z = rebind[Float32](log_scales[primitive_idx, 2])
            variance_x = exp(Float32(2.0) * raw_scale_x)
            variance_y = exp(Float32(2.0) * raw_scale_y)
            variance_z = exp(Float32(2.0) * raw_scale_z)

            quat_r = rebind[Float32](unnormalized_rotations[primitive_idx, 0])
            quat_x = rebind[Float32](unnormalized_rotations[primitive_idx, 1])
            quat_y = rebind[Float32](unnormalized_rotations[primitive_idx, 2])
            quat_z = rebind[Float32](unnormalized_rotations[primitive_idx, 3])
            quat_norm_sq = quat_r * quat_r + quat_x * quat_x + quat_y * quat_y + quat_z * quat_z
            if quat_norm_sq < Float32(1.0e-8):
                contributes = False

            if contributes:
                quat_norm_sq_rcp = Float32(1.0) / quat_norm_sq
                xx = quat_x * quat_x
                yy = quat_y * quat_y
                zz = quat_z * quat_z
                xy = quat_x * quat_y
                xz = quat_x * quat_z
                yz = quat_y * quat_z
                rx = quat_r * quat_x
                ry = quat_r * quat_y
                rz = quat_r * quat_z
                r00 = Float32(1.0) - Float32(2.0) * (yy + zz) * quat_norm_sq_rcp
                r01 = Float32(2.0) * (xy - rz) * quat_norm_sq_rcp
                r02 = Float32(2.0) * (xz + ry) * quat_norm_sq_rcp
                r10 = Float32(2.0) * (xy + rz) * quat_norm_sq_rcp
                r11 = Float32(1.0) - Float32(2.0) * (xx + zz) * quat_norm_sq_rcp
                r12 = Float32(2.0) * (yz - rx) * quat_norm_sq_rcp
                r20 = Float32(2.0) * (xz - ry) * quat_norm_sq_rcp
                r21 = Float32(2.0) * (yz + rx) * quat_norm_sq_rcp
                r22 = Float32(1.0) - Float32(2.0) * (xx + yy) * quat_norm_sq_rcp

                rss00 = r00 * variance_x
                rss01 = r01 * variance_y
                rss02 = r02 * variance_z
                rss10 = r10 * variance_x
                rss11 = r11 * variance_y
                rss12 = r12 * variance_z
                rss20 = r20 * variance_x
                rss21 = r21 * variance_y
                rss22 = r22 * variance_z

                cov3d_00 = rss00 * r00 + rss01 * r01 + rss02 * r02
                cov3d_01 = rss00 * r10 + rss01 * r11 + rss02 * r12
                cov3d_02 = rss00 * r20 + rss01 * r21 + rss02 * r22
                cov3d_11 = rss10 * r10 + rss11 * r11 + rss12 * r12
                cov3d_12 = rss10 * r20 + rss11 * r21 + rss12 * r22
                cov3d_22 = rss20 * r20 + rss21 * r21 + rss22 * r22

                w2c_r1_x = rebind[Float32](world_2_camera[0, 0])
                w2c_r1_y = rebind[Float32](world_2_camera[0, 1])
                w2c_r1_z = rebind[Float32](world_2_camera[0, 2])
                w2c_r1_w = rebind[Float32](world_2_camera[0, 3])
                w2c_r2_x = rebind[Float32](world_2_camera[1, 0])
                w2c_r2_y = rebind[Float32](world_2_camera[1, 1])
                w2c_r2_z = rebind[Float32](world_2_camera[1, 2])
                w2c_r2_w = rebind[Float32](world_2_camera[1, 3])

                x = (
                    w2c_r1_x * mean3d_x
                    + w2c_r1_y * mean3d_y
                    + w2c_r1_z * mean3d_z
                    + w2c_r1_w
                ) / depth
                y = (
                    w2c_r2_x * mean3d_x
                    + w2c_r2_y * mean3d_y
                    + w2c_r2_z * mean3d_z
                    + w2c_r2_w
                ) / depth

                clip_left = (Float32(-0.15) * width - center_x) / focal_x
                clip_right = (Float32(1.15) * width - center_x) / focal_x
                clip_top = (Float32(-0.15) * height - center_y) / focal_y
                clip_bottom = (Float32(1.15) * height - center_y) / focal_y
                x_clipped = min(max(x, clip_left), clip_right)
                y_clipped = min(max(y, clip_top), clip_bottom)

                # Project the covariance with the same Jacobian construction as
                # the CUDA reference so the screen-space conic matches exactly.
                j11 = focal_x / depth
                j13 = -j11 * x_clipped
                j22 = focal_y / depth
                j23 = -j22 * y_clipped

                jw_r1_x = j11 * w2c_r1_x + j13 * w2c_r3_x
                jw_r1_y = j11 * w2c_r1_y + j13 * w2c_r3_y
                jw_r1_z = j11 * w2c_r1_z + j13 * w2c_r3_z
                jw_r2_x = j22 * w2c_r2_x + j23 * w2c_r3_x
                jw_r2_y = j22 * w2c_r2_y + j23 * w2c_r3_y
                jw_r2_z = j22 * w2c_r2_z + j23 * w2c_r3_z

                jwc_r1_x = jw_r1_x * cov3d_00 + jw_r1_y * cov3d_01 + jw_r1_z * cov3d_02
                jwc_r1_y = jw_r1_x * cov3d_01 + jw_r1_y * cov3d_11 + jw_r1_z * cov3d_12
                jwc_r1_z = jw_r1_x * cov3d_02 + jw_r1_y * cov3d_12 + jw_r1_z * cov3d_22
                jwc_r2_x = jw_r2_x * cov3d_00 + jw_r2_y * cov3d_01 + jw_r2_z * cov3d_02
                jwc_r2_y = jw_r2_x * cov3d_01 + jw_r2_y * cov3d_11 + jw_r2_z * cov3d_12
                jwc_r2_z = jw_r2_x * cov3d_02 + jw_r2_y * cov3d_12 + jw_r2_z * cov3d_22

                cov2d_x = jwc_r1_x * jw_r1_x + jwc_r1_y * jw_r1_y + jwc_r1_z * jw_r1_z
                cov2d_y = jwc_r1_x * jw_r2_x + jwc_r1_y * jw_r2_y + jwc_r1_z * jw_r2_z
                cov2d_z = jwc_r2_x * jw_r2_x + jwc_r2_y * jw_r2_y + jwc_r2_z * jw_r2_z
                determinant_raw = cov2d_x * cov2d_z - cov2d_y * cov2d_y
                var kernel_size = DILATION
                comptime if proper_antialiasing:
                    kernel_size = DILATION_PROPER_ANTIALIASING
                cov2d_x += kernel_size
                cov2d_z += kernel_size
                determinant = cov2d_x * cov2d_z - cov2d_y * cov2d_y
                if determinant < MIN_COV2D_DETERMINANT:
                    contributes = False

                if contributes:
                    conic_x = cov2d_z / determinant
                    conic_y = -cov2d_y / determinant
                    conic_z = cov2d_x / determinant
                    comptime if proper_antialiasing:
                        opacity *= sqrt(max(determinant_raw / determinant, Float32(0.0)))
                        comptime if ORIGINAL_OPACITY_INTERPRETATION:
                            if opacity < MIN_ALPHA_THRESHOLD:
                                contributes = False

                if contributes:
                    mean2d_x = x * focal_x + center_x
                    mean2d_y = y * focal_y + center_y
                    # Compute the conservative tile-space bounding box first,
                    # then refine the true touched-tile count below.
                    power_threshold = MAX_POWER_THRESHOLD
                    comptime if ORIGINAL_OPACITY_INTERPRETATION:
                        power_threshold = log(opacity * MIN_ALPHA_THRESHOLD_RCP)
                    cutoff_factor = Float32(2.0) * power_threshold
                    extent_x = max(
                        sqrt(cov2d_x * cutoff_factor) - Float32(0.5),
                        Float32(0.0),
                    )
                    extent_y = max(
                        sqrt(cov2d_z * cutoff_factor) - Float32(0.5),
                        Float32(0.0),
                    )

                    screen_min_x = min(
                        grid_width,
                        max(0, Int((mean2d_x - extent_x) // Float32(TILE_WIDTH))),
                    )
                    screen_max_x = min(
                        grid_width,
                        max(
                            0,
                            Int(
                                ((mean2d_x + extent_x) + Float32(TILE_WIDTH - 1))
                                // Float32(TILE_WIDTH)
                            ),
                        ),
                    )
                    screen_min_y = min(
                        grid_height,
                        max(0, Int((mean2d_y - extent_y) // Float32(TILE_HEIGHT))),
                    )
                    screen_max_y = min(
                        grid_height,
                        max(
                            0,
                            Int(
                                ((mean2d_y + extent_y) + Float32(TILE_HEIGHT - 1))
                                // Float32(TILE_HEIGHT)
                            ),
                        ),
                    )
                    n_touched_tiles_max = (
                        screen_max_x - screen_min_x
                    ) * (screen_max_y - screen_min_y)
                    if n_touched_tiles_max == 0:
                        contributes = False

                if contributes:
                    local_touched_tiles = compute_exact_n_touched_tiles(
                        mean2d_x,
                        mean2d_y,
                        conic_x,
                        conic_y,
                        conic_z,
                        screen_min_x,
                        screen_max_x,
                        screen_min_y,
                        screen_max_y,
                        power_threshold,
                    )
                    if local_touched_tiles == 0:
                        contributes = False

        if contributes:
            # Write the forward state in the same logical contract consumed later
            # by sort, blend, and the existing FasterGS backward path.
            projected_means[primitive_idx, 0] = mean2d_x
            projected_means[primitive_idx, 1] = mean2d_y
            conic_opacity[primitive_idx, 0] = conic_x
            conic_opacity[primitive_idx, 1] = conic_y
            conic_opacity[primitive_idx, 2] = conic_z
            conic_opacity[primitive_idx, 3] = opacity
            num_touched_tiles[primitive_idx] = local_touched_tiles
            screen_bounds[primitive_idx, 0] = UInt16(screen_min_x)
            screen_bounds[primitive_idx, 1] = UInt16(screen_max_x)
            screen_bounds[primitive_idx, 2] = UInt16(screen_min_y)
            screen_bounds[primitive_idx, 3] = UInt16(screen_max_y)

            color = convert_sh_to_color[active_sh_bases](
                sh_coefficients_0,
                sh_coefficients_rest,
                mean3d_x,
                mean3d_y,
                mean3d_z,
                rebind[Float32](camera_position[0]),
                rebind[Float32](camera_position[1]),
                rebind[Float32](camera_position[2]),
                primitive_idx,
            )
            colors_rgb[primitive_idx, 0] = color.x
            colors_rgb[primitive_idx, 1] = color.y
            colors_rgb[primitive_idx, 2] = color.z
            local_depth_key = float_bitcast_to_int32(depth)
            depth_keys[primitive_idx] = local_depth_key


# ================================================================================================ #
#                                   Launch interface                                               #
# ================================================================================================ #


struct PreprocessForwardHelper:
    @staticmethod
    def execute[
        target: StaticString,
        proper_antialiasing: Bool,
        active_sh_bases: Int,
    ](
        projected_means: OutputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: OutputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: OutputTensor[dtype=DType.float32, rank=2, ...],
        primitive_depth: OutputTensor[dtype=DType.float32, rank=1, ...],
        depth_keys: OutputTensor[dtype=DType.int32, rank=1, ...],
        primitive_indices: OutputTensor[dtype=DType.int32, rank=1, ...],
        num_touched_tiles: OutputTensor[dtype=DType.int32, rank=1, ...],
        screen_bounds: OutputTensor[dtype=DType.uint16, rank=2, ...],
        visible_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        instance_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        center_positions: InputTensor[dtype=DType.float32, rank=2, ...],
        log_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        unnormalized_rotations: InputTensor[dtype=DType.float32, rank=2, ...],
        opacities: InputTensor[dtype=DType.float32, rank=2, ...],
        sh_coefficients_0: InputTensor[dtype=DType.float32, rank=3, ...],
        sh_coefficients_rest: InputTensor[dtype=DType.float32, rank=3, ...],
        world_2_camera: InputTensor[dtype=DType.float32, rank=2, ...],
        camera_position: InputTensor[dtype=DType.float32, rank=1, ...],
        near_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        far_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        width: InputTensor[dtype=DType.int32, rank=1, ...],
        height: InputTensor[dtype=DType.int32, rank=1, ...],
        focal_x: InputTensor[dtype=DType.float32, rank=1, ...],
        focal_y: InputTensor[dtype=DType.float32, rank=1, ...],
        center_x: InputTensor[dtype=DType.float32, rank=1, ...],
        center_y: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        """Run the preprocess forward MAX custom op on GPU."""
        comptime if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            n_primitives = Int(center_positions.dim_size[0]())

            # Build explicit runtime row-major tensor views at the custom-op
            # boundary. The GPU kernels only receive typed tensor views; raw
            # pointers do not leak into the kernel signatures.
            center_positions_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                center_positions.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(center_positions.dim_size[0]()), Int(center_positions.dim_size[1]()))),
            )
            log_scales_tensor = LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin](
                log_scales.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(log_scales.dim_size[0]()), Int(log_scales.dim_size[1]()))),
            )
            unnormalized_rotations_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                unnormalized_rotations.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(
                    Index(
                        Int(unnormalized_rotations.dim_size[0]()),
                        Int(unnormalized_rotations.dim_size[1]()),
                    )
                ),
            )
            opacities_tensor = LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin](
                opacities.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(opacities.dim_size[0]()), Int(opacities.dim_size[1]()))),
            )
            sh_coefficients_0_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_3D,
                MutAnyOrigin,
            ](
                sh_coefficients_0.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_3D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(
                    Index(
                        Int(sh_coefficients_0.dim_size[0]()),
                        Int(sh_coefficients_0.dim_size[1]()),
                        Int(sh_coefficients_0.dim_size[2]()),
                    )
                ),
            )
            sh_coefficients_rest_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_3D,
                MutAnyOrigin,
            ](
                sh_coefficients_rest.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_3D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(
                    Index(
                        Int(sh_coefficients_rest.dim_size[0]()),
                        Int(sh_coefficients_rest.dim_size[1]()),
                        Int(sh_coefficients_rest.dim_size[2]()),
                    )
                ),
            )
            world_2_camera_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                world_2_camera.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(world_2_camera.dim_size[0]()), Int(world_2_camera.dim_size[1]()))),
            )
            camera_position_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_1D,
                MutAnyOrigin,
            ](
                camera_position.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(camera_position.dim_size[0]()))),
            )
            projected_means_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                projected_means.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(projected_means.dim_size[0]()), Int(projected_means.dim_size[1]()))),
            )
            conic_opacity_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                conic_opacity.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(conic_opacity.dim_size[0]()), Int(conic_opacity.dim_size[1]()))),
            )
            colors_rgb_tensor = LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin](
                colors_rgb.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(colors_rgb.dim_size[0]()), Int(colors_rgb.dim_size[1]()))),
            )
            primitive_depth_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_1D,
                MutAnyOrigin,
            ](
                primitive_depth.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(primitive_depth.dim_size[0]()))),
            )
            depth_keys_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                depth_keys.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(depth_keys.dim_size[0]()))),
            )
            primitive_indices_tensor = LayoutTensor[
                DType.int32,
                ROW_MAJOR_1D,
                MutAnyOrigin,
            ](
                primitive_indices.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(primitive_indices.dim_size[0]()))),
            )
            num_touched_tiles_tensor = LayoutTensor[
                DType.int32,
                ROW_MAJOR_1D,
                MutAnyOrigin,
            ](
                num_touched_tiles.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(num_touched_tiles.dim_size[0]()))),
            )
            screen_bounds_tensor = LayoutTensor[
                DType.uint16,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                screen_bounds.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(screen_bounds.dim_size[0]()), Int(screen_bounds.dim_size[1]()))),
            )
            scalar_size = Int(1)
            visible_count_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                visible_count.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(scalar_size)),
            )
            instance_count_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                instance_count.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(scalar_size)),
            )
            near_plane_tensor = LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin](
                near_plane.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(scalar_size)),
            )
            far_plane_tensor = LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin](
                far_plane.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(scalar_size)),
            )
            width_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                width.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(scalar_size)),
            )
            height_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                height.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(scalar_size)),
            )
            focal_x_tensor = LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin](
                focal_x.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(scalar_size)),
            )
            focal_y_tensor = LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin](
                focal_y.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(scalar_size)),
            )
            center_x_tensor = LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin](
                center_x.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(scalar_size)),
            )
            center_y_tensor = LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin](
                center_y.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(scalar_size)),
            )

            comptime visible_count_layout = type_of(visible_count_tensor).layout
            comptime instance_count_layout = type_of(instance_count_tensor).layout
            comptime center_positions_layout = type_of(center_positions_tensor).layout
            comptime log_scales_layout = type_of(log_scales_tensor).layout
            comptime unnormalized_rotations_layout = type_of(unnormalized_rotations_tensor).layout
            comptime opacities_layout = type_of(opacities_tensor).layout
            comptime sh_coefficients_0_layout = type_of(sh_coefficients_0_tensor).layout
            comptime sh_coefficients_rest_layout = type_of(sh_coefficients_rest_tensor).layout
            comptime world_2_camera_layout = type_of(world_2_camera_tensor).layout
            comptime camera_position_layout = type_of(camera_position_tensor).layout
            comptime projected_means_layout = type_of(projected_means_tensor).layout
            comptime conic_opacity_layout = type_of(conic_opacity_tensor).layout
            comptime colors_rgb_layout = type_of(colors_rgb_tensor).layout
            comptime primitive_depth_layout = type_of(primitive_depth_tensor).layout
            comptime depth_keys_layout = type_of(depth_keys_tensor).layout
            comptime primitive_indices_layout = type_of(primitive_indices_tensor).layout
            comptime num_touched_tiles_layout = type_of(num_touched_tiles_tensor).layout
            comptime screen_bounds_layout = type_of(screen_bounds_tensor).layout

            gpu_ctx.enqueue_function[
                zero_preprocess_counters_kernel[
                    visible_count_layout,
                    instance_count_layout,
                ],
                zero_preprocess_counters_kernel[
                    visible_count_layout,
                    instance_count_layout,
                ],
            ](
                visible_count_tensor,
                instance_count_tensor,
                grid_dim=1,
                block_dim=1,
            )

            gpu_ctx.enqueue_function[
                preprocess_fwd_kernel[
                    proper_antialiasing,
                    active_sh_bases,
                    center_positions_layout,
                    log_scales_layout,
                    unnormalized_rotations_layout,
                    opacities_layout,
                    sh_coefficients_0_layout,
                    sh_coefficients_rest_layout,
                    world_2_camera_layout,
                    camera_position_layout,
                    projected_means_layout,
                    conic_opacity_layout,
                    colors_rgb_layout,
                    primitive_depth_layout,
                    depth_keys_layout,
                    primitive_indices_layout,
                    num_touched_tiles_layout,
                    screen_bounds_layout,
                ],
                preprocess_fwd_kernel[
                    proper_antialiasing,
                    active_sh_bases,
                    center_positions_layout,
                    log_scales_layout,
                    unnormalized_rotations_layout,
                    opacities_layout,
                    sh_coefficients_0_layout,
                    sh_coefficients_rest_layout,
                    world_2_camera_layout,
                    camera_position_layout,
                    projected_means_layout,
                    conic_opacity_layout,
                    colors_rgb_layout,
                    primitive_depth_layout,
                    depth_keys_layout,
                    primitive_indices_layout,
                    num_touched_tiles_layout,
                    screen_bounds_layout,
                ],
            ](
                # The preprocess kernel owns the full front half of the render
                # graph: projection, culling, SH color evaluation, and dense
                # per-primitive state writes. Visibility recovery happens in a
                # separate reduction pass below so the kernel itself stays
                # pointer-free and easy for MAX to optimize.
                center_positions_tensor,
                log_scales_tensor,
                unnormalized_rotations_tensor,
                opacities_tensor,
                sh_coefficients_0_tensor,
                sh_coefficients_rest_tensor,
                world_2_camera_tensor,
                camera_position_tensor,
                projected_means_tensor,
                conic_opacity_tensor,
                colors_rgb_tensor,
                primitive_depth_tensor,
                depth_keys_tensor,
                primitive_indices_tensor,
                num_touched_tiles_tensor,
                screen_bounds_tensor,
                near_plane_tensor,
                far_plane_tensor,
                width_tensor,
                height_tensor,
                focal_x_tensor,
                focal_y_tensor,
                center_x_tensor,
                center_y_tensor,
                n_primitives,
                grid_dim=div_round_up(n_primitives, 128),
                block_dim=128,
            )
            if n_primitives > 0:
                # Recover scalar counts with a block reduction. This keeps the
                # main projection kernel free of global atomics while avoiding a
                # host readback for visibility or instance counts.
                count_block_count = div_round_up(n_primitives, COUNT_BLOCK_SIZE)
                var block_visible_counts_buffer = gpu_ctx.enqueue_create_buffer[DType.int32](count_block_count)
                var block_instance_counts_buffer = gpu_ctx.enqueue_create_buffer[DType.int32](count_block_count)

                var block_visible_counts_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                    block_visible_counts_buffer.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(count_block_count)),
                )
                var block_instance_counts_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                    block_instance_counts_buffer.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(count_block_count)),
                )

                comptime block_visible_counts_layout = type_of(block_visible_counts_tensor).layout
                comptime block_instance_counts_layout = type_of(block_instance_counts_tensor).layout

                gpu_ctx.enqueue_function[
                    reduce_preprocess_counts_blocks_kernel[
                        num_touched_tiles_layout,
                        block_visible_counts_layout,
                        block_instance_counts_layout,
                    ],
                    reduce_preprocess_counts_blocks_kernel[
                        num_touched_tiles_layout,
                        block_visible_counts_layout,
                        block_instance_counts_layout,
                    ],
                ](
                    num_touched_tiles_tensor,
                    block_visible_counts_tensor,
                    block_instance_counts_tensor,
                    n_primitives,
                    grid_dim=count_block_count,
                    block_dim=COUNT_BLOCK_SIZE,
                )

                gpu_ctx.enqueue_function[
                    finalize_preprocess_counts_kernel[
                        block_visible_counts_layout,
                        block_instance_counts_layout,
                        visible_count_layout,
                        instance_count_layout,
                    ],
                    finalize_preprocess_counts_kernel[
                        block_visible_counts_layout,
                        block_instance_counts_layout,
                        visible_count_layout,
                        instance_count_layout,
                    ],
                ](
                    block_visible_counts_tensor,
                    block_instance_counts_tensor,
                    visible_count_tensor,
                    instance_count_tensor,
                    count_block_count,
                    grid_dim=1,
                    block_dim=1,
                )
        else:
            raise Error("faster_gs_mojo preprocess_fwd currently requires a GPU target")


@compiler.register("preprocess_fwd_pa0_sh1")
struct PreprocessForwardPA0SH1:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        projected_means: OutputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: OutputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: OutputTensor[dtype=DType.float32, rank=2, ...],
        primitive_depth: OutputTensor[dtype=DType.float32, rank=1, ...],
        depth_keys: OutputTensor[dtype=DType.int32, rank=1, ...],
        primitive_indices: OutputTensor[dtype=DType.int32, rank=1, ...],
        num_touched_tiles: OutputTensor[dtype=DType.int32, rank=1, ...],
        screen_bounds: OutputTensor[dtype=DType.uint16, rank=2, ...],
        visible_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        instance_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        center_positions: InputTensor[dtype=DType.float32, rank=2, ...],
        log_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        unnormalized_rotations: InputTensor[dtype=DType.float32, rank=2, ...],
        opacities: InputTensor[dtype=DType.float32, rank=2, ...],
        sh_coefficients_0: InputTensor[dtype=DType.float32, rank=3, ...],
        sh_coefficients_rest: InputTensor[dtype=DType.float32, rank=3, ...],
        world_2_camera: InputTensor[dtype=DType.float32, rank=2, ...],
        camera_position: InputTensor[dtype=DType.float32, rank=1, ...],
        near_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        far_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        width: InputTensor[dtype=DType.int32, rank=1, ...],
        height: InputTensor[dtype=DType.int32, rank=1, ...],
        focal_x: InputTensor[dtype=DType.float32, rank=1, ...],
        focal_y: InputTensor[dtype=DType.float32, rank=1, ...],
        center_x: InputTensor[dtype=DType.float32, rank=1, ...],
        center_y: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        return PreprocessForwardHelper.execute[target, False, 1](
            projected_means,
            conic_opacity,
            colors_rgb,
            primitive_depth,
            depth_keys,
            primitive_indices,
            num_touched_tiles,
            screen_bounds,
            visible_count,
            instance_count,
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane,
            far_plane,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            ctx,
        )


@compiler.register("preprocess_fwd_pa0_sh4")
struct PreprocessForwardPA0SH4:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        projected_means: OutputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: OutputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: OutputTensor[dtype=DType.float32, rank=2, ...],
        primitive_depth: OutputTensor[dtype=DType.float32, rank=1, ...],
        depth_keys: OutputTensor[dtype=DType.int32, rank=1, ...],
        primitive_indices: OutputTensor[dtype=DType.int32, rank=1, ...],
        num_touched_tiles: OutputTensor[dtype=DType.int32, rank=1, ...],
        screen_bounds: OutputTensor[dtype=DType.uint16, rank=2, ...],
        visible_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        instance_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        center_positions: InputTensor[dtype=DType.float32, rank=2, ...],
        log_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        unnormalized_rotations: InputTensor[dtype=DType.float32, rank=2, ...],
        opacities: InputTensor[dtype=DType.float32, rank=2, ...],
        sh_coefficients_0: InputTensor[dtype=DType.float32, rank=3, ...],
        sh_coefficients_rest: InputTensor[dtype=DType.float32, rank=3, ...],
        world_2_camera: InputTensor[dtype=DType.float32, rank=2, ...],
        camera_position: InputTensor[dtype=DType.float32, rank=1, ...],
        near_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        far_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        width: InputTensor[dtype=DType.int32, rank=1, ...],
        height: InputTensor[dtype=DType.int32, rank=1, ...],
        focal_x: InputTensor[dtype=DType.float32, rank=1, ...],
        focal_y: InputTensor[dtype=DType.float32, rank=1, ...],
        center_x: InputTensor[dtype=DType.float32, rank=1, ...],
        center_y: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        return PreprocessForwardHelper.execute[target, False, 4](
            projected_means,
            conic_opacity,
            colors_rgb,
            primitive_depth,
            depth_keys,
            primitive_indices,
            num_touched_tiles,
            screen_bounds,
            visible_count,
            instance_count,
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane,
            far_plane,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            ctx,
        )


@compiler.register("preprocess_fwd_pa0_sh9")
struct PreprocessForwardPA0SH9:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        projected_means: OutputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: OutputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: OutputTensor[dtype=DType.float32, rank=2, ...],
        primitive_depth: OutputTensor[dtype=DType.float32, rank=1, ...],
        depth_keys: OutputTensor[dtype=DType.int32, rank=1, ...],
        primitive_indices: OutputTensor[dtype=DType.int32, rank=1, ...],
        num_touched_tiles: OutputTensor[dtype=DType.int32, rank=1, ...],
        screen_bounds: OutputTensor[dtype=DType.uint16, rank=2, ...],
        visible_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        instance_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        center_positions: InputTensor[dtype=DType.float32, rank=2, ...],
        log_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        unnormalized_rotations: InputTensor[dtype=DType.float32, rank=2, ...],
        opacities: InputTensor[dtype=DType.float32, rank=2, ...],
        sh_coefficients_0: InputTensor[dtype=DType.float32, rank=3, ...],
        sh_coefficients_rest: InputTensor[dtype=DType.float32, rank=3, ...],
        world_2_camera: InputTensor[dtype=DType.float32, rank=2, ...],
        camera_position: InputTensor[dtype=DType.float32, rank=1, ...],
        near_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        far_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        width: InputTensor[dtype=DType.int32, rank=1, ...],
        height: InputTensor[dtype=DType.int32, rank=1, ...],
        focal_x: InputTensor[dtype=DType.float32, rank=1, ...],
        focal_y: InputTensor[dtype=DType.float32, rank=1, ...],
        center_x: InputTensor[dtype=DType.float32, rank=1, ...],
        center_y: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        return PreprocessForwardHelper.execute[target, False, 9](
            projected_means,
            conic_opacity,
            colors_rgb,
            primitive_depth,
            depth_keys,
            primitive_indices,
            num_touched_tiles,
            screen_bounds,
            visible_count,
            instance_count,
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane,
            far_plane,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            ctx,
        )


@compiler.register("preprocess_fwd_pa0_sh16")
struct PreprocessForwardPA0SH16:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        projected_means: OutputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: OutputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: OutputTensor[dtype=DType.float32, rank=2, ...],
        primitive_depth: OutputTensor[dtype=DType.float32, rank=1, ...],
        depth_keys: OutputTensor[dtype=DType.int32, rank=1, ...],
        primitive_indices: OutputTensor[dtype=DType.int32, rank=1, ...],
        num_touched_tiles: OutputTensor[dtype=DType.int32, rank=1, ...],
        screen_bounds: OutputTensor[dtype=DType.uint16, rank=2, ...],
        visible_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        instance_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        center_positions: InputTensor[dtype=DType.float32, rank=2, ...],
        log_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        unnormalized_rotations: InputTensor[dtype=DType.float32, rank=2, ...],
        opacities: InputTensor[dtype=DType.float32, rank=2, ...],
        sh_coefficients_0: InputTensor[dtype=DType.float32, rank=3, ...],
        sh_coefficients_rest: InputTensor[dtype=DType.float32, rank=3, ...],
        world_2_camera: InputTensor[dtype=DType.float32, rank=2, ...],
        camera_position: InputTensor[dtype=DType.float32, rank=1, ...],
        near_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        far_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        width: InputTensor[dtype=DType.int32, rank=1, ...],
        height: InputTensor[dtype=DType.int32, rank=1, ...],
        focal_x: InputTensor[dtype=DType.float32, rank=1, ...],
        focal_y: InputTensor[dtype=DType.float32, rank=1, ...],
        center_x: InputTensor[dtype=DType.float32, rank=1, ...],
        center_y: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        return PreprocessForwardHelper.execute[target, False, 16](
            projected_means,
            conic_opacity,
            colors_rgb,
            primitive_depth,
            depth_keys,
            primitive_indices,
            num_touched_tiles,
            screen_bounds,
            visible_count,
            instance_count,
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane,
            far_plane,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            ctx,
        )


@compiler.register("preprocess_fwd_pa1_sh1")
struct PreprocessForwardPA1SH1:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        projected_means: OutputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: OutputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: OutputTensor[dtype=DType.float32, rank=2, ...],
        primitive_depth: OutputTensor[dtype=DType.float32, rank=1, ...],
        depth_keys: OutputTensor[dtype=DType.int32, rank=1, ...],
        primitive_indices: OutputTensor[dtype=DType.int32, rank=1, ...],
        num_touched_tiles: OutputTensor[dtype=DType.int32, rank=1, ...],
        screen_bounds: OutputTensor[dtype=DType.uint16, rank=2, ...],
        visible_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        instance_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        center_positions: InputTensor[dtype=DType.float32, rank=2, ...],
        log_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        unnormalized_rotations: InputTensor[dtype=DType.float32, rank=2, ...],
        opacities: InputTensor[dtype=DType.float32, rank=2, ...],
        sh_coefficients_0: InputTensor[dtype=DType.float32, rank=3, ...],
        sh_coefficients_rest: InputTensor[dtype=DType.float32, rank=3, ...],
        world_2_camera: InputTensor[dtype=DType.float32, rank=2, ...],
        camera_position: InputTensor[dtype=DType.float32, rank=1, ...],
        near_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        far_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        width: InputTensor[dtype=DType.int32, rank=1, ...],
        height: InputTensor[dtype=DType.int32, rank=1, ...],
        focal_x: InputTensor[dtype=DType.float32, rank=1, ...],
        focal_y: InputTensor[dtype=DType.float32, rank=1, ...],
        center_x: InputTensor[dtype=DType.float32, rank=1, ...],
        center_y: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        return PreprocessForwardHelper.execute[target, True, 1](
            projected_means,
            conic_opacity,
            colors_rgb,
            primitive_depth,
            depth_keys,
            primitive_indices,
            num_touched_tiles,
            screen_bounds,
            visible_count,
            instance_count,
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane,
            far_plane,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            ctx,
        )


@compiler.register("preprocess_fwd_pa1_sh4")
struct PreprocessForwardPA1SH4:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        projected_means: OutputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: OutputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: OutputTensor[dtype=DType.float32, rank=2, ...],
        primitive_depth: OutputTensor[dtype=DType.float32, rank=1, ...],
        depth_keys: OutputTensor[dtype=DType.int32, rank=1, ...],
        primitive_indices: OutputTensor[dtype=DType.int32, rank=1, ...],
        num_touched_tiles: OutputTensor[dtype=DType.int32, rank=1, ...],
        screen_bounds: OutputTensor[dtype=DType.uint16, rank=2, ...],
        visible_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        instance_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        center_positions: InputTensor[dtype=DType.float32, rank=2, ...],
        log_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        unnormalized_rotations: InputTensor[dtype=DType.float32, rank=2, ...],
        opacities: InputTensor[dtype=DType.float32, rank=2, ...],
        sh_coefficients_0: InputTensor[dtype=DType.float32, rank=3, ...],
        sh_coefficients_rest: InputTensor[dtype=DType.float32, rank=3, ...],
        world_2_camera: InputTensor[dtype=DType.float32, rank=2, ...],
        camera_position: InputTensor[dtype=DType.float32, rank=1, ...],
        near_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        far_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        width: InputTensor[dtype=DType.int32, rank=1, ...],
        height: InputTensor[dtype=DType.int32, rank=1, ...],
        focal_x: InputTensor[dtype=DType.float32, rank=1, ...],
        focal_y: InputTensor[dtype=DType.float32, rank=1, ...],
        center_x: InputTensor[dtype=DType.float32, rank=1, ...],
        center_y: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        return PreprocessForwardHelper.execute[target, True, 4](
            projected_means,
            conic_opacity,
            colors_rgb,
            primitive_depth,
            depth_keys,
            primitive_indices,
            num_touched_tiles,
            screen_bounds,
            visible_count,
            instance_count,
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane,
            far_plane,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            ctx,
        )


@compiler.register("preprocess_fwd_pa1_sh9")
struct PreprocessForwardPA1SH9:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        projected_means: OutputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: OutputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: OutputTensor[dtype=DType.float32, rank=2, ...],
        primitive_depth: OutputTensor[dtype=DType.float32, rank=1, ...],
        depth_keys: OutputTensor[dtype=DType.int32, rank=1, ...],
        primitive_indices: OutputTensor[dtype=DType.int32, rank=1, ...],
        num_touched_tiles: OutputTensor[dtype=DType.int32, rank=1, ...],
        screen_bounds: OutputTensor[dtype=DType.uint16, rank=2, ...],
        visible_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        instance_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        center_positions: InputTensor[dtype=DType.float32, rank=2, ...],
        log_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        unnormalized_rotations: InputTensor[dtype=DType.float32, rank=2, ...],
        opacities: InputTensor[dtype=DType.float32, rank=2, ...],
        sh_coefficients_0: InputTensor[dtype=DType.float32, rank=3, ...],
        sh_coefficients_rest: InputTensor[dtype=DType.float32, rank=3, ...],
        world_2_camera: InputTensor[dtype=DType.float32, rank=2, ...],
        camera_position: InputTensor[dtype=DType.float32, rank=1, ...],
        near_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        far_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        width: InputTensor[dtype=DType.int32, rank=1, ...],
        height: InputTensor[dtype=DType.int32, rank=1, ...],
        focal_x: InputTensor[dtype=DType.float32, rank=1, ...],
        focal_y: InputTensor[dtype=DType.float32, rank=1, ...],
        center_x: InputTensor[dtype=DType.float32, rank=1, ...],
        center_y: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        return PreprocessForwardHelper.execute[target, True, 9](
            projected_means,
            conic_opacity,
            colors_rgb,
            primitive_depth,
            depth_keys,
            primitive_indices,
            num_touched_tiles,
            screen_bounds,
            visible_count,
            instance_count,
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane,
            far_plane,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            ctx,
        )


@compiler.register("preprocess_fwd_pa1_sh16")
struct PreprocessForwardPA1SH16:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        projected_means: OutputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: OutputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: OutputTensor[dtype=DType.float32, rank=2, ...],
        primitive_depth: OutputTensor[dtype=DType.float32, rank=1, ...],
        depth_keys: OutputTensor[dtype=DType.int32, rank=1, ...],
        primitive_indices: OutputTensor[dtype=DType.int32, rank=1, ...],
        num_touched_tiles: OutputTensor[dtype=DType.int32, rank=1, ...],
        screen_bounds: OutputTensor[dtype=DType.uint16, rank=2, ...],
        visible_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        instance_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        center_positions: InputTensor[dtype=DType.float32, rank=2, ...],
        log_scales: InputTensor[dtype=DType.float32, rank=2, ...],
        unnormalized_rotations: InputTensor[dtype=DType.float32, rank=2, ...],
        opacities: InputTensor[dtype=DType.float32, rank=2, ...],
        sh_coefficients_0: InputTensor[dtype=DType.float32, rank=3, ...],
        sh_coefficients_rest: InputTensor[dtype=DType.float32, rank=3, ...],
        world_2_camera: InputTensor[dtype=DType.float32, rank=2, ...],
        camera_position: InputTensor[dtype=DType.float32, rank=1, ...],
        near_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        far_plane: InputTensor[dtype=DType.float32, rank=1, ...],
        width: InputTensor[dtype=DType.int32, rank=1, ...],
        height: InputTensor[dtype=DType.int32, rank=1, ...],
        focal_x: InputTensor[dtype=DType.float32, rank=1, ...],
        focal_y: InputTensor[dtype=DType.float32, rank=1, ...],
        center_x: InputTensor[dtype=DType.float32, rank=1, ...],
        center_y: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        return PreprocessForwardHelper.execute[target, True, 16](
            projected_means,
            conic_opacity,
            colors_rgb,
            primitive_depth,
            depth_keys,
            primitive_indices,
            num_touched_tiles,
            screen_bounds,
            visible_count,
            instance_count,
            center_positions,
            log_scales,
            unnormalized_rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            world_2_camera,
            camera_position,
            near_plane,
            far_plane,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            ctx,
        )
