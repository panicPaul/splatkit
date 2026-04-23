import compiler

from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from tensor import InputTensor, OutputTensor
from std.atomic import Atomic
from std.gpu import WARP_SIZE, block_idx, global_idx, thread_idx
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import shuffle_up
from std.gpu.sync import barrier
from std.math import exp
from std.runtime.asyncrt import DeviceContextPtr
from std.utils import Index

from .common import Float2, Float3, div_round_up


comptime TILE_WIDTH = 16
comptime TILE_HEIGHT = 16
comptime BLOCK_SIZE_BLEND = TILE_WIDTH * TILE_HEIGHT
comptime BUCKET_SIZE = 32
comptime MIN_ALPHA_THRESHOLD = Float32(1.0) / Float32(255.0)
comptime ONE_MINUS_ALPHA_EPS = Float32(1.0e-6)

comptime ROW_MAJOR_1D = Layout.row_major(Int())
comptime ROW_MAJOR_2D = Layout.row_major(Int(), Int())
comptime ROW_MAJOR_3D = Layout.row_major(Int(), Int(), Int())

comptime SHARED_BUCKET_I32 = Layout.row_major(BUCKET_SIZE)
comptime SHARED_COLOR_AFTER_TRANS = Layout.row_major(BUCKET_SIZE, 4)
comptime SHARED_GRAD_INFO = Layout.row_major(BUCKET_SIZE, 4)


@always_inline
def zero_f32_kernel(ptr: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var idx = Int(global_idx.x)
    if idx < n:
        ptr[idx] = Float32(0.0)


@always_inline
def atomic_add_f32(ptr: UnsafePointer[Float32, MutAnyOrigin], value: Float32):
    _ = Atomic[DType.float32].fetch_add(ptr, value)


@always_inline
def blend_bwd_kernel(
    grad_projected_means: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    grad_conic_opacity: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    grad_colors_rgb: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    grad_image: LayoutTensor[DType.float32, ROW_MAJOR_3D, MutAnyOrigin],
    image: LayoutTensor[DType.float32, ROW_MAJOR_3D, MutAnyOrigin],
    instance_primitive_indices: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    tile_instance_ranges: LayoutTensor[DType.int32, ROW_MAJOR_2D, MutAnyOrigin],
    tile_bucket_offsets: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    projected_means: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    conic_opacity: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    colors_rgb: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    bg_color: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    tile_final_transmittances: LayoutTensor[
        DType.float32,
        ROW_MAJOR_1D,
        MutAnyOrigin,
    ],
    tile_max_n_processed: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    tile_n_processed: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    bucket_tile_index: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    bucket_color_transmittance: LayoutTensor[
        DType.float32,
        ROW_MAJOR_2D,
        MutAnyOrigin,
    ],
    n_primitives: Int,
    width: Int,
    height: Int,
    grid_width: Int,
    n_buckets: Int,
    proper_antialiasing_flag: Int,
):
    comptime assert WARP_SIZE == 32, "FasterGS blend assumes 32-lane warps"
    var proper_antialiasing = proper_antialiasing_flag != 0
    _ = n_primitives
    _ = n_buckets

    # One warp replays one bucket against the pixels in the tile that produced
    # its saved forward snapshots.
    var bucket_idx = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var current_tile_idx = Int(rebind[Int32](bucket_tile_index[bucket_idx]))
    var tile_start = Int(rebind[Int32](tile_instance_ranges[current_tile_idx, 0]))
    var tile_end = Int(rebind[Int32](tile_instance_ranges[current_tile_idx, 1]))
    var tile_n_primitives = tile_end - tile_start
    var tile_first_bucket_offset = 0
    if current_tile_idx > 0:
        tile_first_bucket_offset = Int(
            rebind[Int32](tile_bucket_offsets[current_tile_idx - 1])
        )
    var tile_bucket_idx = bucket_idx - tile_first_bucket_offset
    if tile_bucket_idx * BUCKET_SIZE >= Int(
        rebind[Int32](tile_max_n_processed[current_tile_idx])
    ):
        return

    var tile_primitive_idx = tile_bucket_idx * BUCKET_SIZE + lane
    var instance_idx = tile_start + tile_primitive_idx
    var valid_primitive = tile_primitive_idx < tile_n_primitives

    # Each lane owns one Gaussian from the current bucket.
    var primitive_idx = 0
    var mean2d = Float2(Float32(0.0), Float32(0.0))
    var conic = Float3(Float32(0.0), Float32(0.0), Float32(0.0))
    var opacity = Float32(0.0)
    var color = Float3(Float32(0.0), Float32(0.0), Float32(0.0))
    var color_grad_factor = Float3(Float32(0.0), Float32(0.0), Float32(0.0))
    if valid_primitive:
        primitive_idx = Int(rebind[Int32](instance_primitive_indices[instance_idx]))
        mean2d = Float2(
            rebind[Float32](projected_means[primitive_idx, 0]),
            rebind[Float32](projected_means[primitive_idx, 1]),
        )
        conic = Float3(
            rebind[Float32](conic_opacity[primitive_idx, 0]),
            rebind[Float32](conic_opacity[primitive_idx, 1]),
            rebind[Float32](conic_opacity[primitive_idx, 2]),
        )
        opacity = rebind[Float32](conic_opacity[primitive_idx, 3])
        var unclamped_r = rebind[Float32](colors_rgb[primitive_idx, 0])
        var unclamped_g = rebind[Float32](colors_rgb[primitive_idx, 1])
        var unclamped_b = rebind[Float32](colors_rgb[primitive_idx, 2])
        color = Float3(
            max(unclamped_r, Float32(0.0)),
            max(unclamped_g, Float32(0.0)),
            max(unclamped_b, Float32(0.0)),
        )
        if unclamped_r >= Float32(0.0):
            color_grad_factor.x = Float32(1.0)
        if unclamped_g >= Float32(0.0):
            color_grad_factor.y = Float32(1.0)
        if unclamped_b >= Float32(0.0):
            color_grad_factor.z = Float32(1.0)

    var background = Float3(
        rebind[Float32](bg_color[0]),
        rebind[Float32](bg_color[1]),
        rebind[Float32](bg_color[2]),
    )

    var dL_dmean2d_accum = Float2(Float32(0.0), Float32(0.0))
    var dL_dconic_accum = Float3(Float32(0.0), Float32(0.0), Float32(0.0))
    var dL_dopacity_accum = Float32(0.0)
    var dL_dcolor_accum = Float3(Float32(0.0), Float32(0.0), Float32(0.0))

    var tile_coord_x = current_tile_idx % grid_width
    var tile_coord_y = current_tile_idx // grid_width
    var start_pixel_x = tile_coord_x * TILE_WIDTH
    var start_pixel_y = tile_coord_y * TILE_HEIGHT

    # Shared staging keeps the same replay surface as the CUDA kernel, but the
    # saved values stay in packed LayoutTensor form.
    var collected_last_contributor = LayoutTensor[
        DType.int32,
        SHARED_BUCKET_I32,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var collected_color_after_transmittance = LayoutTensor[
        DType.float32,
        SHARED_COLOR_AFTER_TRANS,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var collected_grad_info = LayoutTensor[
        DType.float32,
        SHARED_GRAD_INFO,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var last_contributor = Int32(0)
    var color_pixel_after = Float3(Float32(0.0), Float32(0.0), Float32(0.0))
    var transmittance = Float32(0.0)
    var grad_color_pixel = Float3(Float32(0.0), Float32(0.0), Float32(0.0))
    var grad_alpha_common = Float32(0.0)

    for i in range(BLOCK_SIZE_BLEND + BUCKET_SIZE - 1):
        if i % BUCKET_SIZE == 0:
            var local_idx = i + lane
            if local_idx < BLOCK_SIZE_BLEND:
                var local_pixel_x = start_pixel_x + local_idx % TILE_WIDTH
                var local_pixel_y = start_pixel_y + local_idx // TILE_WIDTH

                var snapshot_r = Float32(0.0)
                var snapshot_g = Float32(0.0)
                var snapshot_b = Float32(0.0)
                var snapshot_t = Float32(0.0)
                var color_pixel_w_bg = Float3(
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                )
                var grad_color_pixel_local = Float3(
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                )
                var final_transmittance = Float32(0.0)
                var local_last_contributor = Int32(0)
                if local_pixel_x < width and local_pixel_y < height:
                    var pixel_idx = width * local_pixel_y + local_pixel_x
                    color_pixel_w_bg = Float3(
                        rebind[Float32](image[0, local_pixel_y, local_pixel_x]),
                        rebind[Float32](image[1, local_pixel_y, local_pixel_x]),
                        rebind[Float32](image[2, local_pixel_y, local_pixel_x]),
                    )
                    grad_color_pixel_local = Float3(
                        rebind[Float32](grad_image[0, local_pixel_y, local_pixel_x]),
                        rebind[Float32](grad_image[1, local_pixel_y, local_pixel_x]),
                        rebind[Float32](grad_image[2, local_pixel_y, local_pixel_x]),
                    )
                    final_transmittance = rebind[Float32](
                        tile_final_transmittances[pixel_idx]
                    )
                    var bucket_row = bucket_idx * BLOCK_SIZE_BLEND + local_idx
                    snapshot_r = rebind[Float32](bucket_color_transmittance[bucket_row, 0])
                    snapshot_g = rebind[Float32](bucket_color_transmittance[bucket_row, 1])
                    snapshot_b = rebind[Float32](bucket_color_transmittance[bucket_row, 2])
                    snapshot_t = rebind[Float32](bucket_color_transmittance[bucket_row, 3])
                    local_last_contributor = rebind[Int32](tile_n_processed[pixel_idx])

                collected_color_after_transmittance[lane, 0] = (
                    color_pixel_w_bg.x
                    - final_transmittance * background.x
                    - snapshot_r
                )
                collected_color_after_transmittance[lane, 1] = (
                    color_pixel_w_bg.y
                    - final_transmittance * background.y
                    - snapshot_g
                )
                collected_color_after_transmittance[lane, 2] = (
                    color_pixel_w_bg.z
                    - final_transmittance * background.z
                    - snapshot_b
                )
                collected_color_after_transmittance[lane, 3] = snapshot_t
                collected_grad_info[lane, 0] = grad_color_pixel_local.x
                collected_grad_info[lane, 1] = grad_color_pixel_local.y
                collected_grad_info[lane, 2] = grad_color_pixel_local.z
                collected_grad_info[lane, 3] = (
                    final_transmittance
                    * -(
                        grad_color_pixel_local.x * background.x
                        + grad_color_pixel_local.y * background.y
                        + grad_color_pixel_local.z * background.z
                    )
                )
                collected_last_contributor[lane] = local_last_contributor
        barrier()

        # Lane 0 seeds one pixel state every 32 steps; shuffle_up propagates the
        # replay state through the warp exactly like the CUDA implementation.
        if i > 0:
            last_contributor = shuffle_up(last_contributor, UInt32(1))
            color_pixel_after.x = shuffle_up(color_pixel_after.x, UInt32(1))
            color_pixel_after.y = shuffle_up(color_pixel_after.y, UInt32(1))
            color_pixel_after.z = shuffle_up(color_pixel_after.z, UInt32(1))
            transmittance = shuffle_up(transmittance, UInt32(1))
            grad_color_pixel.x = shuffle_up(grad_color_pixel.x, UInt32(1))
            grad_color_pixel.y = shuffle_up(grad_color_pixel.y, UInt32(1))
            grad_color_pixel.z = shuffle_up(grad_color_pixel.z, UInt32(1))
            grad_alpha_common = shuffle_up(grad_alpha_common, UInt32(1))

        var idx = i - lane
        var idx_in_tile = idx >= 0 and idx < BLOCK_SIZE_BLEND
        var pixel_x = 0
        var pixel_y = 0
        var valid_pixel = False
        if idx_in_tile:
            pixel_x = start_pixel_x + idx % TILE_WIDTH
            pixel_y = start_pixel_y + idx // TILE_WIDTH
            valid_pixel = pixel_x < width and pixel_y < height

        if valid_primitive and valid_pixel and lane == 0 and idx < BLOCK_SIZE_BLEND:
            var shmem_idx = i % BUCKET_SIZE
            last_contributor = rebind[Int32](collected_last_contributor[shmem_idx])
            color_pixel_after = Float3(
                rebind[Float32](collected_color_after_transmittance[shmem_idx, 0]),
                rebind[Float32](collected_color_after_transmittance[shmem_idx, 1]),
                rebind[Float32](collected_color_after_transmittance[shmem_idx, 2]),
            )
            transmittance = rebind[Float32](
                collected_color_after_transmittance[shmem_idx, 3]
            )
            grad_color_pixel = Float3(
                rebind[Float32](collected_grad_info[shmem_idx, 0]),
                rebind[Float32](collected_grad_info[shmem_idx, 1]),
                rebind[Float32](collected_grad_info[shmem_idx, 2]),
            )
            grad_alpha_common = rebind[Float32](collected_grad_info[shmem_idx, 3])

        var skip = (
            not valid_primitive
            or not idx_in_tile
            or not valid_pixel
            or tile_primitive_idx >= Int(last_contributor)
        )
        if skip:
            continue

        var pixel = Float2(
            Float32(pixel_x) + Float32(0.5),
            Float32(pixel_y) + Float32(0.5),
        )
        var delta_x = mean2d.x - pixel.x
        var delta_y = mean2d.y - pixel.y
        var exponent = (
            Float32(-0.5)
            * (
                conic.x * delta_x * delta_x
                + conic.z * delta_y * delta_y
            )
            - conic.y * delta_x * delta_y
        )
        var gaussian = exp(min(exponent, Float32(0.0)))
        var alpha = opacity * gaussian
        if alpha < MIN_ALPHA_THRESHOLD:
            continue

        var blending_weight = transmittance * alpha

        var dL_dcolor = Float3(
            blending_weight * grad_color_pixel.x * color_grad_factor.x,
            blending_weight * grad_color_pixel.y * color_grad_factor.y,
            blending_weight * grad_color_pixel.z * color_grad_factor.z,
        )
        dL_dcolor_accum.x += dL_dcolor.x
        dL_dcolor_accum.y += dL_dcolor.y
        dL_dcolor_accum.z += dL_dcolor.z

        color_pixel_after.x -= blending_weight * color.x
        color_pixel_after.y -= blending_weight * color.y
        color_pixel_after.z -= blending_weight * color.z

        var one_minus_alpha = Float32(1.0) - alpha
        var one_minus_alpha_rcp = Float32(1.0) / max(
            one_minus_alpha,
            ONE_MINUS_ALPHA_EPS,
        )
        var dL_dalpha_from_color = (
            (
                transmittance * color.x
                - color_pixel_after.x * one_minus_alpha_rcp
            )
            * grad_color_pixel.x
            + (
                transmittance * color.y
                - color_pixel_after.y * one_minus_alpha_rcp
            )
            * grad_color_pixel.y
            + (
                transmittance * color.z
                - color_pixel_after.z * one_minus_alpha_rcp
            )
            * grad_color_pixel.z
        )
        var dL_dalpha_from_alpha = grad_alpha_common * one_minus_alpha_rcp
        var dL_dalpha = dL_dalpha_from_color + dL_dalpha_from_alpha

        dL_dopacity_accum += gaussian * dL_dalpha

        var gaussian_grad_helper = -alpha * dL_dalpha
        dL_dconic_accum.x += Float32(0.5) * gaussian_grad_helper * delta_x * delta_x
        dL_dconic_accum.y += Float32(0.5) * gaussian_grad_helper * delta_x * delta_y
        dL_dconic_accum.z += Float32(0.5) * gaussian_grad_helper * delta_y * delta_y
        dL_dmean2d_accum.x += gaussian_grad_helper * (
            conic.x * delta_x + conic.y * delta_y
        )
        dL_dmean2d_accum.y += gaussian_grad_helper * (
            conic.y * delta_x + conic.z * delta_y
        )

        transmittance *= one_minus_alpha

    # The Python runtime expects packed `(a, b, c, opacity)` gradients, so the
    # kernel accumulates directly into that layout.
    if valid_primitive:
        atomic_add_f32(
            grad_projected_means.ptr + primitive_idx * 2,
            dL_dmean2d_accum.x,
        )
        atomic_add_f32(
            grad_projected_means.ptr + primitive_idx * 2 + 1,
            dL_dmean2d_accum.y,
        )
        atomic_add_f32(
            grad_conic_opacity.ptr + primitive_idx * 4,
            dL_dconic_accum.x,
        )
        atomic_add_f32(
            grad_conic_opacity.ptr + primitive_idx * 4 + 1,
            dL_dconic_accum.y,
        )
        atomic_add_f32(
            grad_conic_opacity.ptr + primitive_idx * 4 + 2,
            dL_dconic_accum.z,
        )
        var opacity_grad = dL_dopacity_accum
        if not proper_antialiasing:
            opacity_grad *= opacity * (Float32(1.0) - opacity)
        atomic_add_f32(
            grad_conic_opacity.ptr + primitive_idx * 4 + 3,
            opacity_grad,
        )
        atomic_add_f32(
            grad_colors_rgb.ptr + primitive_idx * 3,
            dL_dcolor_accum.x,
        )
        atomic_add_f32(
            grad_colors_rgb.ptr + primitive_idx * 3 + 1,
            dL_dcolor_accum.y,
        )
        atomic_add_f32(
            grad_colors_rgb.ptr + primitive_idx * 3 + 2,
            dL_dcolor_accum.z,
        )


@compiler.register("blend_bwd")
struct BlendBackward:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        grad_projected_means: OutputTensor[dtype=DType.float32, rank=2, ...],
        grad_conic_opacity: OutputTensor[dtype=DType.float32, rank=2, ...],
        grad_colors_rgb: OutputTensor[dtype=DType.float32, rank=2, ...],
        grad_image: InputTensor[dtype=DType.float32, rank=3, ...],
        image: InputTensor[dtype=DType.float32, rank=3, ...],
        instance_primitive_indices: InputTensor[dtype=DType.int32, rank=1, ...],
        tile_instance_ranges: InputTensor[dtype=DType.int32, rank=2, ...],
        tile_bucket_offsets: InputTensor[dtype=DType.int32, rank=1, ...],
        projected_means: InputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: InputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: InputTensor[dtype=DType.float32, rank=2, ...],
        bg_color: InputTensor[dtype=DType.float32, rank=1, ...],
        tile_final_transmittances: InputTensor[dtype=DType.float32, rank=1, ...],
        tile_max_n_processed: InputTensor[dtype=DType.int32, rank=1, ...],
        tile_n_processed: InputTensor[dtype=DType.int32, rank=1, ...],
        bucket_tile_index: InputTensor[dtype=DType.int32, rank=1, ...],
        bucket_color_transmittance: InputTensor[dtype=DType.float32, rank=2, ...],
        proper_antialiasing_flag: InputTensor[dtype=DType.int32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var n_primitives = projected_means.dim_size[0]()
            var height = image.dim_size[1]()
            var width = image.dim_size[2]()
            var grid_width = div_round_up(width, TILE_WIDTH)
            var grid_height = div_round_up(height, TILE_HEIGHT)
            var tile_count = grid_width * grid_height
            var zero_block_dim = 256

            var projected_grad_size = n_primitives * 2
            if projected_grad_size > 0:
                gpu_ctx.enqueue_function[zero_f32_kernel, zero_f32_kernel](
                    grad_projected_means.unsafe_ptr(),
                    projected_grad_size,
                    grid_dim=div_round_up(projected_grad_size, zero_block_dim),
                    block_dim=zero_block_dim,
                )

            var conic_grad_size = n_primitives * 4
            if conic_grad_size > 0:
                gpu_ctx.enqueue_function[zero_f32_kernel, zero_f32_kernel](
                    grad_conic_opacity.unsafe_ptr(),
                    conic_grad_size,
                    grid_dim=div_round_up(conic_grad_size, zero_block_dim),
                    block_dim=zero_block_dim,
                )

            var color_grad_size = n_primitives * 3
            if color_grad_size > 0:
                gpu_ctx.enqueue_function[zero_f32_kernel, zero_f32_kernel](
                    grad_colors_rgb.unsafe_ptr(),
                    color_grad_size,
                    grid_dim=div_round_up(color_grad_size, zero_block_dim),
                    block_dim=zero_block_dim,
                )

            var n_buckets = bucket_tile_index.dim_size[0]()
            if n_buckets > 0:
                var grad_projected_means_tensor = LayoutTensor[
                    DType.float32,
                    ROW_MAJOR_2D,
                    MutAnyOrigin,
                ](
                    grad_projected_means.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_2D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(
                        Index(
                            grad_projected_means.dim_size[0](),
                            grad_projected_means.dim_size[1](),
                        )
                    ),
                )
                var grad_conic_opacity_tensor = LayoutTensor[
                    DType.float32,
                    ROW_MAJOR_2D,
                    MutAnyOrigin,
                ](
                    grad_conic_opacity.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_2D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(
                        Index(
                            grad_conic_opacity.dim_size[0](),
                            grad_conic_opacity.dim_size[1](),
                        )
                    ),
                )
                var grad_colors_rgb_tensor = LayoutTensor[
                    DType.float32,
                    ROW_MAJOR_2D,
                    MutAnyOrigin,
                ](
                    grad_colors_rgb.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_2D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(
                        Index(
                            grad_colors_rgb.dim_size[0](),
                            grad_colors_rgb.dim_size[1](),
                        )
                    ),
                )
                var grad_image_tensor = LayoutTensor[DType.float32, ROW_MAJOR_3D, MutAnyOrigin](
                    grad_image.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_3D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(
                        Index(
                            grad_image.dim_size[0](),
                            grad_image.dim_size[1](),
                            grad_image.dim_size[2](),
                        )
                    ),
                )
                var image_tensor = LayoutTensor[DType.float32, ROW_MAJOR_3D, MutAnyOrigin](
                    image.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_3D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(
                        Index(
                            image.dim_size[0](),
                            image.dim_size[1](),
                            image.dim_size[2](),
                        )
                    ),
                )
                var instance_primitive_indices_tensor = LayoutTensor[
                    DType.int32,
                    ROW_MAJOR_1D,
                    MutAnyOrigin,
                ](
                    instance_primitive_indices.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(instance_primitive_indices.dim_size[0]())),
                )
                var tile_instance_ranges_tensor = LayoutTensor[
                    DType.int32,
                    ROW_MAJOR_2D,
                    MutAnyOrigin,
                ](
                    tile_instance_ranges.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_2D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(
                        Index(
                            tile_instance_ranges.dim_size[0](),
                            tile_instance_ranges.dim_size[1](),
                        )
                    ),
                )
                var tile_bucket_offsets_tensor = LayoutTensor[
                    DType.int32,
                    ROW_MAJOR_1D,
                    MutAnyOrigin,
                ](
                    tile_bucket_offsets.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(tile_bucket_offsets.dim_size[0]())),
                )
                var projected_means_tensor = LayoutTensor[
                    DType.float32,
                    ROW_MAJOR_2D,
                    MutAnyOrigin,
                ](
                    projected_means.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_2D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(
                        Index(
                            projected_means.dim_size[0](),
                            projected_means.dim_size[1](),
                        )
                    ),
                )
                var conic_opacity_tensor = LayoutTensor[
                    DType.float32,
                    ROW_MAJOR_2D,
                    MutAnyOrigin,
                ](
                    conic_opacity.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_2D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(
                        Index(
                            conic_opacity.dim_size[0](),
                            conic_opacity.dim_size[1](),
                        )
                    ),
                )
                var colors_rgb_tensor = LayoutTensor[
                    DType.float32,
                    ROW_MAJOR_2D,
                    MutAnyOrigin,
                ](
                    colors_rgb.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_2D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(
                        Index(
                            colors_rgb.dim_size[0](),
                            colors_rgb.dim_size[1](),
                        )
                    ),
                )
                var bg_color_tensor = LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin](
                    bg_color.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(bg_color.dim_size[0]())),
                )
                var tile_final_transmittances_tensor = LayoutTensor[
                    DType.float32,
                    ROW_MAJOR_1D,
                    MutAnyOrigin,
                ](
                    tile_final_transmittances.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(tile_count * BLOCK_SIZE_BLEND)),
                )
                var tile_max_n_processed_tensor = LayoutTensor[
                    DType.int32,
                    ROW_MAJOR_1D,
                    MutAnyOrigin,
                ](
                    tile_max_n_processed.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(tile_count)),
                )
                var tile_n_processed_tensor = LayoutTensor[
                    DType.int32,
                    ROW_MAJOR_1D,
                    MutAnyOrigin,
                ](
                    tile_n_processed.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(tile_count * BLOCK_SIZE_BLEND)),
                )
                var bucket_tile_index_tensor = LayoutTensor[
                    DType.int32,
                    ROW_MAJOR_1D,
                    MutAnyOrigin,
                ](
                    bucket_tile_index.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(bucket_tile_index.dim_size[0]())),
                )
                var bucket_color_transmittance_tensor = LayoutTensor[
                    DType.float32,
                    ROW_MAJOR_2D,
                    MutAnyOrigin,
                ](
                    bucket_color_transmittance.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_2D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(
                        Index(
                            bucket_color_transmittance.dim_size[0](),
                            bucket_color_transmittance.dim_size[1](),
                        )
                    ),
                )

                gpu_ctx.enqueue_function[blend_bwd_kernel, blend_bwd_kernel](
                    grad_projected_means_tensor,
                    grad_conic_opacity_tensor,
                    grad_colors_rgb_tensor,
                    grad_image_tensor,
                    image_tensor,
                    instance_primitive_indices_tensor,
                    tile_instance_ranges_tensor,
                    tile_bucket_offsets_tensor,
                    projected_means_tensor,
                    conic_opacity_tensor,
                    colors_rgb_tensor,
                    bg_color_tensor,
                    tile_final_transmittances_tensor,
                    tile_max_n_processed_tensor,
                    tile_n_processed_tensor,
                    bucket_tile_index_tensor,
                    bucket_color_transmittance_tensor,
                    n_primitives,
                    width,
                    height,
                    grid_width,
                    n_buckets,
                    Int(proper_antialiasing_flag.unsafe_ptr()[0]),
                    grid_dim=n_buckets,
                    block_dim=BUCKET_SIZE,
                )
        else:
            raise Error("faster_gs_mojo blend_bwd currently requires a GPU target")
