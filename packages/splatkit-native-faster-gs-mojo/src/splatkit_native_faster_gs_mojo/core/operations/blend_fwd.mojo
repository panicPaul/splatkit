import compiler

from layout import Layout, LayoutTensor, TensorLayout
from layout.runtime_layout import RuntimeLayout
from tensor import InputTensor, OutputTensor
from std.gpu import WARP_SIZE, block_idx, thread_idx
from std.gpu.memory import AddressSpace
from std.gpu.primitives.block import max as block_max
from std.gpu.primitives.block import sum as block_sum
from std.gpu.sync import barrier
from std.math import exp
from std.runtime.asyncrt import DeviceContextPtr
from std.utils import Index

from .common import Float2, Float3, Float4, div_round_up


comptime TILE_WIDTH = 16
comptime TILE_HEIGHT = 16
comptime BLOCK_SIZE_BLEND = TILE_WIDTH * TILE_HEIGHT
comptime BUCKET_SIZE = 32
comptime MIN_ALPHA_THRESHOLD = Float32(1.0) / Float32(255.0)
comptime TRANSMITTANCE_THRESHOLD = Float32(1.0e-4)

comptime PRIMITIVE_BLEND_COLS = 9
comptime TILE_RANGE_START_COL = 0
comptime TILE_RANGE_END_COL = 1

comptime ROW_MAJOR_1D = Layout.row_major(Int())
comptime ROW_MAJOR_2D = Layout.row_major(Int(), Int())
comptime ROW_MAJOR_3D = Layout.row_major(Int(), Int(), Int())

comptime SHARED_BLEND_DATA = Layout.row_major(BLOCK_SIZE_BLEND, PRIMITIVE_BLEND_COLS)


@always_inline
def store_blend_row(
    blend_data: LayoutTensor[
        DType.float32,
        SHARED_BLEND_DATA,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    row_idx: Int,
    projected_means: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    conic_opacity: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    colors_rgb: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    primitive_idx: Int,
):
    """Stage one primitive's blend inputs into shared memory."""
    blend_data[row_idx, 0] = rebind[Float32](projected_means[primitive_idx, 0])
    blend_data[row_idx, 1] = rebind[Float32](projected_means[primitive_idx, 1])
    blend_data[row_idx, 2] = rebind[Float32](conic_opacity[primitive_idx, 0])
    blend_data[row_idx, 3] = rebind[Float32](conic_opacity[primitive_idx, 1])
    blend_data[row_idx, 4] = rebind[Float32](conic_opacity[primitive_idx, 2])
    blend_data[row_idx, 5] = rebind[Float32](conic_opacity[primitive_idx, 3])
    blend_data[row_idx, 6] = max(
        rebind[Float32](colors_rgb[primitive_idx, 0]),
        Float32(0.0),
    )
    blend_data[row_idx, 7] = max(
        rebind[Float32](colors_rgb[primitive_idx, 1]),
        Float32(0.0),
    )
    blend_data[row_idx, 8] = max(
        rebind[Float32](colors_rgb[primitive_idx, 2]),
        Float32(0.0),
    )


@always_inline
def load_blend_mean(
    blend_data: LayoutTensor[
        DType.float32,
        SHARED_BLEND_DATA,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    row_idx: Int,
) -> Float2:
    """Load the staged mean for one primitive."""
    return Float2(
        rebind[Float32](blend_data[row_idx, 0]),
        rebind[Float32](blend_data[row_idx, 1]),
    )


@always_inline
def load_blend_conic_opacity(
    blend_data: LayoutTensor[
        DType.float32,
        SHARED_BLEND_DATA,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    row_idx: Int,
) -> Float4:
    """Load the staged conic and opacity for one primitive."""
    return Float4(
        rebind[Float32](blend_data[row_idx, 2]),
        rebind[Float32](blend_data[row_idx, 3]),
        rebind[Float32](blend_data[row_idx, 4]),
        rebind[Float32](blend_data[row_idx, 5]),
    )


@always_inline
def load_blend_color(
    blend_data: LayoutTensor[
        DType.float32,
        SHARED_BLEND_DATA,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    row_idx: Int,
) -> Float3:
    """Load the staged RGB color for one primitive."""
    return Float3(
        rebind[Float32](blend_data[row_idx, 6]),
        rebind[Float32](blend_data[row_idx, 7]),
        rebind[Float32](blend_data[row_idx, 8]),
    )


@always_inline
def blend_fwd_kernel(
    image: LayoutTensor[DType.float32, ROW_MAJOR_3D, MutAnyOrigin],
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
    instance_primitive_indices: LayoutTensor[
        DType.int32,
        ROW_MAJOR_1D,
        MutAnyOrigin,
    ],
    tile_instance_ranges: LayoutTensor[DType.int32, ROW_MAJOR_2D, MutAnyOrigin],
    tile_bucket_offsets: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    projected_means: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    conic_opacity: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    colors_rgb: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    bg_color: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    width: Int,
    height: Int,
    grid_width: Int,
    n_primitives: Int,
):
    """Blend a tile into the final image and replay-state buffers."""
    comptime assert WARP_SIZE == 32, "FasterGS blend assumes 32-lane warps"
    _ = n_primitives

    # One 16x16 block owns one image tile and one thread shades one pixel.
    var thread_rank = Int(thread_idx.y) * TILE_WIDTH + Int(thread_idx.x)
    var pixel_x = Int(block_idx.x) * TILE_WIDTH + Int(thread_idx.x)
    var pixel_y = Int(block_idx.y) * TILE_HEIGHT + Int(thread_idx.y)
    var inside = pixel_x < width and pixel_y < height
    var pixel = Float2(
        Float32(pixel_x) + Float32(0.5),
        Float32(pixel_y) + Float32(0.5),
    )

    var current_tile_idx = Int(block_idx.y) * grid_width + Int(block_idx.x)
    var tile_start = Int(
        rebind[Int32](tile_instance_ranges[current_tile_idx, TILE_RANGE_START_COL])
    )
    var tile_end = Int(
        rebind[Int32](tile_instance_ranges[current_tile_idx, TILE_RANGE_END_COL])
    )
    var n_points_total = tile_end - tile_start
    var n_buckets = div_round_up(n_points_total, BUCKET_SIZE)
    var bucket_offset = Int(
        Int32(0)
        if current_tile_idx == 0
        else rebind[Int32](tile_bucket_offsets[current_tile_idx - 1])
    )

    var current_bucket_idx = thread_rank
    while current_bucket_idx < n_buckets:
        bucket_tile_index[bucket_offset + current_bucket_idx] = Int32(current_tile_idx)
        current_bucket_idx += BLOCK_SIZE_BLEND

    # Shared staging keeps one packed row per Gaussian even though the external
    # op surface is fully un-packed. That preserves the efficient inner loop
    # structure without leaking any packing across the op boundary.
    var collected_blend_data = LayoutTensor[
        DType.float32,
        SHARED_BLEND_DATA,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var color_pixel = Float3(Float32(0.0), Float32(0.0), Float32(0.0))
    var transmittance = Float32(1.0)
    var n_processed = 0
    var n_processed_and_used = 0
    var done = not inside

    var n_points_remaining = n_points_total
    var current_fetch_idx = tile_start + thread_rank
    var local_bucket_offset = bucket_offset
    while n_points_remaining > 0:
        # Threads that have already converged still participate in the block
        # collectives so the tile exits coherently once every pixel is done.
        var done_count = Int(
            block_sum[block_dim_x=TILE_WIDTH, block_dim_y=TILE_HEIGHT](
                SIMD[DType.int32, 1](Int32(1 if done else 0))
            )[0]
        )
        if done_count == BLOCK_SIZE_BLEND:
            break

        if current_fetch_idx < tile_end:
            var primitive_idx = Int(
                rebind[Int32](instance_primitive_indices[current_fetch_idx])
            )
            store_blend_row(
                collected_blend_data,
                thread_rank,
                projected_means,
                conic_opacity,
                colors_rgb,
                primitive_idx,
            )
        barrier()

        var current_batch_size = min(n_points_remaining, BLOCK_SIZE_BLEND)
        for j in range(current_batch_size):
            if done:
                break

            # Save one replay snapshot per bucket boundary. The backward pass can
            # restart blending from these checkpoints instead of replaying the
            # whole tile from the front.
            if j % BUCKET_SIZE == 0:
                var snapshot_row = local_bucket_offset * BLOCK_SIZE_BLEND + thread_rank
                bucket_color_transmittance[snapshot_row, 0] = color_pixel.x
                bucket_color_transmittance[snapshot_row, 1] = color_pixel.y
                bucket_color_transmittance[snapshot_row, 2] = color_pixel.z
                bucket_color_transmittance[snapshot_row, 3] = transmittance
                local_bucket_offset += 1

            n_processed += 1

            var mean2d = load_blend_mean(collected_blend_data, j)
            var conic_opacity = load_blend_conic_opacity(collected_blend_data, j)
            var color = load_blend_color(collected_blend_data, j)
            var delta_x = mean2d.x - pixel.x
            var delta_y = mean2d.y - pixel.y
            var exponent = (
                Float32(-0.5)
                * (
                    conic_opacity.x * delta_x * delta_x
                    + conic_opacity.z * delta_y * delta_y
                )
                - conic_opacity.y * delta_x * delta_y
            )
            # Clamp the exponent above by zero so we match the CUDA reference's
            # numerically stable Gaussian evaluation path.
            var gaussian = exp(min(exponent, Float32(0.0)))
            var alpha = conic_opacity.w * gaussian
            if alpha < MIN_ALPHA_THRESHOLD:
                continue

            color_pixel.x += transmittance * alpha * color.x
            color_pixel.y += transmittance * alpha * color.y
            color_pixel.z += transmittance * alpha * color.z

            transmittance *= Float32(1.0) - alpha
            n_processed_and_used = n_processed
            if transmittance < TRANSMITTANCE_THRESHOLD:
                done = True
        barrier()

        n_points_remaining -= BLOCK_SIZE_BLEND
        current_fetch_idx += BLOCK_SIZE_BLEND

    if inside:
        color_pixel.x += transmittance * rebind[Float32](bg_color[0])
        color_pixel.y += transmittance * rebind[Float32](bg_color[1])
        color_pixel.z += transmittance * rebind[Float32](bg_color[2])

        var pixel_idx = width * pixel_y + pixel_x
        image[0, pixel_y, pixel_x] = color_pixel.x
        image[1, pixel_y, pixel_x] = color_pixel.y
        image[2, pixel_y, pixel_x] = color_pixel.z
        tile_final_transmittances[pixel_idx] = transmittance
        tile_n_processed[pixel_idx] = Int32(n_processed_and_used)

    var tile_max_processed = Int(
        block_max[block_dim_x=TILE_WIDTH, block_dim_y=TILE_HEIGHT](
            SIMD[DType.int32, 1](Int32(n_processed_and_used))
        )[0]
    )
    if thread_rank == 0:
        tile_max_n_processed[current_tile_idx] = Int32(tile_max_processed)


@always_inline
def blend_fwd_image_only_kernel(
    image: LayoutTensor[DType.float32, ROW_MAJOR_3D, MutAnyOrigin],
    instance_primitive_indices: LayoutTensor[
        DType.int32,
        ROW_MAJOR_1D,
        MutAnyOrigin,
    ],
    tile_instance_ranges: LayoutTensor[DType.int32, ROW_MAJOR_2D, MutAnyOrigin],
    projected_means: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    conic_opacity: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    colors_rgb: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    bg_color: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    width: Int,
    height: Int,
    grid_width: Int,
):
    """Blend a tile into the final image without replay-state writes."""
    comptime assert WARP_SIZE == 32, "FasterGS blend assumes 32-lane warps"

    # This is the viewer/inference path: same per-tile ownership and math as
    # the training kernel, but it skips all replay-state writes entirely.
    var thread_rank = Int(thread_idx.y) * TILE_WIDTH + Int(thread_idx.x)
    var pixel_x = Int(block_idx.x) * TILE_WIDTH + Int(thread_idx.x)
    var pixel_y = Int(block_idx.y) * TILE_HEIGHT + Int(thread_idx.y)
    var inside = pixel_x < width and pixel_y < height
    var pixel = Float2(
        Float32(pixel_x) + Float32(0.5),
        Float32(pixel_y) + Float32(0.5),
    )

    var current_tile_idx = Int(block_idx.y) * grid_width + Int(block_idx.x)
    var tile_start = Int(
        rebind[Int32](tile_instance_ranges[current_tile_idx, TILE_RANGE_START_COL])
    )
    var tile_end = Int(
        rebind[Int32](tile_instance_ranges[current_tile_idx, TILE_RANGE_END_COL])
    )

    var collected_blend_data = LayoutTensor[
        DType.float32,
        SHARED_BLEND_DATA,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var color_pixel = Float3(Float32(0.0), Float32(0.0), Float32(0.0))
    var transmittance = Float32(1.0)
    var done = not inside

    var n_points_remaining = tile_end - tile_start
    var current_fetch_idx = tile_start + thread_rank
    while n_points_remaining > 0:
        var done_count = Int(
            block_sum[block_dim_x=TILE_WIDTH, block_dim_y=TILE_HEIGHT](
                SIMD[DType.int32, 1](Int32(1 if done else 0))
            )[0]
        )
        if done_count == BLOCK_SIZE_BLEND:
            break

        if current_fetch_idx < tile_end:
            var primitive_idx = Int(
                rebind[Int32](instance_primitive_indices[current_fetch_idx])
            )
            store_blend_row(
                collected_blend_data,
                thread_rank,
                projected_means,
                conic_opacity,
                colors_rgb,
                primitive_idx,
            )
        barrier()

        var current_batch_size = min(n_points_remaining, BLOCK_SIZE_BLEND)
        for j in range(current_batch_size):
            if done:
                break

            var mean2d = load_blend_mean(collected_blend_data, j)
            var conic_opacity = load_blend_conic_opacity(collected_blend_data, j)
            var color = load_blend_color(collected_blend_data, j)
            var delta_x = mean2d.x - pixel.x
            var delta_y = mean2d.y - pixel.y
            var exponent = (
                Float32(-0.5)
                * (
                    conic_opacity.x * delta_x * delta_x
                    + conic_opacity.z * delta_y * delta_y
                )
                - conic_opacity.y * delta_x * delta_y
            )
            var gaussian = exp(min(exponent, Float32(0.0)))
            var alpha = conic_opacity.w * gaussian
            if alpha < MIN_ALPHA_THRESHOLD:
                continue

            color_pixel.x += transmittance * alpha * color.x
            color_pixel.y += transmittance * alpha * color.y
            color_pixel.z += transmittance * alpha * color.z
            transmittance *= Float32(1.0) - alpha
            if transmittance < TRANSMITTANCE_THRESHOLD:
                done = True
        barrier()

        n_points_remaining -= BLOCK_SIZE_BLEND
        current_fetch_idx += BLOCK_SIZE_BLEND

    if inside:
        color_pixel.x += transmittance * rebind[Float32](bg_color[0])
        color_pixel.y += transmittance * rebind[Float32](bg_color[1])
        color_pixel.z += transmittance * rebind[Float32](bg_color[2])
        image[0, pixel_y, pixel_x] = color_pixel.x
        image[1, pixel_y, pixel_x] = color_pixel.y
        image[2, pixel_y, pixel_x] = color_pixel.z


# ================================================================================================ #
#                                   Launch interface                                               #
# ================================================================================================ #


@compiler.register("blend_fwd")
struct BlendForward:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        image: OutputTensor[dtype=DType.float32, rank=3, ...],
        tile_final_transmittances: OutputTensor[dtype=DType.float32, rank=1, ...],
        tile_max_n_processed: OutputTensor[dtype=DType.int32, rank=1, ...],
        tile_n_processed: OutputTensor[dtype=DType.int32, rank=1, ...],
        bucket_tile_index: OutputTensor[dtype=DType.int32, rank=1, ...],
        bucket_color_transmittance: OutputTensor[dtype=DType.float32, rank=2, ...],
        instance_primitive_indices: InputTensor[dtype=DType.int32, rank=1, ...],
        tile_instance_ranges: InputTensor[dtype=DType.int32, rank=2, ...],
        tile_bucket_offsets: InputTensor[dtype=DType.int32, rank=1, ...],
        bucket_count: InputTensor[dtype=DType.int32, rank=1, ...],
        projected_means: InputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: InputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: InputTensor[dtype=DType.float32, rank=2, ...],
        bg_color: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        """Run the replay-state-producing blend forward MAX custom op."""
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var height = Int(image.dim_size[1]())
            var width = Int(image.dim_size[2]())
            var grid_width = div_round_up(width, TILE_WIDTH)
            var grid_height = div_round_up(height, TILE_HEIGHT)
            var n_primitives = Int(projected_means.dim_size[0]())
            _ = bucket_count

            var image_tensor = LayoutTensor[DType.float32, ROW_MAJOR_3D, MutAnyOrigin](
                image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_3D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(3, height, width)),
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
                ].row_major(Index(Int(tile_final_transmittances.dim_size[0]()))),
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
                ].row_major(Index(Int(tile_max_n_processed.dim_size[0]()))),
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
                ].row_major(Index(Int(tile_n_processed.dim_size[0]()))),
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
                ].row_major(Index(Int(bucket_tile_index.dim_size[0]()))),
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
                        Int(bucket_color_transmittance.dim_size[0]()),
                        Int(bucket_color_transmittance.dim_size[1]()),
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
                ].row_major(Index(Int(instance_primitive_indices.dim_size[0]()))),
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
                    Index(Int(tile_instance_ranges.dim_size[0]()), Int(tile_instance_ranges.dim_size[1]()))
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
                ].row_major(Index(Int(tile_bucket_offsets.dim_size[0]()))),
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
                ].row_major(Index(Int(projected_means.dim_size[0]()), Int(projected_means.dim_size[1]()))),
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
                ].row_major(Index(Int(conic_opacity.dim_size[0]()), Int(conic_opacity.dim_size[1]()))),
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
                ].row_major(Index(Int(colors_rgb.dim_size[0]()), Int(colors_rgb.dim_size[1]()))),
            )
            var bg_color_tensor = LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin](
                bg_color.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(bg_color.dim_size[0]()))),
            )

            gpu_ctx.enqueue_function[blend_fwd_kernel, blend_fwd_kernel](
                # Blend owns the final image write plus the replay state needed
                # by the existing backward path. All packing stays internal.
                image_tensor,
                tile_final_transmittances_tensor,
                tile_max_n_processed_tensor,
                tile_n_processed_tensor,
                bucket_tile_index_tensor,
                bucket_color_transmittance_tensor,
                instance_primitive_indices_tensor,
                tile_instance_ranges_tensor,
                tile_bucket_offsets_tensor,
                projected_means_tensor,
                conic_opacity_tensor,
                colors_rgb_tensor,
                bg_color_tensor,
                width,
                height,
                grid_width,
                n_primitives,
                grid_dim=(grid_width, grid_height),
                block_dim=(TILE_WIDTH, TILE_HEIGHT),
            )


@compiler.register("blend_fwd_image_only")
struct BlendForwardImageOnly:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        image: OutputTensor[dtype=DType.float32, rank=3, ...],
        instance_primitive_indices: InputTensor[dtype=DType.int32, rank=1, ...],
        tile_instance_ranges: InputTensor[dtype=DType.int32, rank=2, ...],
        tile_bucket_offsets: InputTensor[dtype=DType.int32, rank=1, ...],
        bucket_count: InputTensor[dtype=DType.int32, rank=1, ...],
        projected_means: InputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: InputTensor[dtype=DType.float32, rank=2, ...],
        colors_rgb: InputTensor[dtype=DType.float32, rank=2, ...],
        bg_color: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        """Run the image-only blend MAX custom op."""
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var height = Int(image.dim_size[1]())
            var width = Int(image.dim_size[2]())
            var grid_width = div_round_up(width, TILE_WIDTH)
            var grid_height = div_round_up(height, TILE_HEIGHT)
            var tile_count = Int(tile_instance_ranges.dim_size[0]())

            var image_tensor = LayoutTensor[DType.float32, ROW_MAJOR_3D, MutAnyOrigin](
                image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_3D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(3, height, width)),
            )
            var instance_indices_tensor = LayoutTensor[
                DType.int32,
                ROW_MAJOR_1D,
                MutAnyOrigin,
            ](
                instance_primitive_indices.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(instance_primitive_indices.dim_size[0]()))),
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
                    Index(Int(tile_instance_ranges.dim_size[0]()), Int(tile_instance_ranges.dim_size[1]()))
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
                ].row_major(Index(Int(tile_bucket_offsets.dim_size[0]()))),
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
                ].row_major(Index(Int(projected_means.dim_size[0]()), Int(projected_means.dim_size[1]()))),
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
                ].row_major(Index(Int(conic_opacity.dim_size[0]()), Int(conic_opacity.dim_size[1]()))),
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
                ].row_major(Index(Int(colors_rgb.dim_size[0]()), Int(colors_rgb.dim_size[1]()))),
            )

            var bg_color_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_1D,
                MutAnyOrigin,
            ](
                bg_color.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(3)),
            )

            gpu_ctx.enqueue_function[
                blend_fwd_image_only_kernel,
                blend_fwd_image_only_kernel,
            ](
                image_tensor,
                instance_indices_tensor,
                tile_instance_ranges_tensor,
                projected_means_tensor,
                conic_opacity_tensor,
                colors_rgb_tensor,
                bg_color_tensor,
                width,
                height,
                grid_width,
                grid_dim=(grid_width, grid_height),
                block_dim=(TILE_WIDTH, TILE_HEIGHT),
            )
        else:
            raise Error("faster_gs_mojo blend_fwd currently requires a GPU target")
