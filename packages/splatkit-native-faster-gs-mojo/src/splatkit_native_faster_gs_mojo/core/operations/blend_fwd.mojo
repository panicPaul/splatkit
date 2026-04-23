import compiler

from layout import Layout, LayoutTensor
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

comptime TILE_METADATA_COLS = 3
comptime PRIMITIVE_BLEND_COLS = 9
comptime TILE_RANGE_START_COL = 0
comptime TILE_RANGE_END_COL = 1
comptime TILE_BUCKET_BASE_COL = 2
comptime PRIMITIVE_MEAN_X_COL = 0
comptime PRIMITIVE_MEAN_Y_COL = 1
comptime PRIMITIVE_CONIC_A_COL = 2
comptime PRIMITIVE_CONIC_B_COL = 3
comptime PRIMITIVE_CONIC_C_COL = 4
comptime PRIMITIVE_OPACITY_COL = 5
comptime PRIMITIVE_COLOR_R_COL = 6
comptime PRIMITIVE_COLOR_G_COL = 7
comptime PRIMITIVE_COLOR_B_COL = 8

comptime ROW_MAJOR_1D = Layout.row_major(Int())
comptime ROW_MAJOR_2D = Layout.row_major(Int(), Int())
comptime ROW_MAJOR_3D = Layout.row_major(Int(), Int(), Int())

comptime SHARED_BLEND_DATA = Layout.row_major(BLOCK_SIZE_BLEND, PRIMITIVE_BLEND_COLS)


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
    tile_metadata: LayoutTensor[DType.int32, ROW_MAJOR_2D, MutAnyOrigin],
    primitive_blend_data: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    bg_color: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    width: Int,
    height: Int,
    grid_width: Int,
    n_primitives: Int,
):
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

    # Tile metadata stays packed at the op boundary but is unpacked into local
    # scalars once per tile.
    var current_tile_idx = Int(block_idx.y) * grid_width + Int(block_idx.x)
    var tile_start = Int(
        rebind[Int32](tile_metadata[current_tile_idx, TILE_RANGE_START_COL])
    )
    var tile_end = Int(
        rebind[Int32](tile_metadata[current_tile_idx, TILE_RANGE_END_COL])
    )
    var n_points_total = tile_end - tile_start
    var n_buckets = div_round_up(n_points_total, BUCKET_SIZE)
    var bucket_offset = Int(
        rebind[Int32](tile_metadata[current_tile_idx, TILE_BUCKET_BASE_COL])
    )

    var current_bucket_idx = thread_rank
    while current_bucket_idx < n_buckets:
        bucket_tile_index[bucket_offset + current_bucket_idx] = Int32(current_tile_idx)
        current_bucket_idx += BLOCK_SIZE_BLEND

    # Shared staging keeps one packed row per Gaussian. The inner loop unpacks
    # that row into locals once, which is cheaper than repeatedly re-reading
    # each field from shared memory during the same pixel/Gaussian interaction.
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
            collected_blend_data[thread_rank, PRIMITIVE_MEAN_X_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_MEAN_X_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_MEAN_Y_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_MEAN_Y_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_CONIC_A_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_CONIC_A_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_CONIC_B_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_CONIC_B_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_CONIC_C_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_CONIC_C_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_OPACITY_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_OPACITY_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_COLOR_R_COL] = max(
                rebind[Float32](
                    primitive_blend_data[primitive_idx, PRIMITIVE_COLOR_R_COL]
                ),
                Float32(0.0),
            )
            collected_blend_data[thread_rank, PRIMITIVE_COLOR_G_COL] = max(
                rebind[Float32](
                    primitive_blend_data[primitive_idx, PRIMITIVE_COLOR_G_COL]
                ),
                Float32(0.0),
            )
            collected_blend_data[thread_rank, PRIMITIVE_COLOR_B_COL] = max(
                rebind[Float32](
                    primitive_blend_data[primitive_idx, PRIMITIVE_COLOR_B_COL]
                ),
                Float32(0.0),
            )
        barrier()

        var current_batch_size = min(n_points_remaining, BLOCK_SIZE_BLEND)
        for j in range(current_batch_size):
            if done:
                break

            if j % BUCKET_SIZE == 0:
                var snapshot_row = local_bucket_offset * BLOCK_SIZE_BLEND + thread_rank
                bucket_color_transmittance[snapshot_row, 0] = color_pixel.x
                bucket_color_transmittance[snapshot_row, 1] = color_pixel.y
                bucket_color_transmittance[snapshot_row, 2] = color_pixel.z
                bucket_color_transmittance[snapshot_row, 3] = transmittance
                local_bucket_offset += 1

            n_processed += 1

            var mean2d = Float2(
                rebind[Float32](collected_blend_data[j, PRIMITIVE_MEAN_X_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_MEAN_Y_COL]),
            )
            var conic_opacity = Float4(
                rebind[Float32](collected_blend_data[j, PRIMITIVE_CONIC_A_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_CONIC_B_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_CONIC_C_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_OPACITY_COL]),
            )
            var color = Float3(
                rebind[Float32](collected_blend_data[j, PRIMITIVE_COLOR_R_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_COLOR_G_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_COLOR_B_COL]),
            )
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
    tile_metadata: LayoutTensor[DType.int32, ROW_MAJOR_2D, MutAnyOrigin],
    primitive_blend_data: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    bg_color: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    width: Int,
    height: Int,
    grid_width: Int,
):
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
        rebind[Int32](tile_metadata[current_tile_idx, TILE_RANGE_START_COL])
    )
    var tile_end = Int(
        rebind[Int32](tile_metadata[current_tile_idx, TILE_RANGE_END_COL])
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
            collected_blend_data[thread_rank, PRIMITIVE_MEAN_X_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_MEAN_X_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_MEAN_Y_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_MEAN_Y_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_CONIC_A_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_CONIC_A_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_CONIC_B_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_CONIC_B_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_CONIC_C_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_CONIC_C_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_OPACITY_COL] = rebind[Float32](
                primitive_blend_data[primitive_idx, PRIMITIVE_OPACITY_COL]
            )
            collected_blend_data[thread_rank, PRIMITIVE_COLOR_R_COL] = max(
                rebind[Float32](
                    primitive_blend_data[primitive_idx, PRIMITIVE_COLOR_R_COL]
                ),
                Float32(0.0),
            )
            collected_blend_data[thread_rank, PRIMITIVE_COLOR_G_COL] = max(
                rebind[Float32](
                    primitive_blend_data[primitive_idx, PRIMITIVE_COLOR_G_COL]
                ),
                Float32(0.0),
            )
            collected_blend_data[thread_rank, PRIMITIVE_COLOR_B_COL] = max(
                rebind[Float32](
                    primitive_blend_data[primitive_idx, PRIMITIVE_COLOR_B_COL]
                ),
                Float32(0.0),
            )
        barrier()

        var current_batch_size = min(n_points_remaining, BLOCK_SIZE_BLEND)
        for j in range(current_batch_size):
            if done:
                break

            var mean2d = Float2(
                rebind[Float32](collected_blend_data[j, PRIMITIVE_MEAN_X_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_MEAN_Y_COL]),
            )
            var conic_opacity = Float4(
                rebind[Float32](collected_blend_data[j, PRIMITIVE_CONIC_A_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_CONIC_B_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_CONIC_C_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_OPACITY_COL]),
            )
            var color = Float3(
                rebind[Float32](collected_blend_data[j, PRIMITIVE_COLOR_R_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_COLOR_G_COL]),
                rebind[Float32](collected_blend_data[j, PRIMITIVE_COLOR_B_COL]),
            )
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


@compiler.register("blend_fwd")
struct BlendForward:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        image: OutputTensor[dtype=DType.float32, rank=3, ...],
        forward_state_f32: OutputTensor[dtype=DType.float32, rank=1, ...],
        forward_state_i32: OutputTensor[dtype=DType.int32, rank=1, ...],
        instance_primitive_indices: InputTensor[dtype=DType.int32, rank=1, ...],
        tile_metadata: InputTensor[dtype=DType.int32, rank=2, ...],
        primitive_blend_data: InputTensor[dtype=DType.float32, rank=2, ...],
        bg_color: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var height = image.dim_size[1]()
            var width = image.dim_size[2]()
            var grid_width = div_round_up(width, TILE_WIDTH)
            var grid_height = div_round_up(height, TILE_HEIGHT)
            var tile_count = tile_metadata.dim_size[0]()
            var tile_pixels = tile_count * BLOCK_SIZE_BLEND
            var bucket_total = forward_state_i32.dim_size[0]() - tile_count - tile_pixels
            var n_primitives = primitive_blend_data.dim_size[0]()

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
                forward_state_f32.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(tile_pixels)),
            )
            var tile_max_n_processed_tensor = LayoutTensor[
                DType.int32,
                ROW_MAJOR_1D,
                MutAnyOrigin,
            ](
                forward_state_i32.unsafe_ptr(),
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
                forward_state_i32.unsafe_ptr() + tile_count,
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(tile_pixels)),
            )
            var bucket_tile_index_tensor = LayoutTensor[
                DType.int32,
                ROW_MAJOR_1D,
                MutAnyOrigin,
            ](
                forward_state_i32.unsafe_ptr() + tile_count + tile_pixels,
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(bucket_total)),
            )
            var bucket_color_transmittance_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                forward_state_f32.unsafe_ptr() + tile_pixels,
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(bucket_total * BLOCK_SIZE_BLEND, 4)),
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
            var tile_metadata_tensor = LayoutTensor[
                DType.int32,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                tile_metadata.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(tile_metadata.dim_size[0](), tile_metadata.dim_size[1]())),
            )
            var primitive_blend_data_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                primitive_blend_data.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(
                    Index(
                        primitive_blend_data.dim_size[0](),
                        primitive_blend_data.dim_size[1](),
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

            gpu_ctx.enqueue_function[blend_fwd_kernel, blend_fwd_kernel](
                image_tensor,
                tile_final_transmittances_tensor,
                tile_max_n_processed_tensor,
                tile_n_processed_tensor,
                bucket_tile_index_tensor,
                bucket_color_transmittance_tensor,
                instance_primitive_indices_tensor,
                tile_metadata_tensor,
                primitive_blend_data_tensor,
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
        tile_metadata: InputTensor[dtype=DType.int32, rank=2, ...],
        primitive_blend_data: InputTensor[dtype=DType.float32, rank=2, ...],
        bg_color: InputTensor[dtype=DType.float32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var height = image.dim_size[1]()
            var width = image.dim_size[2]()
            var grid_width = div_round_up(width, TILE_WIDTH)
            var grid_height = div_round_up(height, TILE_HEIGHT)

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
                ].row_major(Index(instance_primitive_indices.dim_size[0]())),
            )
            var tile_metadata_tensor = LayoutTensor[
                DType.int32,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                tile_metadata.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(
                    Index(tile_metadata.dim_size[0](), TILE_METADATA_COLS)
                ),
            )
            var primitive_blend_data_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_2D,
                MutAnyOrigin,
            ](
                primitive_blend_data.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(
                    Index(
                        primitive_blend_data.dim_size[0](),
                        PRIMITIVE_BLEND_COLS,
                    )
                ),
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
                tile_metadata_tensor,
                primitive_blend_data_tensor,
                bg_color_tensor,
                width,
                height,
                grid_width,
                grid_dim=(grid_width, grid_height),
                block_dim=(TILE_WIDTH, TILE_HEIGHT),
            )
        else:
            raise Error("faster_gs_mojo blend_fwd currently requires a GPU target")
