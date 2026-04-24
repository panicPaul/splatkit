import compiler

from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives import block
from std.math import log, max
from std.runtime.asyncrt import DeviceContextPtr
from std.utils import Index
from tensor import InputTensor, OutputTensor

from .common import BUCKET_SIZE, TILE_HEIGHT, TILE_WIDTH, div_round_up, will_primitive_contribute
from .radix_sort import RadixSortWorkspace, device_radix_sort_pairs, device_radix_sort_pairs_ptrs


comptime ROW_MAJOR_1D = Layout.row_major(Int())
comptime ROW_MAJOR_2D = Layout.row_major(Int(), Int())


@always_inline
def zero_i32_1d_kernel(
    tensor: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    size: Int,
):
    """Fill a 1D `int32` tensor with zeros."""
    idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        tensor[idx] = Int32(0)


@always_inline
def zero_i32_2d_kernel(
    tensor: LayoutTensor[DType.int32, ROW_MAJOR_2D, MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    """Fill a 2D `int32` tensor with zeros."""
    idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < rows * cols:
        tensor[idx // cols, idx % cols] = Int32(0)


@always_inline
def prepare_depth_sort_pairs_kernel(
    depth_sort_keys: LayoutTensor[DType.uint32, ROW_MAJOR_1D, MutAnyOrigin],
    sorted_primitive_indices: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    depth_keys: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    primitive_indices: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    visible_count: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    n_primitives: Int,
):
    """Prepare visible `(depth_key, primitive_idx)` pairs for sorting."""
    idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx >= n_primitives:
        return

    n_visible = Int(rebind[Int32](visible_count[0]))
    if idx < n_visible:
        depth_sort_keys[idx] = UInt32(rebind[Int32](depth_keys[idx]))
        sorted_primitive_indices[idx] = rebind[Int32](primitive_indices[idx])
    else:
        depth_sort_keys[idx] = UInt32(4294967295)
        sorted_primitive_indices[idx] = Int32(0)


@always_inline
def gather_sorted_touched_counts_kernel(
    sorted_primitive_indices: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    num_touched_tiles: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    sorted_touched_counts: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    visible_count: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    n_primitives: Int,
):
    """Gather touched-tile counts into depth-sorted primitive order."""
    sorted_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if sorted_idx >= n_primitives:
        return

    if sorted_idx >= Int(rebind[Int32](visible_count[0])):
        sorted_touched_counts[sorted_idx] = Int32(0)
        return

    primitive_idx = Int(rebind[Int32](sorted_primitive_indices[sorted_idx]))
    sorted_touched_counts[sorted_idx] = rebind[Int32](num_touched_tiles[primitive_idx])


@always_inline
def exclusive_prefix_sum_i32_kernel(
    src: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    dst: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    size: Int,
):
    """Write an exclusive prefix sum for a 1D int32 tensor."""
    if Int(block_idx.x * block_dim.x + thread_idx.x) != 0:
        return

    var total = Int32(0)
    for idx in range(size):
        value = rebind[Int32](src[idx])
        dst[idx] = total
        total += value


@always_inline
def exclusive_prefix_sum_i32_blocks_kernel[
    src_layout: Layout,
    dst_layout: Layout,
    block_totals_layout: Layout,
](
    src: LayoutTensor[DType.int32, src_layout, MutAnyOrigin],
    dst: LayoutTensor[DType.int32, dst_layout, MutAnyOrigin],
    block_totals: LayoutTensor[DType.int32, block_totals_layout, MutAnyOrigin],
    size: Int,
):
    """Write per-block exclusive sums and one total per block."""
    comptime SCAN_BLOCK_SIZE = 256
    idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var value = Int32(0)
    if idx < size:
        value = rebind[Int32](src[idx])

    exclusive = block.prefix_sum[
        block_size=SCAN_BLOCK_SIZE,
        exclusive=True,
    ](value)
    total = block.sum[
        block_size=SCAN_BLOCK_SIZE,
        broadcast=True,
    ](value)

    if idx < size:
        dst[idx] = exclusive
    if Int(thread_idx.x) == 0:
        block_totals[Int(block_idx.x)] = total


@always_inline
def add_i32_block_offsets_kernel[
    values_layout: Layout,
    block_offsets_layout: Layout,
](
    values: LayoutTensor[DType.int32, values_layout, MutAnyOrigin],
    block_offsets: LayoutTensor[DType.int32, block_offsets_layout, MutAnyOrigin],
    size: Int,
):
    """Add the scanned block total to each value in that block."""
    idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < size:
        values[idx] += rebind[Int32](block_offsets[Int(block_idx.x)])


@always_inline
def expand_instances_kernel[instance_key_dtype: DType](
    instance_keys: LayoutTensor[instance_key_dtype, ROW_MAJOR_1D, MutAnyOrigin],
    instance_primitive_indices: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    sorted_primitive_indices: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    primitive_offsets: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    sorted_touched_counts: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    screen_bounds: LayoutTensor[DType.uint16, ROW_MAJOR_2D, MutAnyOrigin],
    projected_means: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    conic_opacity: LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin],
    width: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    n_primitives: Int,
    n_instances: Int,
):
    """Expand each visible primitive into tile-major instance keys."""
    sorted_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if sorted_idx >= n_primitives:
        return

    touched_count = Int(rebind[Int32](sorted_touched_counts[sorted_idx]))
    if touched_count <= 0:
        return

    primitive_idx = Int(rebind[Int32](sorted_primitive_indices[sorted_idx]))
    var write_offset = Int(rebind[Int32](primitive_offsets[sorted_idx]))
    var write_end = write_offset + touched_count
    grid_width = div_round_up(Int(rebind[Int32](width[0])), TILE_WIDTH)
    screen_min_x = Int(rebind[UInt16](screen_bounds[primitive_idx, 0]))
    screen_max_x = Int(rebind[UInt16](screen_bounds[primitive_idx, 1]))
    screen_min_y = Int(rebind[UInt16](screen_bounds[primitive_idx, 2]))
    screen_max_y = Int(rebind[UInt16](screen_bounds[primitive_idx, 3]))
    mean_x = rebind[Float32](projected_means[primitive_idx, 0]) - Float32(0.5)
    mean_y = rebind[Float32](projected_means[primitive_idx, 1]) - Float32(0.5)
    conic_x = rebind[Float32](conic_opacity[primitive_idx, 0])
    conic_y = rebind[Float32](conic_opacity[primitive_idx, 1])
    conic_z = rebind[Float32](conic_opacity[primitive_idx, 2])
    opacity = rebind[Float32](conic_opacity[primitive_idx, 3])
    power_threshold = log(opacity * Float32(255.0))

    for tile_y in range(screen_min_y, screen_max_y):
        for tile_x in range(screen_min_x, screen_max_x):
            if write_offset >= write_end or write_offset >= n_instances:
                return
            if will_primitive_contribute(
                mean_x,
                mean_y,
                conic_x,
                conic_y,
                conic_z,
                tile_x,
                tile_y,
                power_threshold,
            ):
                tile_id = tile_y * grid_width + tile_x
                instance_keys[write_offset] = Scalar[instance_key_dtype](tile_id)
                instance_primitive_indices[write_offset] = Int32(primitive_idx)
                write_offset += 1


@always_inline
def fill_instance_padding_kernel[instance_key_dtype: DType](
    instance_keys: LayoutTensor[instance_key_dtype, ROW_MAJOR_1D, MutAnyOrigin],
    instance_primitive_indices: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    instance_capacity: Int,
):
    """Initialize unused instance slots so radix sort keeps them at the end."""
    idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < instance_capacity:
        instance_keys[idx] = Scalar[instance_key_dtype].MAX
        instance_primitive_indices[idx] = Int32(0)


@always_inline
def extract_instance_ranges_kernel[instance_key_dtype: DType](
    instance_keys: LayoutTensor[instance_key_dtype, ROW_MAJOR_1D, MutAnyOrigin],
    tile_instance_ranges: LayoutTensor[DType.int32, ROW_MAJOR_2D, MutAnyOrigin],
    instance_count: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    instance_capacity: Int,
):
    """Recover `[start, end)` ranges for each tile from sorted instance keys."""
    instance_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    actual_instance_count = Int(rebind[Int32](instance_count[0]))
    if instance_idx >= actual_instance_count or instance_idx >= instance_capacity:
        return

    current_tile = Int(rebind[Scalar[instance_key_dtype]](instance_keys[instance_idx]))
    if instance_idx == 0:
        tile_instance_ranges[current_tile, 0] = Int32(0)
    else:
        previous_tile = Int(rebind[Scalar[instance_key_dtype]](instance_keys[instance_idx - 1]))
        if previous_tile != current_tile:
            tile_instance_ranges[previous_tile, 1] = Int32(instance_idx)
            tile_instance_ranges[current_tile, 0] = Int32(instance_idx)
    if instance_idx == actual_instance_count - 1:
        tile_instance_ranges[current_tile, 1] = Int32(actual_instance_count)


@always_inline
def write_bucket_offsets_kernel(
    tile_instance_ranges: LayoutTensor[DType.int32, ROW_MAJOR_2D, MutAnyOrigin],
    tile_bucket_offsets: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    bucket_count: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    tile_count: Int,
):
    """Write inclusive bucket offsets and the final bucket count."""
    if Int(block_idx.x * block_dim.x + thread_idx.x) != 0:
        return

    var total = Int32(0)
    for tile_idx in range(tile_count):
        range_start = Int(rebind[Int32](tile_instance_ranges[tile_idx, 0]))
        range_end = Int(rebind[Int32](tile_instance_ranges[tile_idx, 1]))
        total += Int32(div_round_up(range_end - range_start, BUCKET_SIZE))
        tile_bucket_offsets[tile_idx] = total
    bucket_count[0] = total


# ================================================================================================ #
#                                   Launch interface                                               #
# ================================================================================================ #


@compiler.register("sort_fwd")
struct SortForward:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        instance_primitive_indices: OutputTensor[dtype=DType.int32, rank=1, ...],
        tile_instance_ranges: OutputTensor[dtype=DType.int32, rank=2, ...],
        tile_bucket_offsets: OutputTensor[dtype=DType.int32, rank=1, ...],
        bucket_count: OutputTensor[dtype=DType.int32, rank=1, ...],
        depth_sort_keys_work: OutputTensor[dtype=DType.uint32, rank=1, ...],
        sorted_primitive_indices_work: OutputTensor[dtype=DType.int32, rank=1, ...],
        sorted_touched_counts_work: OutputTensor[dtype=DType.int32, rank=1, ...],
        primitive_offsets_work: OutputTensor[dtype=DType.int32, rank=1, ...],
        depth_keys: InputTensor[dtype=DType.int32, rank=1, ...],
        primitive_indices: InputTensor[dtype=DType.int32, rank=1, ...],
        num_touched_tiles: InputTensor[dtype=DType.int32, rank=1, ...],
        screen_bounds: InputTensor[dtype=DType.uint16, rank=2, ...],
        projected_means: InputTensor[dtype=DType.float32, rank=2, ...],
        conic_opacity: InputTensor[dtype=DType.float32, rank=2, ...],
        visible_count: InputTensor[dtype=DType.int32, rank=1, ...],
        instance_count: InputTensor[dtype=DType.int32, rank=1, ...],
        width: InputTensor[dtype=DType.int32, rank=1, ...],
        height: InputTensor[dtype=DType.int32, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        """Run the sort forward MAX custom op on GPU."""
        comptime if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            n_primitives = Int(depth_keys.dim_size[0]())
            n_visible_capacity = Int(depth_sort_keys_work.dim_size[0]())
            tile_count = Int(tile_instance_ranges.dim_size[0]())
            n_instances = Int(instance_primitive_indices.dim_size[0]())
            visible_count_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                visible_count.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(1)),
            )
            instance_count_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                instance_count.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(1)),
            )
            width_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                width.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(1)),
            )

            # Build explicit runtime layouts at the op boundary. The kernels
            # themselves only see concrete layout tensors, not raw pointers.
            depth_keys_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                depth_keys.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(depth_keys.dim_size[0]()))),
            )
            num_touched_tiles_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                num_touched_tiles.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(num_touched_tiles.dim_size[0]()))),
            )
            primitive_indices_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                primitive_indices.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(primitive_indices.dim_size[0]()))),
            )
            screen_bounds_tensor = LayoutTensor[DType.uint16, ROW_MAJOR_2D, MutAnyOrigin](
                screen_bounds.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(screen_bounds.dim_size[0]()), Int(screen_bounds.dim_size[1]()))),
            )
            projected_means_tensor = LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin](
                projected_means.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(projected_means.dim_size[0]()), Int(projected_means.dim_size[1]()))),
            )
            conic_opacity_tensor = LayoutTensor[DType.float32, ROW_MAJOR_2D, MutAnyOrigin](
                conic_opacity.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(conic_opacity.dim_size[0]()), Int(conic_opacity.dim_size[1]()))),
            )
            instance_primitive_indices_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                instance_primitive_indices.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(instance_primitive_indices.dim_size[0]()))),
            )
            tile_instance_ranges_tensor = LayoutTensor[DType.int32, ROW_MAJOR_2D, MutAnyOrigin](
                tile_instance_ranges.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_2D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(tile_instance_ranges.dim_size[0]()), Int(tile_instance_ranges.dim_size[1]()))),
            )
            tile_bucket_offsets_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                tile_bucket_offsets.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(tile_bucket_offsets.dim_size[0]()))),
            )
            bucket_count_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                bucket_count.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(Int(bucket_count.dim_size[0]()))),
            )
            depth_sort_keys_tensor = LayoutTensor[DType.uint32, ROW_MAJOR_1D, MutAnyOrigin](
                depth_sort_keys_work.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(n_visible_capacity)),
            )
            sorted_primitive_indices_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                sorted_primitive_indices_work.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(n_visible_capacity)),
            )
            sorted_touched_counts_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                sorted_touched_counts_work.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(n_visible_capacity)),
            )
            primitive_offsets_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                primitive_offsets_work.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(n_visible_capacity)),
            )

            # Clear long-lived forward-state outputs before rebuilding the
            # tile-major instance stream.
            gpu_ctx.enqueue_function[
                zero_i32_2d_kernel,
                zero_i32_2d_kernel,
            ](
                tile_instance_ranges_tensor,
                tile_count,
                2,
                grid_dim=div_round_up(max(tile_count * 2, 1), 256),
                block_dim=256,
            )
            gpu_ctx.enqueue_function[
                zero_i32_1d_kernel,
                zero_i32_1d_kernel,
            ](
                tile_bucket_offsets_tensor,
                tile_count,
                grid_dim=div_round_up(max(tile_count, 1), 256),
                block_dim=256,
            )
            gpu_ctx.enqueue_function[
                zero_i32_1d_kernel,
                zero_i32_1d_kernel,
            ](
                bucket_count_tensor,
                1,
                grid_dim=1,
                block_dim=1,
            )

            if n_primitives == 0 or n_instances == 0 or n_visible_capacity == 0:
                return

            var offset_scan_block_count = div_round_up(n_visible_capacity, 256)
            var offset_block_totals_buffer = gpu_ctx.enqueue_create_buffer[DType.int32](offset_scan_block_count)
            var offset_block_offsets_buffer = gpu_ctx.enqueue_create_buffer[DType.int32](offset_scan_block_count)

            var offset_block_totals_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                offset_block_totals_buffer.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(offset_scan_block_count)),
            )
            var offset_block_offsets_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                offset_block_offsets_buffer.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(offset_scan_block_count)),
            )

            gpu_ctx.enqueue_function[
                prepare_depth_sort_pairs_kernel,
                prepare_depth_sort_pairs_kernel,
            ](
                depth_sort_keys_tensor,
                sorted_primitive_indices_tensor,
                depth_keys_tensor,
                primitive_indices_tensor,
                visible_count_tensor,
                n_visible_capacity,
                grid_dim=div_round_up(n_visible_capacity, 256),
                block_dim=256,
            )

            var depth_sort_workspace = RadixSortWorkspace[DType.uint32, DType.int32](
                gpu_ctx,
                n_visible_capacity,
            )
            device_radix_sort_pairs_ptrs[DType.uint32, DType.int32](
                gpu_ctx,
                depth_sort_workspace,
                depth_sort_keys_work.unsafe_ptr(),
                sorted_primitive_indices_work.unsafe_ptr(),
                n_visible_capacity,
            )

            gpu_ctx.enqueue_function[
                gather_sorted_touched_counts_kernel,
                gather_sorted_touched_counts_kernel,
            ](
                sorted_primitive_indices_tensor,
                num_touched_tiles_tensor,
                sorted_touched_counts_tensor,
                visible_count_tensor,
                n_visible_capacity,
                grid_dim=div_round_up(n_visible_capacity, 256),
                block_dim=256,
            )
            gpu_ctx.enqueue_function[
                exclusive_prefix_sum_i32_blocks_kernel[
                    type_of(sorted_touched_counts_tensor).layout,
                    type_of(primitive_offsets_tensor).layout,
                    type_of(offset_block_totals_tensor).layout,
                ],
                exclusive_prefix_sum_i32_blocks_kernel[
                    type_of(sorted_touched_counts_tensor).layout,
                    type_of(primitive_offsets_tensor).layout,
                    type_of(offset_block_totals_tensor).layout,
                ],
            ](
                sorted_touched_counts_tensor,
                primitive_offsets_tensor,
                offset_block_totals_tensor,
                n_visible_capacity,
                grid_dim=offset_scan_block_count,
                block_dim=256,
            )
            gpu_ctx.enqueue_function[
                exclusive_prefix_sum_i32_kernel,
                exclusive_prefix_sum_i32_kernel,
            ](
                offset_block_totals_tensor,
                offset_block_offsets_tensor,
                offset_scan_block_count,
                grid_dim=1,
                block_dim=1,
            )
            gpu_ctx.enqueue_function[
                add_i32_block_offsets_kernel[
                    type_of(primitive_offsets_tensor).layout,
                    type_of(offset_block_offsets_tensor).layout,
                ],
                add_i32_block_offsets_kernel[
                    type_of(primitive_offsets_tensor).layout,
                    type_of(offset_block_offsets_tensor).layout,
                ],
            ](
                primitive_offsets_tensor,
                offset_block_offsets_tensor,
                n_visible_capacity,
                grid_dim=offset_scan_block_count,
                block_dim=256,
            )

            if tile_count < 65536:
                var instance_keys_buffer = gpu_ctx.enqueue_create_buffer[DType.uint16](n_instances)
                var instance_keys_tensor = LayoutTensor[DType.uint16, ROW_MAJOR_1D, MutAnyOrigin](
                    instance_keys_buffer.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(n_instances)),
                )
                comptime fill_kernel = fill_instance_padding_kernel[DType.uint16]
                gpu_ctx.enqueue_function[
                    fill_kernel,
                    fill_kernel,
                ](
                    instance_keys_tensor,
                    instance_primitive_indices_tensor,
                    n_instances,
                    grid_dim=div_round_up(n_instances, 256),
                    block_dim=256,
                )
                comptime expand_kernel = expand_instances_kernel[DType.uint16]
                gpu_ctx.enqueue_function[
                    expand_kernel,
                    expand_kernel,
                ](
                    instance_keys_tensor,
                    instance_primitive_indices_tensor,
                    sorted_primitive_indices_tensor,
                    primitive_offsets_tensor,
                    sorted_touched_counts_tensor,
                    screen_bounds_tensor,
                    projected_means_tensor,
                    conic_opacity_tensor,
                    width_tensor,
                    n_visible_capacity,
                    n_instances,
                    grid_dim=div_round_up(n_visible_capacity, 256),
                    block_dim=256,
                )

                var instance_sort_workspace = RadixSortWorkspace[DType.uint16, DType.int32](
                    gpu_ctx,
                    n_instances,
                )
                device_radix_sort_pairs_ptrs[DType.uint16, DType.int32](
                    gpu_ctx,
                    instance_sort_workspace,
                    instance_keys_buffer.unsafe_ptr(),
                    instance_primitive_indices.unsafe_ptr(),
                    n_instances,
                )

                comptime extract_kernel = extract_instance_ranges_kernel[DType.uint16]
                gpu_ctx.enqueue_function[
                    extract_kernel,
                    extract_kernel,
                ](
                    instance_keys_tensor,
                    tile_instance_ranges_tensor,
                    instance_count_tensor,
                    n_instances,
                    grid_dim=div_round_up(n_instances, 256),
                    block_dim=256,
                )
            else:
                var instance_keys_buffer = gpu_ctx.enqueue_create_buffer[DType.uint32](n_instances)
                var instance_keys_tensor = LayoutTensor[DType.uint32, ROW_MAJOR_1D, MutAnyOrigin](
                    instance_keys_buffer.unsafe_ptr(),
                    RuntimeLayout[
                        ROW_MAJOR_1D,
                        element_type=DType.int32,
                        linear_idx_type=DType.int32,
                    ].row_major(Index(n_instances)),
                )
                comptime fill_kernel = fill_instance_padding_kernel[DType.uint32]
                gpu_ctx.enqueue_function[
                    fill_kernel,
                    fill_kernel,
                ](
                    instance_keys_tensor,
                    instance_primitive_indices_tensor,
                    n_instances,
                    grid_dim=div_round_up(n_instances, 256),
                    block_dim=256,
                )
                comptime expand_kernel = expand_instances_kernel[DType.uint32]
                gpu_ctx.enqueue_function[
                    expand_kernel,
                    expand_kernel,
                ](
                    instance_keys_tensor,
                    instance_primitive_indices_tensor,
                    sorted_primitive_indices_tensor,
                    primitive_offsets_tensor,
                    sorted_touched_counts_tensor,
                    screen_bounds_tensor,
                    projected_means_tensor,
                    conic_opacity_tensor,
                    width_tensor,
                    n_visible_capacity,
                    n_instances,
                    grid_dim=div_round_up(n_visible_capacity, 256),
                    block_dim=256,
                )

                var instance_sort_workspace = RadixSortWorkspace[DType.uint32, DType.int32](
                    gpu_ctx,
                    n_instances,
                )
                device_radix_sort_pairs_ptrs[DType.uint32, DType.int32](
                    gpu_ctx,
                    instance_sort_workspace,
                    instance_keys_buffer.unsafe_ptr(),
                    instance_primitive_indices.unsafe_ptr(),
                    n_instances,
                )

                comptime extract_kernel = extract_instance_ranges_kernel[DType.uint32]
                gpu_ctx.enqueue_function[
                    extract_kernel,
                    extract_kernel,
                ](
                    instance_keys_tensor,
                    tile_instance_ranges_tensor,
                    instance_count_tensor,
                    n_instances,
                    grid_dim=div_round_up(n_instances, 256),
                    block_dim=256,
                )
            gpu_ctx.enqueue_function[
                write_bucket_offsets_kernel,
                write_bucket_offsets_kernel,
            ](
                tile_instance_ranges_tensor,
                tile_bucket_offsets_tensor,
                bucket_count_tensor,
                tile_count,
                grid_dim=1,
                block_dim=1,
            )
            return
        else:
            raise Error("faster_gs_mojo sort_fwd currently requires a GPU target")
