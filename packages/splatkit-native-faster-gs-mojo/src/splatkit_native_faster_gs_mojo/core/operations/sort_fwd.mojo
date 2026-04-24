import compiler

from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from std.gpu import block_dim, block_idx, thread_idx
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
    num_touched_tiles: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    n_primitives: Int,
):
    """Prepare `(depth_key, primitive_idx)` pairs, placing culled primitives last."""
    primitive_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if primitive_idx >= n_primitives:
        return

    if rebind[Int32](num_touched_tiles[primitive_idx]) > Int32(0):
        depth_sort_keys[primitive_idx] = UInt32(rebind[Int32](depth_keys[primitive_idx]))
    else:
        depth_sort_keys[primitive_idx] = UInt32(4294967295)
    sorted_primitive_indices[primitive_idx] = Int32(primitive_idx)


@always_inline
def gather_sorted_touched_counts_kernel(
    sorted_primitive_indices: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    num_touched_tiles: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    sorted_touched_counts: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    n_primitives: Int,
):
    """Gather touched-tile counts into depth-sorted primitive order."""
    sorted_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if sorted_idx >= n_primitives:
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
def expand_instances_kernel(
    instance_keys: LayoutTensor[DType.uint64, ROW_MAJOR_1D, MutAnyOrigin],
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
    """Expand each visible primitive into tile/depth-order instance keys."""
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
                tile_id = UInt64(tile_y * grid_width + tile_x)
                depth_order = UInt64(sorted_idx)
                instance_keys[write_offset] = (tile_id << 32) | depth_order
                instance_primitive_indices[write_offset] = Int32(primitive_idx)
                write_offset += 1


@always_inline
def fill_instance_padding_kernel(
    instance_keys: LayoutTensor[DType.uint64, ROW_MAJOR_1D, MutAnyOrigin],
    instance_primitive_indices: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    instance_capacity: Int,
):
    """Initialize unused instance slots so radix sort keeps them at the end."""
    idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx < instance_capacity:
        instance_keys[idx] = UInt64(18446744073709551615)
        instance_primitive_indices[idx] = Int32(0)


@always_inline
def extract_instance_ranges_u64_kernel(
    instance_keys: LayoutTensor[DType.uint64, ROW_MAJOR_1D, MutAnyOrigin],
    tile_instance_ranges: LayoutTensor[DType.int32, ROW_MAJOR_2D, MutAnyOrigin],
    instance_count: LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin],
    instance_capacity: Int,
):
    """Recover `[start, end)` ranges for each tile from sorted instance keys."""
    instance_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    actual_instance_count = Int(rebind[Int32](instance_count[0]))
    if instance_idx >= actual_instance_count or instance_idx >= instance_capacity:
        return

    current_tile = Int(rebind[UInt64](instance_keys[instance_idx]) >> 32)
    if instance_idx == 0:
        tile_instance_ranges[current_tile, 0] = Int32(0)
    else:
        previous_tile = Int(rebind[UInt64](instance_keys[instance_idx - 1]) >> 32)
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
            tile_count = Int(tile_instance_ranges.dim_size[0]())
            n_instances = Int(instance_primitive_indices.dim_size[0]())
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

            if n_primitives == 0 or n_instances == 0:
                return

            var depth_sort_keys_buffer = gpu_ctx.enqueue_create_buffer[DType.uint32](n_primitives)
            var sorted_primitive_indices_buffer = gpu_ctx.enqueue_create_buffer[DType.int32](n_primitives)
            var sorted_touched_counts_buffer = gpu_ctx.enqueue_create_buffer[DType.int32](n_primitives)
            var primitive_offsets_buffer = gpu_ctx.enqueue_create_buffer[DType.int32](n_primitives)
            var instance_keys_buffer = gpu_ctx.enqueue_create_buffer[DType.uint64](n_instances)

            var depth_sort_keys_tensor = LayoutTensor[DType.uint32, ROW_MAJOR_1D, MutAnyOrigin](
                depth_sort_keys_buffer.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(n_primitives)),
            )
            var sorted_primitive_indices_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                sorted_primitive_indices_buffer.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(n_primitives)),
            )
            var sorted_touched_counts_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                sorted_touched_counts_buffer.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(n_primitives)),
            )
            var primitive_offsets_tensor = LayoutTensor[DType.int32, ROW_MAJOR_1D, MutAnyOrigin](
                primitive_offsets_buffer.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(n_primitives)),
            )
            var instance_keys_tensor = LayoutTensor[DType.uint64, ROW_MAJOR_1D, MutAnyOrigin](
                instance_keys_buffer.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(n_instances)),
            )

            gpu_ctx.enqueue_function[
                fill_instance_padding_kernel,
                fill_instance_padding_kernel,
            ](
                instance_keys_tensor,
                instance_primitive_indices_tensor,
                n_instances,
                grid_dim=div_round_up(n_instances, 256),
                block_dim=256,
            )
            gpu_ctx.enqueue_function[
                prepare_depth_sort_pairs_kernel,
                prepare_depth_sort_pairs_kernel,
            ](
                depth_sort_keys_tensor,
                sorted_primitive_indices_tensor,
                depth_keys_tensor,
                num_touched_tiles_tensor,
                n_primitives,
                grid_dim=div_round_up(n_primitives, 256),
                block_dim=256,
            )

            var depth_sort_workspace = RadixSortWorkspace[DType.uint32, DType.int32](
                gpu_ctx,
                n_primitives,
            )
            device_radix_sort_pairs[DType.uint32, DType.int32](
                gpu_ctx,
                depth_sort_workspace,
                depth_sort_keys_buffer,
                sorted_primitive_indices_buffer,
                n_primitives,
            )

            gpu_ctx.enqueue_function[
                gather_sorted_touched_counts_kernel,
                gather_sorted_touched_counts_kernel,
            ](
                sorted_primitive_indices_tensor,
                num_touched_tiles_tensor,
                sorted_touched_counts_tensor,
                n_primitives,
                grid_dim=div_round_up(n_primitives, 256),
                block_dim=256,
            )
            gpu_ctx.enqueue_function[
                exclusive_prefix_sum_i32_kernel,
                exclusive_prefix_sum_i32_kernel,
            ](
                sorted_touched_counts_tensor,
                primitive_offsets_tensor,
                n_primitives,
                grid_dim=1,
                block_dim=1,
            )
            gpu_ctx.enqueue_function[
                expand_instances_kernel,
                expand_instances_kernel,
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
                n_primitives,
                n_instances,
                grid_dim=div_round_up(n_primitives, 256),
                block_dim=256,
            )

            var instance_sort_workspace = RadixSortWorkspace[DType.uint64, DType.int32](
                gpu_ctx,
                n_instances,
            )
            device_radix_sort_pairs_ptrs[DType.uint64, DType.int32](
                gpu_ctx,
                instance_sort_workspace,
                instance_keys_buffer.unsafe_ptr(),
                instance_primitive_indices.unsafe_ptr(),
                n_instances,
            )

            gpu_ctx.enqueue_function[
                extract_instance_ranges_u64_kernel,
                extract_instance_ranges_u64_kernel,
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
