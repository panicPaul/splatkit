# Radix-sort helper adapted from Bajo GPU sort:
# https://github.com/jgsimard/bajo/tree/main/bajo/sort/gpu
# Bajo also has a faster one-sweep variant there; this multi-pass helper is the
# first correctness-oriented integration point for the FasterGS sort stage.

from std.atomic import Atomic, Ordering
from std.bit import count_trailing_zeros, pop_count
from std.gpu import WARP_SIZE, block_dim, block_idx, grid_dim, lane_id, thread_idx, warp_id
from std.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block, warp
from std.gpu.sync import barrier
from std.memory import stack_allocation
from std.sys.info import bit_width_of


comptime RADIX_ORDERING = Ordering.RELAXED
comptime LANE_LOG = count_trailing_zeros(WARP_SIZE)


@fieldwise_init
struct DoubleBuffer[dtype: DType](Copyable):
    var current: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
    var alternate: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    def swap(mut self):
        swap(self.current, self.alternate)


@always_inline
def ceil_div_int(numerator: Int, denominator: Int) -> Int:
    return (numerator + denominator - 1) // denominator


@always_inline
def circular_shift(val: UInt32) -> UInt32:
    comptime WARP_MASK = UInt32(WARP_SIZE - 1)
    var lid = UInt32(lane_id())
    return warp.shuffle_idx(val, (lid + WARP_MASK) & WARP_MASK)


@always_inline
def warp_level_multi_split[
    keys_dtype: DType,
    BITS_PER_PASS: Int,
    KEYS_PER_THREAD: Int,
](
    keys: InlineArray[Scalar[keys_dtype], KEYS_PER_THREAD],
    lid: Int,
    radix_shift: Scalar[keys_dtype],
    s_warp_hist_ptr: UnsafePointer[
        UInt32,
        MutExternalOrigin,
        address_space=AddressSpace.SHARED,
    ],
) -> InlineArray[UInt32, KEYS_PER_THREAD]:
    comptime RADIX = 2**BITS_PER_PASS
    comptime RADIX_MASK = Scalar[keys_dtype](RADIX - 1)
    comptime mask_dtype = DType.uint64 if WARP_SIZE > 32 else DType.uint32
    comptime MaskInt = SIMD[mask_dtype, 1]

    var offsets = InlineArray[UInt32, KEYS_PER_THREAD](uninitialized=True)
    var lane_mask_lt = (MaskInt(1) << MaskInt(lid)) - 1

    comptime for i in range(KEYS_PER_THREAD):
        var warp_flags: MaskInt = ~MaskInt(0)
        var key = keys[i]
        comptime for k in range(BITS_PER_PASS):
            var bit_is_set = ((key >> (radix_shift + Scalar[keys_dtype](k))) & 1) == 1
            var ballot = warp.vote[mask_dtype](bit_is_set)
            var match_mask = ballot if bit_is_set else ~ballot
            warp_flags &= match_mask

        var bits = UInt32(pop_count(warp_flags & lane_mask_lt))
        var pre_increment_val = UInt32(0)
        if bits == 0:
            var digit = Int((key >> radix_shift) & RADIX_MASK)
            var count = UInt32(pop_count(warp_flags))
            pre_increment_val = Atomic.fetch_add[ordering=RADIX_ORDERING](
                s_warp_hist_ptr + digit,
                count,
            )

        var leader_lane = count_trailing_zeros(warp_flags)
        pre_increment_val = warp.shuffle_idx(pre_increment_val, UInt32(leader_lane))
        offsets[i] = pre_increment_val + bits
    return offsets^


@always_inline
def radix_upsweep[
    keys_dtype: DType,
    BLOCK_SIZE: Int,
    RADIX: Int,
    VEC_WIDTH: Int,
    KEYS_PER_THREAD: Int,
](
    keys_current: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: Scalar[keys_dtype],
):
    comptime PART_SIZE = 512 * KEYS_PER_THREAD
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    comptime PADDED_RADIX = RADIX + 1
    comptime RADIX_MASK = Scalar[keys_dtype](RADIX - 1)

    var tid = thread_idx.x
    var bid = block_idx.x
    var gdim = grid_dim.x
    var lid = lane_id()
    var wid = warp_id()

    var s_global_hist = stack_allocation[
        NUM_WARPS * PADDED_RADIX,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()

    for i in range(tid, NUM_WARPS * PADDED_RADIX, BLOCK_SIZE):
        s_global_hist[i] = UInt32(0)
    barrier()

    var s_warp_hist = s_global_hist + (wid * PADDED_RADIX)

    @always_inline
    def bin_keys[width: Int](i: Int) capturing:
        var digits = keys_current.load[width=width](i)
        digits = (digits >> radix_shift) & RADIX_MASK
        comptime for j in range(width):
            _ = Atomic.fetch_add[ordering=RADIX_ORDERING](
                s_warp_hist + Int(digits[j]),
                UInt32(1),
            )

    var block_start = bid * PART_SIZE
    if bid < gdim - 1:
        for i in range(
            block_start + (tid * VEC_WIDTH),
            block_start + PART_SIZE,
            BLOCK_SIZE * VEC_WIDTH,
        ):
            bin_keys[VEC_WIDTH](i)
    else:
        for i in range(block_start + tid, size, BLOCK_SIZE):
            bin_keys[1](i)
    barrier()

    for i in range(tid, RADIX, BLOCK_SIZE):
        var total_val = UInt32(0)
        comptime for w in range(NUM_WARPS):
            total_val += s_global_hist[w * PADDED_RADIX + i]

        pass_hist[i * gdim + bid] = total_val
        var scan_val = warp.prefix_sum[exclusive=False](total_val)
        s_global_hist[i] = circular_shift(scan_val)
    barrier()

    if tid < WARP_SIZE:
        var idx = tid << LANE_LOG
        var val = UInt32(0)
        if tid < (RADIX >> LANE_LOG):
            val = s_global_hist[idx]
        var exclusive_val = warp.prefix_sum[exclusive=True](val)
        if tid < (RADIX >> LANE_LOG):
            s_global_hist[idx] = exclusive_val
    barrier()

    for i in range(tid, RADIX, BLOCK_SIZE):
        var prev_sum = UInt32(0)
        if lid > 0:
            prev_sum = s_global_hist[i - lid]

        var global_idx = i + (Int(radix_shift) << LANE_LOG)
        _ = Atomic.fetch_add[ordering=RADIX_ORDERING](
            global_hist + global_idx,
            s_global_hist[i] + prev_sum,
        )


@always_inline
def radix_scan[BLOCK_SIZE: Int](
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    thread_blocks: Int,
):
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var reduction = UInt32(0)
    var partitions_end = (thread_blocks // BLOCK_SIZE) * BLOCK_SIZE
    var digit_offset = bid * thread_blocks

    var i = tid
    while i < partitions_end:
        var val = pass_hist[i + digit_offset]
        var exclusive = block.prefix_sum[block_size=BLOCK_SIZE, exclusive=True](val)
        var tile_total = block.sum[block_size=BLOCK_SIZE, broadcast=True](val)
        pass_hist[i + digit_offset] = exclusive + reduction
        reduction += tile_total
        i += BLOCK_SIZE

    var val_tail = UInt32(0)
    var has_data = i < thread_blocks
    if has_data:
        val_tail = pass_hist[i + digit_offset]

    var exclusive_tail = block.prefix_sum[
        block_size=BLOCK_SIZE,
        exclusive=True,
    ](val_tail)
    if has_data:
        pass_hist[i + digit_offset] = exclusive_tail + reduction


@always_inline
def radix_downsweep_pairs[
    keys_dtype: DType,
    vals_dtype: DType,
    BITS_PER_PASS: Int,
    BLOCK_SIZE: Int,
    KEYS_PER_THREAD: Int,
](
    keys_current: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    keys_alternate: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    vals_current: UnsafePointer[Scalar[vals_dtype], MutAnyOrigin],
    vals_alternate: UnsafePointer[Scalar[vals_dtype], MutAnyOrigin],
    global_hist: UnsafePointer[UInt32, MutAnyOrigin],
    pass_hist: UnsafePointer[UInt32, MutAnyOrigin],
    size: Int,
    radix_shift: UInt32,
):
    comptime RADIX = 2**BITS_PER_PASS
    comptime RADIX_MASK = Scalar[keys_dtype](RADIX - 1)
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE
    comptime WARP_PART_SIZE = WARP_SIZE * KEYS_PER_THREAD
    comptime PART_SIZE = NUM_WARPS * WARP_PART_SIZE
    comptime TOTAL_WARP_HISTS_SIZE = NUM_WARPS * RADIX

    var s_warp_histograms = stack_allocation[
        PART_SIZE,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()
    var s_local_histogram = stack_allocation[
        RADIX,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()
    var s_keys = stack_allocation[
        PART_SIZE,
        Scalar[keys_dtype],
        address_space=AddressSpace.SHARED,
    ]()
    var s_vals = stack_allocation[
        PART_SIZE,
        Scalar[vals_dtype],
        address_space=AddressSpace.SHARED,
    ]()

    var tid = thread_idx.x
    var bid = block_idx.x
    var gdim = grid_dim.x
    var lid = lane_id()
    var wid = warp_id()
    var s_warp_hist_ptr = s_warp_histograms + (wid << BITS_PER_PASS)

    for i in range(tid, TOTAL_WARP_HISTS_SIZE, BLOCK_SIZE):
        s_warp_histograms[i] = UInt32(0)
    barrier()

    var keys = InlineArray[Scalar[keys_dtype], KEYS_PER_THREAD](uninitialized=True)
    var bin_sub_part_start = wid * WARP_PART_SIZE
    var bin_part_start = bid * PART_SIZE
    var t_base = lid + bin_sub_part_start + bin_part_start

    var t = t_base
    comptime for i in range(KEYS_PER_THREAD):
        if bid < gdim - 1:
            keys[i] = keys_current[t]
        else:
            keys[i] = keys_current[t] if t < size else Scalar[keys_dtype].MAX
        t += WARP_SIZE
    barrier()

    var offsets = warp_level_multi_split[
        keys_dtype,
        BITS_PER_PASS,
        KEYS_PER_THREAD,
    ](keys, lid, Scalar[keys_dtype](radix_shift), s_warp_hist_ptr)
    barrier()

    if tid < RADIX:
        var reduction = s_warp_histograms[tid]
        for i in range(tid + RADIX, TOTAL_WARP_HISTS_SIZE, RADIX):
            reduction += s_warp_histograms[i]
            s_warp_histograms[i] = reduction - s_warp_histograms[i]

        var sum = warp.prefix_sum[exclusive=False](reduction)
        s_warp_histograms[tid] = circular_shift(sum)
    barrier()

    if tid < WARP_SIZE:
        var idx = tid << LANE_LOG
        var val = UInt32(0)
        if tid < (RADIX >> LANE_LOG):
            val = s_warp_histograms[idx]
        val = warp.prefix_sum[exclusive=True](val)
        if tid < (RADIX >> LANE_LOG):
            s_warp_histograms[idx] = val
    barrier()

    if tid < RADIX:
        var prev_sum = UInt32(0)
        if lid > 0:
            prev_sum = s_warp_histograms[tid - lid]
        s_warp_histograms[tid] += prev_sum
    barrier()

    comptime for i in range(KEYS_PER_THREAD):
        var digit = Int(
            (keys[i] >> Scalar[keys_dtype](radix_shift)) & RADIX_MASK
        )
        if wid > 0:
            offsets[i] += (
                s_warp_histograms[wid * RADIX + digit]
                + s_warp_histograms[digit]
            )
        else:
            offsets[i] += s_warp_histograms[digit]

    if tid < RADIX:
        var global_offset = global_hist[tid + (Int(radix_shift) << LANE_LOG)]
        var pass_offset = pass_hist[tid * gdim + bid]
        s_local_histogram[tid] = (
            global_offset + pass_offset - s_warp_histograms[tid]
        )
    barrier()

    comptime for i in range(KEYS_PER_THREAD):
        s_keys[Int(offsets[i])] = keys[i]
    barrier()

    var vals = InlineArray[Scalar[vals_dtype], KEYS_PER_THREAD](uninitialized=True)
    var digits = InlineArray[UInt8, KEYS_PER_THREAD](uninitialized=True)
    if bid < gdim - 1:
        comptime for i in range(KEYS_PER_THREAD):
            var read_idx = tid + (i * BLOCK_SIZE)
            var key = s_keys[read_idx]
            var digit = Int((key >> Scalar[keys_dtype](radix_shift)) & RADIX_MASK)
            digits[i] = UInt8(digit)
            keys_alternate[Int(s_local_histogram[digit]) + read_idx] = key
        barrier()

        var payload_idx = t_base
        comptime for i in range(KEYS_PER_THREAD):
            vals[i] = vals_current[payload_idx]
            payload_idx += WARP_SIZE

        comptime for i in range(KEYS_PER_THREAD):
            s_vals[Int(offsets[i])] = vals[i]
        barrier()

        comptime for i in range(KEYS_PER_THREAD):
            var read_idx = tid + (i * BLOCK_SIZE)
            var digit = Int(digits[i])
            vals_alternate[Int(s_local_histogram[digit]) + read_idx] = s_vals[read_idx]
    else:
        var final_part_size = size - bin_part_start
        comptime for i in range(KEYS_PER_THREAD):
            var read_idx = tid + (i * BLOCK_SIZE)
            if read_idx < final_part_size:
                var key = s_keys[read_idx]
                var digit = Int((key >> Scalar[keys_dtype](radix_shift)) & RADIX_MASK)
                digits[i] = UInt8(digit)
                keys_alternate[Int(s_local_histogram[digit]) + read_idx] = key
        barrier()

        var payload_idx = t_base
        comptime for i in range(KEYS_PER_THREAD):
            if payload_idx < size:
                vals[i] = vals_current[payload_idx]
            payload_idx += WARP_SIZE

        comptime for i in range(KEYS_PER_THREAD):
            s_vals[Int(offsets[i])] = vals[i]
        barrier()

        comptime for i in range(KEYS_PER_THREAD):
            var read_idx = tid + (i * BLOCK_SIZE)
            if read_idx < final_part_size:
                var digit = Int(digits[i])
                vals_alternate[Int(s_local_histogram[digit]) + read_idx] = s_vals[read_idx]


struct RadixSortWorkspace[
    keys_dtype: DType,
    vals_dtype: DType,
    BITS_PER_PASS: Int = 8,
    KEYS_PER_THREAD: Int = 9,
]:
    var keys_alternate: DeviceBuffer[Self.keys_dtype]
    var vals_alternate: DeviceBuffer[Self.vals_dtype]
    var global_hist: DeviceBuffer[DType.uint32]
    var pass_hist: DeviceBuffer[DType.uint32]

    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        comptime NUM_PASSES = bit_width_of[Self.keys_dtype]() // Self.BITS_PER_PASS
        comptime RADIX = 2**Self.BITS_PER_PASS
        comptime GLOBAL_HIST = NUM_PASSES * RADIX
        comptime PART_SIZE = 512 * Self.KEYS_PER_THREAD
        var gdim = ceil_div_int(size, PART_SIZE)

        self.keys_alternate = ctx.enqueue_create_buffer[Self.keys_dtype](size)
        self.vals_alternate = ctx.enqueue_create_buffer[Self.vals_dtype](size)
        self.global_hist = ctx.enqueue_create_buffer[DType.uint32](GLOBAL_HIST)
        self.pass_hist = ctx.enqueue_create_buffer[DType.uint32](gdim * RADIX)


def device_radix_sort_pairs[
    keys_dtype: DType,
    vals_dtype: DType,
    BITS_PER_PASS: Int = 8,
    KEYS_PER_THREAD: Int = 9,
](
    ctx: DeviceContext,
    mut workspace: RadixSortWorkspace[
        keys_dtype,
        vals_dtype,
        BITS_PER_PASS,
        KEYS_PER_THREAD,
    ],
    mut keys: DeviceBuffer[keys_dtype],
    mut values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    device_radix_sort_pairs_ptrs[
        keys_dtype,
        vals_dtype,
        BITS_PER_PASS,
        KEYS_PER_THREAD,
    ](
        ctx,
        workspace,
        keys.unsafe_ptr(),
        values.unsafe_ptr(),
        size,
    )


def device_radix_sort_pairs_ptrs[
    keys_dtype: DType,
    vals_dtype: DType,
    BITS_PER_PASS: Int = 8,
    KEYS_PER_THREAD: Int = 9,
](
    ctx: DeviceContext,
    mut workspace: RadixSortWorkspace[
        keys_dtype,
        vals_dtype,
        BITS_PER_PASS,
        KEYS_PER_THREAD,
    ],
    keys_input: UnsafePointer[Scalar[keys_dtype], MutAnyOrigin],
    values_input: UnsafePointer[Scalar[vals_dtype], MutAnyOrigin],
    size: Int,
) raises:
    comptime NUM_PASSES = bit_width_of[keys_dtype]() // BITS_PER_PASS
    comptime RADIX = 2**BITS_PER_PASS
    comptime VEC_WIDTH = 4
    comptime PART_SIZE = 512 * KEYS_PER_THREAD
    comptime assert NUM_PASSES % 2 == 0, "in-place pointer sort expects even pass count"
    var gdim = ceil_div_int(size, PART_SIZE)

    var keys_current = keys_input
    var keys_alternate = workspace.keys_alternate.unsafe_ptr()
    var vals_current = values_input
    var vals_alternate = workspace.vals_alternate.unsafe_ptr()
    var global_hist = workspace.global_hist.unsafe_ptr()
    var pass_hist = workspace.pass_hist.unsafe_ptr()

    comptime UPSWEEP_BLOCK_SIZE = 256
    comptime SCAN_BLOCK_SIZE = 256
    comptime DOWNSWEEP_BLOCK_SIZE = 512
    comptime upsweep_kernel = radix_upsweep[
        keys_dtype,
        UPSWEEP_BLOCK_SIZE,
        RADIX,
        VEC_WIDTH,
        KEYS_PER_THREAD,
    ]
    comptime scan_kernel = radix_scan[SCAN_BLOCK_SIZE]
    comptime downsweep_kernel = radix_downsweep_pairs[
        keys_dtype,
        vals_dtype,
        BITS_PER_PASS,
        DOWNSWEEP_BLOCK_SIZE,
        KEYS_PER_THREAD,
    ]

    workspace.global_hist.enqueue_fill(0)
    comptime for pass_idx in range(NUM_PASSES):
        var radix_shift = UInt32(pass_idx * BITS_PER_PASS)
        ctx.enqueue_function[upsweep_kernel, upsweep_kernel](
            keys_current,
            global_hist,
            pass_hist,
            size,
            Scalar[keys_dtype](radix_shift),
            grid_dim=gdim,
            block_dim=UPSWEEP_BLOCK_SIZE,
        )
        ctx.enqueue_function[scan_kernel, scan_kernel](
            pass_hist,
            gdim,
            grid_dim=RADIX,
            block_dim=SCAN_BLOCK_SIZE,
        )
        ctx.enqueue_function[downsweep_kernel, downsweep_kernel](
            keys_current,
            keys_alternate,
            vals_current,
            vals_alternate,
            global_hist,
            pass_hist,
            size,
            radix_shift,
            grid_dim=gdim,
            block_dim=DOWNSWEEP_BLOCK_SIZE,
        )
        swap(keys_current, keys_alternate)
        swap(vals_current, vals_alternate)


def device_radix_sort_pairs[
    keys_dtype: DType,
    vals_dtype: DType,
    BITS_PER_PASS: Int = 8,
    KEYS_PER_THREAD: Int = 9,
](
    ctx: DeviceContext,
    mut keys: DeviceBuffer[keys_dtype],
    mut values: DeviceBuffer[vals_dtype],
    size: Int,
) raises:
    var workspace = RadixSortWorkspace[
        keys_dtype,
        vals_dtype,
        BITS_PER_PASS,
        KEYS_PER_THREAD,
    ](ctx, size)
    device_radix_sort_pairs[
        keys_dtype,
        vals_dtype,
        BITS_PER_PASS,
        KEYS_PER_THREAD,
    ](ctx, workspace, keys, values, size)
