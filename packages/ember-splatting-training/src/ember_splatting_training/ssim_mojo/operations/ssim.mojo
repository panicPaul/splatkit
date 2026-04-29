import compiler

from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from std.gpu import block_idx, thread_idx
from std.gpu.memory import AddressSpace
from std.gpu.sync import barrier
from std.runtime.asyncrt import DeviceContextPtr
from std.utils import Index
from tensor import InputTensor, ManagedTensorSlice, MutableInput, OutputTensor


comptime BLOCK_X = 16
comptime BLOCK_Y = 16
comptime HALO = 5
comptime SHARED_X = BLOCK_X + 2 * HALO
comptime SHARED_Y = BLOCK_Y + 2 * HALO
comptime CONV_X = BLOCK_X
comptime CONV_Y = SHARED_Y
comptime HORIZONTAL_PASSES = (SHARED_Y + BLOCK_Y - 1) // BLOCK_Y
comptime SSIM_C1 = Float32(0.0001)
comptime SSIM_C2 = Float32(0.0009)

comptime ROW_MAJOR_1D = Layout.row_major(Int())
comptime ROW_MAJOR_4D = Layout.row_major(Int(), Int(), Int(), Int())
comptime SHARED_TILE = Layout.row_major(SHARED_Y, SHARED_X, 2)
comptime SHARED_XCONV = Layout.row_major(CONV_Y, CONV_X, 5)
comptime SHARED_BWD_DATA = Layout.row_major(3, SHARED_Y, SHARED_X)
comptime SHARED_BWD_SCRATCH = Layout.row_major(CONV_Y, CONV_X, 3)
comptime MutableInputTensor = ManagedTensorSlice[MutableInput, static_spec=...]


@always_inline
def div_round_up(numerator: Int, denominator: Int) -> Int:
    """Return ceil(numerator / denominator) for positive integer inputs."""
    return (numerator + denominator - 1) // denominator


@always_inline
def gauss_weight(kernel_index: Int) -> Float32:
    """Return one coefficient from fused-ssim's fixed 11-tap Gaussian kernel."""
    if kernel_index == 0:
        return Float32(0.001028380123898387)
    if kernel_index == 1:
        return Float32(0.0075987582094967365)
    if kernel_index == 2:
        return Float32(0.036000773310661316)
    if kernel_index == 3:
        return Float32(0.10936068743467331)
    if kernel_index == 4:
        return Float32(0.21300552785396576)
    if kernel_index == 5:
        return Float32(0.26601171493530273)
    if kernel_index == 6:
        return Float32(0.21300552785396576)
    if kernel_index == 7:
        return Float32(0.10936068743467331)
    if kernel_index == 8:
        return Float32(0.036000773310661316)
    if kernel_index == 9:
        return Float32(0.0075987582094967365)
    return Float32(0.001028380123898387)


@always_inline
def read_image(
    image: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    batch_index: Int,
    channel_index: Int,
    image_y: Int,
    image_x: Int,
    image_height: Int,
    image_width: Int,
) -> Float32:
    """Read an NCHW image pixel with zero padding outside the image extent."""
    if (
        image_x < 0
        or image_x >= image_width
        or image_y < 0
        or image_y >= image_height
    ):
        return Float32(0.0)
    return rebind[Float32](image[batch_index, channel_index, image_y, image_x])


@always_inline
def compute_horizontal_blur_row(
    staged_pixels: LayoutTensor[
        DType.float32,
        SHARED_TILE,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    horizontal_blur: LayoutTensor[
        DType.float32,
        SHARED_XCONV,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    shared_y: Int,
    shared_x: Int,
    output_x: Int,
):
    """Compute one horizontal 11-tap Gaussian row for all SSIM moments."""
    var horizontal_prediction_mean = Float32(0.0)
    var horizontal_prediction_second_moment = Float32(0.0)
    var horizontal_target_mean = Float32(0.0)
    var horizontal_target_second_moment = Float32(0.0)
    var horizontal_cross_moment = Float32(0.0)
    comptime for kernel_offset in range(1, HALO + 1):
        var weight = gauss_weight(HALO - kernel_offset)
        var prediction_left = rebind[Float32](
            staged_pixels[shared_y, shared_x - kernel_offset, 0]
        )
        var target_left = rebind[Float32](
            staged_pixels[shared_y, shared_x - kernel_offset, 1]
        )
        var prediction_right = rebind[Float32](
            staged_pixels[shared_y, shared_x + kernel_offset, 0]
        )
        var target_right = rebind[Float32](
            staged_pixels[shared_y, shared_x + kernel_offset, 1]
        )
        horizontal_prediction_mean += (prediction_left + prediction_right) * weight
        horizontal_prediction_second_moment += (
            prediction_left * prediction_left + prediction_right * prediction_right
        ) * weight
        horizontal_target_mean += (target_left + target_right) * weight
        horizontal_target_second_moment += (
            target_left * target_left + target_right * target_right
        ) * weight
        horizontal_cross_moment += (
            prediction_left * target_left + prediction_right * target_right
        ) * weight

    var center_weight = gauss_weight(HALO)
    var center_prediction = rebind[Float32](staged_pixels[shared_y, shared_x, 0])
    var center_target = rebind[Float32](staged_pixels[shared_y, shared_x, 1])
    horizontal_prediction_mean += center_prediction * center_weight
    horizontal_prediction_second_moment += (
        center_prediction * center_prediction * center_weight
    )
    horizontal_target_mean += center_target * center_weight
    horizontal_target_second_moment += center_target * center_target * center_weight
    horizontal_cross_moment += center_prediction * center_target * center_weight

    horizontal_blur[shared_y, output_x, 0] = horizontal_prediction_mean
    horizontal_blur[shared_y, output_x, 1] = horizontal_prediction_second_moment
    horizontal_blur[shared_y, output_x, 2] = horizontal_target_mean
    horizontal_blur[shared_y, output_x, 3] = horizontal_target_second_moment
    horizontal_blur[shared_y, output_x, 4] = horizontal_cross_moment


@always_inline
def ssim_fwd_kernel(
    ssim_map: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    dm_dmu1: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    dm_dsigma1_sq: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    dm_dsigma12: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    prediction_image: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    target_image: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    channels: Int,
    height: Int,
    width: Int,
):
    """Compute SSIM map and the derivative caches needed by backward.

    One CUDA block owns a 16x16 output tile for one batch image. The block
    loops over channels internally, matching fused-ssim's launch shape so each
    image tile reuses the same block structure across RGB planes.
    """
    var batch_index = Int(block_idx.z)
    var pixel_y = Int(block_idx.y) * BLOCK_Y + Int(thread_idx.y)
    var pixel_x = Int(block_idx.x) * BLOCK_X + Int(thread_idx.x)
    var local_thread = Int(thread_idx.y) * BLOCK_X + Int(thread_idx.x)

    var staged_pixels = LayoutTensor[
        DType.float32,
        SHARED_TILE,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var horizontal_blur = LayoutTensor[
        DType.float32,
        SHARED_XCONV,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    for channel_index in range(channels):
        var tile_start_y = Int(block_idx.y) * BLOCK_Y
        var tile_start_x = Int(block_idx.x) * BLOCK_X
        var shared_pixel_count = SHARED_Y * SHARED_X
        var load_steps = div_round_up(shared_pixel_count, BLOCK_X * BLOCK_Y)
        for load_step in range(load_steps):
            var shared_linear_index = load_step * BLOCK_X * BLOCK_Y + local_thread
            if shared_linear_index < shared_pixel_count:
                var shared_y = shared_linear_index // SHARED_X
                var shared_x = shared_linear_index - shared_y * SHARED_X
                var image_y = tile_start_y + shared_y - HALO
                var image_x = tile_start_x + shared_x - HALO
                staged_pixels[shared_y, shared_x, 0] = read_image(
                    prediction_image,
                    batch_index,
                    channel_index,
                    image_y,
                    image_x,
                    height,
                    width,
                )
                staged_pixels[shared_y, shared_x, 1] = read_image(
                    target_image,
                    batch_index,
                    channel_index,
                    image_y,
                    image_x,
                    height,
                    width,
                )
        barrier()

        var shared_x = Int(thread_idx.x) + HALO
        var output_x = Int(thread_idx.x)
        var first_shared_y = Int(thread_idx.y)
        compute_horizontal_blur_row(
            staged_pixels, horizontal_blur, first_shared_y, shared_x, output_x
        )
        var second_shared_y = first_shared_y + BLOCK_Y
        if second_shared_y < CONV_Y:
            compute_horizontal_blur_row(
                staged_pixels, horizontal_blur, second_shared_y, shared_x, output_x
            )
        barrier()

        if pixel_x < width and pixel_y < height:
            var shared_y = Int(thread_idx.y) + HALO
            var vertical_prediction_mean = Float32(0.0)
            var vertical_prediction_second_moment = Float32(0.0)
            var vertical_target_mean = Float32(0.0)
            var vertical_target_second_moment = Float32(0.0)
            var vertical_cross_moment = Float32(0.0)
            comptime for kernel_offset in range(1, HALO + 1):
                var weight = gauss_weight(HALO - kernel_offset)
                vertical_prediction_mean += (
                    rebind[Float32](
                        horizontal_blur[shared_y - kernel_offset, output_x, 0]
                    )
                    + rebind[Float32](
                        horizontal_blur[shared_y + kernel_offset, output_x, 0]
                    )
                ) * weight
                vertical_prediction_second_moment += (
                    rebind[Float32](
                        horizontal_blur[shared_y - kernel_offset, output_x, 1]
                    )
                    + rebind[Float32](
                        horizontal_blur[shared_y + kernel_offset, output_x, 1]
                    )
                ) * weight
                vertical_target_mean += (
                    rebind[Float32](
                        horizontal_blur[shared_y - kernel_offset, output_x, 2]
                    )
                    + rebind[Float32](
                        horizontal_blur[shared_y + kernel_offset, output_x, 2]
                    )
                ) * weight
                vertical_target_second_moment += (
                    rebind[Float32](
                        horizontal_blur[shared_y - kernel_offset, output_x, 3]
                    )
                    + rebind[Float32](
                        horizontal_blur[shared_y + kernel_offset, output_x, 3]
                    )
                ) * weight
                vertical_cross_moment += (
                    rebind[Float32](
                        horizontal_blur[shared_y - kernel_offset, output_x, 4]
                    )
                    + rebind[Float32](
                        horizontal_blur[shared_y + kernel_offset, output_x, 4]
                    )
                ) * weight
            var center_weight = gauss_weight(HALO)
            vertical_prediction_mean += (
                rebind[Float32](horizontal_blur[shared_y, output_x, 0])
                * center_weight
            )
            vertical_prediction_second_moment += (
                rebind[Float32](horizontal_blur[shared_y, output_x, 1])
                * center_weight
            )
            vertical_target_mean += (
                rebind[Float32](horizontal_blur[shared_y, output_x, 2])
                * center_weight
            )
            vertical_target_second_moment += (
                rebind[Float32](horizontal_blur[shared_y, output_x, 3])
                * center_weight
            )
            vertical_cross_moment += (
                rebind[Float32](horizontal_blur[shared_y, output_x, 4])
                * center_weight
            )

            var prediction_mean = vertical_prediction_mean
            var target_mean = vertical_target_mean
            var prediction_mean_squared = prediction_mean * prediction_mean
            var target_mean_squared = target_mean * target_mean
            var prediction_variance = vertical_prediction_second_moment - prediction_mean_squared
            var target_variance = vertical_target_second_moment - target_mean_squared
            var prediction_target_covariance = vertical_cross_moment - prediction_mean * target_mean
            var luminance_denominator = (
                prediction_mean_squared + target_mean_squared + SSIM_C1
            )
            var contrast_denominator = (
                prediction_variance + target_variance + SSIM_C2
            )
            var luminance_numerator = (
                Float32(2.0) * prediction_mean * target_mean + SSIM_C1
            )
            var contrast_numerator = (
                Float32(2.0) * prediction_target_covariance + SSIM_C2
            )
            var denominator_product = luminance_denominator * contrast_denominator
            var numerator_product = luminance_numerator * contrast_numerator
            var ssim_value = numerator_product / denominator_product
            ssim_map[batch_index, channel_index, pixel_y, pixel_x] = ssim_value
            var double_target_mean = target_mean * Float32(2.0)
            var double_prediction_mean = prediction_mean * Float32(2.0)
            var luminance_squared_product = (
                luminance_denominator * luminance_denominator * contrast_denominator
            )
            var contrast_squared_product = (
                luminance_denominator * contrast_denominator * contrast_denominator
            )
            dm_dmu1[batch_index, channel_index, pixel_y, pixel_x] = (
                (double_target_mean * contrast_numerator) / denominator_product
                - (double_target_mean * luminance_numerator) / denominator_product
                - (double_prediction_mean * numerator_product) / luminance_squared_product
                + (double_prediction_mean * numerator_product) / contrast_squared_product
            )
            dm_dsigma1_sq[batch_index, channel_index, pixel_y, pixel_x] = (
                -numerator_product / contrast_squared_product
            )
            dm_dsigma12[batch_index, channel_index, pixel_y, pixel_x] = (
                (Float32(2.0) * luminance_numerator) / denominator_product
            )
        barrier()


@always_inline
def ssim_bwd_kernel(
    grad_prediction_image: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    prediction_image: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    target_image: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    grad_map: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    dm_dmu1: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    dm_dsigma1_sq: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    dm_dsigma12: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    channels: Int,
    height: Int,
    width: Int,
):
    """Backpropagate an arbitrary SSIM-map gradient into the prediction image.

    The backward pass mirrors the two separable Gaussian reductions from the
    forward pass. It first stages the local derivatives multiplied by the
    upstream per-pixel map gradient, then blurs those derivative fields to
    accumulate every SSIM-window contribution that touches the current pixel.
    """
    var batch_index = Int(block_idx.z)
    var pixel_y = Int(block_idx.y) * BLOCK_Y + Int(thread_idx.y)
    var pixel_x = Int(block_idx.x) * BLOCK_X + Int(thread_idx.x)
    var local_thread = Int(thread_idx.y) * BLOCK_X + Int(thread_idx.x)

    var staged_derivatives = LayoutTensor[
        DType.float32,
        SHARED_BWD_DATA,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var horizontal_derivative_blur = LayoutTensor[
        DType.float32,
        SHARED_BWD_SCRATCH,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    for channel_index in range(channels):
        var prediction_pixel = read_image(
            prediction_image, batch_index, channel_index, pixel_y, pixel_x, height, width
        )
        var target_pixel = read_image(
            target_image, batch_index, channel_index, pixel_y, pixel_x, height, width
        )
        var tile_start_y = Int(block_idx.y) * BLOCK_Y
        var tile_start_x = Int(block_idx.x) * BLOCK_X
        var tile_size = SHARED_Y * SHARED_X
        var steps = div_round_up(tile_size, BLOCK_X * BLOCK_Y)
        for load_step in range(steps):
            var shared_linear_index = load_step * BLOCK_X * BLOCK_Y + local_thread
            if shared_linear_index < tile_size:
                var shared_y = shared_linear_index // SHARED_X
                var shared_x = shared_linear_index - shared_y * SHARED_X
                var image_y = tile_start_y + shared_y - HALO
                var image_x = tile_start_x + shared_x - HALO
                var upstream_map_gradient = read_image(
                    grad_map, batch_index, channel_index, image_y, image_x, height, width
                )
                staged_derivatives[0, shared_y, shared_x] = (
                    read_image(
                        dm_dmu1,
                        batch_index,
                        channel_index,
                        image_y,
                        image_x,
                        height,
                        width,
                    )
                    * upstream_map_gradient
                )
                staged_derivatives[1, shared_y, shared_x] = (
                    read_image(
                        dm_dsigma1_sq,
                        batch_index,
                        channel_index,
                        image_y,
                        image_x,
                        height,
                        width,
                    )
                    * upstream_map_gradient
                )
                staged_derivatives[2, shared_y, shared_x] = (
                    read_image(
                        dm_dsigma12,
                        batch_index,
                        channel_index,
                        image_y,
                        image_x,
                        height,
                        width,
                    )
                    * upstream_map_gradient
                )
        barrier()

        var shared_x = Int(thread_idx.x) + HALO
        for pass_idx in range(HORIZONTAL_PASSES):
            var shared_y = Int(thread_idx.y) + pass_idx * BLOCK_Y
            if shared_y < CONV_Y:
                var horizontal_dmu1_sum = Float32(0.0)
                var horizontal_dvariance_sum = Float32(0.0)
                var horizontal_dcovariance_sum = Float32(0.0)
                comptime for kernel_offset in range(1, HALO + 1):
                    var weight = gauss_weight(HALO - kernel_offset)
                    horizontal_dmu1_sum += (
                        rebind[Float32](
                            staged_derivatives[0, shared_y, shared_x - kernel_offset]
                        )
                        + rebind[Float32](
                            staged_derivatives[0, shared_y, shared_x + kernel_offset]
                        )
                    ) * weight
                    horizontal_dvariance_sum += (
                        rebind[Float32](
                            staged_derivatives[1, shared_y, shared_x - kernel_offset]
                        )
                        + rebind[Float32](
                            staged_derivatives[1, shared_y, shared_x + kernel_offset]
                        )
                    ) * weight
                    horizontal_dcovariance_sum += (
                        rebind[Float32](
                            staged_derivatives[2, shared_y, shared_x - kernel_offset]
                        )
                        + rebind[Float32](
                            staged_derivatives[2, shared_y, shared_x + kernel_offset]
                        )
                    ) * weight
                var center_weight = gauss_weight(HALO)
                horizontal_dmu1_sum += (
                    rebind[Float32](staged_derivatives[0, shared_y, shared_x])
                    * center_weight
                )
                horizontal_dvariance_sum += (
                    rebind[Float32](staged_derivatives[1, shared_y, shared_x])
                    * center_weight
                )
                horizontal_dcovariance_sum += (
                    rebind[Float32](staged_derivatives[2, shared_y, shared_x])
                    * center_weight
                )
                horizontal_derivative_blur[
                    shared_y, Int(thread_idx.x), 0
                ] = horizontal_dmu1_sum
                horizontal_derivative_blur[
                    shared_y, Int(thread_idx.x), 1
                ] = horizontal_dvariance_sum
                horizontal_derivative_blur[
                    shared_y, Int(thread_idx.x), 2
                ] = horizontal_dcovariance_sum
        barrier()

        if pixel_x < width and pixel_y < height:
            var shared_y = Int(thread_idx.y) + HALO
            var output_x = Int(thread_idx.x)
            var blurred_dmu1 = Float32(0.0)
            var blurred_dvariance = Float32(0.0)
            var blurred_dcovariance = Float32(0.0)
            comptime for kernel_offset in range(1, HALO + 1):
                var weight = gauss_weight(HALO - kernel_offset)
                blurred_dmu1 += (
                    rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y - kernel_offset, output_x, 0
                        ]
                    )
                    + rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y + kernel_offset, output_x, 0
                        ]
                    )
                ) * weight
                blurred_dvariance += (
                    rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y - kernel_offset, output_x, 1
                        ]
                    )
                    + rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y + kernel_offset, output_x, 1
                        ]
                    )
                ) * weight
                blurred_dcovariance += (
                    rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y - kernel_offset, output_x, 2
                        ]
                    )
                    + rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y + kernel_offset, output_x, 2
                        ]
                    )
                ) * weight
            var center_weight = gauss_weight(HALO)
            blurred_dmu1 += (
                rebind[Float32](horizontal_derivative_blur[shared_y, output_x, 0])
                * center_weight
            )
            blurred_dvariance += (
                rebind[Float32](horizontal_derivative_blur[shared_y, output_x, 1])
                * center_weight
            )
            blurred_dcovariance += (
                rebind[Float32](horizontal_derivative_blur[shared_y, output_x, 2])
                * center_weight
            )
            grad_prediction_image[batch_index, channel_index, pixel_y, pixel_x] = (
                blurred_dmu1
                + Float32(2.0) * prediction_pixel * blurred_dvariance
                + target_pixel * blurred_dcovariance
            )
        barrier()


@always_inline
def ssim_bwd_mean_kernel(
    grad_prediction_image: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    prediction_image: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    target_image: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    grad_scale: LayoutTensor[DType.float32, ROW_MAJOR_1D, MutAnyOrigin],
    dm_dmu1: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    dm_dsigma1_sq: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    dm_dsigma12: LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin],
    channels: Int,
    height: Int,
    width: Int,
):
    """Backpropagate the scalar mean SSIM loss into the prediction image.

    This variant avoids materializing a full upstream gradient map. The Python
    autograd registration passes one scalar equal to grad_output / numel, and
    every staged derivative is scaled by that value before the same separable
    Gaussian accumulation used by the general backward kernel.
    """
    var batch_index = Int(block_idx.z)
    var pixel_y = Int(block_idx.y) * BLOCK_Y + Int(thread_idx.y)
    var pixel_x = Int(block_idx.x) * BLOCK_X + Int(thread_idx.x)
    var local_thread = Int(thread_idx.y) * BLOCK_X + Int(thread_idx.x)
    var uniform_upstream_scale = rebind[Float32](grad_scale[0])

    var staged_derivatives = LayoutTensor[
        DType.float32,
        SHARED_BWD_DATA,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var horizontal_derivative_blur = LayoutTensor[
        DType.float32,
        SHARED_BWD_SCRATCH,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    for channel_index in range(channels):
        var prediction_pixel = read_image(
            prediction_image, batch_index, channel_index, pixel_y, pixel_x, height, width
        )
        var target_pixel = read_image(
            target_image, batch_index, channel_index, pixel_y, pixel_x, height, width
        )
        var tile_start_y = Int(block_idx.y) * BLOCK_Y
        var tile_start_x = Int(block_idx.x) * BLOCK_X
        var tile_size = SHARED_Y * SHARED_X
        var steps = div_round_up(tile_size, BLOCK_X * BLOCK_Y)
        for load_step in range(steps):
            var shared_linear_index = load_step * BLOCK_X * BLOCK_Y + local_thread
            if shared_linear_index < tile_size:
                var shared_y = shared_linear_index // SHARED_X
                var shared_x = shared_linear_index - shared_y * SHARED_X
                var image_y = tile_start_y + shared_y - HALO
                var image_x = tile_start_x + shared_x - HALO
                staged_derivatives[0, shared_y, shared_x] = (
                    read_image(
                        dm_dmu1,
                        batch_index,
                        channel_index,
                        image_y,
                        image_x,
                        height,
                        width,
                    )
                    * uniform_upstream_scale
                )
                staged_derivatives[1, shared_y, shared_x] = (
                    read_image(
                        dm_dsigma1_sq,
                        batch_index,
                        channel_index,
                        image_y,
                        image_x,
                        height,
                        width,
                    )
                    * uniform_upstream_scale
                )
                staged_derivatives[2, shared_y, shared_x] = (
                    read_image(
                        dm_dsigma12,
                        batch_index,
                        channel_index,
                        image_y,
                        image_x,
                        height,
                        width,
                    )
                    * uniform_upstream_scale
                )
        barrier()

        var shared_x = Int(thread_idx.x) + HALO
        for pass_idx in range(HORIZONTAL_PASSES):
            var shared_y = Int(thread_idx.y) + pass_idx * BLOCK_Y
            if shared_y < CONV_Y:
                var horizontal_dmu1_sum = Float32(0.0)
                var horizontal_dvariance_sum = Float32(0.0)
                var horizontal_dcovariance_sum = Float32(0.0)
                comptime for kernel_offset in range(1, HALO + 1):
                    var weight = gauss_weight(HALO - kernel_offset)
                    horizontal_dmu1_sum += (
                        rebind[Float32](
                            staged_derivatives[0, shared_y, shared_x - kernel_offset]
                        )
                        + rebind[Float32](
                            staged_derivatives[0, shared_y, shared_x + kernel_offset]
                        )
                    ) * weight
                    horizontal_dvariance_sum += (
                        rebind[Float32](
                            staged_derivatives[1, shared_y, shared_x - kernel_offset]
                        )
                        + rebind[Float32](
                            staged_derivatives[1, shared_y, shared_x + kernel_offset]
                        )
                    ) * weight
                    horizontal_dcovariance_sum += (
                        rebind[Float32](
                            staged_derivatives[2, shared_y, shared_x - kernel_offset]
                        )
                        + rebind[Float32](
                            staged_derivatives[2, shared_y, shared_x + kernel_offset]
                        )
                    ) * weight
                var center_weight = gauss_weight(HALO)
                horizontal_dmu1_sum += (
                    rebind[Float32](staged_derivatives[0, shared_y, shared_x])
                    * center_weight
                )
                horizontal_dvariance_sum += (
                    rebind[Float32](staged_derivatives[1, shared_y, shared_x])
                    * center_weight
                )
                horizontal_dcovariance_sum += (
                    rebind[Float32](staged_derivatives[2, shared_y, shared_x])
                    * center_weight
                )
                horizontal_derivative_blur[
                    shared_y, Int(thread_idx.x), 0
                ] = horizontal_dmu1_sum
                horizontal_derivative_blur[
                    shared_y, Int(thread_idx.x), 1
                ] = horizontal_dvariance_sum
                horizontal_derivative_blur[
                    shared_y, Int(thread_idx.x), 2
                ] = horizontal_dcovariance_sum
        barrier()

        if pixel_x < width and pixel_y < height:
            var shared_y = Int(thread_idx.y) + HALO
            var output_x = Int(thread_idx.x)
            var blurred_dmu1 = Float32(0.0)
            var blurred_dvariance = Float32(0.0)
            var blurred_dcovariance = Float32(0.0)
            comptime for kernel_offset in range(1, HALO + 1):
                var weight = gauss_weight(HALO - kernel_offset)
                blurred_dmu1 += (
                    rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y - kernel_offset, output_x, 0
                        ]
                    )
                    + rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y + kernel_offset, output_x, 0
                        ]
                    )
                ) * weight
                blurred_dvariance += (
                    rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y - kernel_offset, output_x, 1
                        ]
                    )
                    + rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y + kernel_offset, output_x, 1
                        ]
                    )
                ) * weight
                blurred_dcovariance += (
                    rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y - kernel_offset, output_x, 2
                        ]
                    )
                    + rebind[Float32](
                        horizontal_derivative_blur[
                            shared_y + kernel_offset, output_x, 2
                        ]
                    )
                ) * weight
            var center_weight = gauss_weight(HALO)
            blurred_dmu1 += (
                rebind[Float32](horizontal_derivative_blur[shared_y, output_x, 0])
                * center_weight
            )
            blurred_dvariance += (
                rebind[Float32](horizontal_derivative_blur[shared_y, output_x, 1])
                * center_weight
            )
            blurred_dcovariance += (
                rebind[Float32](horizontal_derivative_blur[shared_y, output_x, 2])
                * center_weight
            )
            grad_prediction_image[batch_index, channel_index, pixel_y, pixel_x] = (
                blurred_dmu1
                + Float32(2.0) * prediction_pixel * blurred_dvariance
                + target_pixel * blurred_dcovariance
            )
        barrier()


@compiler.register("ssim_fwd")
struct SSIMForward:
    """MAX custom-op entrypoint for SSIM map and derivative-cache creation."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        ssim_map: OutputTensor[dtype=DType.float32, rank=4, ...],
        dm_dmu1: OutputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma1_sq: OutputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma12: OutputTensor[dtype=DType.float32, rank=4, ...],
        prediction_image: InputTensor[dtype=DType.float32, rank=4, ...],
        target_image: InputTensor[dtype=DType.float32, rank=4, ...],
        ctx: DeviceContextPtr,
    ) raises:
        """Launch the GPU forward kernel for one contiguous NCHW tensor pair."""
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var batch = Int(prediction_image.dim_size[0]())
            var channels = Int(prediction_image.dim_size[1]())
            var height = Int(prediction_image.dim_size[2]())
            var width = Int(prediction_image.dim_size[3]())
            var prediction_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                prediction_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var target_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                target_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var ssim_map_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                ssim_map.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dmu1_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dmu1.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma1_sq_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma1_sq.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma12_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma12.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            gpu_ctx.enqueue_function[ssim_fwd_kernel, ssim_fwd_kernel](
                ssim_map_tensor,
                dm_dmu1_tensor,
                dm_dsigma1_sq_tensor,
                dm_dsigma12_tensor,
                prediction_tensor,
                target_tensor,
                channels,
                height,
                width,
                grid_dim=(
                    div_round_up(width, BLOCK_X),
                    div_round_up(height, BLOCK_Y),
                    batch,
                ),
                block_dim=(BLOCK_X, BLOCK_Y),
            )
        else:
            raise Error("ssim_mojo forward currently requires a GPU target")


@compiler.register("ssim_bwd")
struct SSIMBackward:
    """MAX custom-op entrypoint for full SSIM-map backward propagation."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        grad_prediction_image: OutputTensor[dtype=DType.float32, rank=4, ...],
        prediction_image: InputTensor[dtype=DType.float32, rank=4, ...],
        target_image: InputTensor[dtype=DType.float32, rank=4, ...],
        grad_map: InputTensor[dtype=DType.float32, rank=4, ...],
        dm_dmu1: InputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma1_sq: InputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma12: InputTensor[dtype=DType.float32, rank=4, ...],
        ctx: DeviceContextPtr,
    ) raises:
        """Launch the GPU backward kernel for a materialized SSIM gradient map."""
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var batch = Int(prediction_image.dim_size[0]())
            var channels = Int(prediction_image.dim_size[1]())
            var height = Int(prediction_image.dim_size[2]())
            var width = Int(prediction_image.dim_size[3]())
            var grad_prediction_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                grad_prediction_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var prediction_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                prediction_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var target_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                target_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var grad_map_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                grad_map.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dmu1_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                dm_dmu1.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma1_sq_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma1_sq.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma12_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma12.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            gpu_ctx.enqueue_function[ssim_bwd_kernel, ssim_bwd_kernel](
                grad_prediction_tensor,
                prediction_tensor,
                target_tensor,
                grad_map_tensor,
                dm_dmu1_tensor,
                dm_dsigma1_sq_tensor,
                dm_dsigma12_tensor,
                channels,
                height,
                width,
                grid_dim=(
                    div_round_up(width, BLOCK_X),
                    div_round_up(height, BLOCK_Y),
                    batch,
                ),
                block_dim=(BLOCK_X, BLOCK_Y),
            )
        else:
            raise Error("ssim_mojo backward currently requires a GPU target")


@compiler.register("ssim_bwd_mean")
struct SSIMBackwardMean:
    """MAX custom-op entrypoint for mean-reduced SSIM backward propagation."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        grad_prediction_image: OutputTensor[dtype=DType.float32, rank=4, ...],
        prediction_image: InputTensor[dtype=DType.float32, rank=4, ...],
        target_image: InputTensor[dtype=DType.float32, rank=4, ...],
        grad_scale: InputTensor[dtype=DType.float32, rank=1, ...],
        dm_dmu1: InputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma1_sq: InputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma12: InputTensor[dtype=DType.float32, rank=4, ...],
        ctx: DeviceContextPtr,
    ) raises:
        """Launch the GPU backward kernel for scalar mean SSIM."""
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var batch = Int(prediction_image.dim_size[0]())
            var channels = Int(prediction_image.dim_size[1]())
            var height = Int(prediction_image.dim_size[2]())
            var width = Int(prediction_image.dim_size[3]())
            var grad_prediction_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                grad_prediction_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var prediction_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                prediction_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var target_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                target_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var grad_scale_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_1D,
                MutAnyOrigin,
            ](
                grad_scale.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(1)),
            )
            var dm_dmu1_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                dm_dmu1.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma1_sq_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma1_sq.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma12_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma12.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            gpu_ctx.enqueue_function[ssim_bwd_mean_kernel, ssim_bwd_mean_kernel](
                grad_prediction_tensor,
                prediction_tensor,
                target_tensor,
                grad_scale_tensor,
                dm_dmu1_tensor,
                dm_dsigma1_sq_tensor,
                dm_dsigma12_tensor,
                channels,
                height,
                width,
                grid_dim=(
                    div_round_up(width, BLOCK_X),
                    div_round_up(height, BLOCK_Y),
                    batch,
                ),
                block_dim=(BLOCK_X, BLOCK_Y),
            )
        else:
            raise Error("ssim_mojo mean backward currently requires a GPU target")


@compiler.register("ssim_fwd_inplace")
struct SSIMForwardInplace:
    """MAX in-place custom-op entrypoint for SSIM map and derivative-cache creation."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        ssim_map: MutableInputTensor[dtype=DType.float32, rank=4, ...],
        dm_dmu1: MutableInputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma1_sq: MutableInputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma12: MutableInputTensor[dtype=DType.float32, rank=4, ...],
        prediction_image: InputTensor[dtype=DType.float32, rank=4, ...],
        target_image: InputTensor[dtype=DType.float32, rank=4, ...],
        ctx: DeviceContextPtr,
    ) raises:
        """Launch the GPU forward kernel for one contiguous NCHW tensor pair."""
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var batch = Int(prediction_image.dim_size[0]())
            var channels = Int(prediction_image.dim_size[1]())
            var height = Int(prediction_image.dim_size[2]())
            var width = Int(prediction_image.dim_size[3]())
            var prediction_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                prediction_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var target_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                target_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var ssim_map_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                ssim_map.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dmu1_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dmu1.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma1_sq_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma1_sq.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma12_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma12.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            gpu_ctx.enqueue_function[ssim_fwd_kernel, ssim_fwd_kernel](
                ssim_map_tensor,
                dm_dmu1_tensor,
                dm_dsigma1_sq_tensor,
                dm_dsigma12_tensor,
                prediction_tensor,
                target_tensor,
                channels,
                height,
                width,
                grid_dim=(
                    div_round_up(width, BLOCK_X),
                    div_round_up(height, BLOCK_Y),
                    batch,
                ),
                block_dim=(BLOCK_X, BLOCK_Y),
            )
        else:
            raise Error("ssim_mojo forward currently requires a GPU target")

@compiler.register("ssim_bwd_inplace")
struct SSIMBackwardInplace:
    """MAX in-place custom-op entrypoint for full SSIM-map backward propagation."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        grad_prediction_image: MutableInputTensor[dtype=DType.float32, rank=4, ...],
        prediction_image: InputTensor[dtype=DType.float32, rank=4, ...],
        target_image: InputTensor[dtype=DType.float32, rank=4, ...],
        grad_map: InputTensor[dtype=DType.float32, rank=4, ...],
        dm_dmu1: InputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma1_sq: InputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma12: InputTensor[dtype=DType.float32, rank=4, ...],
        ctx: DeviceContextPtr,
    ) raises:
        """Launch the GPU backward kernel for a materialized SSIM gradient map."""
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var batch = Int(prediction_image.dim_size[0]())
            var channels = Int(prediction_image.dim_size[1]())
            var height = Int(prediction_image.dim_size[2]())
            var width = Int(prediction_image.dim_size[3]())
            var grad_prediction_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                grad_prediction_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var prediction_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                prediction_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var target_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                target_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var grad_map_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                grad_map.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dmu1_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                dm_dmu1.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma1_sq_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma1_sq.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma12_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma12.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            gpu_ctx.enqueue_function[ssim_bwd_kernel, ssim_bwd_kernel](
                grad_prediction_tensor,
                prediction_tensor,
                target_tensor,
                grad_map_tensor,
                dm_dmu1_tensor,
                dm_dsigma1_sq_tensor,
                dm_dsigma12_tensor,
                channels,
                height,
                width,
                grid_dim=(
                    div_round_up(width, BLOCK_X),
                    div_round_up(height, BLOCK_Y),
                    batch,
                ),
                block_dim=(BLOCK_X, BLOCK_Y),
            )
        else:
            raise Error("ssim_mojo backward currently requires a GPU target")

@compiler.register("ssim_bwd_mean_inplace")
struct SSIMBackwardMeanInplace:
    """MAX in-place custom-op entrypoint for mean-reduced SSIM backward propagation."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        grad_prediction_image: MutableInputTensor[dtype=DType.float32, rank=4, ...],
        prediction_image: InputTensor[dtype=DType.float32, rank=4, ...],
        target_image: InputTensor[dtype=DType.float32, rank=4, ...],
        grad_scale: InputTensor[dtype=DType.float32, rank=1, ...],
        dm_dmu1: InputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma1_sq: InputTensor[dtype=DType.float32, rank=4, ...],
        dm_dsigma12: InputTensor[dtype=DType.float32, rank=4, ...],
        ctx: DeviceContextPtr,
    ) raises:
        """Launch the GPU backward kernel for scalar mean SSIM."""
        comptime if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            var batch = Int(prediction_image.dim_size[0]())
            var channels = Int(prediction_image.dim_size[1]())
            var height = Int(prediction_image.dim_size[2]())
            var width = Int(prediction_image.dim_size[3]())
            var grad_prediction_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                grad_prediction_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var prediction_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                prediction_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var target_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                target_image.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var grad_scale_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_1D,
                MutAnyOrigin,
            ](
                grad_scale.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_1D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(1)),
            )
            var dm_dmu1_tensor = LayoutTensor[DType.float32, ROW_MAJOR_4D, MutAnyOrigin](
                dm_dmu1.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma1_sq_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma1_sq.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            var dm_dsigma12_tensor = LayoutTensor[
                DType.float32,
                ROW_MAJOR_4D,
                MutAnyOrigin,
            ](
                dm_dsigma12.unsafe_ptr(),
                RuntimeLayout[
                    ROW_MAJOR_4D,
                    element_type=DType.int32,
                    linear_idx_type=DType.int32,
                ].row_major(Index(batch, channels, height, width)),
            )
            gpu_ctx.enqueue_function[ssim_bwd_mean_kernel, ssim_bwd_mean_kernel](
                grad_prediction_tensor,
                prediction_tensor,
                target_tensor,
                grad_scale_tensor,
                dm_dmu1_tensor,
                dm_dsigma1_sq_tensor,
                dm_dsigma12_tensor,
                channels,
                height,
                width,
                grid_dim=(
                    div_round_up(width, BLOCK_X),
                    div_round_up(height, BLOCK_Y),
                    batch,
                ),
                block_dim=(BLOCK_X, BLOCK_Y),
            )
        else:
            raise Error("ssim_mojo mean backward currently requires a GPU target")
