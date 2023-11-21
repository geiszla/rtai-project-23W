import numpy
from scipy.linalg import toeplitz


def get_toeplitz_channel_convolution(kernel, input_size):
    """Compute the Toeplitz matrix for 2D convolution with with the given single-channel kernel
    and input size.

    Args:
        kernel: numpy array with shape (height, width)
        input_size: (height, width)
    """
    kernel_height, kernel_width = kernel.shape
    input_height, input_width = input_size

    # Generate Toeplitz matrices and add them to a list
    toeplitz_list = []

    for column_index in range(kernel_height):
        toeplitz_list.append(
            toeplitz(
                (
                    kernel[column_index, 0],
                    *numpy.zeros(input_width - kernel_width),
                ),
                (
                    *kernel[column_index],
                    *numpy.zeros(input_width - kernel_width),
                ),
            )
        )

    toeplitz_height, toeplitz_width = toeplitz_list[0].shape
    vertical_block_count = input_height - kernel_height + 1

    toeplitz_matrix = numpy.zeros(
        (vertical_block_count, toeplitz_height, input_height, toeplitz_width)
    )

    # Add all the created Toeplitz matrices to one big matrix
    for toeplitz_index, toeplitz_component in enumerate(toeplitz_list):
        for vertical_block_index in range(vertical_block_count):
            toeplitz_matrix[
                vertical_block_index, :, toeplitz_index + vertical_block_index, :
            ] = toeplitz_component

    toeplitz_matrix.shape = (
        vertical_block_count * toeplitz_height,
        input_height * toeplitz_width,
    )

    return toeplitz_matrix


def get_toeplitz_convolution(kernel, input_size):
    """Compute the Toeplitz matrix for 2D convolution with with the given multi-channel kernel
    and input size.

    Args:
        kernel: numpy array with shape (output_channel_count, input_channel_count, height, width)
        input_size: (input_channel_count, height, width)
    """

    output_size = (kernel.shape[0], input_size[1], input_size[2])

    # Initialize the empty Toeplitz matrix
    toeplitz_matrix = numpy.zeros(
        (
            output_size[0],
            int(numpy.prod(output_size[1:])),
            input_size[0],
            int(numpy.prod(input_size[1:])),
        )
    )

    # Calculate the Toeplitz matrix for all input-output channel combinations
    # individually, and place them in a matrix
    for output_channel_index, output_channel_kernels in enumerate(kernel):
        for input_channel_index, channel_kernel in enumerate(output_channel_kernels):
            channel_toeplitz = get_toeplitz_channel_convolution(
                channel_kernel, input_size[1:]
            )

            toeplitz_matrix[
                output_channel_index, :, input_channel_index, :
            ] = channel_toeplitz

    toeplitz_matrix.shape = (numpy.prod(output_size), numpy.prod(input_size))

    return toeplitz_matrix
