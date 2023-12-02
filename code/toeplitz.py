import numpy
from scipy.linalg import toeplitz
import torch

def get_toeplitz_channel_convolution(kernel, input_size, stride=1, padding=1):
    """Compute the Toeplitz matrix for 2D convolution with with the given single-channel kernel
    and input size.

    Args:
        kernel: numpy array with shape (height, width)
        input_size: (height, width)
    """
    kernel_height, kernel_width = kernel.shape
    input_height, input_width = input_size

    padded_input_width = input_width + 2 * padding
    padded_input_height = input_height + 2 * padding

    # Generate Toeplitz matrices and add them to a list
    toeplitz_list = []

    for column_index in range(kernel_height):
        toeplitz_list.append(
            toeplitz(
                (
                    kernel[column_index, 0],
                    *numpy.zeros(padded_input_width - kernel_width),
                ),
                (
                    *kernel[column_index],
                    *numpy.zeros(padded_input_width - kernel_width),
                ),
            )[::stride]
        )

    toeplitz_height, toeplitz_width = toeplitz_list[0].shape
    vertical_block_count = (padded_input_height - kernel_height) // stride + 1

    # Add all the created Toeplitz matrices to one big matrix
    toeplitz_matrix = numpy.zeros(
        (vertical_block_count, toeplitz_height, padded_input_height, toeplitz_width)
    )

    for toeplitz_index, toeplitz_component in enumerate(toeplitz_list):
        for vertical_block_index in range(vertical_block_count):
            toeplitz_matrix[
                vertical_block_index,
                :,
                toeplitz_index + vertical_block_index * stride,
                :,
            ] = toeplitz_component

    toeplitz_matrix.shape = (
        vertical_block_count * toeplitz_height,
        padded_input_height * toeplitz_width,
    )

    # Change matrix to correspond to convolution matrix for padded input
    start_index = padding * padded_input_width + padding
    end_index = start_index + input_height * padded_input_width

    column_indices = []

    for index in range(start_index, end_index, padded_input_width):
        column_indices.extend(list(range(index, index + input_width)))

    toeplitz_matrix = numpy.take(toeplitz_matrix, column_indices, -1)

    return toeplitz_matrix


def get_toeplitz_convolution(kernel, input_size, stride=1, padding=1):
    """Compute the Toeplitz matrix for 2D convolution with with the given multi-channel kernel
    and input size.

    Args:
        kernel: numpy array with shape (output_channel_count, input_channel_count, height, width)
        input_size: (input_channel_count, height, width)
    """
    output_size = (
        kernel.shape[0],
        (input_size[1] - kernel.shape[2] + 2 * padding) // stride + 1,
        (input_size[2] - kernel.shape[3] + 2 * padding) // stride + 1,
    )

    # Initialize the empty Toeplitz matrix
    toeplitz_matrix = numpy.zeros(
        (
            output_size[0],
            output_size[1] * output_size[2],
            input_size[0],
            input_size[1] * input_size[2],
        )
    )

    # Calculate the Toeplitz matrix for all input-output channel combinations
    # individually, and place them in a matrix
    for output_channel_index, output_channel_kernels in enumerate(kernel):
        for input_channel_index, channel_kernel in enumerate(output_channel_kernels):
            channel_toeplitz = get_toeplitz_channel_convolution(
                channel_kernel,
                (input_size[1], input_size[2]),
                stride=stride,
                padding=padding,
            )

            toeplitz_matrix[
                output_channel_index, :, input_channel_index, :
            ] = channel_toeplitz

    toeplitz_matrix.shape = (numpy.prod(output_size), numpy.prod(input_size))

    toeplitz_matrix = torch.from_numpy(toeplitz_matrix).type(torch.float32)

    return toeplitz_matrix
