import sys

import numpy
import torch
from torch.nn.functional import conv2d

from toeplitz import get_toeplitz_channel_convolution, get_toeplitz_convolution

numpy.set_printoptions(precision=2, suppress=True, linewidth=240, threshold=sys.maxsize)

numpy.random.seed(0)

input_matrix = numpy.random.randn(3, 12, 17)
kernel = numpy.random.randn(4 * 3 * 3 * 2).reshape((4, 3, 3, 2))

STRIDE = 2
PADDING = 1


def compare_channel_toeplitz():
    # PyTorch reference implementation
    # pylint: disable=not-callable
    torch_toeplitz = conv2d(
        torch.tensor(input_matrix[:1]).view(
            1, 1, input_matrix.shape[-2], input_matrix.shape[-1]
        ),
        torch.tensor(kernel[:1, :1]),
        stride=STRIDE,
        padding=PADDING,
    ).numpy()

    # Our implementation
    toeplitz = get_toeplitz_channel_convolution(
        kernel[0, 0],
        (input_matrix.shape[1], input_matrix.shape[2]),
        stride=STRIDE,
        padding=PADDING,
    )

    toeplitz = toeplitz.dot(input_matrix[:1].flatten()).reshape(
        1,
        1,
        torch_toeplitz.shape[-2],
        torch_toeplitz.shape[-1],
    )

    print(numpy.sum((toeplitz - torch_toeplitz) ** 2))


def compare_toeplitz():
    # PyTorch reference implementation
    # pylint: disable=not-callable
    torch_toeplitz = conv2d(
        torch.tensor(input_matrix).view((1, *input_matrix.shape[0:])),
        torch.tensor(kernel),
        stride=STRIDE,
        padding=PADDING,
    ).numpy()

    # Our implementation
    toeplitz = get_toeplitz_convolution(
        kernel, input_matrix.shape, stride=STRIDE, padding=PADDING
    )

    output = toeplitz.dot(input_matrix.flatten()).reshape(torch_toeplitz.shape)

    print(numpy.sum((output - torch_toeplitz) ** 2))


compare_channel_toeplitz()
compare_toeplitz()
