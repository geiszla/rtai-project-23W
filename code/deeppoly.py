from torch import nn


class DeepPolyConvolution(nn.Module):

    class DeepPolyReLu(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.relu()
