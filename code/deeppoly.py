from abc import abstractmethod

import numpy
import torch
from toeplitz import get_toeplitz_convolution
from torch import nn


class DeepPolyBase(nn.Module):
    def __init__(self, prev = None, next = None):
        super(DeepPolyBase, self).__init__()

        # Double linking the network layers
        self.prev_layer = prev
        self.next_layer = next

        # Each layer has a box represented by upper and lower bound
        self.upper_bound = None
        self.lower_bound = None

        # Each layer has additional constraints
        self.constraints = None

    def compute_new_box(self, prev_ub, prev_lb):
        """
        Given the upper and lower bounds of previous layer,
        compute the new upper and lower bounds for this layer.
        """
        lb = prev_lb.repeat(self.weight.shape[0], 1)
        ub = prev_ub.repeat(self.weight.shape[0], 1)
        assert lb.shape == ub.shape == self.weight.shape

        # When computing the new lower/upper bounds, we need to take into account the sign of the
        # weight. Effectively, the expression that we want to overapproximate is:
        # x_1 * w_1 + x_2 * w_2 + ... + x_d * w_d,
        # where each x_i is overapproximated/abstracted by the box [lb_i, ub_i], i.e.
        # the concrete value of the neuron x_i can be any number from the interval [lb_i, ub_i].
        mul_lb = torch.where(self.weight > 0, lb, ub)
        mul_ub = torch.where(self.weight > 0, ub, lb)

        lb = (mul_lb * self.weight).sum(dim=1)
        ub = (mul_ub * self.weight).sum(dim=1)
        assert lb.shape == ub.shape == self.bias.shape

        if self.bias is not None:
            lb += self.bias
            ub += self.bias

        self.lower_bound = lb.unsqueeze(0)
        self.upper_bound = ub.unsqueeze(0)


    @abstractmethod
    def forward(self, upper_bound, lower_bound):
        # Pushing the previous box through this layer
        # Obtain the new box
        return
    
class DeepPolyLinear(DeepPolyBase):
    def __init__(self, fc):
        super(DeepPolyLinear, self).__init__()
        self.weight = fc.weight
        self.bias = fc.bias

    def forward(self, prev_ub, prev_lb):
        """
        Pushing the previous box through this layer.
        Calling backsubstitution to compute additional constraints.
        """
        assert prev_lb == prev_ub
        assert len(prev_lb) == 2
        assert (
            prev_lb.shape[0] == 1
            and prev_lb.shape[1] == self.weight.shape[1]
        )
        self.compute_new_box(prev_ub, prev_lb)

class DeepPolyFlatten(DeepPolyBase):
    def __init__(self):
        super(DeepPolyFlatten, self).__init__()

    def forward(self, prev_lb, prev_ub):
        flatten = nn.Flatten()
        lb = flatten(prev_lb)
        ub = flatten(prev_ub)
        self.lower_bound = lb
        self.upper_bound = ub

class DeepPolyShape(DeepPolyBase):
    def __init__(self, input, eps):
        upper_bound = input + eps
        upper_bound.clamp_(min=0, max=1)
        lower_bound = input - eps
        lower_bound.clamp_(min=0, max=1)

        super(DeepPolyShape, self).__init__(upper_bound, lower_bound)

    def check_postcondition(self, y) -> bool:
        upper_bound = self.upper_bound.squeeze()
        lower_bound = self.lower_bound.squeeze()
        mask = torch.ones_like(upper_bound, dtype=torch.bool)
        mask[y] = False
        max_value = torch.max(torch.masked_select(upper_bound, mask))
        return max_value < lower_bound[y]

    def forward(self, x):
        pass


class DeepPolyConvolution(nn.Module):
    def __init__(self, layer, input_shape):
        self.weights = get_toeplitz_convolution(
            layer.weight.data,
            (layer.in_channels, input_shape[-2], input_shape[-1]),
            stride=layer.stride[0],
            padding=layer.padding[0],
        ).T

        self.bias = torch.repeat_interleave(
            layer.bias.data, self.output_shape[-1] * self.output_shape[-2]
        )

        super().__init__()

    def forward(self, x):
        self.network()

class DeepPolyReLu(DeepPolyBase):
    def __init__(self, ub, lb):
        super(DeepPolyReLu, self).__init__(ub, lb)
        self.upper_bound_slope = torch.full_like(
            self.upper_bound, float("nan"), dtype=torch.float32
        )
        self.lower_bound_slope = torch.full_like(
            self.upper_bound, float("nan"), dtype=torch.float32
        )

    def forward(self, x):
        # Compute DeepPoly slopes
        self.compute_relu_slopes(self.upper_bound, self.lower_bound)
        # Update box bounds
        self.update_upper_and_lower_bound_()

    def update_upper_and_lower_bound_(self):
        # After computing the slopes we
        # update box bounds
        self.upper_bound[self.upper_bound < 0] = 0
        self.lower_bound[self.lower_bound < 0] = 0

    def compute_relu_slopes(self, ub, lb):
        # Computes upper and lower bound slopes
        self.upper_bound_slope = self.compute_upper_bound_slope(ub, lb)
        self.lower_bound_slope = self.compute_lower_bound_slopes(ub, lb)

    def compute_upper_bound_slope(self, ub, lb):
        # Compute upper slope for all crossing ReLus
        # All other slopes stay nan
        ub_slopes = torch.full_like(ub, float("nan"), dtype=torch.float32)

        # Division by zero is not possible due to crossing constraint: ub != lb
        ub_slopes[self.crossing_relu_mask()] = ub[self.crossing_relu_mask()] / (
            ub[self.crossing_relu_mask()] - lb[self.crossing_relu_mask()]
        )

        # Check if slopes valid
        assert ub_slopes.shape == ub.shape
        assert (ub_slopes < 0).sum() == 0
        assert torch.isnan(ub_slopes).sum() == self.crossing_relu_mask().sum()

        return ub_slopes

    def compute_lower_bound_slopes(self, ub, lb):
        # Compute lower slope for all crossing ReLus based on two DeepPoly variants
        # All other slopes stay nan
        lb_slopes = torch.full_like(lb, float("nan"), dtype=torch.float32)

        assert (
            self.deep_poly_variant_1_mask().sum()
            + self.deep_poly_variant_2_mask().sum()
            == self.crossing_relu_mask().sum()
        )

        # Compute slope
        lb_slopes[self.deep_poly_variant_1_mask()] = 0
        lb_slopes[self.deep_poly_variant_2_mask()] = 1

        # Check if slope is valid
        assert lb_slopes.shape == lb.shape
        print(lb_slopes)
        assert torch.isnan(lb_slopes).sum() == self.crossing_relu_mask().sum()

        return lb_slopes

    def positive_relu_mask(self):
        return (self.upper_bound >= 0) & (self.lower_bound >= 0)

    def negative_relu_mask(self):
        return self.upper_bound <= 0

    def crossing_relu_mask(self):
        return (self.upper_bound > 0) & (self.lower_bound < 0)

    def deep_poly_variant_1_mask(self):
        return self.crossing_relu_mask() & (self.upper_bound <= abs(self.lower_bound))

    def deep_poly_variant_2_mask(self):
        return self.crossing_relu_mask() & (self.upper_bound > abs(self.lower_bound))

class Constraints():
    upper_constraints = None
    lower_constraints = None
    upper_bias = None
    lower_bias = None

    def __init__(self, uc, lc, ub, lb):
        self.upper_constraints = uc
        self.lower_constraints = lc
        self.upper_bias = ub
        self.lower_bias = lb
    
def main():
    ub = torch.tensor([[-1, 1], [2, 3], [2, 1]])
    lb = torch.tensor([[-3, -1], [1, -1], [1, -1]])
    reluLayer = DeepPolyReLu(ub, lb)
    print("Upper bound", reluLayer.upper_bound)
    print("Lower bound", reluLayer.lower_bound)
    print("Crossing ReLU mask", reluLayer.crossing_relu_mask())
    reluLayer.forward(3)
    print("Upper bound slope", reluLayer.upper_bound_slope)
    print("Lower bound slope", reluLayer.lower_bound_slope)
    print("Updated upper bound", reluLayer.upper_bound)
    print("Updated lower bound", reluLayer.lower_bound)


if __name__ == "__main__":
    main()
