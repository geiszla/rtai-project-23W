from abc import abstractmethod

import numpy
import torch
from torch import nn


class DeepPolyBase(nn.Module):
    def __init__(self, upper_bound, lower_bound):
        super(DeepPolyBase, self).__init__()
        assert upper_bound.shape == lower_bound.shape
        assert (lower_bound > upper_bound).sum() == 0
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    @abstractmethod
    def forward(self, x):
        return x


class DeepPolyLinear(nn.Module):
    def __init__(self, fc):
        super(DeepPolyLinear, self).__init__()
        self.weight = fc.weight
        self.bias = fc.bias

    def forward(self, x):
        assert x.lower_bound.shape == x.upper_bound.shape
        assert len(x.lower_bound.shape) == 2
        assert (
            x.lower_bound.shape[0] == 1
            and x.lower_bound.shape[1] == self.weight.shape[1]
        )

        lb = x.lower_bound.repeat(self.weight.shape[0], 1)
        ub = x.upper_bound.repeat(self.weight.shape[0], 1)
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

        x.lower_bound = lb.unsqueeze(0)
        x.upper_bound = ub.unsqueeze(0)

        return x


class DeepPolyFlatten(nn.Module):
    def __init__(self):
        super(DeepPolyFlatten, self).__init__()

    def forward(self, x):
        flatten = nn.Flatten()
        lb = flatten(x.lower_bound)
        ub = flatten(x.upper_bound)
        x.lower_bound = lb
        x.upper_bound = ub
        return x


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


class DeepPolyConvolution(DeepPolyBase):
    def __init__(self, ub, lb):
        super().__init__(ub, lb)

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
