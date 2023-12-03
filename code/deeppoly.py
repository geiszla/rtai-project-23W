from abc import abstractmethod

import numpy
import torch
from toeplitz import get_toeplitz_convolution
from torch import nn


class DeepPolyBase(nn.Module):
    def __init__(self, prev=None, next=None):
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
    def box_from_constraints(self, constraints):
        # Given the constraints of this layer
        # compute the box of this layer
        return

    @abstractmethod
    def backsubstitution(self, constraints):
        # Given the constraints from the previous layers
        # compute the constraints of this layer
        return

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

    def backsubstitution(self, constraints=None):
        """
        Given the constraints from the previous layers
        compute the constraints of this layer.
        """
        if constraints is None:
            return Constraints(self.weight.T, self.weight.T, self.bias, self.bias)

        assert isinstance(constraints, Constraints)

        # Compute the new constraints and bias
        pos_weight, neg_weight = self.pos_neg_split(self.weight.T)
        new_upper_constraints, new_lower_constraints = self.compute_new_constraints(
            pos_weight, neg_weight, constraints
        )

        pos_weight, neg_weight = self.pos_neg_split(self.weight)
        upper_bias, lower_bias = self.compute_new_bias(
            pos_weight, neg_weight, constraints
        )

        # Update the constraints
        updated_constraints = Constraints(
            new_upper_constraints, new_lower_constraints, upper_bias, lower_bias
        )

        return updated_constraints

    def pos_neg_split(self, matrix):
        # Split the weight matrix into positive and negative weights
        pos_weight = torch.where(matrix > 0, matrix, 0)
        neg_weight = torch.where(matrix < 0, matrix, 0)
        return pos_weight, neg_weight

    def compute_new_constraints(self, pos_weight, neg_weight, constraints):
        # Compute the new constraints
        new_upper_constraints = (
            constraints.upper_constraints @ pos_weight
            + constraints.lower_constraints @ neg_weight
        )
        new_lower_constraints = (
            constraints.upper_constraints @ neg_weight
            + constraints.lower_constraints @ pos_weight
        )
        return new_upper_constraints, new_lower_constraints

    def compute_new_bias(self, pos_weight, neg_weight, constraints):
        # Compute the new bias
        upper_bias = (
            pos_weight @ constraints.upper_bias + neg_weight @ constraints.lower_bias
        )
        lower_bias = (
            pos_weight @ constraints.lower_bias + neg_weight @ constraints.upper_bias
        )
        return upper_bias, lower_bias

    def swap(self, upper_matrix, lower_matrix):
        # Swap the upper and lower bound if the weight is negative
        upper_matrix = torch.where(self.weight > 0, upper_matrix, lower_matrix)
        lower_matrix = torch.where(self.weight > 0, lower_matrix, upper_matrix)
        return upper_matrix, lower_matrix

    def box_from_constraints(self, prev_ub, prev_lb, constraints):
        """
        Given the constraints of this layer
        compute the box of this layer.
        """
        if constraints is None:
            return self.compute_new_box(prev_ub, prev_lb)

        assert isinstance(constraints, Constraints)

        pos_uc, neg_uc = self.pos_neg_split(constraints.upper_constraints.T)
        pos_lc, neg_lc = self.pos_neg_split(constraints.lower_constraints.T)

        # Compute the new upper and lower bound
        new_upper_bound = (
            pos_uc @ prev_ub + neg_uc @ prev_lb + constraints.upper_bias.unsqueeze(0).T
        )
        new_lower_bound = (
            pos_lc @ prev_lb + neg_lc @ prev_ub + constraints.lower_bias.unsqueeze(0).T
        )

        # Update the box
        self.upper_bound = new_upper_bound  # .unsqueeze(0)
        self.lower_bound = new_lower_bound  # .unsqueeze(0)

    def forward(self, inputs):
        """
        Pushing the previous box through this layer.
        Calling backsubstitution to compute additional constraints.
        """
        orig_ub, orig_lb, prev_ub, prev_lb, constraints = inputs

        constraints = self.backsubstitution(constraints)
        self.box_from_constraints(orig_ub, orig_lb, constraints)

        return orig_ub, orig_lb, self.upper_bound, self.lower_bound, constraints


class DeepPolyFlatten(DeepPolyBase):
    def __init__(self):
        super(DeepPolyFlatten, self).__init__()

    def forward(self, inputs):
        """
        Pushing the previous box through this layer.
        """
        orig_ub, orig_lb, prev_ub, prev_lb, constraints = inputs
        # print("flatten layer")
        flatten = nn.Flatten()
        lb = flatten(prev_lb).T
        ub = flatten(prev_ub).T
        self.lower_bound = lb
        self.upper_bound = ub
        return (
            self.upper_bound,
            self.lower_bound,
            self.upper_bound,
            self.lower_bound,
            constraints,
        )


class DeepPolyConvolution(DeepPolyLinear):
    def __init__(self, layer):
        self.weight = None
        self.bias = None
        super().__init__(self)
        self.layer = layer

    def forward(self, inputs):
        print("################################# conv layer")
        orig_ub, orig_lb, prev_ub, prev_lb, constraints = inputs
        assert prev_ub.shape == prev_lb.shape

        _, input_height, input_width = prev_ub.shape
        _, _, kernel_height, kernel_width = self.layer.weight.data.shape

        stride = self.layer.stride[0]
        padding = self.layer.padding[0]

        self.weight = get_toeplitz_convolution(
            self.layer.weight.data,
            (self.layer.in_channels, input_height, input_width),
            stride=stride,
            padding=padding,
        )

        output_shape = self.weight.shape[0]
        bias_shape = self.layer.bias.data.shape[0]

        assert output_shape % bias_shape == 0
        repetition = output_shape // bias_shape

        self.bias = torch.repeat_interleave(self.layer.bias.data, repetition)

        print("bias", self.bias.shape)
        print("weight", self.weight.shape)
        print("self.layer.bias.data", self.layer.bias.data.shape)

        # in case this is the first layer we need to flatten and transpose the input
        flatten = nn.Flatten()
        if constraints is None:
            new_inputs = (
                flatten(orig_lb).T,
                flatten(orig_ub).T,
                flatten(prev_lb).T,
                flatten(prev_ub).T,
                constraints,
            )
        else:
            new_inputs = inputs

        # Use linear forward pass
        linear_result = super().forward((new_inputs))

        output_height = (input_height - kernel_height + 2 * padding) // stride + 1
        output_width = (input_width - kernel_width + 2 * padding) // stride + 1
        output_shape = (1, self.layer.out_channels, output_height, output_width)

        self.upper_bound = self.upper_bound.view(output_shape)
        self.lower_bound = self.lower_bound.view(output_shape)

        linear_result[2] = self.upper_bound
        linear_result[3] = self.lower_bound

        return linear_result


class DeepPolyReLu(DeepPolyBase):
    def __init__(self, ub=None, lb=None):
        super(DeepPolyReLu, self).__init__(ub, lb)

        self.upper_bound_slope = None
        self.lower_bound_slope = None

    def forward(self, inputs):
        orig_ub, orig_lb, prev_ub, prev_lb, constraints = inputs
        assert prev_ub.shape == prev_lb.shape
        self.upper_bound = prev_ub
        self.lower_bound = prev_lb

        self.upper_bound_slope = torch.full_like(
            self.upper_bound, float("nan"), dtype=torch.float32
        )
        self.lower_bound_slope = torch.full_like(
            self.upper_bound, float("nan"), dtype=torch.float32
        )

        # Compute DeepPoly slopes
        self.compute_relu_slopes(self.upper_bound, self.lower_bound)
        # Update box bounds
        self.update_upper_and_lower_bound_()

        return orig_ub, orig_lb, self.upper_bound, self.lower_bound, constraints

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
        # assert torch.isnan(ub_slopes).sum() == self.crossing_relu_mask().sum()

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
        # print(lb_slopes)
        # assert torch.isnan(lb_slopes).sum() == self.crossing_relu_mask().sum()

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


class Constraints:
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
