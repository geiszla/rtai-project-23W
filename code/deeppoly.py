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
        print("pos_weight linear ", pos_weight.shape)
        print("constraints.upper_bias linear ", constraints.upper_bias.shape)
        upper_bias = (
            pos_weight @ constraints.upper_bias + neg_weight @ constraints.lower_bias + self.bias
        )
        lower_bias = (
            pos_weight @ constraints.lower_bias + neg_weight @ constraints.upper_bias + self.bias
        )
        return upper_bias, lower_bias


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
        print("linear layer")
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
        print("flatten layer")
        flatten = nn.Flatten()
        lb = flatten(prev_lb).T
        ub = flatten(prev_ub).T
        self.lower_bound = lb
        self.upper_bound = ub

        if constraints is None:
            return (
                self.upper_bound,
                self.lower_bound,
                self.upper_bound,
                self.lower_bound,
                constraints,
            )
        else:
            return (
                orig_ub,
                orig_lb,
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
        print("conv layer")
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

        # in case this is the first layer we need to flatten and transpose the input
        flatten = nn.Flatten()
        if constraints is None:
            new_inputs = (
                flatten(orig_ub).T,
                flatten(orig_lb).T,
                flatten(prev_ub).T,
                flatten(prev_lb).T,
                constraints,
            )
        else:
            new_inputs = inputs

        # Use linear forward pass
        linear_result = super().forward((new_inputs))
        linear_result = list(linear_result)

        output_height = (input_height - kernel_height + 2 * padding) // stride + 1
        output_width = (input_width - kernel_width + 2 * padding) // stride + 1
        output_shape = (self.layer.out_channels, output_height, output_width)

        self.upper_bound = self.upper_bound.view(output_shape)
        self.lower_bound = self.lower_bound.view(output_shape)

        linear_result[2] = self.upper_bound
        linear_result[3] = self.lower_bound

        return tuple(linear_result)


class DeepPolyReLu(DeepPolyBase):
    def __init__(self, layer):
        super(DeepPolyReLu, self).__init__()

        self.upper_bound_slope = None
        self.lower_bound_slope = None

    def forward(self, inputs):
        print("relu layer")
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
        # Compute constraints
        new_upper_constraints, new_lower_constraints = self.updated_constraints(constraints)
        # Compute bias
        upper_bias, lower_bias = self.updated_bias(constraints, prev_lb)
        # Update box bounds
        self.update_upper_and_lower_bound_()

        constraints = Constraints(
            new_upper_constraints, new_lower_constraints, upper_bias, lower_bias
        )

        return orig_ub, orig_lb, self.upper_bound, self.lower_bound, constraints

    def updated_constraints(self, constraints):
        """
        Given the constraints from the previous layers,
        compute the constraints of this layer.
        """
        assert isinstance(constraints, Constraints)

        num_constraints = constraints.upper_constraints.shape[0]

        # Compute the new constraints for crossing ReLUs
        upper_bound_slope_repeat = self.upper_bound_slope.T.repeat(num_constraints, 1)
        lower_bound_slope_repeat = self.lower_bound_slope.T.repeat(num_constraints, 1)

        new_upper_constraints = constraints.upper_constraints * upper_bound_slope_repeat
        new_lower_constraints = constraints.lower_constraints * lower_bound_slope_repeat

        positive_relu_mask_repeat = self.positive_relu_mask().T.repeat(num_constraints, 1)
        negative_relu_mask_repeat = self.negative_relu_mask().T.repeat(num_constraints, 1)

        # Compute the new constraints for positive ReLUs
        new_upper_constraints[positive_relu_mask_repeat] = constraints.upper_constraints[positive_relu_mask_repeat]
        new_lower_constraints[positive_relu_mask_repeat] = constraints.lower_constraints[positive_relu_mask_repeat]

        # Compute the new constraints for negative ReLUs
        new_upper_constraints[negative_relu_mask_repeat] = 0
        new_lower_constraints[negative_relu_mask_repeat] = 0

        print("new_upper_constraints", new_upper_constraints)

        return new_upper_constraints, new_lower_constraints

    def updated_bias(self, constraints, prev_lb):
        """
        Given the constraints from the previous layers
        compute the bias of this layer.
        """
        assert isinstance(constraints, Constraints)

        # Compute the new bias for crossing ReLUs
        upper_bound_slope = self.upper_bound_slope.view(-1)
        lower_bound_slope = self.lower_bound_slope.view(-1)
        positive_relu_mask = self.positive_relu_mask().view(-1)
        negative_relu_mask = self.negative_relu_mask().view(-1)

        upper_bias = torch.mul(constraints.upper_bias.view(-1), upper_bound_slope) + torch.mul(upper_bound_slope, prev_lb.view(-1))
        lower_bias = torch.mul(constraints.lower_bias.view(-1), lower_bound_slope)

        # Compute the new bias for positive ReLUs
        upper_bias[positive_relu_mask] = constraints.upper_bias[positive_relu_mask]
        lower_bias[positive_relu_mask] = constraints.lower_bias[positive_relu_mask]

        # Compute the new bias for negative ReLUs
        upper_bias[negative_relu_mask] = 0
        lower_bias[negative_relu_mask] = 0

        # Check if bias is valid
        assert upper_bias.shape == lower_bias.shape
        # check there are no nan values
        assert upper_bias.isnan().sum() == 0

        return upper_bias, lower_bias
    
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

class DeepPolyLeakyReLu(DeepPolyBase):
    def __init__(self, layer):
        super(DeepPolyLeakyReLu, self).__init__()
        self.leaky_relu_slope = layer.negative_slope

    def forward(self, inputs):
        orig_ub, orig_lb, prev_ub, prev_lb, constraints = inputs
        self.prev_ub = prev_ub
        self.prev_lb = prev_lb

        # Compute simple box first
        self.upper_bound, self.lower_bound = self.compute_new_box()

        # Compute constraints constisting of upper and lower bound slopes and biases
        self.constraints = self.compute_constraints()

    def compute_new_box(self):
        """
        Compute a simple box from upper bound and lower bound of previous layer.
        This box can be tighented later by using the constraints.
        """
        assert self.prev_ub.shape == self.prev_lb.shape
        assert torch.all(self.prev_ub > self.prev_lb)

        # Initialize with NaN
        upper_bound = torch.full_like(self.prev_ub, float('nan'), dtype=torch.float32)
        lower_bound = torch.full_like(self.prev_lb, float('nan'), dtype=torch.float32)

        # Upper bound
        upper_bound[self.prev_ub > 0] = self.prev_ub[self.prev_ub > 0]
        upper_bound[self.prev_ub <= 0] = self.leaky_relu_slope * self.prev_ub[self.prev_ub <= 0]

        # Lower bound
        lower_bound[self.prev_lb > 0] = self.prev_lb[self.prev_lb > 0]
        lower_bound[self.prev_lb <= 0] = self.leaky_relu_slope * self.prev_lb[self.prev_lb <= 0]

        assert upper_bound.shape == lower_bound.shape == self.prev_ub.shape == self.prev_lb.shape
        assert (upper_bound < lower_bound).sum() == 0
        assert upper_bound.isnan().sum() == 0
        assert lower_bound.isnan().sum() == 0

        return upper_bound, lower_bound
    
    def compute_slopes(self):
        """
        Compute the slopes of the upper and lower bound for the constraints.
        Use two different cases for leaky relu slope bigger or smaller than 1.
        """
        if self.leaky_relu_slope >= 1:
            return self.compute_slope_bigger_one()
        else:
            return self.compute_slope_smaller_one()
    
    def compute_bias(self):
        """
        Compute the bias of the upper and lower bound for the constraints.
        Use two different cases for leaky relu slope bigger or smaller than 1.
        """
        if self.leaky_relu_slope >= 1:
            return self.compute_bias_bigger_one()
        else:
            return self.compute_bias_smaller_one()
        
    def compute_constraints(self):
        """
        Computes all constraints for the leaky relu layer.
        The constraints consist of upper and lower bound slopes and biases.
        They can be used to tighten the box bounds during backsubstitution.
        """
        assert self.prev_ub.shape == self.prev_lb.shape
        assert torch.all(self.prev_ub >= self.prev_lb)

        upper_slopes, lower_slopes = self.compute_slopes()
        upper_bias, lower_bias = self.compute_bias()

        assert upper_slopes.shape == lower_slopes.shape == self.prev_ub.shape == self.prev_lb.shape
        assert upper_bias.shape == lower_bias.shape == self.prev_ub.shape == self.prev_lb.shape


        return Constraints(upper_slopes, lower_slopes, upper_bias, lower_bias)

    def crossing_relu_mask(self):
        return (self.prev_ub> 0) & (self.prev_lb < 0)
    
    def compute_slope_bigger_one(self):

        # Initialize with NaN
        upper_slopes = torch.full_like(self.prev_ub, float('nan'), dtype=torch.float32)
        lower_slopes = torch.full_like(self.prev_lb, float('nan'), dtype=torch.float32)

        # Upper slopes
        upper_slopes[self.crossing_relu_mask()] = 1

        # Lower slopes
        # (ub - a*lb) / (ub - lb)
        rise = self.prev_ub[self.crossing_relu_mask()] - (self.leaky_relu_slope * self.prev_lb[self.crossing_relu_mask()])
        run = self.prev_ub[self.crossing_relu_mask()] - self.prev_lb[self.crossing_relu_mask()]
        lower_slopes[self.crossing_relu_mask()] = rise / run

        # Only crossing relus have slopes, else nan
        return upper_slopes, lower_slopes
    
    def compute_slope_smaller_one(self):
        # Andras implementation
        raise NotImplementedError

    def compute_bias_bigger_one(self):
        # Initialize with NaN
        upper_bias = torch.full_like(self.prev_ub, float('nan'), dtype=torch.float32)
        lower_bias = torch.full_like(self.prev_lb, float('nan'), dtype=torch.float32)

        # Get the indices of crossing relus
        crossing_indices = self.crossing_relu_mask()

        # Upper bias
        upper_bias[crossing_indices] = 0

        # Lower bias
        numerator = torch.square(self.prev_ub[crossing_indices]) - self.prev_ub[crossing_indices] * self.prev_lb[crossing_indices]
        denominator = (self.prev_ub[crossing_indices] - self.leaky_relu_slope * self.prev_lb[crossing_indices])
        lower_bias[crossing_indices] = self.prev_ub[crossing_indices] - (numerator / denominator)

        # Only crossing relus have biases, else nan
        return upper_bias, lower_bias
    
    def compute_bias_smaller_one(self):
        # Andras implementation
        raise NotImplementedError


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
    prev_ub = torch.tensor([[-1, 1], [2, 3], [2, 1]], dtype=torch.float32)
    prev_lb = torch.tensor([[-3, -1], [1, -1], [1, -1]], dtype=torch.float32)
    reluLayer = nn.LeakyReLU(negative_slope=2)

    verifier = DeepPolyLeakyReLu(reluLayer)
    verifier.forward((None, None, prev_ub, prev_lb, None))
    ub, lb = verifier.compute_new_box()
    print("Prev upper bound", prev_ub)
    print("Prev lower bound", prev_lb)
    print("Upper bound", ub)
    print("Lower bound", lb)
    print("Crossing relu mask", verifier.crossing_relu_mask())
    us, ls = verifier.compute_slopes()
    print("Upper slopes", us)
    print("Lower slopes", ls)
    ubias, lbias = verifier.compute_bias()
    print("Upper bias", ubias)
    print("Lower bias", lbias)

if __name__ == "__main__":
    main()
