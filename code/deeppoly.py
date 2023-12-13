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

    def box_from_constraints(self, orig_ub, orig_lb, constraints):
        """
        Given the constraints of this layer
        compute the box of this layer.
        """
        assert torch.all(orig_ub >= orig_lb)

        if constraints is None:
            return self.compute_new_box(orig_ub, orig_lb)

        assert isinstance(constraints, Constraints)

        pos_uc, neg_uc = self.pos_neg_split(constraints.upper_constraints.T)
        pos_lc, neg_lc = self.pos_neg_split(constraints.lower_constraints.T)

        # Compute the new upper and lower bound
        print("without bias: ", pos_uc @ orig_ub + neg_uc @ orig_lb >= pos_lc @ orig_lb + neg_lc @ orig_ub)
        print("upper bias: ", constraints.upper_bias)
        print("lower bias: ", constraints.lower_bias)
        print("upper bias < lower bias: ", constraints.upper_bias < constraints.lower_bias)
    
        new_upper_bound = (
            pos_uc @ orig_ub + neg_uc @ orig_lb + constraints.upper_bias.unsqueeze(0).T
        )
        new_lower_bound = (
            pos_lc @ orig_lb + neg_lc @ orig_ub + constraints.lower_bias.unsqueeze(0).T
        )

        # Update the box
        self.upper_bound = new_upper_bound  # .unsqueeze(0)
        self.lower_bound = new_lower_bound  # .unsqueeze(0)

        print("upper bound < lower bound: ", self.upper_bound < self.lower_bound)

        assert torch.all(self.upper_bound >= self.lower_bound)
        assert self.upper_bound.shape == self.lower_bound.shape

    def pos_neg_split(self, matrix):
        """
        Split the weight matrix into positive and negative weights
        """
        pos_weight = torch.where(matrix > 0, matrix, 0)
        neg_weight = torch.where(matrix < 0, matrix, 0)
        return pos_weight, neg_weight

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

    def forward(self, inputs):
        """
        Pushing the previous box through this layer.
        Calling backsubstitution to compute additional constraints.
        """
        print("linear layer")
        orig_ub, orig_lb, prev_ub, prev_lb, constraints = inputs

        assert torch.all(prev_ub >= prev_lb)
        assert prev_ub.shape == prev_lb.shape

        constraints = self.backsubstitution(constraints)
        self.box_from_constraints(orig_ub, orig_lb, constraints)

        assert torch.all(self.upper_bound >= self.lower_bound)
        assert self.upper_bound.shape == self.lower_bound.shape

        return orig_ub, orig_lb, self.upper_bound, self.lower_bound, constraints
    
    def backsubstitution(self, constraints=None):
        """
        Given the constraints from the previous layers
        compute the constraints of this layer.
        """
        # First layer
        if constraints is None:
            return Constraints(self.weight.T, self.weight.T, self.bias, self.bias)

        # All other layers
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
        print("multiply", (pos_weight @ constraints.upper_bias).shape)
        print(self.bias.shape)

        upper_bias = (
            pos_weight @ constraints.upper_bias + neg_weight @ constraints.lower_bias + self.bias
        )
        lower_bias = (
            pos_weight @ constraints.lower_bias + neg_weight @ constraints.upper_bias + self.bias
        )
        return upper_bias, lower_bias

    def box_from_constraints(self, orig_ub, orig_lb, constraints):
        """
        Given the constraints of this layer
        compute the box of this layer.
        """
        assert torch.all(orig_ub >= orig_lb)

        if constraints is None:
            return self.compute_new_box(orig_ub, orig_lb)

        assert isinstance(constraints, Constraints)

        pos_uc, neg_uc = self.pos_neg_split(constraints.upper_constraints.T)
        pos_lc, neg_lc = self.pos_neg_split(constraints.lower_constraints.T)

        # Compute the new upper and lower bound
        #print(pos_uc @ orig_ub + neg_uc @ orig_lb >= pos_lc @ orig_lb + neg_lc @ orig_ub)
        #print(constraints.upper_bias)
        #print(constraints.lower_bias)
    
        new_upper_bound = (
            pos_uc @ orig_ub + neg_uc @ orig_lb + constraints.upper_bias.unsqueeze(0).T
        )
        new_lower_bound = (
            pos_lc @ orig_lb + neg_lc @ orig_ub + constraints.lower_bias.unsqueeze(0).T
        )

        # Update the box
        self.upper_bound = new_upper_bound  # .unsqueeze(0)
        self.lower_bound = new_lower_bound  # .unsqueeze(0)

        assert torch.all(self.upper_bound >= self.lower_bound)
        assert self.upper_bound.shape == self.lower_bound.shape


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
        orig_ub, orig_lb, prev_ub, prev_lb, prev_constraints = inputs

        assert torch.all(prev_ub >= prev_lb)
        assert prev_ub.shape == prev_lb.shape

        # Compute simple box first
        self.compute_new_box(prev_ub, prev_lb)

        # Compute constraints using backsubstitution
        constraints = self.backsubstitution(prev_constraints, prev_ub, prev_lb)

        # Update box bounds with constraints
        self.box_from_constraints(orig_ub, orig_lb, constraints)

        return orig_ub, orig_lb, self.upper_bound, self.lower_bound, prev_constraints

    def compute_new_box(self, prev_ub, prev_lb):
        """
        Compute a simple box only from previous upper and lower bound.
        Save them in class variables.
        """
        self.upper_bound = prev_ub
        self.lower_bound = prev_lb

        self.upper_bound[self.upper_bound < 0] = 0
        self.lower_bound[self.lower_bound < 0] = 0
        #print("compute_new_box")
        #print("prev_ub", prev_ub)
        #print("prev_lb", prev_lb)
        #print("upper_bound", self.upper_bound)
        #print("lower_bound", self.lower_bound)
    
    def backsubstitution(self, prev_constraints, prev_ub, prev_lb):

        # Compute DeepPoly slopes
        self.compute_relu_slopes(prev_ub, prev_lb)

        # Compute constraints
        new_upper_constraints, new_lower_constraints = self.updated_constraints(prev_constraints)

        # Compute bias
        upper_bias, lower_bias = self.compute_bias(prev_constraints, prev_ub, prev_lb)

        new_constraints = Constraints(
            new_upper_constraints, new_lower_constraints, upper_bias, lower_bias
        )

        return new_constraints

    def updated_constraints(self, prev_constraints):
        """
        Given the constraints from the previous layers,
        compute the constraints of this layer.
        """
        assert isinstance(prev_constraints, Constraints)

        new_upper_constraints = torch.full_like(self.upper_bound_slope, float("nan"), dtype=torch.float32)
        new_lower_constraints = torch.full_like(self.lower_bound_slope, float("nan"), dtype=torch.float32)

        diag_ub = torch.diag(self.upper_bound_slope.view(-1))
        diag_lb = torch.diag(self.lower_bound_slope.view(-1))

        #print("Computing new constraints")
        #print("previous upper constraints", prev_constraints.upper_constraints.shape)
        #print("previous lower constraints", prev_constraints.lower_constraints.shape)
        #print("diag_ub", diag_ub)
        #print("diag_lb", diag_lb)

        new_upper_constraints = prev_constraints.upper_constraints.where(prev_constraints.upper_constraints >= 0, 0) @ diag_ub \
                                + prev_constraints.lower_constraints.where(prev_constraints.lower_constraints < 0, 0) @ diag_lb
        
        new_lower_constraints = prev_constraints.lower_constraints.where(prev_constraints.lower_constraints >= 0, 0) @ diag_lb \
                                + prev_constraints.upper_constraints.where(prev_constraints.upper_constraints < 0, 0) @ diag_ub

        #print("new_upper_constraints", new_upper_constraints)
        #print("new_lower_constraints", new_lower_constraints)

        assert new_upper_constraints.shape == prev_constraints.upper_constraints.shape
        assert new_lower_constraints.shape == prev_constraints.lower_constraints.shape
        assert torch.isnan(new_upper_constraints).sum() == 0
        assert torch.isnan(new_lower_constraints).sum() == 0

        return new_upper_constraints, new_lower_constraints

    def compute_bias(self, prev_constraints, prev_ub, prev_lb):
        """
        Given the constraints from the previous layers
        compute the bias of this layer.
        """
        assert isinstance(prev_constraints, Constraints)

        # print("####### upper_slope_crossing", upper_slope_crossing.shape)
        # print("####### upper_slope_crossing", upper_slope_crossing)

        this_layer_upper_bias = (self.upper_bound_slope * (-1) * prev_lb).view(-1)
        this_layer_lower_bias = torch.zeros_like(prev_lb).view(-1)

        #print("Computing new bias")
        #print("upper_bound_slopes", self.upper_bound_slope)
        #print("prev_lb", prev_lb)
        #print("this layer upper bias", this_layer_upper_bias)
        #print("this layer lower bias", this_layer_lower_bias)
        #print("this layer upper bias shape", this_layer_upper_bias.shape)
        #print("prev layer upper bias shape", prev_constraints.upper_bias.shape)
        #print("upper_bound_slope shape", self.upper_bound_slope.shape)

        upper_bias = self.upper_bound_slope.view(-1) * prev_constraints.upper_bias \
                        + this_layer_upper_bias
        
        lower_bias = self.lower_bound_slope.view(-1) * prev_constraints.lower_bias \
                        + this_layer_lower_bias
        
        #print("total upper_bias", upper_bias)
        #print("total lower_bias", lower_bias)
        #print("upper_bias shape", upper_bias.shape)
        #print("lower_bias shape", lower_bias.shape)

        assert upper_bias.shape == prev_constraints.upper_bias.shape
        assert lower_bias.shape == prev_constraints.lower_bias.shape
        assert upper_bias.isnan().sum() == 0
        assert lower_bias.isnan().sum() == 0

        return upper_bias, lower_bias
        
    def compute_relu_slopes(self, prev_ub, prev_lb):
        """
        Computes the slopes of the upper and lower bound for the constraints.
        Stores them in class variables.
        """
        print(prev_ub >= prev_lb)

        assert prev_ub.shape == prev_lb.shape
        assert torch.all(prev_ub >= prev_lb)

        # Compute upper and lower bound slopes
        self.upper_bound_slope = self.compute_upper_bound_slope(prev_ub, prev_lb)
        self.lower_bound_slope = self.compute_lower_bound_slopes(prev_ub, prev_lb)

        # Check if slopes valid
        assert self.upper_bound_slope.shape == self.lower_bound_slope.shape == prev_ub.shape
        assert (self.upper_bound_slope < 0).sum() == 0
        assert (self.lower_bound_slope < 0).sum() == 0
        assert torch.isnan(self.upper_bound_slope).sum() == 0
        assert torch.isnan(self.lower_bound_slope).sum() == 0
        
        #print("compute_relu_slopes")
        #print("prev_ub", prev_ub)
        #print("prev_lb", prev_lb)
        #print("upper_bound_slope", self.upper_bound_slope)
        #print("lower_bound_slope", self.lower_bound_slope)

    def compute_upper_bound_slope(self, ub, lb):
        # Compute upper slope for all crossing ReLus
        ub_slopes = torch.full_like(ub, float("nan"), dtype=torch.float32)

        ub_slopes[self.positive_relu_mask()] = 1
        ub_slopes[self.negative_relu_mask()] = 0

        # Division by zero is not possible due to crossing constraint: ub != lb
        ub_slopes[self.crossing_relu_mask()] = ub[self.crossing_relu_mask()] / (
            ub[self.crossing_relu_mask()] - lb[self.crossing_relu_mask()]
        )
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

        lb_slopes[self.positive_relu_mask()] = 1
        lb_slopes[self.negative_relu_mask()] = 0

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
        upper_bound = torch.full_like(self.prev_ub, float("nan"), dtype=torch.float32)
        lower_bound = torch.full_like(self.prev_lb, float("nan"), dtype=torch.float32)

        # Upper bound
        upper_bound[self.prev_ub > 0] = self.prev_ub[self.prev_ub > 0]
        upper_bound[self.prev_ub <= 0] = (
            self.leaky_relu_slope * self.prev_ub[self.prev_ub <= 0]
        )

        # Lower bound
        lower_bound[self.prev_lb > 0] = self.prev_lb[self.prev_lb > 0]
        lower_bound[self.prev_lb <= 0] = (
            self.leaky_relu_slope * self.prev_lb[self.prev_lb <= 0]
        )

        assert (
            upper_bound.shape
            == lower_bound.shape
            == self.prev_ub.shape
            == self.prev_lb.shape
        )
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

        assert (
            upper_slopes.shape
            == lower_slopes.shape
            == self.prev_ub.shape
            == self.prev_lb.shape
        )
        assert (
            upper_bias.shape
            == lower_bias.shape
            == self.prev_ub.shape
            == self.prev_lb.shape
        )

        return Constraints(upper_slopes, lower_slopes, upper_bias, lower_bias)

    def crossing_relu_mask(self):
        return (self.prev_ub > 0) & (self.prev_lb < 0)

    def compute_slope_bigger_one(self):
        # Initialize with NaN
        upper_slopes = torch.full_like(self.prev_ub, float("nan"), dtype=torch.float32)
        lower_slopes = torch.full_like(self.prev_lb, float("nan"), dtype=torch.float32)

        # Upper slopes
        upper_slopes[self.crossing_relu_mask()] = 1

        # Lower slopes
        # (ub - a*lb) / (ub - lb)
        rise = self.prev_ub[self.crossing_relu_mask()] - (
            self.leaky_relu_slope * self.prev_lb[self.crossing_relu_mask()]
        )
        run = (
            self.prev_ub[self.crossing_relu_mask()]
            - self.prev_lb[self.crossing_relu_mask()]
        )
        lower_slopes[self.crossing_relu_mask()] = rise / run

        # Only crossing relus have slopes, else nan
        return upper_slopes, lower_slopes

    def compute_slope_smaller_one(self):
        upper_slopes = torch.full_like(self.prev_ub, float("nan"), dtype=torch.float32)
        lower_slopes = torch.full_like(self.prev_lb, float("nan"), dtype=torch.float32)

        # Compute slopes
        upper_slopes[self.crossing_relu_mask()] = self.prev_ub[
            self.crossing_relu_mask()
        ] / (
            self.prev_ub[self.crossing_relu_mask()]
            - self.prev_lb[self.crossing_relu_mask()]
        )

        lower_slopes[self.crossing_relu_mask()] = 1

        return lower_slopes

    def compute_bias_bigger_one(self):
        # Initialize with NaN
        upper_bias = torch.full_like(self.prev_ub, float("nan"), dtype=torch.float32)
        lower_bias = torch.full_like(self.prev_lb, float("nan"), dtype=torch.float32)

        # Get the indices of crossing relus
        crossing_indices = self.crossing_relu_mask()

        # Upper bias
        upper_bias[crossing_indices] = 0

        # Lower bias
        numerator = (
            torch.square(self.prev_ub[crossing_indices])
            - self.prev_ub[crossing_indices] * self.prev_lb[crossing_indices]
        )
        denominator = (
            self.prev_ub[crossing_indices]
            - self.leaky_relu_slope * self.prev_lb[crossing_indices]
        )
        lower_bias[crossing_indices] = self.prev_ub[crossing_indices] - (
            numerator / denominator
        )

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
