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
        # print("-----------linear layer-----------")
        orig_ub, orig_lb, prev_ub, prev_lb, constraints = inputs
        

        assert torch.all(prev_ub >= prev_lb)
        assert prev_ub.shape == prev_lb.shape

        constraints = self.backsubstitution(constraints)
        # print("constraints: ")
        # print("uc:", constraints.upper_constraints)
        # print("lc:", constraints.lower_constraints)
        # print("ub:", constraints.upper_bias)
        # print("lb", constraints.lower_bias)
        self.box_from_constraints(orig_ub, orig_lb, constraints)
        # print("upper bound: ", self.upper_bound)
        # print("lower bound: ", self.lower_bound)

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

        # print percentage of weights that are zero
        # print("percentage of weights that are zero: ", (self.weight == 0).sum().item() / self.weight.numel())

        # Compute the new constraints and bias
        pos_weight, neg_weight = self.pos_neg_split(self.weight.T)
        # print("compute new constraints start")
        new_upper_constraints, new_lower_constraints = self.compute_new_constraints(
            pos_weight, neg_weight, constraints
        )
        # print("compute new constraints end")

        pos_weight, neg_weight = self.pos_neg_split(self.weight)
        # print("compute new bias start")
        upper_bias, lower_bias = self.compute_new_bias(
            pos_weight, neg_weight, constraints
        )
        # print("compute new bias end")
        # Update the constraints
        updated_constraints = Constraints(
            new_upper_constraints, new_lower_constraints, upper_bias, lower_bias
        )

        return updated_constraints

    def compute_new_constraints(self, pos_weight, neg_weight, constraints):
        # Compute the new constraints
        # print("pos_weight", pos_weight.shape)
        # print("constraints.upper_constraints", constraints.upper_constraints.shape)
        new_upper_constraints = (
            constraints.upper_constraints @ pos_weight
            + constraints.lower_constraints @ neg_weight
        )
        # new_upper_constraints = torch.sparse.mm(constraints.upper_constraints.to_sparse(), pos_weight) \
        #                         + torch.sparse.mm(constraints.lower_constraints.to_sparse(), neg_weight)

        # print("new_upper_constraints", new_upper_constraints.shape)
        new_lower_constraints = (
            constraints.upper_constraints @ neg_weight
            + constraints.lower_constraints @ pos_weight
        )
        return new_upper_constraints, new_lower_constraints

    def compute_new_bias(self, pos_weight, neg_weight, constraints):
        # Compute the new bias

        upper_bias = (
            pos_weight @ constraints.upper_bias
            + neg_weight @ constraints.lower_bias
            + self.bias
        )
        lower_bias = (
            pos_weight @ constraints.lower_bias
            + neg_weight @ constraints.upper_bias
            + self.bias
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
        # print("---------flatten layer-------------")

        flatten = nn.Flatten(start_dim=0)
        lb = flatten(prev_lb).unsqueeze(1)
        ub = flatten(prev_ub).unsqueeze(1)
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
        # print("--------------conv layer--------------")
        orig_ub, orig_lb, prev_ub, prev_lb, constraints = inputs
        assert prev_ub.shape == prev_lb.shape

        # check wether input shape has 3 or 4 dimensions
        if len(prev_ub.shape) == 3:
            _, input_height, input_width = prev_ub.shape
        elif len(prev_ub.shape) == 2:
            channels = self.layer.in_channels
            # squareroot of width 
            input_width = int((prev_ub.shape[0] // channels) ** 0.5)
            input_height = input_width
            assert type(input_width) == int
        else:
            raise ValueError("Input shape error in conv layer") 
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
        flatten = nn.Flatten(start_dim=0)
        if constraints is None:
            # print("orig_ub", orig_ub.shape)
            # print("flatten orig_ub", flatten(orig_ub).unsqueeze(1).shape)
            new_inputs = (
                flatten(orig_ub).unsqueeze(1),
                flatten(orig_lb).unsqueeze(1),
                flatten(prev_ub).unsqueeze(1),
                flatten(prev_lb).unsqueeze(1),
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
    def __init__(self, layer, input_size):
        super(DeepPolyReLu, self).__init__()

        # Initialize the alpha learnable parameters
        self.alpha = torch.nn.Parameter(torch.rand(input_size))
        self.alpha.requires_grad = True

    def forward(self, inputs):
        # print("-------------relu layer---------------")
        orig_ub, orig_lb, prev_ub, prev_lb, prev_constraints = inputs

        assert torch.all(prev_ub >= prev_lb)
        assert prev_ub.shape == prev_lb.shape

        self.prev_ub = prev_ub
        self.prev_lb = prev_lb

        
        # Compute constraints using backsubstitution
        # print("backsubstitution start")
        constraints = self.backsubstitution(prev_constraints)
        # print("backsubstitution end")
        # print("upper bound slope", self.upper_bound_slope)
        # print("lower bound slope", self.lower_bound_slope)
        # print("constraints: ")
        # print("Test")
        # print("uc:", constraints.upper_constraints)
        # print("lc:", constraints.lower_constraints)
        # print("ub:", constraints.upper_bias)
        # print("lb", constraints.lower_bias)


        # Update box bounds with constraints
        self.box_from_constraints(orig_ub, orig_lb, constraints)
        # print("upper bound: ", self.upper_bound)
        # print("lower bound: ", self.lower_bound)
        return orig_ub, orig_lb, self.upper_bound, self.lower_bound, constraints
    
    def backsubstitution(self, prev_constraints):

        # Compute DeepPoly slopes
        self.compute_relu_slopes()
        # Compute bias
        self.compute_bias()

        # Compute constraints
        new_upper_constraints, new_lower_constraints, upper_bias, lower_bias = self.updated_constraints(prev_constraints)

        new_constraints = Constraints(
            new_upper_constraints, new_lower_constraints, upper_bias, lower_bias
        )

        return new_constraints

    def updated_constraints(self, prev_constraints):
        """
        Given the constraints from the previous layers,
        compute the constraints of this layer.
        """
        # assert isinstance(prev_constraints, Constraints)

        # Update the factor matrix
        #diag_ub = torch.diag(self.upper_bound_slope.view(-1))
        #diag_lb = torch.diag(self.lower_bound_slope.view(-1))

        new_upper_constraints = prev_constraints.upper_constraints * (self.upper_bound_slope.view(-1).repeat(prev_constraints.upper_constraints.shape[0], 1))
        #print("> upper constraint start")
        #print("diag_ub", diag_ub.shape) 
        #new_upper_constraints = prev_constraints.upper_constraints @ diag_ub
        

        #assert torch.all(new_upper_constraints == new_upper_constraints2)
        
        #new_lower_constraints = prev_constraints.lower_constraints @ diag_lb
        new_lower_constraints = prev_constraints.lower_constraints * (self.lower_bound_slope.view(-1).repeat(prev_constraints.lower_constraints.shape[0], 1))

        # update the bias
        upper_bias = self.upper_bound_slope.view(-1) * prev_constraints.upper_bias \
                        + self.this_layer_upper_bias
        
        lower_bias = self.lower_bound_slope.view(-1) * prev_constraints.lower_bias \
                        + self.this_layer_lower_bias


        assert new_upper_constraints.shape == prev_constraints.upper_constraints.shape
        assert new_lower_constraints.shape == prev_constraints.lower_constraints.shape
        assert torch.isnan(new_upper_constraints).sum() == 0
        assert torch.isnan(new_lower_constraints).sum() == 0

        return new_upper_constraints, new_lower_constraints, upper_bias, lower_bias

    def compute_bias(self):
        """
        Given the constraints from the previous layers
        compute the bias of this layer.
        """

        self.this_layer_upper_bias = (self.upper_bound_slope * (-1) * self.prev_lb)
        self.this_layer_lower_bias = torch.zeros_like(self.prev_lb).view(-1)
        self.this_layer_upper_bias[~self.crossing_relu_mask()] = 0
        self.this_layer_upper_bias = self.this_layer_upper_bias.view(-1)

        
    def compute_relu_slopes(self):
        """
        Computes the slopes of the upper and lower bound for the constraints.
        Stores them in class variables.
        """
        assert self.prev_ub.shape == self.prev_lb.shape
        assert torch.all(self.prev_ub >= self.prev_lb)

        # Compute upper and lower bound slopes
        self.upper_bound_slope = self.compute_upper_bound_slope()
        self.lower_bound_slope = self.compute_lower_bound_slopes()

        # Check if slopes valid
        assert self.upper_bound_slope.shape == self.lower_bound_slope.shape == self.prev_ub.shape
        assert (self.upper_bound_slope < 0).sum() == 0
        assert (self.lower_bound_slope < 0).sum() == 0
        assert torch.isnan(self.upper_bound_slope).sum() == 0
        assert torch.isnan(self.lower_bound_slope).sum() == 0

    def compute_upper_bound_slope(self):
        # Compute upper slope for all crossing ReLus
        ub_slopes = torch.full_like(self.prev_ub, float("nan"), dtype=torch.float32)

        ub_slopes[self.positive_relu_mask()] = 1
        ub_slopes[self.negative_relu_mask()] = 0

        # Division by zero is not possible due to crossing constraint: ub != lb
        ub_slopes[self.crossing_relu_mask()] = self.prev_ub[self.crossing_relu_mask()] / (
            self.prev_ub[self.crossing_relu_mask()] - self.prev_lb[self.crossing_relu_mask()]
        )
        return ub_slopes

    def compute_lower_bound_slopes(self):
        # Compute lower slope for all crossing ReLus based on two DeepPoly variants
        # All other slopes stay nan
        lb_slopes = torch.full_like(self.prev_lb, float("nan"), dtype=torch.float32)

        assert (
            self.deep_poly_variant_1_mask().sum()
            + self.deep_poly_variant_2_mask().sum()
            == self.crossing_relu_mask().sum()
        )

        # Compute slope
        lb_slopes[self.deep_poly_variant_1_mask()] = 0
        lb_slopes[self.deep_poly_variant_2_mask()] = self.alpha

        lb_slopes[self.positive_relu_mask()] = 1
        lb_slopes[self.negative_relu_mask()] = 0

        return lb_slopes

    def positive_relu_mask(self):
        return (self.prev_ub >= 0) & (self.prev_lb >= 0)

    def negative_relu_mask(self):
        return self.prev_ub <= 0

    def crossing_relu_mask(self):
        return (self.prev_ub > 0) & (self.prev_lb < 0)

    def deep_poly_variant_1_mask(self):
        return self.crossing_relu_mask() & (self.prev_ub <= abs(self.prev_lb))

    def deep_poly_variant_2_mask(self):
        return self.crossing_relu_mask() & (self.prev_ub > abs(self.prev_lb))


class DeepPolyLeakyReLu(DeepPolyReLu):
    def __init__(self, layer):
        super(DeepPolyLeakyReLu, self).__init__(layer)
        self.leaky_relu_slope = layer.negative_slope
        self.prev_ub = None
        self.prev_lb = None

    def compute_relu_slopes(self):
        """
        Compute the slopes of the upper and lower bound for the constraints.
        Use two different cases for leaky relu slope bigger or smaller than 1.
        Set class variables.
        """
        if self.leaky_relu_slope >= 1:
            ubs, lbs = self.compute_slope_bigger_one()
        else:
            ubs, lbs = self.compute_slope_smaller_one()
        
        # Set class variables
        self.upper_bound_slope = ubs
        self.lower_bound_slope = lbs

    def compute_bias(self): 
        """
        Compute the bias of the upper and lower bound for the constraints.
        Use two different cases for leaky relu slope bigger or smaller than 1.
        """
        if self.leaky_relu_slope >= 1:
            ub, lb = self.compute_bias_bigger_one()
        else:
            ub, lb = self.compute_bias_smaller_one()

        self.this_layer_upper_bias = ub.view(-1)
        self.this_layer_lower_bias = lb.view(-1)

    def compute_slope_bigger_one(self):

        # Initialize with NaN
        upper_slopes = torch.full_like(self.prev_ub, float("nan"), dtype=torch.float32)
        lower_slopes = torch.full_like(self.prev_lb, float("nan"), dtype=torch.float32)

        # Upper slopes
        upper_slopes[self.crossing_relu_mask()] = 1
        upper_slopes[self.positive_relu_mask()] = 1
        upper_slopes[self.negative_relu_mask()] = self.leaky_relu_slope

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
        lower_slopes[self.positive_relu_mask()] = 1
        lower_slopes[self.negative_relu_mask()] = self.leaky_relu_slope

        # Check if slopes valid
        assert torch.isnan(upper_slopes).sum() == 0
        assert torch.isnan(lower_slopes).sum() == 0

        return upper_slopes, lower_slopes

    def compute_slope_smaller_one(self):
        # Initialize with NaN
        upper_slopes = torch.full_like(self.prev_ub, float("nan"), dtype=torch.float32)
        lower_slopes = torch.full_like(self.prev_lb, float("nan"), dtype=torch.float32)

        # Upper slopes
        # (ub - a*lb) / (ub - lb)
        rise = self.prev_ub - (
            self.leaky_relu_slope * self.prev_lb
        )
        run = (
            self.prev_ub
            - self.prev_lb
        )
        upper_slopes[self.crossing_relu_mask()] = rise[self.crossing_relu_mask()] / run[self.crossing_relu_mask()]
        upper_slopes[self.positive_relu_mask()] = 1
        upper_slopes[self.negative_relu_mask()] = self.leaky_relu_slope

        # Lower slopes
        lower_slopes[self.crossing_relu_mask()] = 1
        lower_slopes[self.positive_relu_mask()] = 1
        lower_slopes[self.negative_relu_mask()] = self.leaky_relu_slope

        return upper_slopes, lower_slopes

    def compute_bias_bigger_one(self):
        # Upper bias
        upper_bias = torch.zeros_like(self.prev_ub, dtype=torch.float32)

        # Lower bias

        # Initialize with NaN
        lower_bias = torch.full_like(self.prev_lb, float("nan"), dtype=torch.float32)

        # b = u - m*u
        lower_bias = self.prev_ub - self.lower_bound_slope * self.prev_ub

        # Set all non-crossing ReLUs to 0
        lower_bias[self.positive_relu_mask()] = 0
        lower_bias[self.negative_relu_mask()] = 0

        return upper_bias, lower_bias

    def compute_bias_smaller_one(self):
        # Upper bias
        # b = u - m*u
        upper_bias = self.prev_ub - self.upper_bound_slope * self.prev_ub
        upper_bias[self.positive_relu_mask()] = 0
        upper_bias[self.negative_relu_mask()] = 0

        # Lower bias
        lower_bias = torch.zeros_like(self.prev_lb, dtype=torch.float32)
    
        return upper_bias, lower_bias

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
    reluLayer = nn. LeakyReLU(0.5)

    verifier = DeepPolyLeakyReLu(reluLayer)
    verifier.prev_ub = prev_ub
    verifier.prev_lb = prev_lb


    print("Prev upper bound", verifier.prev_ub)
    print("Prev lower bound", verifier.prev_lb)
    print("Crossing relu mask", verifier.crossing_relu_mask())
    # print("Positive relu mask", verifier.positive_relu_mask())
    # print("Negative relu mask", verifier.negative_relu_mask())
    verifier.compute_relu_slopes()
    print("Upper slopes", verifier.upper_bound_slope)
    print("Lower slopes", verifier.lower_bound_slope)
    verifier.compute_bias()
    print("Upper bias", verifier.this_layer_upper_bias)
    print("Lower bias", verifier.this_layer_lower_bias)


if __name__ == "__main__":
    main()
