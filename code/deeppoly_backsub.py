from abc import abstractmethod

import torch
from toeplitz import get_convolution_output_size, get_toeplitz_convolution
from torch import nn


class DeepPolyBase(nn.Module):
    def __init__(self, previous_layer=None):
        super().__init__()

        # Double linking the network layers
        self.previous_layer = previous_layer

        # Each layer has a box represented by upper and lower bound
        self.upper_bound = None
        self.lower_bound = None

        # Each layer has additional constraints
        self.constraints = None

    def calculate_new_bounds(
        self, prev_ub, prev_lb, weight_ub, weight_lb, bias_ub, bias_lb
    ):
        """Calculate new bounds for this layer"""
        assert torch.all(prev_ub >= prev_lb)

        pos_uc, neg_uc = self.pos_neg_split(weight_ub)
        pos_lc, neg_lc = self.pos_neg_split(weight_lb)

        # Compute the new upper and lower bound

        new_upper_bound = pos_uc @ prev_ub + neg_uc @ prev_lb + bias_ub
        new_lower_bound = pos_lc @ prev_lb + neg_lc @ prev_ub + bias_lb

        # Update the box
        self.upper_bound = new_upper_bound
        self.lower_bound = new_lower_bound

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
    def backsubstitution(self, weight_ub, weight_lb, bias_ub, bias_lb):
        # Given the constraints from the previous layers
        # compute the constraints of this layer
        return

    @abstractmethod
    def forward(self, upper_bound, lower_bound):
        # Pushing the previous box through this layer
        # Obtain the new box
        return


class DeepPolyLinear(DeepPolyBase):
    def __init__(self, fc, previous_layer):
        super().__init__(previous_layer)

        self.weight = fc.weight
        self.bias = fc.bias

    def forward(self, inputs):
        """
        Pushing the previous box through this layer.
        Calling backsubstitution to compute additional constraints.
        """
        # print("-----------linear layer-----------")
        orig_ub, orig_lb, prev_ub, prev_lb = inputs

        assert torch.all(prev_ub >= prev_lb)
        assert prev_ub.shape == prev_lb.shape

        (
            first_layer,
            weight_ub,
            weight_lb,
            bias_ub,
            bias_lb,
        ) = (
            self.previous_layer.backsubstitution(
                self.weight, self.weight, self.bias, self.bias
            )
            if self.previous_layer is not None
            else (None, self.weight, self.weight, self.bias, self.bias)
        )

        self.calculate_new_bounds(
            first_layer.upper_bound if first_layer is not None else orig_ub,
            first_layer.lower_bound if first_layer is not None else orig_lb,
            weight_ub,
            weight_lb,
            bias_ub,
            bias_lb,
        )

        return orig_ub, orig_lb, self.upper_bound, self.lower_bound

    def backsubstitution(self, weight_ub, weight_lb, bias_ub, bias_lb):
        new_weight_ub = weight_ub @ self.weight
        new_weight_lb = weight_lb @ self.weight

        new_bias_ub = weight_ub @ self.bias + bias_ub
        new_bias_lb = weight_lb @ self.bias + bias_lb

        if self.previous_layer is not None:
            return self.previous_layer.backsubstitution(
                new_weight_ub,
                new_weight_lb,
                new_bias_ub,
                new_bias_lb,
            )

        return (
            self.previous_layer,
            new_weight_ub,
            new_weight_lb,
            new_bias_ub,
            new_bias_lb,
        )


class DeepPolyFlatten(DeepPolyBase):
    def __init__(self, previous_layer):
        super().__init__(previous_layer)

    def forward(self, inputs):
        """
        Pushing the previous box through this layer.
        """
        orig_ub, orig_lb, prev_ub, prev_lb = inputs
        # print("---------flatten layer-------------")

        self.lower_bound = torch.flatten(prev_lb)
        self.upper_bound = torch.flatten(prev_ub)

        if self.previous_layer is None:
            return (
                self.upper_bound,
                self.lower_bound,
                self.upper_bound,
                self.lower_bound,
            )

        return (orig_ub, orig_lb, self.upper_bound, self.lower_bound)

    def backsubstitution(self, weight_ub, weight_lb, bias_ub, bias_lb):
        previous_layer = self if self.previous_layer is None else self.previous_layer
        return (previous_layer, weight_ub, weight_lb, bias_ub, bias_lb)


class DeepPolyConvolution(DeepPolyLinear):
    def __init__(self, layer, true_shape, previous_layer):
        self.weight = None
        self.bias = None

        super().__init__(self, previous_layer)

        self.layer = layer
        self.true_shape = true_shape

    def forward(self, inputs):
        # print("--------------conv layer--------------")
        orig_ub, orig_lb, prev_ub, prev_lb = inputs
        assert prev_ub.shape == prev_lb.shape

        # # check wether input shape has 3 or 4 dimensions
        # if len(prev_ub.shape) == 3:
        #     _, input_height, input_width = prev_ub.shape
        # elif len(prev_ub.shape) == 2:
        #     channels = self.layer.in_channels
        #     # squareroot of width
        #     input_width = int((prev_ub.shape[0] // channels) ** 0.5)
        #     input_height = input_width
        #     assert type(input_width) == int
        # else:
        #     raise ValueError("Input shape error in conv layer")

        # _, _, kernel_height, kernel_width = self.layer.weight.data.shape

        stride = self.layer.stride[0]
        padding = self.layer.padding[0]

        self.weight = get_toeplitz_convolution(
            self.layer.weight.data,
            self.true_shape,
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

        if self.previous_layer is None:
            # print("orig_ub", orig_ub.shape)
            # print("flatten orig_ub", flatten(orig_ub).unsqueeze(1).shape)
            new_inputs = (
                flatten(orig_ub),
                flatten(orig_lb),
                flatten(prev_ub),
                flatten(prev_lb),
            )
        else:
            new_inputs = inputs

        # Use linear forward pass
        linear_result = super().forward((new_inputs))

        return linear_result
        # linear_result = list(linear_result)

        # output_height = get_convolution_output_size(
        #     input_height, kernel_height, stride, padding
        # )
        # output_width = get_convolution_output_size(
        #     input_width, kernel_width, stride, padding
        # )

        # output_shape = (self.layer.out_channels, output_height, output_width)

        # self.upper_bound = self.upper_bound.view(output_shape)
        # self.lower_bound = self.lower_bound.view(output_shape)

        # linear_result[2] = self.upper_bound
        # linear_result[3] = self.lower_bound

        # return tuple(linear_result)


class DeepPolyReLu(DeepPolyBase):
    def __init__(self, layer, input_shape, previous_layer):
        super().__init__(previous_layer)

        # Initialize the alpha learnable parameters
        self.alpha = nn.Parameter(torch.zeros(input_shape))
        self.alpha.requires_grad = True

    def forward(self, inputs):
        # print("-------------relu layer---------------")
        orig_ub, orig_lb, prev_ub, prev_lb = inputs

        assert torch.all(prev_ub >= prev_lb)
        assert prev_ub.shape == prev_lb.shape

        self.prev_ub = prev_ub
        self.prev_lb = prev_lb

        # Compute DeepPoly slopes
        self.compute_relu_slopes()

        # Compute bias
        self.compute_bias()

        (
            first_layer,
            weight_ub,
            weight_lb,
            bias_ub,
            bias_lb,
        ) = (
            self.previous_layer.backsubstitution(
                torch.diag(self.upper_bound_slope),
                torch.diag(self.lower_bound_slope),
                self.this_layer_upper_bias,
                self.this_layer_lower_bias,
            )
            if self.previous_layer is not None
            else (None, self.weight, self.weight, self.bias, self.bias)
        )

        self.calculate_new_bounds(
            first_layer.upper_bound if first_layer is not None else orig_ub,
            first_layer.lower_bound if first_layer is not None else orig_lb,
            weight_ub,
            weight_lb,
            bias_ub,
            bias_lb,
        )

        return orig_ub, orig_lb, self.upper_bound, self.lower_bound

    def backsubstitution(self, weight_ub, weight_lb, bias_ub, bias_lb):
        pos_uc, neg_uc = self.pos_neg_split(weight_ub)
        pos_lc, neg_lc = self.pos_neg_split(weight_lb)

        upper_bound_slope = torch.diag(self.upper_bound_slope)
        lower_bound_slope = torch.diag(self.lower_bound_slope)

        new_weight_ub = pos_uc @ upper_bound_slope + neg_uc @ lower_bound_slope
        new_weight_lb = pos_lc @ lower_bound_slope + neg_lc @ upper_bound_slope

        new_bias_ub = (
            bias_ub
            + pos_uc @ self.this_layer_upper_bias
            + neg_uc @ self.this_layer_lower_bias
        )

        new_bias_lb = (
            bias_lb
            + pos_lc @ self.this_layer_lower_bias
            + neg_lc @ self.this_layer_upper_bias
        )

        if self.previous_layer is not None:
            return self.previous_layer.backsubstitution(
                new_weight_ub, new_weight_lb, new_bias_ub, new_bias_lb
            )

        return (
            self.previous_layer,
            new_weight_ub,
            new_weight_lb,
            new_bias_ub,
            new_bias_lb,
        )

    def compute_bias(self):
        """
        Given the constraints from the previous layers
        compute the bias of this layer.
        """

        self.this_layer_upper_bias = self.upper_bound_slope * (-1) * self.prev_lb
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
        self.upper_bound_slope = self.compute_upper_bound_slopes()
        self.lower_bound_slope = self.compute_lower_bound_slopes()

        # Check if slopes valid
        assert (
            self.upper_bound_slope.shape
            == self.lower_bound_slope.shape
            == self.prev_ub.shape
        )
        assert (self.upper_bound_slope < 0).sum() == 0
        assert (self.lower_bound_slope < 0).sum() == 0
        assert torch.isnan(self.upper_bound_slope).sum() == 0
        assert torch.isnan(self.lower_bound_slope).sum() == 0

    def compute_upper_bound_slopes(self):
        # Compute upper slope for all crossing ReLus
        ub_slopes = torch.full_like(self.prev_ub, float("nan"), dtype=torch.float32)

        ub_slopes[self.positive_relu_mask()] = 1
        ub_slopes[self.negative_relu_mask()] = 0

        # Division by zero is not possible due to crossing constraint: ub != lb
        ub_slopes[self.crossing_relu_mask()] = self.prev_ub[
            self.crossing_relu_mask()
        ] / (
            self.prev_ub[self.crossing_relu_mask()]
            - self.prev_lb[self.crossing_relu_mask()]
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
        lb_slopes[self.deep_poly_variant_2_mask()] = 1

        # self.alpha = nn.Parameter(lb_slopes.clone())

        # print first 10 values of alpha
        # print("alpha", self.alpha[:10])

        lb_slopes[self.crossing_relu_mask()] = self.alpha[self.crossing_relu_mask()]

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
    def __init__(self, layer, input_shape, previous_layer):
        super(DeepPolyLeakyReLu, self).__init__(layer, input_shape, previous_layer)

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
        upper_slopes[self.crossing_relu_mask()] = self.alpha[self.crossing_relu_mask()]
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
        rise = self.prev_ub - (self.leaky_relu_slope * self.prev_lb)
        run = self.prev_ub - self.prev_lb
        upper_slopes[self.crossing_relu_mask()] = (
            rise[self.crossing_relu_mask()] / run[self.crossing_relu_mask()]
        )
        upper_slopes[self.positive_relu_mask()] = 1
        upper_slopes[self.negative_relu_mask()] = self.leaky_relu_slope

        # Lower slopes
        lower_slopes[self.crossing_relu_mask()] = self.alpha[self.crossing_relu_mask()]
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
