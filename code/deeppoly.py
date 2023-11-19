import torch
from torch import nn
from abc import ABC, abstractmethod

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
        super(DeepPolyConvolution, self).__init__()
    
    def forward(self, x):
        pass

class DeepPolyReLu(DeepPolyBase):

    def __init__(self, ub, lb):
        super(DeepPolyReLu, self).__init__(ub, lb)
        self.upper_bound_slope = torch.full_like(self.upper_bound, float('nan'))
        self.lower_bound_slope = torch.full_like(self.upper_bound, float('nan'))

    def forward(self, x):
        # Case 1: positive ReLu
        # We can skip this as the upper bound and lower bound stay the same

        # Case 2: negative ReLu
        # Set ub = lb = 0
        negative_mask = self.upper_bound < 0
        self.upper_bound[negative_mask] = 0
        self.lower_bound[negative_mask] = 0

        # Case 3: crossing ReLu
        # Use DeepPoly to compute constraints

        # Determine variants
        deep_poly_variant_1_mask = self.upperbound >= abs(self.lower_bound)
        deep_poly_variant_2_mask = self.upperbound < abs(self.lower_bound)
        assert (deep_poly_variant_1_mask + deep_poly_variant_2_mask).sum() == self.upper_bound.size

        # Compute upper_bound_slope (same for both variants)
        self.upper_bound_slope = self.compute_upper_bound_slope(self.upper_bound, self.lower_bound)
    
    def compute_upper_bound_slope(self, ub, lb):
        ub_slopes = torch.full_like(ub, float('nan'))

        # Precaution against divison by zero
        ub_equal_lb_mask = ub == lb
        ub_not_equal_lb_mask = ub != lb

        # Compute slope
        ub_slopes[ub_equal_lb_mask] = 0
        ub_slopes[ub_not_equal_lb_mask] = ub[ub_not_equal_lb_mask] / (ub[ub_not_equal_lb_mask] - lb[ub_not_equal_lb_mask])

        # Check if slope is valid
        assert ub_slopes.shape == ub.shape
        assert (ub_slopes < 0).sum() == 0
        assert torch.nan(ub_slopes).sum() == 0

        return ub_slopes



class DeepPolyLinear(nn.Module):

    def __init__(self, fc):
        super(DeepPolyLinear, self).__init__()
        self.weight = fc.weight
        self.bias = fc.bias

    def forward(self, x):
        assert x.lower_bound.shape == x.upper_bound.shape
        assert len(x.lower_bound.shape) == 2
        assert x.lower_bound.shape[0] == 1 and x.lower_bound.shape[1] == self.weight.shape[1]

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

