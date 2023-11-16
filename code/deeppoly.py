import torch
from torch import nn
from abc import ABC, abstractmethod

class DeepPolyBase(nn.Module):
    
    def __init__(self, ub, lb):
        super(DeepPolyBase, self).__init__()
        assert ub.shape == lb.shape
        assert (lb > ub).sum() == 0
        self.upper_bound = ub
        self.lower_bound = lb

    @abstractmethod
    def forward(self, x):
        return x
        
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