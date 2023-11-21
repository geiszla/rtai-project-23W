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
        self.upper_bound_slope = torch.full_like(self.upper_bound, float('nan'), dtype=torch.float32)
        self.lower_bound_slope = torch.full_like(self.upper_bound, float('nan'), dtype=torch.float32)

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
        ub_slopes = torch.full_like(ub, float('nan'), dtype=torch.float32)

        # Division by zero is not possible due to crossing constraint: ub != lb
        ub_slopes[self.crossing_relu_mask()] = ub[self.crossing_relu_mask()] / (ub[self.crossing_relu_mask()] - lb[self.crossing_relu_mask()])

        # Check if slopes valid
        assert ub_slopes.shape == ub.shape
        assert (ub_slopes < 0).sum() == 0
        assert torch.isnan(ub_slopes).sum() == self.crossing_relu_mask().sum()

        return ub_slopes
    
    def compute_lower_bound_slopes(self, ub, lb):
        # Compute lower slope for all crossing ReLus based on two DeepPoly variants
        # All other slopes stay nan
        lb_slopes = torch.full_like(lb, float('nan'), dtype=torch.float32)

        assert self.deep_poly_variant_1_mask().sum() + self.deep_poly_variant_2_mask().sum() == self.crossing_relu_mask().sum()

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
        return (self.upper_bound <= 0)
    
    def crossing_relu_mask(self):
        return (self.upper_bound > 0) & (self.lower_bound < 0)
    
    def deep_poly_variant_1_mask(self):
        return self.crossing_relu_mask() & (self.upper_bound <= abs(self.lower_bound))
    
    def deep_poly_variant_2_mask(self):
        return self.crossing_relu_mask() & (self.upper_bound > abs(self.lower_bound))
    
def main():
    ub = torch.tensor([[-1,1], [2,3], [2,1]])
    lb = torch.tensor([[-3,-1], [1,-1], [1,-1]])
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