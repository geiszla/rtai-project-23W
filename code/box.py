import torch
import torch.nn as nn

class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) / 0.3081


class View(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def get_model() -> nn.Sequential:
    # Add the data normalization as a first "layer" to the network
    # this allows us to search for adversarial examples to the real image, rather than
    # to the normalized image
    net = nn.Sequential(
        nn.Linear(28 * 28, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    return nn.Sequential(Normalize(), View(), *net)



class AbstractBox:

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb
        self.ub = ub

    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> 'AbstractBox':
        lb = x - eps
        lb.clamp_(min=0, max=1)

        ub = x + eps
        ub.clamp_(min=0, max=1)

        return AbstractBox(lb, ub)

    def propagate_normalize(self, normalize: Normalize) -> 'AbstractBox':
        # Follows from the rules in the lecture.
        lb = normalize(self.lb)
        ub = normalize(self.ub)
        return AbstractBox(lb, ub)

    def propagate_view(self, view: View) -> 'AbstractBox':
        lb = view(self.lb)
        ub = view(self.ub)
        return AbstractBox(lb, ub)

    def propagate_flatten(self, flatten: nn.Flatten) -> 'AbstractBox':
        lb = flatten(self.lb)
        ub = flatten(self.ub)
        return AbstractBox(lb, ub)

    def propagate_linear(self, fc: nn.Linear) -> 'AbstractBox':
        assert self.lb.shape == self.ub.shape
        assert len(self.lb.shape) == 2
        assert self.lb.shape[0] == 1 and self.lb.shape[1] == fc.weight.shape[1]

        lb = self.lb.repeat(fc.weight.shape[0], 1)
        ub = self.ub.repeat(fc.weight.shape[0], 1)
        assert lb.shape == ub.shape == fc.weight.shape

        # When computing the new lower/upper bounds, we need to take into account the sign of the
        # weight. Effectively, the expression that we want to overapproximate is:
        # x_1 * w_1 + x_2 * w_2 + ... + x_d * w_d,
        # where each x_i is overapproximated/abstracted by the box [lb_i, ub_i], i.e.
        # the concrete value of the neuron x_i can be any number from the interval [lb_i, ub_i].
        mul_lb = torch.where(fc.weight > 0, lb, ub)
        mul_ub = torch.where(fc.weight > 0, ub, lb)

        lb = (mul_lb * fc.weight).sum(dim=1)
        ub = (mul_ub * fc.weight).sum(dim=1)
        assert lb.shape == ub.shape == fc.bias.shape

        if fc.bias is not None:
            lb += fc.bias
            ub += fc.bias

        lb = lb.unsqueeze(0)
        ub = ub.unsqueeze(0)

        return AbstractBox(lb, ub)

    def propagate_relu(self, relu: nn.ReLU) -> 'AbstractBox':
        # Follows from the rules in the lecture.
        lb = relu(self.lb)
        ub = relu(self.ub)
        return AbstractBox(lb, ub)

    def check_postcondition(self, y) -> bool:
        ub = self.ub.squeeze()
        lb = self.lb.squeeze()
        mask = torch.ones_like(ub, dtype=torch.bool)
        mask[y] = False
        max_value = torch.max(torch.masked_select(ub, mask))
        return max_value < lb[y]


def certify_sample(model, x, y, eps) -> bool:
    box = AbstractBox.construct_initial_box(x, eps)
    for layer in model:
        if isinstance(layer, Normalize):
            box = box.propagate_normalize(layer)
        elif isinstance(layer, View):
            box = box.propagate_view(layer)
        elif isinstance(layer, nn.Flatten):
            box = box.propagate_flatten(layer)
        elif isinstance(layer, nn.Linear):
            box = box.propagate_linear(layer)
        elif isinstance(layer, nn.ReLU):
            box = box.propagate_relu(layer)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
    return box.check_postcondition(y)