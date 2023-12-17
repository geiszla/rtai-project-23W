import argparse

import torch
import torch.nn as nn

# from box import certify_sample, AbstractBox
from deeppoly import DeepPolyConvolution, DeepPolyFlatten, DeepPolyLinear, DeepPolyReLu
from networks import get_network
from torch import optim
from utils.loading import parse_spec
# from box import certify_sample, AbstractBox
from deeppoly import DeepPolyLinear, DeepPolyFlatten, DeepPolyReLu, DeepPolyLeakyReLu, DeepPolyConvolution

DEVICE = "cpu"


# def analyze(
#     net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
# ) -> bool:
#     return certify_sample(net, inputs, true_label, eps)


def check_postcondition(upper_bound, lower_bound, true_label):
    mask = torch.ones_like(upper_bound, dtype=torch.bool)
    mask[true_label] = False
    max_value = torch.max(torch.masked_select(upper_bound, mask))
    # print("max_value", max_value)
    # print("lower_bound", lower_bound)
    # print("true_label", true_label)
    # print("upper_bound", upper_bound)
    return max_value < lower_bound[true_label]


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    layers = []

    prev_layer = None

    for layer in net:
        if isinstance(layer, nn.Flatten):
            poly_layer = DeepPolyFlatten()
        elif isinstance(layer, nn.Linear):
            poly_layer = DeepPolyLinear(layer)
        elif isinstance(layer, nn.ReLU):
            if prev_layer is None:
                raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
            poly_layer = DeepPolyReLu(layer)
        elif isinstance(layer, nn.LeakyReLU):
            if prev_layer is None:
                raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
            poly_layer = DeepPolyLeakyReLu(layer)
        elif isinstance(layer, nn.Conv2d):
            poly_layer = DeepPolyConvolution(layer)
        else:
            raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

        layers.append(poly_layer)
        prev_layer = poly_layer

    polynet = nn.Sequential(*layers)

    upper_bound = inputs + eps
    upper_bound.clamp_(min=0, max=1)
    lower_bound = inputs - eps
    lower_bound.clamp_(min=0, max=1)

    # upper_bound, lower_bound, _constraints = polynet((upper_bound, lower_bound, None))
    _orig_ub, _orig_lb, upper_bound, lower_bound, _constraints = polynet(
        (upper_bound, lower_bound, upper_bound, lower_bound, None)
    )
    # print("upper_bound", upper_bound)
    # print("lower_bound", lower_bound)
    # print("true_label", true_label)

    optimizer = optim.Adam(polynet.parameters(), lr=0.7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

    for _ in range(10):
        optimizer.zero_grad()

        result = check_postcondition(upper_bound, lower_bound, true_label)

        if (result > 0).all():
            return True

        loss = torch.log(-result[result < 0]).max()
        loss.backward()
        optimizer.step()

        for parameter in polynet.parameters():
            if parameter.requires_grad:
                parameter.data.clamp_(0, 1)

        if scheduler.get_last_lr()[0] > 0.1:
            scheduler.step()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)
    # print(net)
    print(args.net)
    print(args.spec)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")



if __name__ == "__main__":
    main()
