import argparse

import numpy as np
import torch
import torch.nn as nn

# from box import certify_sample, AbstractBox
from deeppoly_backsub import (
    DeepPolyConvolution,
    DeepPolyFlatten,
    DeepPolyLeakyReLu,
    DeepPolyLinear,
    DeepPolyReLu,
)

# from deeppoly import DeepPolyReLu
from networks import get_network
from toeplitz import get_convolution_output_size
from torch import optim
from utils.loading import parse_spec

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
    return lower_bound[true_label] - max_value  # max_value < lower_bound[true_label]


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    layers = []

    prev_layer = None
    prev_poly_layer = None

    prev_layer_output_shape = inputs.shape
    prev_layer_flattened_shape = inputs.shape

    for layer in net:
        if isinstance(layer, nn.Flatten):
            poly_layer = DeepPolyFlatten(prev_poly_layer)
            prev_layer_output_shape = (np.prod(prev_layer_output_shape),)
        elif isinstance(layer, nn.Linear):
            poly_layer = DeepPolyLinear(layer, prev_poly_layer)
            prev_layer_output_shape = (layer.out_features,)
        elif isinstance(layer, nn.ReLU):
            if prev_layer is None:
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

            poly_layer = DeepPolyReLu(
                layer, prev_layer_flattened_shape, prev_poly_layer
            )
        elif isinstance(layer, nn.LeakyReLU):
            if prev_layer is None:
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

            poly_layer = DeepPolyLeakyReLu(
                layer, prev_layer_flattened_shape, prev_poly_layer
            )
        elif isinstance(layer, nn.Conv2d):
            poly_layer = DeepPolyConvolution(
                layer, prev_layer_output_shape, prev_poly_layer
            )

            output_height = get_convolution_output_size(
                prev_layer_output_shape[-1],
                layer.kernel_size[-1],
                layer.stride[0],
                layer.padding[0],
            )

            output_width = get_convolution_output_size(
                prev_layer_output_shape[-2],
                layer.kernel_size[-2],
                layer.stride[0],
                layer.padding[0],
            )

            prev_layer_output_shape = (layer.out_channels, output_height, output_width)
        else:
            raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

        prev_layer_flattened_shape = (
            np.prod(prev_layer_output_shape)
            if isinstance(layer, nn.Conv2d)
            else prev_layer_output_shape
        )

        layers.append(poly_layer)

        prev_layer = layer
        prev_poly_layer = poly_layer

    polynet = nn.Sequential(*layers)

    upper_bound = inputs + eps
    upper_bound.clamp_(min=0, max=1)
    lower_bound = inputs - eps
    lower_bound.clamp_(min=0, max=1)

    # upper_bound, lower_bound, _constraints = polynet((upper_bound, lower_bound, None))

    # print("upper_bound", upper_bound)
    # print("lower_bound", lower_bound)
    # print("true_label", true_label)

    (
            _orig_ub,
            _orig_lb,
            upper_bound_result,
            lower_bound_result,
        ) = polynet((upper_bound, lower_bound, upper_bound, lower_bound))
    
    result = check_postcondition(upper_bound_result, lower_bound_result, true_label)

    # for idx in range(len(polynet)):
    #     layer = polynet[idx]
    #     print("Layer: ", idx)
    #     layer.constraints.print_constraints()

    print(lower_bound_result)


    # optimizer = optim.Adam(polynet.parameters(), lr=0.7)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

    # is_trainable = False

    # for name, param in polynet.named_parameters():
    #     if "alpha" not in name:
    #         param.requires_grad = False
    #     else:
    #         is_trainable = True

    # # for name, param in polynet.named_parameters():
    # #     print(name, param.requires_grad)

    # for _ in range(100):
    #     (
    #         _orig_ub,
    #         _orig_lb,
    #         upper_bound_result,
    #         lower_bound_result,
    #     ) = polynet((upper_bound, lower_bound, upper_bound, lower_bound))
    #     optimizer.zero_grad()



    #     result = check_postcondition(upper_bound_result, lower_bound_result, true_label)

    #     if result > 0 or not is_trainable:
    #         return result > 0

    #     # print("Alpha:", polynet[2].alpha.data[:10])

    #     loss = torch.log(-result)
    #     loss.backward()
    #     optimizer.step()

    #     print(f"loss: {loss.item()}")

    #     if scheduler.get_last_lr()[0] > 0.1:
    #         scheduler.step()

    #     for parameter in polynet.parameters():
    #         if parameter.requires_grad:
    #             parameter.data = parameter.data.clamp_(0, 1)

    return result > 0


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
    print(net)
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
