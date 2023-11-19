import argparse
import torch
import torch.nn as nn


from networks import get_network
from utils.loading import parse_spec
# from box import certify_sample, AbstractBox
from deeppoly import DeepPolyLinear, DeepPolyFlatten, DeepPolyShape, DeepPolyReLu

DEVICE = "cpu"


# def analyze(
#     net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
# ) -> bool:
#     return certify_sample(net, inputs, true_label, eps)



def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:

    layers = []

    for layer in net:
        if isinstance(layer, nn.Flatten):
            poly_layer = DeepPolyFlatten()
        elif isinstance(layer, nn.Linear):
            poly_layer = DeepPolyLinear(layer)
        # elif isinstance(layer, nn.ReLU):
        #   poly_layer = DeepPolyReLu(layer)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        layers.append(poly_layer)

    polynet = nn.Sequential(*layers)
    x = DeepPolyShape(inputs, eps)

    x = polynet(x)

    return x.check_postcondition(true_label)

    


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
