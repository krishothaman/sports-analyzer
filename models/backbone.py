"""The 'eyes': a frozen, pretrained ResNet-18 that turns an image into 512 numbers.

Shared across sports and across phases. Nothing in this file knows what a
basketball is -- it was trained on general photographs, and its job is only to
describe what is in a picture. The sport-specific knowledge lives in the head.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

# ResNet-18's penultimate layer is 512 wide. Swapping to a different backbone
# means changing this constant along with build_backbone.
FEATURE_DIM = 512


def build_backbone(device):
    """Return (frozen feature extractor, matching preprocessing transform)."""
    # DEFAULT is torchvision's best available ImageNet checkpoint for this
    # architecture: weights learned from 1.2 million labelled photographs. That
    # training is the thing we are borrowing -- it is why 520 basketball frames
    # is enough to build a classifier, when learning to see from scratch would
    # need hundreds of thousands.
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # resnet18 normally ends in Linear(512 -> 1000) to name an ImageNet class.
    # We do not want a class name -- "basketball" is not one of the 1000, and
    # "crowd of people" is not the question we are asking. We want the 512
    # numbers feeding that final layer: the description the network built before
    # committing to an answer. Identity is a layer that returns its input
    # unchanged, so replacing fc with it hands us those 512 numbers directly.
    model.fc = nn.Identity()

    # Freeze. Without this, gradients from the head would flow back into the
    # backbone and the optimizer would edit weights that took days of GPU time
    # to learn -- using 364 training frames to do it. The result would be worse
    # eyes, not better ones.
    for param in model.parameters():
        param.requires_grad = False

    # eval() matters here in a way it did not for MnistCNN. ResNet-18 contains
    # batchnorm layers, which in train mode update running statistics from
    # whatever data passes through. That would quietly change a model we just
    # declared frozen, with no error and no warning.
    model.eval()
    model.to(device)

    # The weights carry their own preprocessing: resize the short side to 256,
    # centre-crop 224, scale to 0-1, normalise by ImageNet's mean and std.
    # Using anything else feeds the network inputs shaped unlike its training
    # data, and accuracy drops for no visible reason.
    return model, weights.transforms()
