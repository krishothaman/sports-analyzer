"""The 'rulebook': the only part of Phase 2 that learns.

One linear layer, 512 inputs to 2 outputs. That is 1,026 parameters against the
backbone's 11 million. All the visual understanding was paid for by ImageNet;
this layer only has to decide which combinations of those 512 numbers mean
"basketball is happening right now".

A single linear layer on frozen features has a name: a linear probe. If it
works, that is evidence the backbone's features already separate the classes
cleanly -- worth knowing before building anything larger, because if a probe
cannot do it, a deeper head usually cannot rescue it either.
"""

import torch.nn as nn

from models.backbone import FEATURE_DIM


class FrameHead(nn.Module):
    def __init__(self, num_classes=2, feature_dim=FEATURE_DIM):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x arrives as [B, 512] cached features -- no convolution here, the
        # backbone already did all of that. Out: [B, 2] raw logits, exactly like
        # MnistCNN returned [B, 10]. CrossEntropyLoss applies softmax itself.
        return self.fc(x)
