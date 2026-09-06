"""The MNIST CNN.

Deliberately small and readable. Every layer's output shape is written in a
comment, because shape arithmetic is where beginners get stuck -- and where
the bugs live.
"""

import torch.nn as nn


class MnistCNN(nn.Module):
    """A small convolutional network: 28x28 grayscale image -> 10 digit scores."""

    def __init__(self):
        super().__init__()

        # Conv2d(in_channels, out_channels, kernel_size, padding)
        # 1 input channel (grayscale) -> 16 output channels. "16 channels" means
        # 16 different 3x3 filters, each learning to detect a different pattern.
        # padding=1 keeps the output the same width/height as the input.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # ReLU: replace every negative number with 0. This is the ONLY nonlinear
        # step, and without it stacked conv layers collapse mathematically into
        # a single conv layer -- the network could then only learn straight-line
        # relationships and would fail on anything interesting.
        self.relu = nn.ReLU()

        # MaxPool: take the largest value in each 2x2 block, halving width and
        # height. Cuts computation and makes the network tolerant to small shifts.
        self.pool = nn.MaxPool2d(2)

        # After two conv+pool rounds: 32 channels of 7x7 -> 32*7*7 = 1568 numbers.
        # Those become the input to ordinary dense layers that do the final vote.
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # x arrives as [B, 1, 28, 28]
        x = self.pool(self.relu(self.conv1(x)))   # -> [B, 16, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))   # -> [B, 32,  7,  7]

        # Flatten every image's feature maps into one long vector, keeping the
        # batch dimension. start_dim=1 means "flatten everything except dim 0".
        x = x.flatten(start_dim=1)                # -> [B, 1568]

        x = self.relu(self.fc1(x))                # -> [B, 64]
        x = self.fc2(x)                           # -> [B, 10]

        # Raw logits, deliberately. CrossEntropyLoss applies softmax itself;
        # doing it here as well would apply it twice and break training.
        return x
