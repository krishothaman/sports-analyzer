"""MNIST data loading.

Kept separate from the model so the two concerns stay independent: this file
knows about datasets and never about network architecture.
"""

import torch
from torchvision import datasets, transforms

# Every image passes through this pipeline before the network sees it.
transform = transforms.Compose([
    # PIL image (0-255 ints) -> float tensor scaled to 0.0-1.0, shape [1, 28, 28].
    # The leading 1 is the channel dimension: MNIST is grayscale so there is one
    # channel. A colour video frame would have three.
    transforms.ToTensor(),

    # Shift and scale so values are centred near zero instead of near 0.13.
    # These two constants are the actual mean and standard deviation of the
    # MNIST training set. Networks train faster and more stably on inputs that
    # are small and centred -- it keeps gradients in a sane range.
    transforms.Normalize((0.1307,), (0.3081,)),
])


def get_loaders(batch_size=64, test_batch_size=1000):
    """Download MNIST (once) and wrap it in DataLoaders.

    Returns (train_loader, test_loader).
    """
    # download=True only downloads if ./data does not already have the files.
    # train=True is the 60,000 images we learn from; train=False is the 10,000
    # held-out images we never train on, used purely to detect overfitting.
    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # shuffle=True on training data matters. If the network saw all the 0s, then
    # all the 1s, each update would be biased toward whichever digit is on
    # screen. Shuffling makes every batch a fair sample of the whole dataset.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    # The test set is not shuffled -- order is irrelevant when only scoring.
    # It uses a larger batch because no gradients are computed, so it is cheap.
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader
