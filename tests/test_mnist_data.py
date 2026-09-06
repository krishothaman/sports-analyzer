"""Tests for MNIST data loading.

These check the SHAPE CONTRACT -- that data comes out in the form the model
expects. Shape mismatches are the single most common bug in ML code, and
they are cheap to catch here instead of three layers deep in a training run.
"""

import torch

from mnist.data import get_loaders


def test_train_batch_has_expected_shape():
    train_loader, _ = get_loaders(batch_size=8)
    images, labels = next(iter(train_loader))

    # [batch, channels, height, width] -- 1 channel because MNIST is grayscale
    assert images.shape == (8, 1, 28, 28)
    assert labels.shape == (8,)


def test_labels_are_digits_zero_to_nine():
    train_loader, _ = get_loaders(batch_size=64)
    _, labels = next(iter(train_loader))

    assert labels.dtype == torch.int64
    assert labels.min() >= 0
    assert labels.max() <= 9


def test_images_are_normalized_not_raw_pixels():
    """Raw pixels are 0-255 ints. After ToTensor they are 0-1 floats, and
    after Normalize they straddle zero. Seeing negative values proves the
    normalization step actually ran -- forgetting it is a classic bug that
    makes training mysteriously bad rather than obviously broken."""
    train_loader, _ = get_loaders(batch_size=64)
    images, _ = next(iter(train_loader))

    assert images.dtype == torch.float32
    assert images.min() < 0, "expected normalized values, got un-normalized"
    assert images.max() < 5, "values look like raw pixels, not normalized"


def test_dataset_sizes_are_the_known_mnist_split():
    """MNIST is a fixed, standard dataset: exactly 60k train, 10k test.
    Wrong counts mean a corrupt or partial download."""
    train_loader, test_loader = get_loaders(batch_size=64, test_batch_size=1000)

    assert len(train_loader.dataset) == 60_000
    assert len(test_loader.dataset) == 10_000
