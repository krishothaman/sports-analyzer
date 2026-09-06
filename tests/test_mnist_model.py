"""Tests for the MNIST CNN.

These verify the model's shape contract using random noise as input. We are
not testing that it predicts correctly -- an untrained network predicts
nothing useful. We are testing that data of the right shape goes in and data
of the right shape comes out, which is what actually breaks in practice.
"""

import torch

from mnist.model import MnistCNN


def test_forward_pass_returns_one_score_per_digit():
    model = MnistCNN()
    batch = torch.randn(4, 1, 28, 28)  # 4 fake images

    out = model(batch)

    # 10 numbers per image: one score per digit 0-9
    assert out.shape == (4, 10)


def test_output_is_logits_not_probabilities():
    """The model returns raw scores, NOT probabilities. This matters: PyTorch's
    CrossEntropyLoss applies softmax internally, so applying it here too would
    do it twice and quietly cripple training. Logits are unbounded, so a batch
    of random input should produce at least one negative value."""
    model = MnistCNN()
    out = model(torch.randn(32, 1, 28, 28))

    assert out.min() < 0, "outputs look like probabilities; they should be raw logits"


def test_gradients_reach_the_first_layer():
    """Runs one backward pass and checks a gradient actually arrived at the
    earliest layer. If the network is wired wrong -- a detached tensor, a
    broken connection -- gradients silently stop flowing and the model never
    learns, with no error message. This catches that."""
    model = MnistCNN()
    out = model(torch.randn(2, 1, 28, 28))
    out.sum().backward()

    assert model.conv1.weight.grad is not None
    assert model.conv1.weight.grad.abs().sum() > 0
