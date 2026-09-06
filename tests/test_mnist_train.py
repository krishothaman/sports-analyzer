"""Tests for evaluate().

evaluate() is pure counting, so it can be tested exactly -- no training, no
randomness, no "about 98%". We feed it a fake model whose answers we already
know, and check the arithmetic.

train_one_epoch() is deliberately NOT tested here. Its correctness shows up as
"does the loss actually go down", which is what the real training run measures.
"""

import torch

from mnist.train import evaluate

DEVICE = torch.device("cpu")


class AlwaysPredictsThree(torch.nn.Module):
    """A stand-in model that answers '3' for every image, very confidently."""

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 10)
        logits[:, 3] = 10.0
        return logits


def loader_with_labels(labels):
    """Build a DataLoader of random images carrying the labels we specify."""
    images = torch.randn(len(labels), 1, 28, 28)
    dataset = torch.utils.data.TensorDataset(images, torch.tensor(labels))
    # batch_size=2 so the labels span multiple batches -- this catches the
    # mistake of returning one batch's score instead of the whole loader's.
    return torch.utils.data.DataLoader(dataset, batch_size=2)


def test_counts_partial_credit():
    # Two of the four labels are 3, so a model that always says 3 scores 50%.
    loader = loader_with_labels([3, 3, 1, 0])

    assert evaluate(AlwaysPredictsThree(), loader, DEVICE) == 0.5


def test_perfect_score_is_one():
    loader = loader_with_labels([3, 3, 3, 3])

    assert evaluate(AlwaysPredictsThree(), loader, DEVICE) == 1.0


def test_no_correct_answers_is_zero():
    loader = loader_with_labels([0, 1, 2, 4])

    assert evaluate(AlwaysPredictsThree(), loader, DEVICE) == 0.0
