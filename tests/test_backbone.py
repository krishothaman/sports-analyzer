"""Contracts for the frozen backbone.

All three of these guard silent failures. None of them would raise an error in
normal use -- they would just make the model quietly worse, which is the hardest
kind of bug to notice.
"""

import torch

from models.backbone import FEATURE_DIM, build_backbone


def test_backbone_turns_an_image_into_512_numbers():
    model, _ = build_backbone(torch.device("cpu"))
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, FEATURE_DIM)


def test_backbone_is_frozen():
    # The whole Phase 2 premise is that only the head learns. If any backbone
    # parameter could accumulate gradients, training would drift the eyes using
    # a few hundred frames -- and nothing would report it.
    model, _ = build_backbone(torch.device("cpu"))
    assert all(not p.requires_grad for p in model.parameters())


def test_backbone_is_in_eval_mode():
    # ResNet-18 has batchnorm. In train mode it updates running statistics from
    # our frames, which changes a model we declared frozen.
    model, _ = build_backbone(torch.device("cpu"))
    assert not model.training


def test_same_image_always_gives_the_same_features():
    # Feature caching is only valid if the backbone is deterministic. If this
    # ever fails, cached features and freshly computed ones would disagree and
    # training would learn from one while inference used the other.
    model, _ = build_backbone(torch.device("cpu"))
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(model(image), model(image))
