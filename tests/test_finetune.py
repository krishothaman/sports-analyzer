"""Splitting the backbone must change nothing until the tail is trained.

The whole fine-tuning plan rests on one claim: frozen trunk, then tail, is the
same network as before. If it weren't, a "fine-tuned" result would be measured
against a model that never existed, and a token cache would describe clips
differently from the features every earlier number came from.
"""

import pytest
import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s

from clips.finetune import finetune_checkpoint_path, token_cache_path
from clips.predict import CHECKPOINT
from clips.data import cache_path
from models.backbone import VideoTail, VideoTrunk, split_video_backbone


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    net = mvit_v2_s(weights=None)
    net.head = nn.Identity()
    return net.eval()


def test_trunk_then_tail_is_the_whole_network(model):
    clip = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        whole = model(clip)
        tokens, thw = VideoTrunk(model)(clip)
        split = VideoTail(model)(tokens, thw)
    assert split.shape == whole.shape == (1, 768)
    assert torch.allclose(split, whole, atol=1e-5)


def test_only_the_tail_can_learn():
    trunk, tail, _, dim = split_video_backbone("mvit_v2_s", "cpu", "squash", pretrained=False)
    assert dim == 768
    assert not any(p.requires_grad for p in trunk.parameters())
    assert all(p.requires_grad for p in tail.parameters())
    tail_weights = sum(p.numel() for p in tail.parameters())
    trunk_weights = sum(p.numel() for p in trunk.parameters())
    assert tail_weights < trunk_weights


def test_a_saved_tail_loads_into_a_whole_model():
    # clips/predict.py loads a fine-tuned tail into the full backbone this way.
    torch.manual_seed(1)
    trained, fresh = mvit_v2_s(weights=None), mvit_v2_s(weights=None)
    VideoTail(fresh).load_state_dict(VideoTail(trained).state_dict())
    assert torch.equal(fresh.blocks[-1].mlp[0].weight, trained.blocks[-1].mlp[0].weight)
    assert torch.equal(fresh.norm.weight, trained.norm.weight)
    assert not torch.equal(fresh.blocks[0].mlp[0].weight, trained.blocks[0].mlp[0].weight)


def test_other_backbones_are_refused():
    with pytest.raises(SystemExit):
        split_video_backbone("mc3_18", "cpu", pretrained=False)


def test_fine_tuning_never_writes_over_phase_5_files():
    for extra in ([], ["jitter"], ["jitter", "hard"]):
        assert finetune_checkpoint_path(extra) != CHECKPOINT
    for extra in (None, "jitter", "hard"):
        assert token_cache_path(extra) != cache_path("mvit_v2_s", "hoop")
