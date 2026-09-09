"""Contracts for the two clip heads.

The pair of shuffle tests below are the ones that matter. The phase compares an
order-blind head against an order-aware one, and that comparison only means
anything if each head really is what it claims to be. Without these, a tie
between them could equally mean "motion does not help" or "the GRU quietly
ignored the motion", and there would be no way to tell which.
"""

import torch

from clips.data import CLASSES
from clips.model import GRUHead, MeanPoolHead, build_head

BATCH, FRAMES, DIM = 4, 16, 512


def a_batch(seed=0):
    torch.manual_seed(seed)
    return torch.randn(BATCH, FRAMES, DIM)


def test_both_heads_turn_a_clip_into_one_score_per_class():
    for head in (MeanPoolHead(len(CLASSES)), GRUHead(len(CLASSES))):
        head.eval()
        assert head(a_batch()).shape == (BATCH, len(CLASSES))


def test_mean_pooling_cannot_tell_the_frames_apart():
    # The control. Averaging is order-independent, so shuffling the 16 frames
    # must change nothing at all. If this ever fails, the "order-blind baseline"
    # is not order-blind and the whole comparison is void.
    head = MeanPoolHead(len(CLASSES))
    head.eval()
    clip = a_batch()
    shuffled = clip[:, torch.randperm(FRAMES), :]

    with torch.no_grad():
        assert torch.allclose(head(clip), head(shuffled), atol=1e-5)


def test_the_gru_does_notice_the_order():
    # The other half. A GRU that returned the same answer for shuffled frames
    # would be a second mean-pool wearing a costume, and "the GRU did not beat
    # the baseline" would be uninterpretable.
    head = GRUHead(len(CLASSES))
    head.eval()
    clip = a_batch()
    shuffled = clip[:, torch.flip(torch.arange(FRAMES), dims=[0]), :]

    with torch.no_grad():
        assert not torch.allclose(head(clip), head(shuffled), atol=1e-4)


def test_the_baseline_really_is_the_smaller_model():
    # The comparison is only informative while mean-pool is the cheap one. If a
    # future edit grew it past the GRU, "the simple head did just as well" would
    # stop meaning what it is reported to mean.
    small = sum(p.numel() for p in MeanPoolHead(len(CLASSES)).parameters())
    large = sum(p.numel() for p in GRUHead(len(CLASSES)).parameters())
    assert small < large


def test_dropout_is_off_when_evaluating():
    # Dropout during eval makes the test score random, so two runs of the same
    # weights would disagree and neither would be the model's real accuracy.
    head = GRUHead(len(CLASSES))
    head.eval()
    clip = a_batch()

    with torch.no_grad():
        assert torch.allclose(head(clip), head(clip))


def test_build_head_refuses_an_unknown_name():
    # A typo in --head must not silently fall back to some default, or a run
    # gets filed under the wrong architecture in the results table.
    try:
        build_head("lstm", len(CLASSES))
    except SystemExit:
        return
    raise AssertionError("an unknown head name should stop the run")
