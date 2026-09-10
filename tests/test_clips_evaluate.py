"""Contracts for the raw-video evaluator's bookkeeping. No model is loaded."""

import pytest
import torch

from clips.evaluate import average_probs, dedup_rows, score


def row(clip_id, start, label, match="match03"):
    return {"clip_id": clip_id, "match_id": match, "start_sec": str(start), "label": label}


def test_a_play_marked_twice_counts_once():
    rows = [row("a", 10.0, "two_pointer"), row("b", 10.5, "two_pointer")]
    assert [r["clip_id"] for r in dedup_rows(rows)] == ["a"]


def test_different_plays_close_together_are_both_kept():
    # A basket and the free throw that follows it are two plays, however close.
    rows = [row("a", 10.0, "two_pointer"), row("b", 10.5, "free_throw"),
            row("c", 20.0, "two_pointer"), row("d", 10.2, "two_pointer", match="match02")]
    assert sorted(r["clip_id"] for r in dedup_rows(rows)) == ["a", "b", "c", "d"]


def test_background_clips_are_never_dropped():
    rows = [row("a", 10.0, "none"), row("b", 10.3, "none")]
    assert len(dedup_rows(rows)) == 2


def test_averaging_several_looks_gives_one_probability_vector():
    probs = [torch.tensor([0.8, 0.2]), torch.tensor([0.4, 0.6])]
    assert torch.allclose(average_probs(probs), torch.tensor([0.6, 0.4]))
    assert average_probs(p for p in probs).sum().item() == pytest.approx(1.0)


def test_scoring_uses_the_most_likely_class_then_its_group():
    # The same trap clips/predict.py guards: field goals pooled (0.72) outweigh
    # none (0.26), but none is the single most likely class, so it is scored none.
    classes = ["two_pointer", "three_pointer", "dunk", "free_throw", "block", "steal", "none"]
    probs = {"a": torch.tensor([0.24, 0.24, 0.24, 0.02, 0.0, 0.0, 0.26])}
    matrix = score([row("a", 1.0, "none")], lambda r: probs[r["clip_id"]], classes)
    assert matrix[0, 0].item() == 1        # truth none, called none


def test_clips_with_no_usable_look_are_left_out_not_guessed():
    classes = ["two_pointer", "three_pointer", "dunk", "free_throw", "block", "steal", "none"]
    matrix = score([row("a", 1.0, "none")], lambda r: None, classes)
    assert matrix.sum().item() == 0
