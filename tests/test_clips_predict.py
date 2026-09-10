"""Contracts for the prediction script's bookkeeping.

No model is loaded here. What is tested is everything around the models: which
moments get looked at, how seven probabilities become three, how neighbouring
windows become one event, and whether a head trained on the wrong view is let
through. Each of those can be wrong while the output still looks like a result.
"""

import argparse

import pytest
import torch

from clips.predict import (PRE, format_time, goal_probabilities, load_head,
                           merge_events, parse_time, window_starts)
from ingest.cut import CLIP_SECONDS

CLASSES = ["two_pointer", "three_pointer", "dunk", "free_throw", "block", "steal", "none"]
GROUPS = {"none": ["none", "block", "steal"],
          "field_goal": ["two_pointer", "three_pointer", "dunk"],
          "free_throw": ["free_throw"]}


@pytest.mark.parametrize("text, seconds", [("12:31", 751.0), ("1:02:03", 3723.0),
                                           ("95", 95.0), ("0:07.5", 7.5)])
def test_times_are_read_the_way_a_person_writes_them(text, seconds):
    assert parse_time(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "-5"])
def test_a_bad_time_is_refused_rather_than_guessed(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_time(text)


def test_times_print_in_the_timeline_format():
    assert format_time(751.0) == "00:12:31"
    assert format_time(3723.4) == "01:02:03"


def test_a_field_goal_is_the_sum_of_its_three_kinds():
    # The head can be split 40/40 between two and three and still be 80% sure
    # it saw a field goal. That is the point of reporting at the goal level.
    probs = torch.tensor([0.4, 0.4, 0.0, 0.1, 0.0, 0.0, 0.1])
    goals = goal_probabilities(probs, CLASSES, GROUPS)
    assert goals["field_goal"] == pytest.approx(0.8)
    assert goals["free_throw"] == pytest.approx(0.1)
    assert sum(goals.values()) == pytest.approx(1.0)


def test_scan_windows_tile_the_stretch_and_stay_inside_it():
    starts = window_starts(100.0, 110.0)
    assert starts == [100.0, 102.0, 104.0, 106.0, 108.0]
    assert starts[-1] + CLIP_SECONDS <= 110.0


def test_a_stretch_shorter_than_one_clip_has_no_windows():
    assert window_starts(100.0, 101.0) == []


def test_neighbouring_windows_with_the_same_answer_become_one_event():
    # A shot that straddles two windows must not be reported twice.
    events = merge_events([(10.0, "field_goal", 0.6), (12.0, "field_goal", 0.9)])
    assert events == [{"t": 12.0, "event": "field_goal", "confidence": 0.9}]


def test_different_answers_or_a_gap_keep_events_apart():
    events = merge_events([(10.0, "field_goal", 0.6), (12.0, "free_throw", 0.7),
                           (20.0, "free_throw", 0.8)])
    assert [e["event"] for e in events] == ["field_goal", "free_throw", "free_throw"]


def test_a_moment_sits_where_training_put_the_event():
    # ingest/cut.py's default --pre. If these drift apart the model is asked
    # about clips shifted from anything it was trained on.
    assert 0 < PRE < CLIP_SECONDS
    assert PRE == 1.5


def test_a_head_trained_on_the_whole_court_view_is_refused(tmp_path):
    # Same backbone, same shapes -- it would load and run and be wrong.
    path = tmp_path / "head.pt"
    torch.save({"state_dict": {}, "head": "clip", "backbone": "mvit_v2_s",
                "crop": ["squash"], "feature_dim": 768, "classes": CLASSES}, path)
    with pytest.raises(SystemExit):
        load_head(str(path), "cpu")


def test_a_missing_head_says_how_to_train_one(tmp_path):
    with pytest.raises(SystemExit, match="clips.train"):
        load_head(str(tmp_path / "nope.pt"), "cpu")
