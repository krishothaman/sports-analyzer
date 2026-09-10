"""Contracts for the prediction script's bookkeeping.

No model is loaded here. What is tested is everything around the models: which
moments get looked at, how seven probabilities become three, how neighbouring
windows become one event, and whether a head trained on the wrong view is let
through. Each of those can be wrong while the output still looks like a result.
"""

import argparse

import pytest
import torch

from clips.predict import (PRE, build_parser, format_time, goal_answer,
                           goal_probabilities, load_head, merge_events, parse_time,
                           window_starts)
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


def test_the_answer_is_decided_the_way_the_results_were_scored():
    # The trap this guards: field goals pooled together (0.25 x 3 = 0.75)
    # outweigh `none` (0.25 alone) -- yet `none` is the single most likely class,
    # and grouped_report, which produced every published number, would call
    # this clip `none`. Deciding by the sums ships a different model from the
    # one that was measured; on seed 0 it moved every goal class by 8-11 points.
    probs = torch.tensor([0.24, 0.24, 0.24, 0.02, 0.0, 0.0, 0.26])
    label, confidence = goal_answer(probs, CLASSES, GROUPS)
    assert label == "none"
    assert confidence == pytest.approx(0.26)
    assert goal_probabilities(probs, CLASSES, GROUPS)["field_goal"] > confidence


def test_the_answer_agrees_with_grouped_report():
    # Same rule as models/metrics.py, checked against it directly on random heads.
    from clips.train import GOAL_GROUPS
    from models.metrics import grouped_report

    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(200, len(CLASSES)), dim=1)
    targets = torch.randint(0, len(CLASSES), (200,))
    matrix = grouped_report(probs.argmax(dim=1), targets, CLASSES, GOAL_GROUPS, "check")

    names = list(GOAL_GROUPS)
    ours = torch.zeros_like(matrix)
    for p, t in zip(probs, targets):
        truth = next(g for g, members in GOAL_GROUPS.items() if CLASSES[t] in members)
        ours[names.index(truth), names.index(goal_answer(p, CLASSES, GOAL_GROUPS)[0])] += 1
    assert torch.equal(ours, matrix)


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


def test_the_live_play_filter_is_off_unless_asked_for():
    # On match03 it threw out 5 of 8 real scoring plays -- wide shots it called
    # not_game. It has to be something a user opts into, not a default.
    assert build_parser().parse_args(["v.mp4", "--at", "1"]).live_filter is False
    assert build_parser().parse_args(["v.mp4", "--at", "1", "--live-filter"]).live_filter


class FakePredictor:
    """Answers with the clip's start time as its 'probability', so averaging is visible."""

    def __init__(self):
        self.starts = []

    def predict(self, frames):
        self.starts.append(frames[0])
        return torch.tensor([frames[0], 1.0]), True


def fake_read_clip(cap, start, fps):
    return [start] * 16 if start < 1000 else [None] * 16


def test_one_look_by_default_exactly_as_phase_5(monkeypatch):
    import clips.predict as predict
    monkeypatch.setattr(predict, "read_clip", fake_read_clip)
    fake = FakePredictor()
    [(moment, probs, found)] = list(predict.run(fake, None, 30.0, [10.0]))
    assert fake.starts == [10.0]
    assert moment == 10.0 + PRE and probs[0].item() == 10.0 and found


def test_several_shifts_are_averaged_into_one_answer(monkeypatch):
    import clips.predict as predict
    monkeypatch.setattr(predict, "read_clip", fake_read_clip)
    fake = FakePredictor()
    [(_, probs, _)] = list(predict.run(fake, None, 30.0, [10.0], (-0.25, 0.0, 0.25)))
    assert fake.starts == [9.75, 10.0, 10.25]
    assert probs[0].item() == pytest.approx(10.0)


def test_shifts_off_either_end_of_the_video_are_skipped(monkeypatch):
    import clips.predict as predict
    monkeypatch.setattr(predict, "read_clip", fake_read_clip)
    fake = FakePredictor()
    list(predict.run(fake, None, 30.0, [0.0], (-0.25, 0.0, 0.25)))
    assert fake.starts == [0.0, 0.25]
    assert list(predict.run(fake, None, 30.0, [2000.0])) == []   # past the end


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
