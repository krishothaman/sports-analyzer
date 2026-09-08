"""Contracts for reading the score bug.

The bias throughout is towards missing an event rather than inventing one. A
missed basket costs one training example. A fabricated one teaches the model
that something happened when nothing did, and there is no later stage that can
notice the difference.
"""

from ingest.scoreboard import (POINTS_TO_LABEL, calibrate_lag, load_layout,
                               score_changes, to_marks)


def readings(*triples):
    return [(float(t), h, a) for t, h, a in triples]


def test_a_confirmed_three_point_change_is_detected():
    found = score_changes(readings((0, 0, 0), (1, 0, 0), (2, 3, 0), (3, 3, 0)))
    assert found == [(2.0, 3)]


def test_a_change_is_ignored_until_it_is_confirmed():
    # One frame reading 3-0 and then reverting is a misread digit, not a basket.
    assert score_changes(readings((0, 0, 0), (1, 3, 0), (2, 0, 0), (3, 0, 0))) == []


def test_the_timestamp_is_when_the_new_score_first_appeared():
    # Not when it was confirmed -- the basket happened before the graphic
    # updated, so the earliest sighting is the closest we have.
    found = score_changes(readings((0, 0, 0), (5, 0, 0), (10, 2, 0), (11, 2, 0)))
    assert found[0][0] == 10.0


def test_each_point_value_maps_to_the_right_class():
    for points, expected in ((1, "free_throw"), (2, "two_pointer"), (3, "three_pointer")):
        found = score_changes(readings((0, 0, 0), (1, points, 0), (2, points, 0)))
        assert POINTS_TO_LABEL[found[0][1]] == expected


def test_a_four_point_jump_is_rejected():
    # No single basket scores four. This is a misread, and marking it would
    # invent an event of a class that does not exist.
    assert score_changes(readings((0, 0, 0), (1, 4, 0), (2, 4, 0))) == []


def test_a_score_going_backwards_is_rejected():
    assert score_changes(readings((0, 5, 0), (1, 2, 0), (2, 2, 0))) == []


def test_both_teams_scoring_at_once_is_rejected():
    # Impossible in basketball, so it means the reader lost sync.
    assert score_changes(readings((0, 0, 0), (1, 2, 3), (2, 2, 3))) == []


def test_away_team_baskets_are_detected_too():
    found = score_changes(readings((0, 0, 0), (1, 0, 2), (2, 0, 2)))
    assert found == [(1.0, 2)]


def test_a_run_of_baskets_is_detected_as_separate_events():
    found = score_changes(readings((0, 0, 0), (1, 3, 0), (2, 3, 0),
                                   (3, 3, 2), (4, 3, 2), (5, 6, 2), (6, 6, 2)))
    assert [points for _, points in found] == [3, 2, 3]


def test_a_gap_where_the_bug_was_hidden_does_not_invent_an_event():
    # Readings simply stop while the graphic is off screen. When it returns the
    # score may have moved by more than one basket -- that is not markable,
    # because we do not know when the baskets happened.
    assert score_changes(readings((0, 0, 0), (30, 7, 4), (31, 7, 4))) == []


def test_lag_is_measured_against_the_owners_own_marks():
    # The graphic trails the ball. Rather than guessing that delay, take the
    # median offset from marks a human made watching the same footage.
    manual = [{"timestamp_sec": "100.0", "label": "three_pointer"},
              {"timestamp_sec": "200.0", "label": "two_pointer"}]
    lag, matched = calibrate_lag([(101.5, 3), (201.5, 2)], manual)
    assert matched == 2
    assert abs(lag - 1.5) < 1e-6


def test_lag_calibration_reports_nothing_when_there_is_no_overlap():
    lag, matched = calibrate_lag([(101.5, 3)], [])
    assert lag is None and matched == 0


def test_lag_calibration_ignores_marks_of_a_different_class():
    manual = [{"timestamp_sec": "100.0", "label": "dunk"}]
    lag, matched = calibrate_lag([(101.5, 3)], manual)
    assert lag is None and matched == 0


def test_marks_are_shifted_back_by_the_lag():
    marks = to_marks([(101.5, 3)], "m1", lag=1.5)
    assert marks == [{"match_id": "m1", "timestamp_sec": "100.00",
                      "label": "three_pointer"}]


def test_a_mark_never_lands_before_the_video_starts():
    marks = to_marks([(0.5, 2)], "m1", lag=1.5)
    assert float(marks[0]["timestamp_sec"]) == 0.0


def test_layout_scales_to_a_different_video_width():
    # The boxes were measured on an 854-wide stream. A 1280-wide copy of the
    # same broadcast has the same graphic, larger.
    base = load_layout("paris2024", 854)
    wide = load_layout("paris2024", 1708)
    assert [v * 2 for v in base["home"]] == list(wide["home"])


def test_a_change_across_a_long_blind_spot_is_not_marked():
    # The graphic hides during the replay that follows a basket. If it stays
    # hidden across two free throws the score reappears +2, and marking that
    # as a two_pointer labels footage that contains no two-pointer. Refuse.
    assert score_changes(readings((0, 0, 0), (60, 2, 0), (61, 2, 0))) == []


def test_the_score_still_advances_after_a_refused_gap():
    # Refusing to mark must not desynchronise the reader: the next basket after
    # the blind spot is still detected relative to the new score.
    found = score_changes(readings((0, 0, 0), (60, 2, 0), (61, 2, 0),
                                   (62, 4, 0), (63, 4, 0)))
    assert found == [(62.0, 2)]


def test_a_change_within_the_gap_limit_is_still_marked():
    found = score_changes(readings((0, 0, 0), (5, 2, 0), (6, 2, 0)))
    assert found == [(5.0, 2)]
