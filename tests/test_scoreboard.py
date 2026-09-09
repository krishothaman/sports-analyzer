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
    assert [v * 2 for v in base["states"][0]["home"]] == \
           list(wide["states"][0]["home"])


def test_digit_bounds_scale_with_the_picture_too():
    # Scaling the crops but not the glyph size bounds would refuse every digit
    # on any stream that is not the width the layout was measured at -- and it
    # would do it silently, as a total absence of readings.
    base = load_layout("paris2024", 854)
    wide = load_layout("paris2024", 1708)
    assert [v * 2 for v in base["digit_height"]] == list(wide["digit_height"])
    assert [v * 2 for v in base["digit_width"]] == list(wide["digit_width"])


def test_each_layout_has_its_own_template_set():
    # The two broadcasts use different typefaces. A shared set would let a glyph
    # from one production match a digit from the other.
    from ingest.scoreboard import LAYOUTS, templates_path
    paths = {templates_path(name) for name in LAYOUTS}
    assert len(paths) == len(LAYOUTS)


def test_every_layout_declares_the_geometry_the_reader_needs():
    # A layout added without these reads as "no scoreboard anywhere in the
    # match" rather than as an error, which is the hardest failure to notice.
    from ingest.scoreboard import LAYOUTS
    for name, layout in LAYOUTS.items():
        assert layout["states"], name
        for state in layout["states"]:
            assert set(state) == {"home", "away"}, name
        assert layout["present"]["mode"] in ("bright_box", "dark_cell"), name
        assert layout["digit_height"][0] < layout["digit_height"][1], name
        assert layout["digit_width"][0] < layout["digit_width"][1], name


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


def test_a_recap_replaying_an_earlier_score_invents_nothing():
    # Broadcasts cut to highlight montages that replay earlier moments with the
    # old score on screen. A reader that trusts every number walks the score
    # back up through the recap and marks a basket for each step -- this match
    # ends 95-91 but a naive pass claimed 186 points of scoring plays.
    # A live score never goes down, so readings below the confirmed score are
    # not the game and are ignored outright.
    found = score_changes(readings((0, 10, 10), (1, 10, 10),
                                   (2, 4, 6), (3, 6, 6), (4, 8, 8),   # the recap
                                   (5, 12, 10), (6, 12, 10)))
    assert found == [(5.0, 2)]


def test_the_score_still_tracks_after_a_recap():
    found = score_changes(readings((0, 10, 10), (1, 10, 10),
                                   (2, 2, 2), (3, 4, 4),
                                   (4, 10, 13), (5, 10, 13)))
    assert found == [(4.0, 3)]


def hand(timestamp, label):
    return {"match_id": "m1", "timestamp_sec": str(timestamp), "label": label}


def auto(timestamp, label):
    return {"match_id": "m1", "timestamp_sec": f"{timestamp:.2f}", "label": label}


def test_a_hand_marked_dunk_suppresses_the_readers_two_pointer():
    # The same basket under two names. Keeping both would put two clips of
    # identical footage in the dataset with conflicting labels -- and it would
    # do it to dunk, the class with the fewest examples to spare.
    from ingest.scoreboard import drop_already_marked
    kept = drop_already_marked([auto(100.0, "two_pointer")], [hand(100.5, "dunk")])
    assert kept == []


def test_a_hand_marked_steal_does_not_suppress_the_layup_after_it():
    # A steal is not a basket. The fast break it starts scores a second or two
    # later and is a separate, real event.
    from ingest.scoreboard import drop_already_marked
    kept = drop_already_marked([auto(102.0, "two_pointer")], [hand(100.0, "steal")])
    assert len(kept) == 1


def test_an_unrelated_basket_survives():
    from ingest.scoreboard import drop_already_marked
    kept = drop_already_marked([auto(500.0, "three_pointer")], [hand(100.0, "dunk")])
    assert len(kept) == 1


def test_an_impossible_jump_is_not_adopted_as_the_new_score():
    # The failure this exists for: one misread showed 197 where the score was
    # ~107. The jump was correctly refused as a basket, but was still adopted as
    # the current score -- and because a live score never goes down, every real
    # reading afterwards was then rejected as a recap. The reader went silent
    # for the last ten minutes of the match and reported no error at all.
    found = score_changes(readings((0, 10, 5), (1, 10, 5),
                                   (2, 197, 5), (3, 197, 5), (4, 197, 5),
                                   (5, 12, 5), (6, 12, 5)))
    assert found == [(5.0, 2)]


def test_a_real_score_move_behind_a_long_hide_is_still_adopted():
    # The legitimate resync must survive: the graphic hides for a minute, the
    # score genuinely moves by more than one basket, and we pick up from the new
    # value without marking anything.
    found = score_changes(readings((0, 10, 5), (1, 10, 5),
                                   (90, 18, 9), (91, 18, 9),
                                   (92, 21, 9), (93, 21, 9)))
    assert found == [(92.0, 3)]


def test_the_reader_keeps_working_after_an_impossible_reading():
    # Not just "does not mark it" -- it must not go deaf either.
    found = score_changes(readings((0, 0, 0), (1, 0, 0),
                                   (2, 250, 0), (3, 250, 0),
                                   (4, 2, 0), (5, 2, 0),
                                   (6, 5, 0), (7, 5, 0)))
    assert [p for _, p in found] == [2, 3]
