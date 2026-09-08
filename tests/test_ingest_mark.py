"""Contracts for mark bookkeeping.

The OpenCV loop is a thin shell around these functions precisely so they can be
tested without a window. Anything that touches your marks lives here.
"""

from ingest.mark import (MARK_KEYS, add_mark, format_clock, load_marks,
                         save_marks, summary_line, tally)


def test_add_mark_appends_and_formats_the_timestamp():
    marks, accepted = add_mark([], "m1", 12.3456, "dunk")
    assert accepted
    assert marks == [{"match_id": "m1", "timestamp_sec": "12.35", "label": "dunk"}]


def test_a_double_tap_of_the_same_key_is_ignored():
    # Holding a key a fraction too long, or a nervous second press, must not
    # become two events -- that would put two near-identical clips in the
    # dataset and quietly double the weight of one moment.
    marks, _ = add_mark([], "m1", 100.0, "block")
    marks, accepted = add_mark(marks, "m1", 100.4, "block")
    assert not accepted
    assert len(marks) == 1


def test_two_different_classes_at_the_same_moment_are_both_kept():
    # A steal and the dunk that follows it are separate events a moment apart.
    # The duplicate guard is per class and must not swallow them.
    marks, _ = add_mark([], "m1", 100.0, "steal")
    marks, accepted = add_mark(marks, "m1", 100.4, "dunk")
    assert accepted
    assert len(marks) == 2


def test_the_same_class_far_enough_apart_is_kept():
    marks, _ = add_mark([], "m1", 100.0, "two_pointer")
    marks, accepted = add_mark(marks, "m1", 101.0, "two_pointer")
    assert accepted
    assert len(marks) == 2


def test_insertion_order_is_preserved_so_undo_removes_what_you_just_added():
    # You can seek backwards and mark an earlier moment. Undo has to remove that
    # mark, not whichever one happens to be latest on the clock.
    marks, _ = add_mark([], "m1", 500.0, "dunk")
    marks, _ = add_mark(marks, "m1", 100.0, "steal")
    assert marks[-1]["label"] == "steal"


def test_marks_round_trip_through_the_csv(tmp_path):
    path = tmp_path / "m1.csv"
    marks, _ = add_mark([], "m1", 61.5, "three_pointer")
    marks, _ = add_mark(marks, "m1", 90.25, "exclude")
    save_marks(marks, str(path))
    assert load_marks(str(path)) == marks


def test_loading_a_file_that_does_not_exist_yet_returns_nothing():
    # First run of a new match: no file, no crash.
    assert load_marks("data/events/definitely-not-a-real-match.csv") == []


def test_exclude_is_a_real_key_and_is_counted_separately():
    # 'exclude' is the pressure valve that keeps misses out of the background
    # sample. If it ever stops being a mark key, missed shots silently start
    # being labelled 'none'.
    assert "exclude" in MARK_KEYS.values()
    counts = tally([{"label": "dunk"}, {"label": "exclude"}, {"label": "dunk"}])
    assert counts == {"dunk": 2, "exclude": 1}


def test_summary_line_puts_exclude_last():
    line = summary_line([{"label": "exclude"}, {"label": "dunk"}])
    assert line.index("dunk") < line.index("excl")


def test_format_clock_is_minutes_and_seconds():
    assert format_clock(0) == "00:00"
    assert format_clock(61.9) == "01:01"
    assert format_clock(3600) == "60:00"
