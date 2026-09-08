"""Contracts for clip planning.

The guard-band test is the one that matters. Nothing crashes if background is
sampled on top of an event -- you simply get clips labelled 'none' that contain
the motion the model is supposed to fire on, and the affected class quietly
underperforms with no visible cause.
"""

from ingest.cut import (BACKGROUND_LABEL, CLIP_FRAMES, CLIP_SECONDS,
                        background_starts, clip_id_for, event_starts,
                        frame_indices)


def marks(*pairs):
    return [{"match_id": "m1", "timestamp_sec": str(t), "label": label}
            for t, label in pairs]


def test_a_clip_is_sixteen_frames_spanning_two_seconds():
    indices = frame_indices(10.0, video_fps=25.0)
    assert len(indices) == CLIP_FRAMES
    assert indices[0] == 250                       # 10.0s at 25 fps
    assert indices[-1] == 297                      # 10.0 + 15/8 = 11.875s
    # 16 frames at 8 fps spans 15 gaps, not 16 -- 1.875s of video, and the
    # clip's 2 seconds is that plus the last frame's own eighth.
    assert abs((indices[-1] - indices[0]) / 25.0 - (CLIP_SECONDS - 1 / 8)) < 0.02


def test_frames_within_a_clip_are_evenly_spaced():
    indices = frame_indices(0.0, video_fps=24.0)   # 24/8 = exactly 3 frames apart
    gaps = {b - a for a, b in zip(indices, indices[1:])}
    assert gaps == {3}


def test_the_clip_sits_behind_the_mark():
    # You press the key when you SEE the outcome, roughly 0.4s late, and the
    # useful motion happened before that. A mark at 100s with pre=1.5 must
    # cover 98.5s-100.5s, not 100s-102s.
    (start, label), = event_starts(marks((100.0, "dunk")), pre=1.5)
    assert start == 98.5
    assert start + CLIP_SECONDS == 100.5
    assert label == "dunk"


def test_exclude_marks_never_become_clips():
    # A missed shot is not an event...
    starts = event_starts(marks((100.0, "exclude"), (200.0, "dunk")), pre=1.5)
    assert [label for _, label in starts] == ["dunk"]


def test_exclude_marks_still_block_background():
    # ...but it is not background either. A missed three looks almost exactly
    # like a made one; sampling it as 'none' would teach the model that one
    # picture is both classes at once.
    starts = background_starts(300.0, marks((100.0, "exclude")), guard=4.0)
    centres = [s + CLIP_SECONDS / 2 for s in starts]
    assert all(abs(c - 100.0) >= 4.0 for c in centres)


def test_no_background_window_falls_inside_the_guard_band():
    every = marks((50.0, "dunk"), (120.0, "steal"), (121.5, "two_pointer"))
    starts = background_starts(400.0, every, guard=4.0)
    assert starts
    for start in starts:
        centre = start + CLIP_SECONDS / 2
        for mark in every:
            assert abs(centre - float(mark["timestamp_sec"])) >= 4.0


def test_background_windows_stay_inside_the_video():
    starts = background_starts(10.0, [], guard=4.0)
    assert all(start + CLIP_SECONDS <= 10.0 for start in starts)


def test_background_windows_do_not_overlap_each_other():
    starts = background_starts(60.0, [], guard=4.0)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(gap >= CLIP_SECONDS for gap in gaps)


def test_clip_ids_are_deterministic_so_recutting_overwrites():
    # Re-cutting after more marking must land on the same directories, not
    # accumulate a second copy of every clip.
    assert clip_id_for("m1", 98.5) == clip_id_for("m1", 98.5)
    assert clip_id_for("m1", 98.5) != clip_id_for("m1", 98.6)
    assert clip_id_for("m1", 98.5).split("_")[0] == "m1"


def test_background_label_is_not_a_mark_key():
    # 'none' must never be something the owner can press -- it is sampled.
    from ingest.mark import MARK_KEYS
    assert BACKGROUND_LABEL not in MARK_KEYS.values()


def test_background_is_never_sampled_beyond_the_reviewed_stretch():
    # The owner watched 22 minutes of a 108-minute match. "No mark here" only
    # means "nothing happened" for the part they actually watched. Sampling the
    # rest would turn every unmarked dunk into a 'none' clip -- label noise
    # aimed at the rarest classes, and invisible in every metric.
    starts = background_starts(1325.0, marks((100.0, "dunk")), guard=4.0)
    assert starts
    assert max(starts) + CLIP_SECONDS <= 1325.0


def test_reviewed_until_prefers_the_saved_watch_position(tmp_path, monkeypatch):
    import ingest.cut as cut
    monkeypatch.setattr(cut, "load_position", lambda path: 1325.0)
    assert cut.reviewed_until_for("m1", marks((100.0, "dunk")), 6501.0) == 1325.0


def test_reviewed_until_falls_back_to_the_last_mark(monkeypatch):
    import ingest.cut as cut
    monkeypatch.setattr(cut, "load_position", lambda path: None)
    assert cut.reviewed_until_for("m1", marks((800.0, "dunk")), 6501.0) == 800.0


def test_reviewed_until_never_exceeds_the_video():
    import ingest.cut as cut
    assert cut.reviewed_until_for("m1", [], 500.0, override=9999.0) == 500.0
