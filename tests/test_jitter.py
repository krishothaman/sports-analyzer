"""Phase 6's extra training clips: the copies are right, and nothing old is touched.

Rule zero of Phase 6 is that the Phase 5 model survives whatever is tried. These
tests are that rule written down: the extra clips go to new files, only ever
into training, and the shipped weights, manifest, detections and cache cannot
be written by any of the new paths.
"""

import os

import pytest

from clips.data import EXTRA_MANIFESTS, add_extra_rows, cache_path
from clips.predict import CHECKPOINT, PRE
from clips.train import checkpoint_path
from ingest.cut import CLIP_MANIFEST
from ingest.hoop import HOOPS_CSV, detections_path
from ingest.jitter import CUT_PRE, PRES, jitter_rows


def row(clip_id, start, label, match="match01"):
    return {"clip_id": clip_id, "match_id": match, "start_sec": f"{start:.2f}",
            "label": label, "n_frames": 16}


def test_each_event_is_copied_with_the_play_at_every_offset():
    copies = jitter_rows([row("m_1", 100.0, "two_pointer")], test_matches=set())
    # The mark is 1.5 s into the original: 101.5. Each copy puts it `pre` in.
    assert [c["start_sec"] for c in copies] == ["100.50", "100.25", "99.75", "99.50"]
    assert [c["clip_id"] for c in copies] == ["m_1_p100", "m_1_p125", "m_1_p175", "m_1_p200"]
    assert {c["label"] for c in copies} == {"two_pointer"}


def test_background_is_not_copied():
    assert jitter_rows([row("m_1", 100.0, "none")], test_matches=set()) == []


def test_a_test_match_event_is_refused_not_skipped():
    with pytest.raises(ValueError, match="test match"):
        jitter_rows([row("m_1", 100.0, "free_throw", match="match03")], {"match03"})


def test_a_copy_that_would_start_before_the_video_is_dropped():
    copies = jitter_rows([row("m_1", 0.0, "dunk")], test_matches=set())
    assert all(float(c["start_sec"]) >= 0 for c in copies)
    assert len(copies) == 2           # pre 1.0 and 1.25 fit; 1.75 and 2.0 do not


def test_the_copies_centre_on_where_training_and_prediction_put_the_event():
    assert CUT_PRE == PRE
    assert min(PRES) < CUT_PRE < max(PRES)


def test_extra_clips_from_a_test_match_are_refused():
    train, test = [row("a", 1, "none")], [row("b", 1, "none", match="match03")]
    with pytest.raises(SystemExit, match="training only"):
        add_extra_rows(train, test, [row("c", 5, "two_pointer", match="match03")])


def test_extra_clips_join_the_training_side_and_leave_the_test_side_alone():
    train, test = [row("a", 1, "none")], [row("b", 1, "none", match="match03")]
    combined = add_extra_rows(train, test, [row("c", 5, "two_pointer")])
    assert [r["clip_id"] for r in combined] == ["a", "c"]
    assert [r["clip_id"] for r in test] == ["b"]


# ---- rule zero: no new path can land on a file the Phase 5 model depends on ----

def same(a, b):
    return os.path.abspath(a) == os.path.abspath(b)


def test_extra_manifests_never_overwrite_the_original():
    assert not any(same(path, CLIP_MANIFEST) for path in EXTRA_MANIFESTS.values())


def test_extra_detections_get_their_own_file():
    assert same(detections_path(None), HOOPS_CSV)
    assert same(detections_path(CLIP_MANIFEST), HOOPS_CSV)
    for path in EXTRA_MANIFESTS.values():
        assert not same(detections_path(path), HOOPS_CSV)
    assert same(detections_path(EXTRA_MANIFESTS["jitter"]),
                os.path.join("data", "hoops_jitter.csv"))


def test_extra_caches_never_overwrite_the_original():
    original = cache_path("mvit_v2_s", "hoop")
    for name in EXTRA_MANIFESTS:
        assert not same(cache_path("mvit_v2_s", "hoop", name), original)


def test_a_run_with_extra_clips_cannot_overwrite_the_shipped_head():
    assert checkpoint_path("mvit_v2_s", "hoop", "clip") == CHECKPOINT
    for name in EXTRA_MANIFESTS:
        assert checkpoint_path("mvit_v2_s", "hoop", "clip", [name]) != CHECKPOINT
    assert checkpoint_path("mvit_v2_s", "hoop", "clip", ["jitter", "hard"]) != CHECKPOINT
