"""Mining false alarms: what counts as one, what a reviewer can say, what is kept."""

import pytest

from ingest.mine import (GUARD, collect_rows, far_from_marks, has_verdicts,
                         parse_verdict, refuse_test_match)


def test_a_window_near_any_mark_is_not_a_candidate():
    # Window 10-12 s, centre 11 s. A mark 3 s away is inside the 4 s guard.
    assert not far_from_marks(10.0, [14.0])
    assert far_from_marks(10.0, [11.0 + GUARD, 11.0 - GUARD - 0.5])
    assert far_from_marks(10.0, [])


@pytest.mark.parametrize("text, label", [("none", "none"), ("N", "none"), ("miss", "none"),
                                         ("2", "two_pointer"), ("3", "three_pointer"),
                                         ("d", "dunk"), ("f", "free_throw"),
                                         ("free_throw", "free_throw"), ("s", "skip"),
                                         ("", None), ("  ", None)])
def test_verdicts_a_reviewer_can_type(text, label):
    assert parse_verdict(text) == label


def test_an_unknown_verdict_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        parse_verdict("maybe")


def candidate(n, start, verdict):
    return {"n": str(n), "start_sec": f"{start:.2f}", "verdict": verdict}


def test_only_reviewed_candidates_become_training_clips():
    rows = collect_rows([candidate(1, 100.0, "n"), candidate(2, 200.0, ""),
                         candidate(3, 300.0, "s"), candidate(4, 400.0, "3")],
                        "match01", tests={"match03"})
    assert [(r["start_sec"], r["label"]) for r in rows] == [("100.00", "none"),
                                                            ("400.00", "three_pointer")]
    # Their own ids, so they can never collide with an original clip's folder.
    assert all(r["clip_id"].endswith("_hard") for r in rows)


def test_a_bad_verdict_names_the_candidate():
    with pytest.raises(SystemExit, match="candidate 7"):
        collect_rows([candidate(7, 100.0, "maybe")], "match01", tests=set())


def test_test_matches_are_never_mined_or_collected():
    with pytest.raises(SystemExit, match="test match"):
        refuse_test_match("match03", {"match03"})
    with pytest.raises(SystemExit, match="test match"):
        collect_rows([candidate(1, 100.0, "n")], "match03", tests={"match03"})


def test_the_owners_review_work_is_detected_so_it_is_never_overwritten(tmp_path):
    path = tmp_path / "candidates.csv"
    assert not has_verdicts(str(path))
    path.write_text("n,start_sec,verdict\n1,10.00,\n", encoding="utf-8")
    assert not has_verdicts(str(path))
    path.write_text("n,start_sec,verdict\n1,10.00,\n2,20.00,n\n", encoding="utf-8")
    assert has_verdicts(str(path))
