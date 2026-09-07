"""Contracts for splitting and label mapping.

The split test is the important one. A leaking split does not crash -- it
produces a *higher* accuracy number, which is exactly why it has to be caught
by a test rather than by noticing something looks wrong.
"""

from frames.data import CLASS_TO_INDEX, LABEL_MAP, chronological_split


def rows(match_id, timestamps, label="game"):
    return [{"match_id": match_id, "filename": f"{match_id}_{i}.jpg",
             "timestamp_sec": str(t), "label": label}
            for i, t in enumerate(timestamps)]


def test_split_puts_earlier_frames_in_train_and_later_in_test():
    train, test = chronological_split(rows("m1", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
                                      train_frac=0.7)
    assert len(train) == 7
    assert len(test) == 3
    assert max(float(r["timestamp_sec"]) for r in train) < \
           min(float(r["timestamp_sec"]) for r in test)


def test_split_is_chronological_even_when_the_manifest_is_out_of_order():
    # The manifest is written in labelling order, which is not timestamp order
    # once frames have been skipped, relabelled or undone.
    train, test = chronological_split(rows("m1", [50, 10, 90, 30, 70, 0, 20, 80, 40, 60]),
                                      train_frac=0.7)
    assert max(float(r["timestamp_sec"]) for r in train) < \
           min(float(r["timestamp_sec"]) for r in test)


def test_each_match_is_split_independently():
    # With several matches, every match must contribute to both sides. Splitting
    # the pooled list would let one whole match become the test set, which
    # measures generalisation across matches -- a different, harder question.
    both = rows("m1", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) + \
           rows("m2", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    train, test = chronological_split(both, train_frac=0.7)
    assert {r["match_id"] for r in train} == {"m1", "m2"}
    assert {r["match_id"] for r in test} == {"m1", "m2"}


def test_no_frame_appears_on_both_sides():
    train, test = chronological_split(rows("m1", list(range(0, 200, 10))), train_frac=0.7)
    assert not {r["filename"] for r in train} & {r["filename"] for r in test}


def test_every_raw_label_maps_to_a_real_class():
    # The manifest holds three labels; the model has two. If a raw label ever
    # maps to a class the model does not have, CLASS_TO_INDEX would raise deep
    # inside dataset construction with no useful message.
    assert set(LABEL_MAP.values()) <= set(CLASS_TO_INDEX)


def test_crowd_and_graphic_both_become_not_game():
    assert LABEL_MAP["crowd"] == "not_game"
    assert LABEL_MAP["graphic"] == "not_game"
    assert LABEL_MAP["game"] == "game"
