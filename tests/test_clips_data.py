"""Contracts for clip splitting.

Same shape as tests/test_frames_data.py, one level up. The leakage tests matter
for the same reason they did in Phase 2: a leaking split does not crash, it
raises the accuracy number, so nothing about the output looks wrong.
"""

import pytest

from clips.data import (CLASS_TO_INDEX, CLASSES, MIN_MATCHES_FOR_MATCH_SPLIT,
                        chronological_split, composition, match_split,
                        split_clips)


def clips(match_id, starts, label="none"):
    return [{"clip_id": f"{match_id}_{int(s * 100):08d}", "match_id": match_id,
             "start_sec": str(s), "label": label, "n_frames": "16"}
            for s in starts]


def many_matches(n):
    rows = []
    for i in range(n):
        rows.extend(clips(f"m{i}", [0.0, 10.0, 20.0]))
    return rows


def test_no_match_appears_on_both_sides_of_a_match_split():
    # The tripwire. Clips from one game share arena, lighting, jerseys and
    # camera crew; a model that learns the venue scores well and then collapses
    # on new footage.
    train, test = match_split(many_matches(5))
    assert {r["match_id"] for r in train} & {r["match_id"] for r in test} == set()


def test_a_match_split_keeps_every_clip():
    rows = many_matches(5)
    train, test = match_split(rows)
    assert len(train) + len(test) == len(rows)
    assert train and test


def test_match_split_is_used_once_there_are_enough_matches():
    train, test = split_clips(many_matches(MIN_MATCHES_FOR_MATCH_SPLIT))
    assert {r["match_id"] for r in train} & {r["match_id"] for r in test} == set()


def test_a_single_match_falls_back_to_a_chronological_split(capsys):
    # The fallback is allowed, but it must be impossible to use without seeing
    # that the resulting score is not a generalisation estimate.
    train, test = split_clips(clips("m1", [float(i) for i in range(0, 100, 10)]))
    warning = capsys.readouterr().out
    assert "NOT a generalisation estimate" in warning
    assert max(float(r["start_sec"]) for r in train) < \
           min(float(r["start_sec"]) for r in test)


def test_the_chronological_fallback_still_never_shares_a_clip():
    rows = clips("m1", [float(i) for i in range(0, 200, 10)])
    train, test = chronological_split(rows)
    assert not {r["clip_id"] for r in train} & {r["clip_id"] for r in test}


def test_chronological_split_works_when_the_manifest_is_out_of_order():
    # ingest/cut.py writes events first and background second, so the manifest
    # is never in time order.
    rows = clips("m1", [50.0, 10.0, 90.0, 30.0, 70.0, 0.0, 20.0, 80.0, 40.0, 60.0])
    train, test = chronological_split(rows)
    assert max(float(r["start_sec"]) for r in train) < \
           min(float(r["start_sec"]) for r in test)


def test_every_class_has_an_index_and_background_is_one_of_them():
    assert set(CLASS_TO_INDEX) == set(CLASSES)
    assert len(CLASS_TO_INDEX) == len(CLASSES)
    assert "none" in CLASSES


def test_the_six_event_classes_match_the_marking_keys():
    # If a key is added to mark.py without adding the class here, those clips
    # are silently dropped at load time and the owner's work disappears.
    from ingest.mark import MARK_KEYS
    marked = {name for name in MARK_KEYS.values() if name != "exclude"}
    assert marked == set(CLASSES) - {"none"}


def test_composition_counts_by_label():
    rows = clips("m1", [0.0, 10.0], "dunk") + clips("m1", [20.0], "none")
    assert composition(rows) == {"dunk": 2, "none": 1}


@pytest.mark.parametrize("n", [1, 2])
def test_too_few_matches_never_silently_produces_a_match_split(n, capsys):
    split_clips(many_matches(n))
    assert "chronological" in capsys.readouterr().out


def test_a_rare_class_is_weighted_above_a_common_one():
    # Without this, the cheapest way to cut the loss is to answer 'none' to
    # everything: ~58% accuracy, nothing learned. The gradient from four block
    # clips cannot outvote the gradient from four hundred background ones.
    from clips.data import class_weights
    rows = clips("m1", [float(i) for i in range(100)], "none") + \
        clips("m1", [200.0, 210.0], "block")
    weights = class_weights(rows)
    assert weights[CLASS_TO_INDEX["block"]] > weights[CLASS_TO_INDEX["none"]]


def test_a_class_with_no_examples_gets_no_weight():
    # 1/0 is not a number, and a class with no clips contributes no gradient
    # anyway. It must not become inf and poison the loss.
    from clips.data import class_weights
    weights = class_weights(clips("m1", [0.0, 1.0], "none"))
    assert weights[CLASS_TO_INDEX["dunk"]] == 0.0
    assert torch_is_finite(weights)


def torch_is_finite(t):
    import torch
    return bool(torch.isfinite(t).all())


def a_cache(tmp_path, monkeypatch, **blob):
    """Write a feature cache to a temp file and point cache_path at it."""
    import torch

    from clips import data as clips_data

    path = tmp_path / "cache.pt"
    torch.save(blob, path)
    monkeypatch.setattr(clips_data, "cache_path", lambda *_: str(path))
    return clips_data


def test_a_cache_built_by_another_backbone_is_refused(tmp_path, monkeypatch):
    # The guard that matters most in Phase 5. There are now several caches on
    # disk holding different-sized vectors from different models. Training on
    # the wrong one does not necessarily raise -- it can just quietly produce a
    # number, and a wrong number that looks fine is this project's recurring
    # failure mode.
    import torch

    clips_data = a_cache(tmp_path, monkeypatch, keys=["a"],
                         features=torch.randn(1, 768), backbone="mvit_v2_s",
                         crop="center", feature_dim=768)

    with pytest.raises(SystemExit):
        clips_data.load_cache("mc3_18", "center")


def test_a_cache_built_with_another_crop_is_refused(tmp_path, monkeypatch):
    # centre-crop and squash features are the same shape and the same backbone,
    # so nothing downstream would ever notice the mix-up. That is exactly why
    # the crop has to be recorded and checked.
    import torch

    clips_data = a_cache(tmp_path, monkeypatch, keys=["a"],
                         features=torch.randn(1, 768), backbone="mvit_v2_s",
                         crop="center", feature_dim=768)

    with pytest.raises(SystemExit):
        clips_data.load_cache("mvit_v2_s", "squash")


def test_the_matching_cache_loads_and_reports_its_dimension(tmp_path, monkeypatch):
    import torch

    clips_data = a_cache(tmp_path, monkeypatch, keys=["a", "b"],
                         features=torch.randn(2, 768), backbone="mvit_v2_s",
                         crop="center", feature_dim=768)

    lookup, feature_dim = clips_data.load_cache("mvit_v2_s", "center")
    assert feature_dim == 768
    assert set(lookup) == {"a", "b"}


def test_phase_fours_cache_still_loads_without_the_new_keys(tmp_path, monkeypatch):
    # The existing data/features/clips.pt was written before backbone and crop
    # were recorded. It must keep working, or the 53.92% baseline stops being
    # reproducible and the whole comparison loses its control.
    import torch

    clips_data = a_cache(tmp_path, monkeypatch, keys=["a"],
                         features=torch.randn(1, 16, 512))

    lookup, feature_dim = clips_data.load_cache("resnet18", "center")
    assert feature_dim == 512
    assert lookup["a"].shape == (16, 512)


def test_a_clip_loads_as_sixteen_uint8_frames(tmp_path, monkeypatch):
    # The handoff into torchvision's video preprocessing, which wants
    # [T, C, H, W] uint8. Anything else fails deep inside the transform.
    from PIL import Image

    from clips import data as clips_data
    from ingest.cut import CLIP_FRAMES

    directory = tmp_path / "m1" / "m1_00000000"
    directory.mkdir(parents=True)
    for i in range(CLIP_FRAMES):
        Image.new("RGB", (455, 256), (i * 10, 0, 0)).save(directory / f"{i:02d}.jpg")

    monkeypatch.setattr(clips_data, "CLIPS_ROOT", str(tmp_path))
    clip = clips_data.load_clip_tensor(
        {"match_id": "m1", "clip_id": "m1_00000000"})

    assert clip.shape == (CLIP_FRAMES, 3, 256, 455)
    assert clip.dtype.is_floating_point is False


def test_folded_classes_become_background_and_keep_their_clips(tmp_path, monkeypatch):
    # --fold block steal relabels, it does not drop. The clips are genuinely
    # "not a scoring play", and dropping them would quietly shrink the test set
    # and make the next number incomparable with the last one.
    from clips import data as clips_data

    rows = (clips("m1", [0.0], "block") + clips("m1", [1.0], "steal") +
            clips("m1", [2.0], "dunk"))
    monkeypatch.setattr(clips_data, "load_clip_manifest", lambda *_: rows)

    folded = clips_data.load_clips(fold=("block", "steal"))
    assert [r["label"] for r in folded] == ["none", "none", "dunk"]
    assert len(folded) == len(rows)


def test_two_views_are_joined_into_one_longer_vector(tmp_path, monkeypatch):
    # squash + hoop: the court then the rim, in that order, for every clip. If
    # the order ever varied between clips, the head's weights would be reading
    # the rim with the court's half and nothing would raise.
    import torch

    from clips import data as clips_data

    paths, stored = {}, {}
    for view in ("squash", "hoop"):
        stored[view] = torch.randn(2, 768)
        paths[view] = tmp_path / f"{view}.pt"
        torch.save({"keys": ["a", "b"], "features": stored[view],
                    "backbone": "mvit_v2_s", "crop": view, "feature_dim": 768},
                   paths[view])
    monkeypatch.setattr(clips_data, "cache_path", lambda _, view: str(paths[view]))

    lookup, feature_dim = clips_data.load_views("mvit_v2_s", ["squash", "hoop"])
    assert feature_dim == 1536
    assert torch.equal(lookup["a"][:768], stored["squash"][0])
    assert torch.equal(lookup["a"][768:], stored["hoop"][0])


def test_the_hoop_view_reads_the_close_ups_and_resizes_them_whole():
    # The close-ups are already square. Centre-cropping them would throw away
    # the ball's arc above the rim -- the very thing they were cut to show.
    from clips.data import VIEWS
    from ingest.hoop import HOOP_ROOT
    assert VIEWS["hoop"] == (HOOP_ROOT, "squash")


def test_weights_are_computed_from_the_rows_they_are_given():
    # clips/train.py passes the TRAIN split only. Deriving them from everything
    # would feed the test set's class balance into training.
    from clips.data import class_weights
    balanced = clips("m1", [0.0, 1.0], "none") + clips("m1", [2.0, 3.0], "dunk")
    weights = class_weights(balanced)
    assert abs(weights[CLASS_TO_INDEX["none"]] - weights[CLASS_TO_INDEX["dunk"]]) < 1e-6
