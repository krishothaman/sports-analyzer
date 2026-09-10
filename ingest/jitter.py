"""Copies of every training event, cut at other offsets -- more timings, no new labels.

ingest/cut.py puts every event exactly 1.5 s into its 2-second clip, so the
model has only ever seen a shot at one moment of the window. Asked about a
moment one frame off, it can change its mind: in Phase 5 a 0.03 s shift turned
a field goal into `none`. A real scan lands wherever it lands.

This writes a second manifest holding each TRAINING event again at four other
offsets (the event 1.0, 1.25, 1.75 and 2.0 s into the clip). Same label, same
play, different framing -- the model learns the play rather than the timing.

Test-match events are never copied: the test set stays exactly as it was, so
the numbers stay comparable with every earlier phase. The original manifest is
never written to.

    python -m ingest.jitter
    python -m ingest.hoop --manifest data/clip_manifest_jitter.csv
    python -m clips.data --backbone mvit_v2_s --crop hoop --extra jitter
    python -m clips.train --backbone mvit_v2_s --crop hoop --head clip --fold block steal --epochs 100 --seed 0 --extra jitter
"""

import os

from clips.data import EXTRA_MANIFESTS, composition, load_clips, split_clips
from ingest.cut import BACKGROUND_LABEL, CLIP_MANIFEST, save_manifest

# ingest/cut.py's default --pre: where the event sits in every original clip.
CUT_PRE = 1.5
# Where the copies put it. Symmetric around 1.5, and never under 1.0 s in, so
# every copy still holds the run-up before the shot.
PRES = (1.0, 1.25, 1.75, 2.0)


def jitter_rows(rows, test_matches, pres=PRES, cut_pre=CUT_PRE):
    """One copy of every event row per offset in `pres`. Background is not copied.

    Refuses rows from test matches outright rather than skipping them: a
    caller passing test rows has a bug, and copying them would leak the test
    set into training.
    """
    copies = []
    for row in rows:
        if row["label"] == BACKGROUND_LABEL:
            continue
        if row["match_id"] in test_matches:
            raise ValueError(f"{row['clip_id']} is from test match {row['match_id']} "
                             f"-- test clips are never copied into training")
        mark = float(row["start_sec"]) + cut_pre
        for pre in pres:
            start = mark - pre
            if start < 0:
                continue
            copies.append({"clip_id": f"{row['clip_id']}_p{round(pre * 100):03d}",
                           "match_id": row["match_id"], "start_sec": f"{start:.2f}",
                           "label": row["label"], "n_frames": row["n_frames"]})
    return copies


def main():
    path = EXTRA_MANIFESTS["jitter"]
    if os.path.abspath(path) == os.path.abspath(CLIP_MANIFEST):
        raise SystemExit("refusing to overwrite the original clip manifest")

    train, test = split_clips(load_clips(), quiet=True)
    test_matches = {row["match_id"] for row in test}
    copies = jitter_rows(train, test_matches)
    save_manifest(copies, path)

    events = sum(1 for row in train if row["label"] != BACKGROUND_LABEL)
    print(f"{events} training events -> {len(copies)} copies at offsets {PRES}")
    print("  " + "  ".join(f"{name}={n}" for name, n in sorted(composition(copies).items())))
    print(f"test matches {sorted(test_matches)} untouched; wrote {path}")
    print(f"next: python -m ingest.hoop --manifest {path}")


if __name__ == "__main__":
    main()
