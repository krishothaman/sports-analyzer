"""Score the model the way it is used: from raw video, on the held-out match.

clips/train.py scores cached features of clips cut at exactly one offset. The
product reads raw video at whatever moment it is asked about. This runs the real
prediction path (clips.predict.Predictor) on every test clip, optionally from
several shifted starts, and reports goal-level recall and precision for:

    centre    the clip start as cut (shift 0) -- must reproduce clips/train.py
    averaged  the class probabilities averaged over every --shifts start

each on the test set as labelled, and with double-marked plays removed.

    python -m clips.evaluate --shifts -0.25 0 0.25

About 1.5 s per clip per shift: 293 test clips x 3 shifts is ~22 minutes.
"""

import argparse

import cv2
import torch

from clips.data import load_clips, split_clips
from clips.predict import CHECKPOINT, Predictor, average_probs
from clips.train import GOAL_GROUPS
from ingest.cut import BACKGROUND_LABEL
from ingest.hoop import read_clip, video_path
from models.metrics import group_matrix, print_report

# Same-label events closer than this are one play marked twice.
DUPLICATE_WITHIN = 1.0


def dedup_rows(rows, within=DUPLICATE_WITHIN):
    """Drop event clips that repeat the previous same-label event within `within` s.

    match03 holds 23 plays marked twice -- once by hand, once by the scoreboard
    reader -- under a second apart. Counting both scores one basket twice.
    Background clips are never dropped: they were sampled, not marked.
    """
    kept, last = [], {}
    for row in sorted(rows, key=lambda r: (r["match_id"], float(r["start_sec"]))):
        if row["label"] == BACKGROUND_LABEL:
            kept.append(row)
            continue
        key = (row["match_id"], row["label"])
        start = float(row["start_sec"])
        previous = last.get(key)
        last[key] = start
        if previous is None or start - previous >= within:
            kept.append(row)
    return kept


def predict_rows(predictor, rows, shifts):
    """{clip_id: {shift: probs}} from raw video.

    A shift that would start before the video or run past its end is left out
    for that clip rather than padded.
    """
    results = {}
    total = len(rows)
    done = 0
    for match_id in sorted({row["match_id"] for row in rows}):
        cap = cv2.VideoCapture(video_path(match_id))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for row in sorted((r for r in rows if r["match_id"] == match_id),
                          key=lambda r: float(r["start_sec"])):
            per_shift = {}
            for shift in shifts:
                start = float(row["start_sec"]) + shift
                if start < 0:
                    continue
                frames = read_clip(cap, start, fps)
                if any(frame is None for frame in frames):
                    continue
                probs, _ = predictor.predict(frames)
                if probs is not None:
                    per_shift[shift] = probs
            results[row["clip_id"]] = per_shift
            done += 1
            print(f"  {done}/{total} clips", end="\r", flush=True)
        cap.release()
    print()
    return results


def score(rows, probs_for, classes):
    """Goal-level confusion matrix, from probs_for(row) -> [C] probabilities or None."""
    preds, targets = [], []
    for row in rows:
        probs = probs_for(row)
        if probs is None:
            continue
        # The single most likely class, then its group: grouped_report's rule.
        preds.append(int(probs.argmax()))
        targets.append(classes.index(row["label"]))
    return group_matrix(torch.tensor(preds, dtype=torch.long),
                        torch.tensor(targets, dtype=torch.long), classes, GOAL_GROUPS)


def summary(name, matrix):
    """One table row: recall and precision per goal group."""
    cells = []
    for i in range(len(GOAL_GROUPS)):
        support = matrix[i, :].sum().item()
        called = matrix[:, i].sum().item()
        hits = matrix[i, i].item()
        cells.append(f"{hits / support if support else 0:>7.1%} "
                     f"{hits / called if called else 0:>6.1%}")
    correct = matrix.diagonal().sum().item()
    total = matrix.sum().item()
    return f"{name:<22}" + "  ".join(cells) + f"   {correct}/{total}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shifts", nargs="+", type=float, default=[0.0], metavar="SEC",
                        help="clip start offsets to average over, e.g. -0.25 0 0.25")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    _, test_rows = split_clips(load_clips(), quiet=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(test_rows)} test clips x {len(args.shifts)} shift(s) "
          f"{args.shifts}, about {len(test_rows) * len(args.shifts) * 1.5 / 60:.0f} min")
    predictor = Predictor(device, args.checkpoint)
    classes = predictor.classes

    results = predict_rows(predictor, test_rows, args.shifts)

    variants = []
    if 0.0 in args.shifts:
        variants.append(("centre", lambda row: results[row["clip_id"]].get(0.0)))
    if len(args.shifts) > 1:
        variants.append(("averaged", lambda row: average_probs(
            results[row["clip_id"]].values()) if results[row["clip_id"]] else None))

    header = "  ".join(f"{group:>7} {'prec':>6}" for group in GOAL_GROUPS)
    print(f"\n{'':<22}{header}   right  (recall, precision per goal class)")
    for name, probs_for in variants:
        for label, rows in (("", test_rows), (" dedup", dedup_rows(test_rows))):
            print(summary(name + label, score(rows, probs_for, classes)))

    # The full matrix for the centre view, to set beside clips/train.py's report.
    if 0.0 in args.shifts:
        print("\ncentre, as labelled -- should match clips/train.py for this checkpoint:")
        print_report(score(test_rows, variants[0][1], classes), list(GOAL_GROUPS))


if __name__ == "__main__":
    main()
