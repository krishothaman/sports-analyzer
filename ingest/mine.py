"""Find the model's false alarms in the training matches, for the owner to review.

About half of the Phase 5 model's "field goal" calls are wrong. The likely cause
is missed shots: a miss looks like a make until the last few frames, and the
training set holds few of them labelled `none`. The way to teach the difference
is to show it the misses it currently gets wrong.

This scans a stretch of a TRAINING match that the owner has already watched and
marked, and keeps every window the model calls a scoring play that is nowhere
near any mark (hand or scoreboard). Each of those is a false alarm, or a play
nobody marked. Which one is a person's call, so it writes a contact sheet per
candidate -- 4 frames of the broadcast above the 16 hoop close-ups the model saw
-- and a CSV with an empty `verdict` column:

    none  (or n, miss)   nothing scored here -- a miss, a pass, a rebound
    two_pointer (2)  three_pointer (3)  dunk (d)  free_throw (f)
                         a real play nobody marked
    skip  (or s)         can't tell -- leave it out

    python -m ingest.mine --match-id match01 --from 20:00 --to 40:00
    ... fill in data/review/match01/candidates.csv ...
    python -m ingest.mine --collect
    python -m ingest.hoop --manifest data/clip_manifest_hard.csv

Test matches are refused: mining them would train on the test set.
"""

import argparse
import csv
import glob
import os

import cv2
import torch
from PIL import Image, ImageDraw

from clips.data import CLASSES, EXTRA_MANIFESTS, load_clips, split_clips
from clips.predict import (PRE, Predictor, format_time, goal_answer, parse_time,
                           run, window_starts)
from ingest.cut import (BACKGROUND_LABEL, CLIP_FRAMES, CLIP_MANIFEST, CLIP_SECONDS,
                        clip_id_for, reviewed_until_for, save_manifest)
from ingest.hoop import DETECT_POSITIONS, close_ups, read_clip, track, video_path
from ingest.mark import EVENTS_ROOT, load_marks
from ingest.scoreboard import auto_events_path

REVIEW_ROOT = os.path.join("data", "review")
# ingest/cut.py's default --guard: how far background clips stay from any mark.
GUARD = 4.0
# Most confident first. The most confident false alarms teach the most, and a
# review longer than this stops being careful.
MAX_CANDIDATES = 60
FIELDS = ["n", "t", "start_sec", "predicted", "confidence", "sheet", "verdict"]

VERDICTS = {"none": BACKGROUND_LABEL, "n": BACKGROUND_LABEL, "miss": BACKGROUND_LABEL,
            "2": "two_pointer", "3": "three_pointer", "d": "dunk", "f": "free_throw",
            "skip": "skip", "s": "skip"}
VERDICTS.update({name: name for name in CLASSES})


def parse_verdict(text):
    """A reviewer's verdict -> a class name, 'skip', or None if not reviewed yet."""
    key = (text or "").strip().lower()
    if not key:
        return None
    if key not in VERDICTS:
        raise ValueError(f"unknown verdict '{text}' -- use one of {sorted(VERDICTS)}")
    return VERDICTS[key]


def far_from_marks(start, mark_times, guard=GUARD):
    """True if a window's centre is at least `guard` s from every mark -- ingest/cut.py's rule."""
    centre = start + CLIP_SECONDS / 2
    return all(abs(centre - t) >= guard for t in mark_times)


def test_matches():
    _, test = split_clips(load_clips(), quiet=True)
    return {row["match_id"] for row in test}


def refuse_test_match(match_id, tests):
    if match_id in tests:
        raise SystemExit(f"{match_id} is a test match -- mining it would train on the "
                         f"test set")


def has_verdicts(path):
    """Whether a candidates file already holds any of the owner's review work."""
    if not os.path.exists(path):
        return False
    with open(path, newline="", encoding="utf-8") as fh:
        return any((row.get("verdict") or "").strip() for row in csv.DictReader(fh))


def collect_rows(candidates, match_id, tests):
    """Reviewed candidates -> manifest rows. Unreviewed and skipped ones are left out."""
    refuse_test_match(match_id, tests)
    rows = []
    for candidate in candidates:
        try:
            label = parse_verdict(candidate.get("verdict"))
        except ValueError as err:
            raise SystemExit(f"{match_id} candidate {candidate.get('n')}: {err}") from None
        if label in (None, "skip"):
            continue
        start = float(candidate["start_sec"])
        rows.append({"clip_id": f"{clip_id_for(match_id, start)}_hard",
                     "match_id": match_id, "start_sec": f"{start:.2f}",
                     "label": label, "n_frames": CLIP_FRAMES})
    return rows


def contact_sheet(frames, views, title):
    """4 broadcast frames above the 16 close-ups the model read, with a title bar."""
    side, bar = 256, 22
    wide = [frames[pos].resize((side, side * 9 // 16)) for pos in DETECT_POSITIONS]
    top = side * 9 // 16
    sheet = Image.new("RGB", (4 * side, bar + top + 4 * side), "black")
    for i, image in enumerate(wide):
        sheet.paste(image, (i * side, bar))
    for i, view in enumerate(views):
        sheet.paste(view.resize((side, side)), ((i % 4) * side, bar + top + (i // 4) * side))
    ImageDraw.Draw(sheet).text((6, 5), title, fill="yellow")
    return sheet


def all_mark_times(match_id):
    marks = (load_marks(os.path.join(EVENTS_ROOT, f"{match_id}.csv")) +
             load_marks(auto_events_path(match_id)))
    return marks, [float(mark["timestamp_sec"]) for mark in marks]


def mine(match_id, start, end, limit, device):
    refuse_test_match(match_id, test_matches())
    out_dir = os.path.join(REVIEW_ROOT, match_id)
    csv_path = os.path.join(out_dir, "candidates.csv")
    if has_verdicts(csv_path):
        raise SystemExit(f"{csv_path} already holds review verdicts -- move it aside "
                         f"first; it will not be overwritten")

    cap = cv2.VideoCapture(video_path(match_id))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    marks, times = all_mark_times(match_id)
    reviewed = reviewed_until_for(match_id, marks, duration)
    if end > reviewed:
        raise SystemExit(f"--to {format_time(end)} is past what has been reviewed "
                         f"({format_time(reviewed)}): an unmarked play there is not a "
                         f"false alarm, just unwatched")

    starts = [s for s in window_starts(start, end) if far_from_marks(s, times)]
    print(f"{match_id} {format_time(start)}-{format_time(end)}: {len(starts)} windows "
          f"clear of every mark, about {len(starts) * 1.5 / 60:.0f} min")
    predictor = Predictor(device)

    found = []
    for moment, probs, _ in run(predictor, cap, fps, starts):
        label, confidence = goal_answer(probs, predictor.classes)
        if label != BACKGROUND_LABEL:
            found.append((moment - PRE, label, confidence))
    print(f"\n{len(found)} windows called a scoring play with no mark near them")

    found = sorted(sorted(found, key=lambda c: -c[2])[:limit])
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for n, (clip_start, label, confidence) in enumerate(found, 1):
        frames = read_clip(cap, clip_start, fps)
        detections = predictor.detector.find([frames[pos] for pos in DETECT_POSITIONS])
        centres = track([det[:2] if det else None for det in detections])
        name = f"{n:03d}_{int(clip_start)}.jpg"
        title = (f"#{n}  {format_time(clip_start + PRE)}  model says {label} "
                 f"{confidence:.0%} -- what really happened?")
        contact_sheet(frames, close_ups(frames, centres), title).save(
            os.path.join(out_dir, name), quality=85)
        rows.append({"n": n, "t": format_time(clip_start + PRE),
                     "start_sec": f"{clip_start:.2f}", "predicted": label,
                     "confidence": f"{confidence:.2f}", "sheet": name, "verdict": ""})
    cap.release()

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} candidates -> {csv_path} (sheets alongside)")
    print("fill in `verdict` for each, then: python -m ingest.mine --collect")


def collect():
    path = EXTRA_MANIFESTS["hard"]
    if os.path.abspath(path) == os.path.abspath(CLIP_MANIFEST):
        raise SystemExit("refusing to overwrite the original clip manifest")

    tests = test_matches()
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(REVIEW_ROOT, "*", "candidates.csv"))):
        match_id = os.path.basename(os.path.dirname(csv_path))
        with open(csv_path, newline="", encoding="utf-8") as fh:
            candidates = list(csv.DictReader(fh))
        collected = collect_rows(candidates, match_id, tests)
        print(f"{match_id}: {len(collected)} of {len(candidates)} candidates reviewed and kept")
        rows += collected
    if not rows:
        raise SystemExit("no reviewed candidates -- fill in the verdict column first")

    save_manifest(rows, path)
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print("  " + "  ".join(f"{name}={n}" for name, n in sorted(counts.items())))
    print(f"wrote {path}\nnext: python -m ingest.hoop --manifest {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", help="a training match to mine")
    parser.add_argument("--from", dest="start", type=parse_time, metavar="TIME")
    parser.add_argument("--to", dest="end", type=parse_time, metavar="TIME")
    parser.add_argument("--max", type=int, default=MAX_CANDIDATES,
                        help=f"keep the most confident N (default {MAX_CANDIDATES})")
    parser.add_argument("--collect", action="store_true",
                        help="turn reviewed candidates into data/clip_manifest_hard.csv")
    args = parser.parse_args()

    if args.collect:
        collect()
        return
    if not (args.match_id and args.start is not None and args.end is not None):
        parser.error("give --match-id, --from and --to (or --collect)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mine(args.match_id, args.start, args.end, args.max, device)


if __name__ == "__main__":
    main()
