"""Turn marks into clips -- the second half of mark-then-cut.

Reads data/events/<match>.csv, works out every clip that should exist, then walks
the video ONCE and writes them all. One pass matters: seeking to each clip
separately would re-decode from the nearest keyframe every time and turn minutes
into hours.

Three things happen here that are worth understanding:

1. The clip sits BEHIND the mark. You press a key when you see the outcome, which
   is already about 0.4s late, and the informative motion -- the drive, the
   gather, the rise -- happened before that. So a mark at t becomes the window
   [t - pre, t - pre + 2s].

2. Background is sampled, never marked. Any 2-second window that stays clear of
   every mark is a candidate for the 'none' class, which is how ~80% of the
   dataset gets built without a single keystroke.

3. Those candidates are sifted by the Phase 2 frame filter. Adverts, crowd shots
   and graphics must not enter the dataset as "basketball with nothing
   happening". This is the job Phase 2 was built for.

Event clips are NOT filtered. You saw them and marked them; a filter
false-negative silently deleting a real dunk would be worse than a stray frame.
"""

import argparse
import csv
import glob
import os
import random
import shutil

import cv2
import torch
from PIL import Image

from frames.data import CLASSES as FRAME_CLASSES
from frames.predict import classify_images, load_filter
from ingest.mark import EVENTS_ROOT, load_marks

# 16 frames at 8 fps = exactly 2 seconds. 8 fps is enough to see a shooting
# motion; 25 fps would trade three times the storage for near-duplicate frames.
CLIP_FRAMES = 16
SAMPLE_FPS = 8.0
CLIP_SECONDS = CLIP_FRAMES / SAMPLE_FPS

# 'exclude' is a hole in the timeline: not an event, and not offered as
# background either. That is the whole reason the key exists.
EXCLUDE_LABEL = "exclude"
BACKGROUND_LABEL = "none"

CLIPS_ROOT = os.path.join("data", "clips")
CLIP_MANIFEST = os.path.join("data", "clip_manifest.csv")
CLIP_FIELDS = ["clip_id", "match_id", "start_sec", "label", "n_frames"]

# Short side 256 leaves room for the backbone's 224 centre crop while cutting
# each JPEG to roughly a fifth of the original's bytes.
SHORT_SIDE = 256


def clip_id_for(match_id, start_sec):
    """Deterministic from the time, so re-cutting overwrites rather than duplicates."""
    return f"{match_id}_{int(round(start_sec * 100)):08d}"


def frame_indices(start_sec, video_fps):
    """The 16 source frame numbers making up one clip."""
    return [int(round((start_sec + i / SAMPLE_FPS) * video_fps))
            for i in range(CLIP_FRAMES)]


def event_starts(marks, pre):
    """[(start_sec, label)] for every real event mark, excludes dropped."""
    starts = []
    for mark in marks:
        if mark["label"] == EXCLUDE_LABEL:
            continue
        starts.append((float(mark["timestamp_sec"]) - pre, mark["label"]))
    return starts


def background_starts(duration, marks, guard, stride=CLIP_SECONDS):
    """Windows whose centre stays `guard` seconds clear of every mark.

    The guard band is the important part. A window overlapping a dunk's run-up
    would be labelled 'none' while containing the very motion that should fire
    the dunk class -- teaching the model to suppress exactly what we want it to
    detect. Excludes count as marks here for the same reason: a missed three is
    not background.
    """
    times = sorted(float(m["timestamp_sec"]) for m in marks)

    starts = []
    start = 0.0
    while start + CLIP_SECONDS <= duration:
        centre = start + CLIP_SECONDS / 2
        if all(abs(centre - t) >= guard for t in times):
            starts.append(start)
        start += stride
    return starts


def resize_short_side(image, target=SHORT_SIDE):
    height, width = image.shape[:2]
    scale = target / min(height, width)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)),
                      interpolation=cv2.INTER_AREA)


def extract_clips(video_path, plan, video_fps):
    """Write every planned clip in one sequential pass.

    `plan` is [(clip_id, start_sec, label)]. Returns
    (written, middle_frames) where middle_frames maps clip_id -> PIL image for
    the clip's centre frame, used afterwards to sift background candidates.
    """
    # frame number -> [(clip_id, position within the clip)]
    wanted = {}
    for clip_id, start_sec, _ in plan:
        for position, index in enumerate(frame_indices(start_sec, video_fps)):
            wanted.setdefault(index, []).append((clip_id, position))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    written = {clip_id: 0 for clip_id, _, _ in plan}
    middle_frames = {}
    last_wanted = max(wanted) if wanted else -1

    frame_index = 0
    done = 0
    while frame_index <= last_wanted:
        # grab() advances without fully decoding. We only pay for a decode on the
        # frames a clip actually needs -- a few thousand out of 162,000.
        if not cap.grab():
            break

        targets = wanted.get(frame_index)
        if targets:
            ok, frame = cap.retrieve()
            if ok:
                small = resize_short_side(frame)
                for clip_id, position in targets:
                    out_dir = os.path.join(CLIPS_ROOT, clip_id.split("_")[0], clip_id)
                    os.makedirs(out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(out_dir, f"{position:02d}.jpg"), small)
                    written[clip_id] += 1

                    if position == CLIP_FRAMES // 2:
                        middle_frames[clip_id] = Image.fromarray(
                            cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

                done += len(targets)
                print(f"  extracting {done}/{sum(len(v) for v in wanted.values())} frames",
                      end="\r", flush=True)

        frame_index += 1

    cap.release()
    return written, middle_frames


def sift_background(candidates, middle_frames, device, keep):
    """Drop candidates the Phase 2 filter says are not live game footage.

    Returns (kept_ids, rejected_ids). Only the centre frame is classified: a
    2-second window is either broadcast basketball throughout or it is not, and
    checking one frame costs a sixteenth of checking all of them.
    """
    ids = [clip_id for clip_id in candidates if clip_id in middle_frames]
    if not ids:
        return [], []

    bundle = load_filter(device)
    predicted, _ = classify_images([middle_frames[i] for i in ids], device, bundle)

    game_index = FRAME_CLASSES.index("game")
    survivors = [clip_id for clip_id, pred in zip(ids, predicted.tolist())
                 if pred == game_index]
    rejected = [clip_id for clip_id in ids if clip_id not in set(survivors)]

    # Trim to the target only after filtering, so rejects do not eat the quota.
    if keep is not None and len(survivors) > keep:
        rejected.extend(survivors[keep:])
        survivors = survivors[:keep]

    return survivors, rejected


def remove_clip(clip_id):
    path = os.path.join(CLIPS_ROOT, clip_id.split("_")[0], clip_id)
    shutil.rmtree(path, ignore_errors=True)


def save_manifest(rows, path=CLIP_MANIFEST):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLIP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_clip_manifest(path=CLIP_MANIFEST):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def cut_match(video_path, match_id, args, device):
    marks = load_marks(os.path.join(EVENTS_ROOT, f"{match_id}.csv"))
    if not marks:
        raise SystemExit(f"no marks for {match_id} -- run: python -m ingest.mark")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / video_fps if total_frames else 0.0
    cap.release()

    events = event_starts(marks, args.pre)
    events = [(s, label) for s, label in events if s >= 0 and s + CLIP_SECONDS <= duration]

    target_background = int(round(len(events) * args.bg_ratio))
    candidates = background_starts(duration, marks, args.guard)
    # Oversample so the filter's rejects do not leave us short of the target.
    rng = random.Random(args.seed)
    if len(candidates) > target_background * 2:
        candidates = rng.sample(candidates, target_background * 2)
    candidates.sort()

    plan = [(clip_id_for(match_id, s), s, label) for s, label in events]
    plan += [(clip_id_for(match_id, s), s, BACKGROUND_LABEL) for s in candidates]

    print(f"{match_id}: {len(events)} event clips, {len(candidates)} background "
          f"candidates for a target of {target_background}")

    written, middle_frames = extract_clips(video_path, plan, video_fps)
    print()

    candidate_ids = [clip_id for clip_id, _, label in plan if label == BACKGROUND_LABEL]
    kept, rejected = sift_background(candidate_ids, middle_frames, device,
                                     target_background)
    for clip_id in rejected:
        remove_clip(clip_id)

    keep_ids = set(kept)
    rows = []
    short = 0
    for clip_id, start_sec, label in plan:
        if label == BACKGROUND_LABEL and clip_id not in keep_ids:
            continue
        # A clip missing frames means the decoder stalled. Drop it rather than
        # padding: a short clip would silently change what the model sees.
        if written.get(clip_id, 0) != CLIP_FRAMES:
            remove_clip(clip_id)
            short += 1
            continue
        rows.append({"clip_id": clip_id, "match_id": match_id,
                     "start_sec": f"{start_sec:.2f}", "label": label,
                     "n_frames": CLIP_FRAMES})

    return rows, len(rejected), short


def main():
    parser = argparse.ArgumentParser(description="Cut marked events into clips.")
    parser.add_argument("--match-id", default="match01")
    parser.add_argument("--video", default=None,
                        help="path to the video (default: data/video/<match_id>.mp4)")
    parser.add_argument("--pre", type=float, default=1.5,
                        help="seconds of run-up before your keypress "
                             f"(clip is always {CLIP_SECONDS:g}s, so post = {CLIP_SECONDS:g} - pre)")
    parser.add_argument("--guard", type=float, default=4.0,
                        help="keep background this far from every mark")
    parser.add_argument("--bg-ratio", type=float, default=2.0,
                        help="background clips per event clip")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    video = args.video or os.path.join("data", "video", f"{args.match_id}.mp4")
    if not os.path.exists(video):
        found = glob.glob(os.path.join("data", "video", f"{args.match_id}.*"))
        if not found:
            raise SystemExit(f"no video found for {args.match_id} in data/video/")
        video = found[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cutting on {device}   pre={args.pre}s  post={CLIP_SECONDS - args.pre:g}s")

    rows, rejected, short = cut_match(video, args.match_id, args, device)

    # Re-cutting replaces this match's rows and leaves other matches alone.
    existing = [r for r in load_clip_manifest() if r["match_id"] != args.match_id]
    save_manifest(existing + rows)

    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    print(f"\n{len(rows)} clips written for {args.match_id}")
    for name in sorted(counts, key=lambda n: (n == BACKGROUND_LABEL, n)):
        print(f"  {name:<16}{counts[name]:>5}")
    print(f"\nframe filter rejected {rejected} background candidates as not-game")
    if short:
        print(f"{short} clips dropped for missing frames")
    print(f"manifest: {CLIP_MANIFEST}")
    print(f"\nspot-check one:  data/clips/{args.match_id}/<clip_id>/")


if __name__ == "__main__":
    main()
