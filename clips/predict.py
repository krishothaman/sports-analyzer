"""Point the trained model at a match video and a moment; get back what happened.

Everything before this worked on clips someone had already cut, and features
someone had already cached. This is the first path that starts from raw video,
so every stage runs, in order, on a moment the pipeline has never seen:

    video --(read 16 frames at 8 fps)--> Phase 2 filter: is this live play?
          --(if not: answer none, skip the rest)--> OWLv2 finds the hoop
          --(256x256 close-up, cut exactly as ingest/hoop.py cut it)-->
          frozen MViTv2-S --> 768 numbers --> the trained linear head
          --> none / field_goal / free_throw

    python -m clips.predict data/video/match03.mp4 --at 12:31 --at 40:05
    python -m clips.predict data/video/match03.mp4 --from 10:00 --to 15:00 --json timeline.json

The answer is reported at the goal level only. The head does emit seven classes,
but Phase 5 measured that it cannot tell a two from a three and over-says
`dunk`, so printing the fine-grained guess would be printing noise with a
confident-looking percentage next to it.
"""

import argparse
import io
import json
import os

import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from clips.data import VIEWS
from clips.model import build_head
from clips.train import GOAL_GROUPS
from frames.data import CLASSES as FRAME_CLASSES
from frames.predict import classify_images, load_filter
from ingest.cut import CLIP_FRAMES, CLIP_SECONDS
from ingest.hoop import (CROP_QUALITY, DETECT_POSITIONS, HoopDetector,
                         close_ups, read_clip, track)
from models.backbone import build_video_backbone

BACKBONE = "mvit_v2_s"
VIEW = "hoop"
# The file clips/train.py writes for --backbone mvit_v2_s --crop hoop --head clip.
CHECKPOINT = f"clip_head_{BACKBONE}_{VIEW}_clip.pt"

# Seconds of clip before the moment asked about. Training clips were cut with
# ingest/cut.py's default --pre 1.5, so the event sits 1.5s into the 2s window;
# asking about a moment puts it in the same place.
PRE = 1.5

# Measured on this project's laptop GPU: OWLv2 on four frames dominates.
SECONDS_PER_WINDOW = 1.5


def parse_time(text):
    """'12:31' -> 751.0. Also takes 'h:mm:ss' and plain seconds."""
    parts = text.strip().split(":")
    try:
        if not 1 <= len(parts) <= 3:
            raise ValueError
        values = [float(part) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{text}' is not a time -- use seconds, mm:ss or h:mm:ss") from None
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(f"'{text}' is negative")

    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def format_time(seconds):
    whole = int(round(seconds))
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def goal_probabilities(probs, classes, groups=GOAL_GROUPS):
    """Add up the seven class probabilities into the goal groups.

    Adding is legitimate here because the classes in a group are mutually
    exclusive: P(field goal) is P(two) + P(three) + P(dunk). It is also why the
    goal-level answer can be confident while the head is torn between two and
    three -- that indecision stays inside the group.
    """
    return {group: sum(probs[classes.index(name)].item() for name in members)
            for group, members in groups.items()}


def goal_answer(probs, classes, groups=GOAL_GROUPS):
    """The goal-level answer, decided exactly the way the reported results were scored.

    clips/train.py's grouped_report takes the head's single most likely class
    and then maps it to its group, and every number in docs/phase-5-notes.md was
    scored that way. Picking the group with the largest *summed* probability
    sounds equivalent and is not: three field-goal classes pooled together
    outvote `none` far more often (seed 0: field goals 85% instead of 74%,
    `none` 59% instead of 68%). That would ship a different model from the one
    that was measured.

    Returns (group, that group's summed probability).
    """
    best = classes[int(probs.argmax())]
    group = next(name for name, members in groups.items() if best in members)
    return group, goal_probabilities(probs, classes, groups)[group]


def window_starts(start, end, every=CLIP_SECONDS):
    """Clip start times tiling [start, end], each clip wholly inside it."""
    starts, t = [], start
    while t + CLIP_SECONDS <= end + 1e-9:
        starts.append(round(t, 3))
        t += every
    return starts


def merge_events(windows, every=CLIP_SECONDS):
    """Collapse neighbouring windows that report the same event into one.

    `windows` is [(moment, label, confidence)] in time order, `none` already
    dropped. A shot that straddles two windows would otherwise be listed twice.
    The merged event keeps the moment of its most confident window.
    """
    events = []
    for moment, label, confidence in windows:
        last = events[-1] if events else None
        if last and last["event"] == label and moment - last["_end"] <= every + 1e-6:
            last["_end"] = moment
            if confidence > last["confidence"]:
                last["t"], last["confidence"] = moment, confidence
        else:
            events.append({"t": moment, "event": label,
                           "confidence": confidence, "_end": moment})
    return [{key: value for key, value in event.items() if key != "_end"}
            for event in events]


def load_head(path, device):
    """The trained head, refusing one that was trained on a different view.

    A head trained on the whole-court view would load without complaint -- the
    shapes are identical -- and then read close-ups as if they were courts.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"no trained head at {path} -- train it first:\n"
            f"  python -m clips.train --backbone {BACKBONE} --crop {VIEW} --head clip "
            f"--fold block steal --epochs 100 --seed 0")

    saved = torch.load(path, map_location=device)
    if saved.get("backbone") != BACKBONE or saved.get("crop") != [VIEW]:
        raise SystemExit(
            f"{path} was trained on {saved.get('backbone')} / {saved.get('crop')}, "
            f"but this script feeds it {BACKBONE} / ['{VIEW}'] -- retrain it")

    classes = saved["classes"]
    head = build_head(saved["head"], len(classes), saved["feature_dim"]).to(device)
    head.load_state_dict(saved["state_dict"])
    head.eval()
    return head, classes


def as_stored(view):
    """The close-up as training saw it: after a JPEG save and reload.

    Training read its close-ups back off disk, compression and all. Skipping
    that here would hand the model slightly cleaner pixels than it ever learned
    from -- a small difference, but a free one to remove.
    """
    buffer = io.BytesIO()
    view.save(buffer, format="JPEG", quality=CROP_QUALITY)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class Predictor:
    """The models, loaded once: live-play filter, hoop finder, eyes, and rulebook."""

    def __init__(self, device, checkpoint=CHECKPOINT):
        self.device = device
        # Head first: it is the quick one to load and the likeliest to be missing.
        self.head, self.classes = load_head(checkpoint, device)
        # Phase 2's game / not_game filter. ingest/cut.py used it to throw
        # close-ups, replays and crowd shots out of the background clips, so the
        # head never saw any during training. A scan meets them constantly, and
        # a model asked about footage unlike anything it learned from answers
        # confidently and wrongly -- the first scan called player close-ups free
        # throws at 97%. Filtering here asks it only the questions it was
        # trained on.
        self.live_filter = load_filter(device)
        self.detector = HoopDetector(device)
        _, mode = VIEWS[VIEW]
        self.backbone, self.transform, _ = build_video_backbone(BACKBONE, device, mode)

    def is_live(self, frames):
        """Is the clip's centre frame live game footage? Same check as ingest/cut.py."""
        predicted, _ = classify_images([frames[CLIP_FRAMES // 2]], self.device,
                                       self.live_filter)
        return FRAME_CLASSES[predicted[0].item()] == "game"

    def features(self, frames):
        """16 full-resolution frames -> (768-d feature, whether a hoop was found)."""
        detections = self.detector.find([frames[pos] for pos in DETECT_POSITIONS])
        centres = track([det[:2] if det else None for det in detections])
        clip = torch.stack([pil_to_tensor(as_stored(view))
                            for view in close_ups(frames, centres)])
        with torch.no_grad():
            feature = self.backbone(self.transform(clip).unsqueeze(0).to(self.device))
        return feature[0], centres is not None

    def predict(self, frames):
        """-> (probability per class, or None if not live play; hoop found)."""
        if not self.is_live(frames):
            return None, False
        feature, found = self.features(frames)
        with torch.no_grad():
            # Softmax only to make the numbers readable; training never needed it.
            probs = torch.softmax(self.head(feature.unsqueeze(0)), dim=1)[0].cpu()
        return probs, found


def run(predictor, cap, fps, clip_starts):
    """Predict every clip start; yield (moment, class probabilities or None, hoop found)."""
    for n, start in enumerate(clip_starts, 1):
        frames = read_clip(cap, start, fps)
        moment = start + PRE
        if any(frame is None for frame in frames):
            print(f"  {format_time(moment)}  past the end of the video -- skipped")
            continue
        probs, found = predictor.predict(frames)
        if len(clip_starts) > 1:
            print(f"  {n}/{len(clip_starts)} windows", end="\r", flush=True)
        yield moment, probs, found


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--at", action="append", type=parse_time, default=[],
                        metavar="TIME", help="a moment to classify (repeatable)")
    parser.add_argument("--from", dest="start", type=parse_time, metavar="TIME",
                        help="scan a stretch of the match, from here...")
    parser.add_argument("--to", dest="end", type=parse_time, metavar="TIME",
                        help="...to here")
    parser.add_argument("--every", type=float, default=CLIP_SECONDS,
                        help=f"seconds between scan windows (default {CLIP_SECONDS:g}, "
                             f"back to back)")
    parser.add_argument("--json", metavar="PATH", help="also write the timeline as JSON")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    scanning = args.start is not None or args.end is not None
    if scanning and (args.start is None or args.end is None):
        parser.error("--from and --to go together")
    if not scanning and not args.at:
        parser.error("give --at TIME, or --from TIME --to TIME")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    if scanning:
        starts = window_starts(args.start, args.end, args.every)
        if not starts:
            raise SystemExit(f"--from/--to must span at least {CLIP_SECONDS:g}s")
    else:
        starts = [max(0.0, moment - PRE) for moment in args.at]

    print(f"{len(starts)} window(s), about {len(starts) * SECONDS_PER_WINDOW / 60:.1f} "
          f"min at ~{SECONDS_PER_WINDOW}s each (a whole match would be hours at this rate)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading OWLv2, {BACKBONE} and {args.checkpoint} on {device}")
    predictor = Predictor(device, args.checkpoint)

    results = list(run(predictor, cap, fps, starts))
    cap.release()
    groups = list(GOAL_GROUPS)

    if scanning:
        windows = []
        for moment, probs, _ in results:
            if probs is None:
                continue
            label, confidence = goal_answer(probs, predictor.classes)
            if label != "none":
                windows.append((moment, label, confidence))
        events = merge_events(windows, args.every)
        not_live = sum(1 for _, probs, _ in results if probs is None)
        print(f"\n{not_live}/{len(results)} windows were not live play "
              f"(close-ups, replays, graphics) and were skipped")
        print(f"{len(events)} event(s) between {format_time(args.start)} and "
              f"{format_time(args.end)}:")
    else:
        events = []
        print(f"\n{'moment':<10}{'answer':<12}{'conf':>6}   " +
              "  ".join(f"{g:>10}" for g in groups) + "   hoop")
        for moment, probs, found in results:
            if probs is None:
                print(f"{format_time(moment):<10}not live play (close-up, replay "
                      f"or graphic) -- not asked")
                continue
            goals = goal_probabilities(probs, predictor.classes)
            label, confidence = goal_answer(probs, predictor.classes)
            events.append({"t": moment, "event": label, "confidence": confidence})
            print(f"{format_time(moment):<10}{label:<12}{confidence:>6.0%}   " +
                  "  ".join(f"{goals[g]:>10.0%}" for g in groups) +
                  f"   {'yes' if found else 'NO'}")

    if scanning:
        for event in events:
            print(f"  {format_time(event['t'])}  {event['event']:<12}{event['confidence']:.0%}")

    if args.json:
        timeline = [{"t": format_time(e["t"]), "event": e["event"],
                     "confidence": round(e["confidence"], 2)} for e in events]
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(timeline, fh, indent=2)
        print(f"\nwrote {len(timeline)} event(s) to {args.json}")


if __name__ == "__main__":
    main()
