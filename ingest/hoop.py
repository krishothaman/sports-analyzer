"""Find the hoop, and cut a close-up of it -- the idea taken from shot-tracking pipelines.

Phase 5's first result explained itself once the frames were looked at at full
resolution. The model finds free throws -- a scene that holds still for ten
seconds -- and misses field goals, because a made jump shot is a ball a few
pixels across dropping through a rim in the corner of an 854x480 wide shot. By
the time a frame reaches the backbone (455 wide on disk, squashed to 224) the
rim is about 15 pixels wide. The shot was in the clip. The model could not see it.

Every pipeline that tracks shots well starts the same way: find the hoop, then
look closely there. We take the idea, not their detectors. The YOLO ball-and-hoop
models in those repos were trained on phone footage of outdoor courts; the
broadcast ones detect players, the ball and court lines but not the hoop; and
ultralytics is AGPL. Instead:

  OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0) is an *open-vocabulary*
  detector. You give it the text "a basketball hoop" and it draws a box -- no
  training, no labelled hoops. A spike on 12 mid-shot frames across all three
  broadcasts put the box on the rim in all 12.

For every clip in the manifest this re-reads the 16 frames at FULL resolution,
detects the hoop on four of them, interpolates its position across the rest (the
camera pans), and writes a 256x256 native-resolution crop to data/clips_hoop/.
That is roughly four times the pixels on the rim that the squashed full frame
gets. Detections go to data/hoops.csv.

    python -m ingest.hoop
    python -m ingest.hoop --match-id match03
"""

import argparse
import csv
import glob
import os

import cv2
import torch
from PIL import Image

from ingest.cut import CLIP_FRAMES, frame_indices, load_clip_manifest

HOOP_ROOT = os.path.join("data", "clips_hoop")
HOOPS_CSV = os.path.join("data", "hoops.csv")

DETECTOR = "google/owlv2-base-patch16-ensemble"
# The spike also tried "a basketball backboard". It found a player once and a
# shot clock once. The rim query was right every time, so it is the only one.
QUERY = "a basketball hoop"

# Detect on four of the sixteen frames and interpolate between them. Detection is
# the slow step (~0.3s a frame) and the camera does not jump in two seconds.
DETECT_POSITIONS = (0, 5, 10, 15)

# OWLv2's scores run low in absolute terms: the spike's correct boxes scored
# 0.12 to 0.47. Below this, a frame is treated as having no hoop in it.
MIN_SCORE = 0.08

# Crop side, in pixels of a 480-line source. A rim is ~30px wide there, so this
# frames it with room for the ball's arc above and the finish below. Scaled with
# the source height so a 720p match gets the same field of view.
CROP_SIDE = 256
REFERENCE_HEIGHT = 480
# Where the rim sits inside the crop, from the top. A third of the way down keeps
# the ball arriving from above and the players underneath.
HOOP_ROW = 1 / 3

HOOP_FIELDS = (["clip_id", "match_id", "label", "found"] +
               [f"{key}{pos}" for pos in DETECT_POSITIONS for key in "xys"])


def track(detections, positions=DETECT_POSITIONS, length=CLIP_FRAMES):
    """The hoop's centre in every frame, interpolated from the frames that found it.

    `detections` holds one (cx, cy) or None per entry of `positions`. Frames
    between two detections get a straight-line blend; frames before the first or
    after the last hold the nearest one. None if the hoop was never found.
    """
    known = [(pos, det) for pos, det in zip(positions, detections) if det is not None]
    if not known:
        return None

    centres = []
    for i in range(length):
        before = [(pos, det) for pos, det in known if pos <= i]
        after = [(pos, det) for pos, det in known if pos >= i]
        if not before:
            centres.append(after[0][1])
        elif not after:
            centres.append(before[-1][1])
        else:
            (p0, d0), (p1, d1) = before[-1], after[0]
            w = 0.0 if p0 == p1 else (i - p0) / (p1 - p0)
            centres.append((d0[0] + w * (d1[0] - d0[0]), d0[1] + w * (d1[1] - d0[1])))
    return centres


def crop_box(centre, frame_width, frame_height, side=CROP_SIDE, hoop_row=HOOP_ROW):
    """(x0, y0, x1, y1) of a square with the hoop at `hoop_row`, kept inside the frame.

    Near an edge the square slides rather than shrinks -- the hoop is then off
    centre, but the crop is always the same size and always real pixels. A
    padded crop would teach the model that black bars mean something.
    """
    side = min(side, frame_width, frame_height)
    cx, cy = centre
    x0 = round(cx - side / 2)
    y0 = round(cy - side * hoop_row)
    x0 = min(max(x0, 0), frame_width - side)
    y0 = min(max(y0, 0), frame_height - side)
    return x0, y0, x0 + side, y0 + side


class HoopDetector:
    """OWLv2 asked one question: where is the basketball hoop?"""

    def __init__(self, device):
        # Imported here, not at the top, so that the pure geometry above -- and
        # its tests -- never pay for loading transformers.
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = device
        self.processor = Owlv2Processor.from_pretrained(DETECTOR)
        self.model = Owlv2ForObjectDetection.from_pretrained(DETECTOR).to(device).eval()

    def find(self, images):
        """Best hoop in each image, as (cx, cy, score) -- or None if nothing scored."""
        inputs = self.processor(text=[[QUERY]] * len(images), images=images,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # OWLv2 pads each image to a square (bottom and right) before detecting,
        # so its boxes are in the padded square's coordinates. With the padding
        # on the far edges those are the original pixel coordinates.
        sides = torch.tensor([[max(image.size)] * 2 for image in images],
                             device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=MIN_SCORE, target_sizes=sides)

        found = []
        for result in results:
            if len(result["scores"]) == 0:
                found.append(None)
                continue
            best = result["scores"].argmax()
            x0, y0, x1, y1 = result["boxes"][best].tolist()
            found.append(((x0 + x1) / 2, (y0 + y1) / 2, result["scores"][best].item()))
        return found


def video_path(match_id):
    found = glob.glob(os.path.join("data", "video", f"{match_id}.*"))
    if not found:
        raise SystemExit(f"no video for {match_id} under data/video/")
    return found[0]


def read_clip(cap, start_sec, fps):
    """The 16 frames of one clip at full resolution, as RGB images.

    Seeks once, then walks forward. ingest/cut.py makes one pass through the
    whole match instead, because it is writing every clip in the video; here we
    need ~60 consecutive frames around each of 807 clips, and a seek per clip is
    far cheaper than decoding three matches end to end.
    """
    indices = frame_indices(start_sec, fps)
    wanted = set(indices)
    frames = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    current = indices[0]
    while current <= indices[-1]:
        # grab() advances without the colour conversion; only the 16 frames we
        # keep pay for retrieve().
        if not cap.grab():
            break
        if current in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames[current] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current += 1

    return [frames.get(i) for i in indices]


def write_crops(row, frames, centres):
    out_dir = os.path.join(HOOP_ROOT, row["match_id"], row["clip_id"])
    os.makedirs(out_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        side = round(CROP_SIDE * frame.size[1] / REFERENCE_HEIGHT)
        if centres is None:
            # No hoop anywhere in the clip -- usually a background clip mid-pan.
            # Hand over the whole frame, so "no close-up available" is itself
            # something the model gets to see, rather than a crop of nothing.
            view = frame.resize((CROP_SIDE, CROP_SIDE))
        else:
            view = frame.crop(crop_box(centres[i], frame.size[0], frame.size[1], side))
            if view.size != (CROP_SIDE, CROP_SIDE):
                view = view.resize((CROP_SIDE, CROP_SIDE))
        view.save(os.path.join(out_dir, f"{i:02d}.jpg"), quality=90)


def record(row, detections, found):
    rec = {"clip_id": row["clip_id"], "match_id": row["match_id"],
           "label": row["label"], "found": int(found)}
    for pos, det in zip(DETECT_POSITIONS, detections):
        rec[f"x{pos}"], rec[f"y{pos}"], rec[f"s{pos}"] = (
            (round(det[0], 1), round(det[1], 1), round(det[2], 3)) if det else ("", "", ""))
    return rec


def cut_match(match_id, rows, detector):
    cap = cv2.VideoCapture(video_path(match_id))
    if not cap.isOpened():
        raise SystemExit(f"could not open video for {match_id}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    records = []
    ordered = sorted(rows, key=lambda r: float(r["start_sec"]))
    for n, row in enumerate(ordered, 1):
        frames = read_clip(cap, float(row["start_sec"]), fps)
        if any(frame is None for frame in frames):
            print(f"\n  skipped {row['clip_id']}: could not read all {CLIP_FRAMES} frames")
            continue

        detections = detector.find([frames[pos] for pos in DETECT_POSITIONS])
        centres = track([det[:2] if det else None for det in detections])
        write_crops(row, frames, centres)
        records.append(record(row, detections, centres is not None))
        print(f"  {match_id}: {n}/{len(ordered)} clips", end="\r", flush=True)

    cap.release()
    print()
    return records


def save_records(records, replaced_matches, path=HOOPS_CSV):
    """Write detections, keeping rows for matches this run did not touch."""
    kept = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            kept = [r for r in csv.DictReader(fh) if r["match_id"] not in replaced_matches]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HOOP_FIELDS)
        writer.writeheader()
        writer.writerows(kept + records)


def summarise(records):
    """How often a hoop was found, per label -- the detector's own report card."""
    by_label = {}
    for rec in records:
        by_label.setdefault(rec["label"], []).append(rec["found"])

    print("\nhoop found in:")
    for label, found in sorted(by_label.items()):
        print(f"  {label:<16}{sum(found):>5}/{len(found):<5}{sum(found) / len(found):>6.0%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", nargs="*", default=None,
                        help="only these matches (default: every match in the manifest)")
    args = parser.parse_args()

    rows = load_clip_manifest()
    if not rows:
        raise SystemExit("no clips in the manifest -- run: python -m ingest.cut")
    matches = args.match_id or sorted({row["match_id"] for row in rows})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {DETECTOR} on {device}")
    detector = HoopDetector(device)

    records = []
    for match_id in matches:
        match_rows = [row for row in rows if row["match_id"] == match_id]
        print(f"{match_id}: {len(match_rows)} clips")
        records += cut_match(match_id, match_rows, detector)

    save_records(records, set(matches))
    summarise(records)
    print(f"\ncrops in {HOOP_ROOT}/, detections in {HOOPS_CSV}")
    print("next: python -m clips.data --backbone mvit_v2_s --crop hoop")


if __name__ == "__main__":
    main()
