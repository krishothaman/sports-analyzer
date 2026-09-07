# Phase 2: Frame Classifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** A classifier that looks at one video frame and says whether it shows live game action, the
crowd/bench, or a graphic/advert — trained on the owner's own labelled frames from a real basketball
match.

**Architecture:** A frozen ImageNet-pretrained ResNet-18 acts as the "eyes", converting each frame to
512 numbers. Those numbers are computed once and cached to disk. A single `nn.Linear(512, 3)` layer —
the "rulebook" — is the only thing that trains. This is transfer learning with a frozen backbone, and
it is the same shape the clip classifier takes in Phase 4.

**Tech Stack:** Python (venv), PyTorch 2.11+cu128, torchvision 0.26, OpenCV (`opencv-python`), yt-dlp.
No scikit-learn, no matplotlib — the confusion matrix is written by hand because writing it is the point.

**Spec:** `docs/superpowers/specs/2026-09-06-sports-highlight-analyzer-design.md` (§5.2 eyes/rulebook
split, §5.5 feature caching, §9 roadmap row 2)

## Global Constraints

- **The owner runs every training, labelling and inference command.** Claude writes and explains code;
  the owner executes it. Never hand over a command the owner cannot roughly describe.
- **Explain, do not quiz.** Write the code and explain what each line does and why. No fill-in-the-blank
  exercises — the owner does not yet write PyTorch unaided.
- **No `Co-Authored-By` trailer on any commit in this repository.**
- **The repository is a public portfolio piece.** No video files, no extracted frames, no cached
  features, no absolute paths containing the username. All large artefacts live under `data/` and are
  gitignored.
- **The backbone is frozen.** `requires_grad = False` on every backbone parameter, `model.eval()`
  always. If the backbone trains during Phase 2, that is a bug.
- **Splits are chronological, never random.** See Task 6 — random splitting leaks and is the easiest
  way to produce a fake accuracy number.

## Scope adjustment from the spec

Spec §9 row 2 describes the Phase 2 filter as "game-action vs crowd/replay/ad". **Replay is dropped
from Phase 2 and deferred to Phase 4.** A replay frame is visually identical to a live game frame —
same court, same players, same camera position. The signals that mark a replay (the wipe transition,
slow motion, the on-screen bug) are all *temporal*, and a single-frame classifier cannot see them.
Keeping the class would train a detector that cannot work. Phase 4's temporal head sees 16 frames at
once and is the right place for it.

Phase 2 therefore ships three classes:

| Label | Key | Means |
|---|---|---|
| `game` | `g` | Live play, wide or mid court camera, ball in play |
| `crowd` | `c` | Audience, bench, coach, player close-up off the run of play, huddle |
| `graphic` | `x` | Adverts, scoreboard cards, station bumpers, title screens, black frames |

---

## File structure

```
models/
  backbone.py          frozen ResNet-18 feature extractor, shared with Phase 4+
frames/
  extract.py           video file -> sampled JPEG frames + index CSV
  label.py             keyboard-driven labelling tool -> manifest CSV
  data.py              manifest -> chronological split -> cached features -> DataLoaders
  model.py             the trainable head: Linear(512 -> 3)
  train.py             training loop + confusion matrix
  predict.py           run the finished filter on frames
tests/
  test_backbone.py     shape and frozen-ness contracts
  test_frames_data.py  split correctness (the leakage tripwire)
data/                  (gitignored) video/, frames/, features/, manifest.csv
```

`frames/` deliberately mirrors `mnist/` — `data.py`, `model.py`, `train.py`, `predict.py` mean the same
things they did in Phase 1. `models/backbone.py` sits outside `frames/` because Phase 4 reuses it
unchanged.

---

## Task 0: Dependencies and skeleton

**Files:**
- Modify: `requirements.txt`, `.gitignore`
- Create: `models/__init__.py`, `frames/__init__.py`

- [ ] **Step 1: Add the two new dependencies**

Append to `requirements.txt`:

```
opencv-python>=4.10
yt-dlp>=2024.8.6
```

`opencv-python` decodes video and draws the labelling window. `yt-dlp` downloads the match. Neither
touches training.

- [ ] **Step 2: Keep the big artefacts out of the public repo**

Append to `.gitignore`:

```
# Phase 2+ media and derived artefacts (large, and not ours to redistribute)
data/video/
data/frames/
data/features/
data/*.csv
```

The manifest CSV is excluded too: it is derived data referencing local paths. Phase 3 revisits whether
a manifest ships with the repo.

- [ ] **Step 3: Create the packages**

```bash
mkdir -p models frames data/video data/frames data/features && touch models/__init__.py frames/__init__.py
```

- [ ] **Step 4: OWNER RUNS — install**

```bash
pip install -r requirements.txt
```

Expected: opencv-python and yt-dlp install; torch is already satisfied and is not reinstalled.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore models/__init__.py frames/__init__.py
git commit -m "Phase 2 Task 0: dependencies and package skeleton"
```

---

## Task 1: Get a match video

**Files:**
- Create: `docs/phase-2-video-notes.md`

**Interfaces:**
- Produces: a video file at `data/video/<match_id>.mp4` that Task 2 reads.

- [ ] **Step 1: OWNER CHOOSES a video**

Requirements for a usable first video:
- A **full or half game**, not a highlight reel. Highlight reels are ~100% `game` class, which leaves
  nothing to classify. Broadcast footage with adverts, crowd cuts and scoreboard cards is what is wanted.
- **20 minutes minimum.** Shorter gives too few distinct scenes.
- **Broadcast camera**, not amateur handheld from the stands.
- 720p is plenty. 1080p costs disk for no accuracy gain — frames are downscaled to 224x224 anyway.

- [ ] **Step 2: OWNER RUNS — download**

```bash
yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" -o "data/video/match01.%(ext)s" "PASTE_URL_HERE"
```

The video stays local. It is gitignored and never pushed — this repository is public and the footage
is not ours to redistribute.

- [ ] **Step 3: Record what was used**

`docs/phase-2-video-notes.md`:

```markdown
# Phase 2 video source

| match_id | source | length | resolution | notes |
|---|---|---|---|---|
| match01 | (URL or description) | (mm:ss) | 720p | first labelled match |

Footage kept local under `data/video/`, gitignored. Recorded here so results stay
reproducible without redistributing the video.
```

- [ ] **Step 4: Commit**

```bash
git add docs/phase-2-video-notes.md
git commit -m "Phase 2 Task 1: record match video source"
```

---

## Task 2: Extract frames

**Files:**
- Create: `frames/extract.py`

**Interfaces:**
- Consumes: `data/video/<match_id>.mp4`
- Produces: `data/frames/<match_id>/<match_id>_00042.jpg` plus `data/frames/<match_id>/index.csv`
  with columns `filename,timestamp_sec`. Task 3 reads that index.

- [ ] **Step 1: Write the extractor**

`frames/extract.py`:

```python
"""Sample still frames from a match video.

One frame every few seconds is plenty for a frame classifier: consecutive frames
of a 30fps video are near-identical, so sampling densely would mean labelling the
same picture over and over.
"""

import argparse
import csv
import os

import cv2

DEFAULT_EVERY_SECONDS = 3.0


def extract(video_path, out_dir, match_id, every_seconds=DEFAULT_EVERY_SECONDS):
    """Write one JPEG every `every_seconds` of video. Returns the number saved."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * every_seconds)))

    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.csv")

    saved = 0
    frame_index = 0

    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "timestamp_sec"])

        while True:
            # grab() advances to the next frame without fully decoding it, which is
            # far cheaper than read(). We only pay for a full decode via retrieve()
            # on the frames we actually keep.
            if not cap.grab():
                break

            if frame_index % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    name = f"{match_id}_{saved:05d}.jpg"
                    cv2.imwrite(os.path.join(out_dir, name), frame)
                    writer.writerow([name, f"{frame_index / fps:.2f}"])
                    saved += 1

            frame_index += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Sample frames from a match video.")
    parser.add_argument("video", help="path to the video file")
    parser.add_argument("--match-id", default="match01")
    parser.add_argument("--every", type=float, default=DEFAULT_EVERY_SECONDS,
                        help="seconds between sampled frames")
    args = parser.parse_args()

    out_dir = os.path.join("data", "frames", args.match_id)
    saved = extract(args.video, out_dir, args.match_id, args.every)
    print(f"saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: OWNER RUNS — extract**

```bash
python -m frames.extract data/video/match01.mp4 --match-id match01 --every 3
```

Expected: a 30-minute video at one frame per 3 seconds gives ~600 frames. Aim for 400-800. If the
count is far outside that, adjust `--every` and rerun.

Open a few JPEGs and confirm they look like basketball. A codec problem shows up here as green or
scrambled images, and it is much cheaper to catch now than after labelling.

- [ ] **Step 3: Commit**

```bash
git add frames/extract.py
git commit -m "Phase 2 Task 2: sample frames from match video"
```

---

## Task 3: The labelling tool

**Files:**
- Create: `frames/label.py`

**Interfaces:**
- Consumes: `data/frames/<match_id>/index.csv`
- Produces: `data/manifest.csv` with columns `match_id,filename,timestamp_sec,label`. Task 6 reads it.

- [ ] **Step 1: Write the tool**

`frames/label.py`:

```python
"""Keyboard-driven frame labelling.

Opens each unlabelled frame in a window. One keypress assigns a class and advances.
Every label is written to disk immediately, so quitting halfway loses nothing and
rerunning resumes where you stopped.
"""

import argparse
import csv
import os

import cv2

LABELS = {
    ord("g"): "game",
    ord("c"): "crowd",
    ord("x"): "graphic",
}
QUIT_KEYS = {ord("q"), 27}          # q or Escape
UNDO_KEY = ord("u")

MANIFEST = os.path.join("data", "manifest.csv")
MANIFEST_FIELDS = ["match_id", "filename", "timestamp_sec", "label"]


def load_manifest(path=MANIFEST):
    """Return {(match_id, filename): row} for everything already labelled."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["match_id"], r["filename"]): r for r in csv.DictReader(fh)}


def save_manifest(rows, path=MANIFEST):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)


def read_index(match_dir):
    with open(os.path.join(match_dir, "index.csv"), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def annotate(image, caption):
    """Burn the instructions into the top of the displayed image."""
    banner = image.copy()
    cv2.rectangle(banner, (0, 0), (banner.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(banner, caption, (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return banner


def main():
    parser = argparse.ArgumentParser(description="Label sampled frames by keypress.")
    parser.add_argument("--match-id", default="match01")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many new labels (0 = no limit)")
    args = parser.parse_args()

    match_dir = os.path.join("data", "frames", args.match_id)
    rows = load_manifest()
    entries = read_index(match_dir)

    todo = [e for e in entries if (args.match_id, e["filename"]) not in rows]
    print(f"{len(rows)} already labelled, {len(todo)} to go")
    print("keys:  g = game    c = crowd    x = graphic    u = undo    q = save and quit")

    history = []
    done = 0

    for entry in todo:
        if args.limit and done >= args.limit:
            break

        image = cv2.imread(os.path.join(match_dir, entry["filename"]))
        if image is None:
            continue

        # Scale tall frames down so the window fits on screen.
        if image.shape[0] > 720:
            scale = 720 / image.shape[0]
            image = cv2.resize(image, None, fx=scale, fy=scale)

        seconds = float(entry["timestamp_sec"])
        caption = f"{done + 1}/{len(todo)}   t={seconds:.0f}s   g/c/x   u=undo   q=quit"
        cv2.imshow("label", annotate(image, caption))

        # waitKey(0) blocks until a key is pressed. The & 0xFF masks off high bits
        # some platforms set, leaving a plain ASCII code.
        key = cv2.waitKey(0) & 0xFF

        if key in QUIT_KEYS:
            break

        if key == UNDO_KEY:
            if history:
                rows.pop(history.pop(), None)
                save_manifest(rows)
                done -= 1
            continue

        if key not in LABELS:
            continue

        record_key = (args.match_id, entry["filename"])
        rows[record_key] = {
            "match_id": args.match_id,
            "filename": entry["filename"],
            "timestamp_sec": entry["timestamp_sec"],
            "label": LABELS[key],
        }
        history.append(record_key)
        save_manifest(rows)
        done += 1

    cv2.destroyAllWindows()

    counts = {}
    for row in rows.values():
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"\n{len(rows)} labelled total: {counts}")


if __name__ == "__main__":
    main()
```

The undo path rewrites the whole manifest rather than tracking a diff. At a few hundred rows that is
instant, and simple beats clever for a tool used once.

- [ ] **Step 2: Commit before labelling**

```bash
git add frames/label.py
git commit -m "Phase 2 Task 3: keyboard-driven frame labelling tool"
```

---

## Task 4: Label the frames

**Files:**
- Create: `docs/label-guide.md`

- [ ] **Step 1: Write the label guide**

`docs/label-guide.md`:

```markdown
# Frame label guide — Phase 2

Three classes. When a frame is genuinely ambiguous, pick the class a viewer would say the frame
is *about*, and stay consistent — consistency matters more than any individual call.

## game
Live play. Wide or mid broadcast camera, court visible, ball in play or about to be.
Includes inbounds, free throws, players moving up court, and dead-ball moments still shot
from the game camera.

## crowd
Anything shot away from live play. Audience, bench, coaches, huddles, referees in conference,
a player's face in close-up, celebration cutaways, the tunnel.

## graphic
Frames dominated by rendered content rather than the arena. Adverts, scoreboard and stat cards,
station bumpers, sponsor stings, title screens, black or transition frames.

## Rules for the hard cases
- **Score bug over live play** -> `game`. Nearly every broadcast frame has an overlay; the overlay
  alone never makes a frame `graphic`.
- **Close-up of a player during play** -> `crowd`. The court is not readable, so the frame carries
  no game information.
- **Replay of live action** -> `game`. Replays are deferred to Phase 4; for a single frame, `game`
  is the honest call because that is what the pixels show.
- **Advert board filling the shot** -> `game` if players are visible, else `graphic`.
```

- [ ] **Step 2: OWNER RUNS — label**

```bash
python -m frames.label --match-id match01
```

Target: **at least 300 labelled frames**, and **at least 40 in the smallest class**. A class with five
examples cannot be learned and makes the confusion matrix unreadable.

Broadcast basketball runs roughly 60-70% `game`, so `graphic` is the class likely to fall short.

Labelling 300 frames at about a second each takes ten minutes. Do it in one sitting if possible —
consistency drifts across sessions, and inconsistent labels cap accuracy no matter how good the model is.

- [ ] **Step 3: OWNER RUNS — check the class balance**

```bash
python -m frames.label --match-id match01 --limit 0
```

Rerunning prints the totals and exits quickly once everything is labelled. Expected output resembles
`312 labelled total: {'game': 197, 'crowd': 74, 'graphic': 41}`. If the smallest class is under 40,
label more frames.

- [ ] **Step 4: Commit**

```bash
git add docs/label-guide.md
git commit -m "Phase 2 Task 4: frame label guide"
```

The manifest itself stays gitignored.

---

## Task 5: The frozen backbone

**Files:**
- Create: `models/backbone.py`
- Test: `tests/test_backbone.py`

**Interfaces:**
- Produces: `build_backbone(device) -> (nn.Module, preprocess_transform)` and `FEATURE_DIM = 512`.
  Tasks 6 and 8 consume both.

- [ ] **Step 1: Write the failing test**

`tests/test_backbone.py`:

```python
import torch

from models.backbone import FEATURE_DIM, build_backbone


def test_backbone_turns_an_image_into_512_numbers():
    model, _ = build_backbone(torch.device("cpu"))
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, FEATURE_DIM)


def test_backbone_is_frozen():
    # The whole Phase 2 premise is that only the head learns. If any backbone
    # parameter can accumulate gradients, training would silently drift the eyes.
    model, _ = build_backbone(torch.device("cpu"))
    assert all(not p.requires_grad for p in model.parameters())


def test_backbone_is_in_eval_mode():
    # ResNet-18 has batchnorm. In train mode it would update running statistics
    # from our frames -- a silent change to a model we declared frozen.
    model, _ = build_backbone(torch.device("cpu"))
    assert not model.training
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
pytest tests/test_backbone.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'models.backbone'`.

- [ ] **Step 3: Write the backbone**

`models/backbone.py`:

```python
"""The 'eyes': a frozen, pretrained ResNet-18 that turns an image into 512 numbers.

Shared across sports and across phases. Nothing in this file knows what a
basketball is -- it was trained on general photographs and its job is only to
describe what is in a picture. The sport-specific knowledge lives in the head.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

# ResNet-18's penultimate layer is 512 wide. Swapping to a different backbone
# means changing this constant along with build_backbone.
FEATURE_DIM = 512


def build_backbone(device):
    """Return (frozen feature extractor, matching preprocessing transform)."""
    # DEFAULT is torchvision's best available ImageNet checkpoint for this
    # architecture: weights learned from 1.2 million labelled photographs.
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # resnet18 normally ends in Linear(512 -> 1000) to name an ImageNet class.
    # We do not want a class name, we want the 512 numbers feeding that layer --
    # the description the network built before committing to an answer. Identity
    # is a layer that returns its input unchanged, so replacing fc with it hands
    # us those 512 numbers directly.
    model.fc = nn.Identity()

    # Freeze. Without this, backprop from the head would flow into the backbone
    # and the optimizer would edit weights that took days of GPU time to learn.
    for param in model.parameters():
        param.requires_grad = False

    # eval() matters here in a way it did not for MnistCNN: ResNet-18 has
    # batchnorm layers, which behave differently in train mode.
    model.eval()
    model.to(device)

    # The weights carry their own preprocessing: resize to 256, centre crop 224,
    # scale to 0-1, normalise by ImageNet's mean and std. Using anything else
    # feeds the network inputs shaped unlike its training data, and accuracy
    # drops for no visible reason.
    return model, weights.transforms()
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/test_backbone.py -v
```

Expected: 3 passed. The first run downloads ~45 MB of ResNet-18 weights and needs internet; later runs
use the cached copy in `~/.cache/torch`.

- [ ] **Step 5: Commit**

```bash
git add models/backbone.py tests/test_backbone.py
git commit -m "Phase 2 Task 5: frozen ResNet-18 backbone with freeze contracts"
```

---

## Task 6: Splits and the feature cache

**Files:**
- Create: `frames/data.py`
- Test: `tests/test_frames_data.py`

**Interfaces:**
- Consumes: `data/manifest.csv`, `models.backbone.build_backbone`
- Produces: `CLASSES`, `chronological_split(rows, train_frac)`, `build_cache(...)`,
  `get_loaders(batch_size)`. Tasks 7 and 8 consume `CLASSES` and `get_loaders`.

- [ ] **Step 1: Write the failing test**

`tests/test_frames_data.py`:

```python
from frames.data import chronological_split


def rows(match_id, timestamps):
    return [{"match_id": match_id, "filename": f"{match_id}_{i}.jpg",
             "timestamp_sec": str(t), "label": "game"}
            for i, t in enumerate(timestamps)]


def test_split_puts_earlier_frames_in_train_and_later_in_test():
    train, test = chronological_split(rows("m1", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
                                      train_frac=0.7)
    assert len(train) == 7
    assert len(test) == 3
    assert max(float(r["timestamp_sec"]) for r in train) < \
           min(float(r["timestamp_sec"]) for r in test)


def test_split_is_chronological_even_when_the_manifest_is_out_of_order():
    # The manifest is written in labelling order, which is not necessarily
    # timestamp order once frames have been relabelled or added later.
    train, test = chronological_split(rows("m1", [50, 10, 90, 30, 70, 0, 20, 80, 40, 60]),
                                      train_frac=0.7)
    assert max(float(r["timestamp_sec"]) for r in train) < \
           min(float(r["timestamp_sec"]) for r in test)


def test_each_match_is_split_independently():
    # With several matches, every match contributes to both sides -- otherwise
    # the test set could end up being one whole match and measure something else.
    both = rows("m1", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) + \
           rows("m2", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    train, test = chronological_split(both, train_frac=0.7)
    assert {r["match_id"] for r in train} == {"m1", "m2"}
    assert {r["match_id"] for r in test} == {"m1", "m2"}
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
pytest tests/test_frames_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'frames.data'`.

- [ ] **Step 3: Write the data module**

`frames/data.py`:

```python
"""Manifest -> chronological split -> cached features -> DataLoaders.

The backbone is frozen, so a given frame's 512 numbers can never change. We
compute them once, write them to disk, and every training run afterwards reads
arrays instead of decoding JPEGs. Training drops from minutes to under a second.
"""

import csv
import os

import torch
from PIL import Image

from models.backbone import FEATURE_DIM, build_backbone

CLASSES = ["game", "crowd", "graphic"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

MANIFEST = os.path.join("data", "manifest.csv")
CACHE = os.path.join("data", "features", "frames.pt")
FRAMES_ROOT = os.path.join("data", "frames")


def load_manifest(path=MANIFEST):
    with open(path, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["label"] in CLASS_TO_INDEX]


def chronological_split(rows, train_frac=0.7):
    """Split each match by time: earliest frames train, latest frames test.

    A random split would leak. Frames three seconds apart show the same
    possession, the same players, the same camera angle -- near-duplicates. Put
    one in train and its neighbour in test and the model can score well by
    recognising a moment it has already seen, which measures memory, not
    understanding. Cutting on time puts the whole shared moment on one side.

    This is the single-video stand-in for the match-level splitting that Phase 3
    uses once there are several matches.
    """
    by_match = {}
    for row in rows:
        by_match.setdefault(row["match_id"], []).append(row)

    train, test = [], []
    for match_id in sorted(by_match):
        ordered = sorted(by_match[match_id], key=lambda r: float(r["timestamp_sec"]))
        cut = int(len(ordered) * train_frac)
        train.extend(ordered[:cut])
        test.extend(ordered[cut:])
    return train, test


def build_cache(rows, device, batch_size=32, cache_path=CACHE):
    """Run every frame through the backbone once and save the results."""
    model, preprocess = build_backbone(device)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    keys, chunks = [], []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = []
        for row in batch:
            path = os.path.join(FRAMES_ROOT, row["match_id"], row["filename"])
            # convert("RGB") guards against greyscale or palettised JPEGs, which
            # would arrive with the wrong number of channels.
            images.append(preprocess(Image.open(path).convert("RGB")))
            keys.append((row["match_id"], row["filename"]))

        stacked = torch.stack(images).to(device)
        with torch.no_grad():
            chunks.append(model(stacked).cpu())

        print(f"  features: {min(start + batch_size, len(rows))}/{len(rows)}", end="\r")

    features = torch.cat(chunks)
    torch.save({"keys": keys, "features": features}, cache_path)
    print(f"\ncached {features.shape[0]} x {features.shape[1]} features to {cache_path}")
    return features


def load_cache(cache_path=CACHE):
    blob = torch.load(cache_path, map_location="cpu")
    return {key: blob["features"][i] for i, key in enumerate(blob["keys"])}


def _dataset(rows, lookup):
    features = torch.stack([lookup[(r["match_id"], r["filename"])] for r in rows])
    labels = torch.tensor([CLASS_TO_INDEX[r["label"]] for r in rows])
    return torch.utils.data.TensorDataset(features, labels)


def get_loaders(batch_size=32, train_frac=0.7):
    """Return (train_loader, test_loader) over cached features."""
    rows = load_manifest()
    lookup = load_cache()
    train_rows, test_rows = chronological_split(rows, train_frac)

    train_loader = torch.utils.data.DataLoader(
        _dataset(train_rows, lookup), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        _dataset(test_rows, lookup), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def main():
    """Build the feature cache. Run once after labelling, and again after adding labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest()
    print(f"{len(rows)} labelled frames; extracting {FEATURE_DIM}-d features on {device}")
    build_cache(rows, device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

```bash
pytest tests/test_frames_data.py -v
```

Expected: 3 passed.

- [ ] **Step 5: OWNER RUNS — build the cache**

```bash
python -m frames.data
```

Expected: a progress counter, then `cached 312 x 512 features to data/features/frames.pt`. Takes a few
seconds on GPU. The file is roughly `frames x 512 x 4 bytes` — under a megabyte for 300 frames.

**Rerun this after labelling more frames.** The cache does not update itself, and a stale cache means
`get_loaders` raises `KeyError` on the newly labelled files.

- [ ] **Step 6: Commit**

```bash
git add frames/data.py tests/test_frames_data.py
git commit -m "Phase 2 Task 6: chronological splits and frozen-feature cache"
```

---

## Task 7: The head, training, and the confusion matrix

**Files:**
- Create: `frames/model.py`, `frames/train.py`

**Interfaces:**
- Consumes: `frames.data.get_loaders`, `frames.data.CLASSES`, `models.backbone.FEATURE_DIM`
- Produces: `frame_head.pt` at the repository root. Task 8 loads it.

- [ ] **Step 1: Write the head**

`frames/model.py`:

```python
"""The 'rulebook': the only part of Phase 2 that learns.

One linear layer, 512 inputs to 3 outputs. That is 1,539 parameters against the
backbone's 11 million. All the visual understanding was already paid for; this
layer only has to decide which combinations of those 512 numbers mean 'crowd'.

A single linear layer on frozen features has a name -- a linear probe. If it
works, it is evidence the backbone's features already separate the classes
cleanly, which is exactly what we want to know before building anything larger.
"""

import torch.nn as nn

from models.backbone import FEATURE_DIM


class FrameHead(nn.Module):
    def __init__(self, num_classes=3, feature_dim=FEATURE_DIM):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x arrives as [B, 512] cached features -- no convolution here, the
        # backbone already did that. Out: [B, 3] raw logits, same as MnistCNN.
        return self.fc(x)
```

- [ ] **Step 2: Write the training script**

`frames/train.py`:

```python
"""Train the frame classifier head on cached features.

The five steps are identical to mnist/train.py. Only the data changed: 512-number
descriptions of basketball frames instead of 28x28 pixel grids.
"""

import torch
import torch.nn as nn

from frames.data import CLASSES, get_loaders
from frames.model import FrameHead

EPOCHS = 30
LEARNING_RATE = 1e-3


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        # ---- THE FIVE STEPS ----
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def collect_predictions(model, loader, device):
    """Return (predictions, labels) over a whole loader."""
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features.to(device))
            preds.append(outputs.argmax(dim=1).cpu())
            targets.append(labels)

    return torch.cat(preds), torch.cat(targets)


def confusion_matrix(preds, targets, num_classes):
    """matrix[true][predicted] = count.

    Accuracy compresses every kind of mistake into one number. This does not:
    it shows which classes get confused for which, which is what tells you what
    to do next. 'crowd' guessed as 'game' and 'game' guessed as 'crowd' are
    different problems with different fixes.
    """
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true, pred in zip(targets, preds):
        matrix[true.item(), pred.item()] += 1
    return matrix


def print_report(matrix, classes):
    width = max(len(name) for name in classes) + 2

    print("\nconfusion matrix (rows = truth, columns = prediction)")
    print(" " * width + "".join(f"{name:>10}" for name in classes))
    for i, name in enumerate(classes):
        counts = "".join(f"{matrix[i, j].item():>10}" for j in range(len(classes)))
        print(f"{name:<{width}}{counts}")

    print(f"\n{'class':<{width}}{'precision':>11}{'recall':>9}{'support':>9}")
    for i, name in enumerate(classes):
        # Recall: of the frames that really were this class, how many did we
        # catch? Precision: when we said this class, how often were we right?
        support = matrix[i, :].sum().item()
        predicted_as = matrix[:, i].sum().item()
        hits = matrix[i, i].item()
        recall = hits / support if support else 0.0
        precision = hits / predicted_as if predicted_as else 0.0
        print(f"{name:<{width}}{precision:>10.1%}{recall:>9.1%}{support:>9}")

    total = matrix.sum().item()
    correct = matrix.diagonal().sum().item()
    print(f"\noverall accuracy {correct}/{total} = {correct / total:.2%}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, test_loader = get_loaders()
    model = FrameHead(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch % 5 == 0 or epoch == EPOCHS:
            preds, targets = collect_predictions(model, test_loader, device)
            acc = (preds == targets).float().mean().item()
            print(f"epoch {epoch:>3}/{EPOCHS}  loss {avg_loss:.4f}  test accuracy {acc:.2%}")

    preds, targets = collect_predictions(model, test_loader, device)
    print_report(confusion_matrix(preds, targets, len(CLASSES)), CLASSES)

    torch.save(model.state_dict(), "frame_head.pt")
    print("\nsaved weights to frame_head.pt")


if __name__ == "__main__":
    main()
```

`EPOCHS = 30` where MNIST used 3: there are ~200 training frames here against 60,000, so an epoch is a
handful of batches. Thirty passes over 300 cached vectors still finishes in about a second.

- [ ] **Step 3: OWNER RUNS — train**

```bash
python -m frames.train
```

Expected: loss falling, test accuracy in the **80-95%** range, then the confusion matrix.

**Do not expect 98%.** MNIST had 60,000 clean examples of a tidy problem. This has ~300 frames of a
messy one, labelled by hand in ten minutes. 85% on 300 labels is a good result and says the frozen
features carry real signal.

**How to read the confusion matrix.** Row = what the frame actually was, column = what the model said.
The diagonal is correct answers; everything off it is a mistake with a direction:
- `crowd` frames predicted `game` — usually mid-shots where some court is visible. Genuinely ambiguous.
- `graphic` frames predicted `game` — often adverts painted on the court surround. Also fair.
- `game` frames predicted `graphic` — more suspicious. Check whether those frames were mislabelled.

Low recall on the smallest class almost always means too few examples, not a broken model. Label more
of that class, rerun `python -m frames.data`, and retrain.

- [ ] **Step 4: Commit**

```bash
git add frames/model.py frames/train.py
git commit -m "Phase 2 Task 7: linear head, training loop and confusion matrix"
```

---

## Task 8: Use the filter

**Files:**
- Create: `frames/predict.py`

**Interfaces:**
- Consumes: `frame_head.pt`, `models.backbone.build_backbone`, `frames.data.CLASSES`
- Produces: nothing later tasks import. Phase 5 will call the same logic over a whole match.

- [ ] **Step 1: Write the predictor**

`frames/predict.py`:

```python
"""Classify frames with the trained filter.

Unlike training, this path has no cache: a new frame has never been through the
backbone, so both halves run -- backbone for the 512 numbers, head for the answer.
This is the shape Phase 5 uses over a full match.
"""

import argparse
import glob
import os

import torch
from PIL import Image

from frames.data import CLASSES
from frames.model import FrameHead
from models.backbone import build_backbone


def classify(paths, device):
    backbone, preprocess = build_backbone(device)

    head = FrameHead(num_classes=len(CLASSES)).to(device)
    head.load_state_dict(torch.load("frame_head.pt", map_location=device))
    head.eval()

    images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)

    with torch.no_grad():
        features = backbone(images)      # [N, 512]
        logits = head(features)          # [N, 3]
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)

    return predicted.cpu(), confidence.cpu()


def main():
    parser = argparse.ArgumentParser(description="Classify frames as game / crowd / graphic.")
    parser.add_argument("pattern", help="a JPEG path or a glob such as 'data/frames/match01/*.jpg'")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))[:args.limit]
    if not paths:
        raise SystemExit(f"no files matched {args.pattern}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted, confidence = classify(paths, device)

    print(f"{'frame':<28}{'prediction':>12}{'conf':>8}")
    for path, pred, conf in zip(paths, predicted, confidence):
        print(f"{os.path.basename(path):<28}{CLASSES[pred.item()]:>12}{conf.item():>7.1%}")

    kept = sum(1 for p in predicted if CLASSES[p.item()] == "game")
    print(f"\n{kept}/{len(paths)} frames would pass the game-action filter")
```

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 2: OWNER RUNS — see it work**

```bash
python -m frames.predict "data/frames/match01/*.jpg" --limit 20
```

Open a few of the named JPEGs and check the calls by eye. As in Phase 1, the interesting cases are the
mistakes: a wrong answer at low confidence is a model that knew it was unsure; a wrong answer at 99%
is the failure mode that breaks threshold filtering later.

- [ ] **Step 3: Commit**

```bash
git add frames/predict.py
git commit -m "Phase 2 Task 8: run the frame filter on new frames"
```

---

## Task 9: Close out Phase 2

- [ ] **Step 1: Record the result**

Append to `docs/phase-2-video-notes.md`:

```markdown
## Result

| labelled frames | train / test | test accuracy | weakest class |
|---|---|---|---|
| (n) | (n) / (n) | (x%) | (class, recall) |

Backbone: ResNet-18 ImageNet (frozen). Head: Linear(512 -> 3), (n) epochs.
Split: chronological, 70/30 within the match.
```

- [ ] **Step 2: Confirm understanding**

1. Why is the backbone frozen, and what would go wrong if it were not?
2. What does caching features buy, and why is it only safe while the backbone is frozen?
3. Why is the split chronological rather than random?
4. In the confusion matrix, what is the difference between precision and recall for one class?
5. Which class is weakest, and what would you do about it?

- [ ] **Step 3: Commit and push**

```bash
git add docs/phase-2-video-notes.md
git commit -m "Phase 2: frame classifier complete"
git push
```

Then Phase 3 begins: the same labelling loop, but over 2-second clips instead of single frames, and
~1,800 of them. Per spec §9 it is the attrition risk — the tooling built here is what shortens it.

---

## Self-Review

**Spec coverage.** Covers spec §9 roadmap row 2 and exercises §5.2 (eyes/rulebook split) and §5.5
(feature caching) on real data. `models/backbone.py` is written to the interface Phase 4 needs, so the
backbone is not rewritten later. Open question §10.2 (backbone choice) is deliberately left open —
ResNet-18 is used here and measured; comparing alternatives is only worth doing once there is a
baseline number to compare against, which this phase produces.

**Deviation recorded.** Replay is deferred from Phase 2 to Phase 4, with the reason stated at the top
of this plan. The spec's roadmap row 2 should be read as amended by that section.

**Placeholder scan.** No TBDs. Every code step carries complete runnable code. The two owner-supplied
values — the video URL and the resulting numbers in Task 9 — are marked as owner inputs, not as
unfinished work.

**Type consistency.** `build_backbone(device) -> (Module, transform)` and `FEATURE_DIM` are defined in
Task 5 and consumed unchanged in Tasks 6 and 8. `CLASSES` and `get_loaders` are defined in Task 6 and
consumed unchanged in Tasks 7 and 8. `FrameHead(num_classes=...)` is defined in Task 7 and constructed
identically in Task 8. Manifest columns (`match_id`, `filename`, `timestamp_sec`, `label`) are written
in Task 3 and read in Task 6 under the same names.
