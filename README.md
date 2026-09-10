# Basketball Highlight Analyzer

Point it at a basketball broadcast and a moment (or a stretch of the match), and it tells you
whether a **field goal**, a **free throw**, or **nothing** happened there. The long-term goal is a
full-match timeline of tagged events. This repo gets as far as a model that somewhat works on
footage it has never seen.

Built from scratch as a machine-learning learning project: every phase has notes explaining
what was tried, what failed and why.

> **Status: paused after Phase 5.** The model works well enough to demo. It isn't reliable enough
> to use unattended. See [If this is picked up again](#if-this-is-picked-up-again).

## Results

Tested on a match from a **different broadcaster** than the two it was trained on, so there's no
shared arena, lighting or camera crew. Figures are recall, averaged over five training seeds:

| | field goal | free throw | nothing |
|---|---|---|---|
| always say "nothing" (baseline) | 0% | 0% | 100% |
| whole-court view | 22.5% | 80.0% | 82.1% |
| **hoop close-up (the shipped model)** | **75.3%** | **67.6%** | **65.5%** |

It finds about 3 in 4 made baskets and 2 in 3 free throws. The weak spots, stated plainly:

- **About half of its "field goal" calls are false alarms** (52% precision). The likely cause is
  missed shots, which count as "nothing" but look like makes until the last few frames.
- **It can't tell a two from a three.** That needs knowing where the shooter stood, i.e. court-line
  detection, which isn't built.
- **Free throws are 12 points worse** on the close-up than on the whole-court view.
- **Scanning continuous footage is much harder than the test set.** Over a 5-minute
  stretch with 8 scoring plays, it found 7 (5 labelled correctly) and made 28 false
  calls, about four for every real play. Asking about a known moment works (8 of 12
  right in a spot check on raw video); an automatic timeline doesn't yet.

The full story, including the experiments that didn't work, is in
[docs/phase-5-notes.md](docs/phase-5-notes.md).

## How it works

```
match video
   │  read 16 frames over 2 seconds
   ▼
OWLv2 finds "a basketball hoop"        (zero-shot detector, not trained here)
   │  cut a 256×256 close-up of the rim
   ▼
MViTv2-S video transformer, frozen     (pretrained on Kinetics-400, not trained here)
   │  768 numbers describing the clip
   ▼
one linear layer                       (the only part trained: ~5k weights)
   ▼
nothing / field goal / free throw
```

The key finding: in a broadcast wide shot the rim is about **15 pixels** wide by the time it
reaches the model, too small to see a ball go through. Zooming in on the hoop took field goals
from 22% to 75%.

## Running it

Needs an NVIDIA GPU (developed on an 8 GB RTX 4060 Laptop) and Python 3.

```bash
python -m venv venv
./venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

Match footage isn't distributed with this repo. With videos under `data/video/`, the pipeline
runs in this order:

```bash
python -m ingest.scoreboard --match-id match01   # score changes -> event marks, automatically
python -m ingest.mark data/video/match01.mp4     # hand marks for what the scoreboard can't see
python -m ingest.cut --match-id match01          # marks -> 2-second clips
python -m ingest.hoop                            # hoop close-ups for every clip
python -m clips.data --backbone mvit_v2_s --crop hoop
python -m clips.train --backbone mvit_v2_s --crop hoop --head clip --fold block steal --epochs 100 --seed 0
```

Then ask it about a moment, or scan a stretch:

```bash
python -m clips.predict data/video/match03.mp4 --at 36:01 --at 37:05
python -m clips.predict data/video/match03.mp4 --from 33:55 --to 38:55 --json timeline.json
```

`--live-filter` skips windows that Phase 2's frame filter calls close-ups or replays. It halves
the false calls in a scan, but it's off by default: it was trained on one broadcast, and asked
about the test match's 8 real scoring plays it skipped 5 of them.

Each 2-second window takes about 1.5 seconds, mostly spent in the hoop detector. That makes a
5-minute stretch take about 4 minutes, and a whole match several hours.

Tests: `python -m pytest tests/`

## How it got here

| phase | what | notes |
|---|---|---|
| 0–1 | CUDA PyTorch set up; a digit classifier (MNIST) to learn the training loop | |
| 2 | frame classifier: is this live play? (frozen ResNet-18 + one layer) | [phase-2](docs/phase-2-video-notes.md) |
| 3 | three matches labelled into 807 clips; a scoreboard reader labels scoring plays automatically | [phase-3](docs/phase-3-pilot-notes.md) |
| 4 | clip classifiers on per-frame features: **fail** the cross-broadcaster test | [phase-4](docs/phase-4-notes.md) |
| 5 | video transformer backbone, then hoop close-ups: **pass** | [phase-5](docs/phase-5-notes.md) |

## If this is picked up again

In order of cost versus likely payoff:

1. **Unfreeze the backbone's last block** so it can adapt to broadcast footage. Minutes of
   training, and no new data.
2. **Scan smarter:** overlapping windows (`--every 1`), and a live-play filter retrained on
   frames from all three broadcasts (the current one only knows match01's). Also deduplicate
   the test manifest, which counts 23 plays twice (details in the Phase 5 notes).
3. **More training data.** [Basketball-51](https://www.kaggle.com/datasets/sarbagyashakya/basketball-51-dataset)
   has thousands of made twos, threes and free throws from 51 NBA broadcasts. Check its licence
   first.
4. **Two versus three:** detect court keypoints, map the shooter onto a court diagram and check
   which side of the arc they stood.
5. **The pragmatic hybrid:** `ingest/scoreboard.py` already reads score changes, so +2/+3/+1 is
   known exactly. Let the scoreboard say *that* points were scored and the video say *what kind*
   (dunks, and blocks and steals, which change no score).

## Credits

- [MViTv2-S](https://pytorch.org/vision/stable/models/video_mvit.html) Kinetics-400 weights, via
  torchvision
- [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) (Apache-2.0), via Hugging Face
  `transformers`
