# Phase 5 notes: swapping the eyes, and finding the hoop

## The goal, as agreed

A decently working model for at least three classes, measured on the held-out
FIBA match (`match03`), never on footage the model trained on:

| goal class | made of |
|---|---|
| `none` | `none`, `block`, `steal` |
| `field_goal` | `two_pointer`, `three_pointer`, `dunk` |
| `free_throw` | `free_throw` |

Bar: every goal class above 50% recall. Two-versus-three and dunks are
refinements inside `field_goal`, reported but not required.

**Where Phase 4's model stands on that goal** (resnet18 + GRU): 58.0% overall,
`field_goal` 20.5%, `free_throw` **0%**.

---

## Step 1: a backbone trained on motion

ResNet-18 (ImageNet, one still frame at a time) replaced by **MViTv2-S** from
torchvision: Kinetics-400, 34.5M parameters, 80.8% top-1, reads all 16 frames at
once. No new dependency. It takes exactly `[B, 3, 16, 224, 224]` and its eval
recipe samples at 7.5 fps against our 8 -- Phase 3's clip format needed no change.

### What the untrained weights already knew (`clips/probe_kinetics.py`)

Kinetics-400 includes `dunking basketball`. Before training anything:

| our label | P(dunking basketball) | most common Kinetics guess |
|---|---|---|
| dunk (18) | **0.542** | dunking basketball (94%) |
| two_pointer | 0.367 | dunking basketball (55%) |
| everything else | 0.279 | playing basketball |

A real signal, but a weak one: *anything near the rim* scores high on
"dunking". That later explained a failure -- see below.

### The crop was throwing the basket away

The torchvision preset resizes the short side to 256 and centre-crops 224.
Stored frames are 455x256, so that keeps the middle 49% of the width and drops
both sidelines -- where a broadcast camera puts the hoop during a shot. Every
result since Phase 2 was computed through that crop. `tests/test_backbone.py`
proves it: a clip black except for a bright left edge comes back with a
standard deviation of exactly 0.0.

Five seeds each, `block` and `steal` folded into `none`, 100 epochs, frozen
MViTv2-S + one linear layer (dumb baseline on this test set: **58.0%**):

| view | 5-class accuracy | free_throw recall |
|---|---|---|
| centre crop | 51.7% (49.2 - 54.6) | 70 - 74% |
| **squash whole frame** | **62.9% (60.8 - 65.5)** | **80%** |

Squash beats the baseline in all five seeds; centre loses to it in all five.
With the centre crop, the head learned "near the rim means dunk" and dumped 86
clips into `dunk` -- the probe's weak signal, amplified by class weighting.

**Free throws went from 0% to 80%.** That is the backbone.

### What was still broken: field goals are not being seen

Squash, seed 0: 29 of 37 two-pointers and 20 of 33 three-pointers were called
`none`. They were not confused with each other -- they were missed.

Full-resolution frames from inside a test-match two-pointer show the shot is in
the window (clip timing is right). The hoop sits in the top-left corner of an
854x480 frame, about 60px across. By the time a frame reaches the backbone --
455 wide on disk, squashed to 224 -- the rim is about **15 pixels**. Free throws
work because they look the same for ten seconds; a made jump shot is a few
pixels for half a second.

---

## Step 2: find the hoop, look closely there

Taken from the shot-tracking pipelines (and the racquet-sports analyzer). The
idea, not their parts:

- their YOLO ball-and-hoop weights were trained on phone footage of outdoor courts
- the broadcast repos detect players, the ball and court keypoints, but not the hoop
- `ultralytics` is AGPL-3.0

Instead **OWLv2** (`google/owlv2-base-patch16-ensemble`, Apache-2.0), an
open-vocabulary detector: ask for "a basketball hoop", get a box. No training,
no labelled hoops. Spike: 12 mid-shot frames, 4 from each broadcast -- the box
landed on the rim in **12 of 12**. The "backboard" query found a player once and
a shot clock once, so it was dropped.

`ingest/hoop.py` re-reads every clip's 16 frames at full resolution, detects on
4 of them, interpolates the hoop across the rest, and writes a 256x256
native-resolution close-up with the rim a third of the way down -- about 4x the
pixels on the rim. The same frozen MViTv2-S turns the close-up into 768 numbers,
which the head reads alone or joined to the whole-court view (squash, 768).

Cost: OWLv2 is the bottleneck (GPU at 100%). One process ran at 2.7 s/clip;
three in parallel, one per match, about 1.5 s/clip overall -- ~20 minutes for
807 clips, once.

### Results

OWLv2 found a hoop in 94-100% of clips for every label, on all three
broadcasts (lowest: `two_pointer` on match02, 33 of 35).

Five seeds each, `block` and `steal` folded into `none`, 100 epochs, recall on
the goal classes, held-out match03 (293 clips: 170 none, 73 field goals, 50 free
throws):

| view | goal accuracy | `field_goal` | `free_throw` | `none` |
|---|---|---|---|---|
| squash (whole court) | 66.9% | 22.5% (13.7 - 30.1) | 80.0% | 82.1% |
| **hoop (close-up)** | 68.3% | **75.3% (72.6 - 76.7)** | 67.6% (62 - 72) | 65.5% (63.5 - 68.2) |
| squash + hoop | 70.9% | 39.7% (35.6 - 46.6) | 76.0% | 82.8% |

**The hoop view clears the bar: every goal class above 50%, in every seed.**
Field goals went from 22% to 75% -- the rim was the missing information, and
looking at it closely was enough. This is the model Phase 5 set out to build.

What that costs, read from the seed-0 matrix:

| truth \ predicted | none | field_goal | free_throw |
|---|---|---|---|
| none | 116 | 41 | 13 |
| field_goal | 10 | 54 | 9 |
| free_throw | 8 | 8 | 34 |

- **`field_goal` precision is 52%.** Of 103 clips called a field goal, 41 were
  `none`. Background near the rim now looks like scoring. The likely cause is
  missed shots -- the labels come from the scoreboard, so a miss is `none` and
  looks like a make until the last frames -- but that is unverified.
- **Free throws lost 12 points** against the whole-court view, which sees the
  lane lined up at the line.
- **Inside `field_goal`, two-versus-three is not learned.** At seven classes the
  head still dumps into `dunk` (24 of 37 two-pointers, 31 `none` clips); the
  collapse to the goal hides that, which is why the goal is reported separately
  rather than instead.

**Joining both views does worse on the goal, not better.** Squash + hoop has the
best overall accuracy, but only because it goes back to answering `none` -- field
goals fall to 40%. With 1,536 inputs and 514 training clips, a single linear
layer has room to lean on whichever half fits the training matches best, and the
whole-court half is what said "background" before. More data, or a head that is
forced to use both halves, might change that; with this data, the close-up alone
is the model.

Noise: one test field goal is 1.4 points of recall, so the five-seed ranges
above are about three clips wide.

---

## Searched and not used

HuggingFace, via its API (every basketball model and dataset):

- **No model classifies shots in broadcast footage.** The only basketball video
  classifiers (`yerx/videomae-...`) predict free-throw made vs missed.
- `saveerjain/basketball-events`: real shot-type labels (2-pt jump shot, layup,
  dunk, 3-pt), but 543 clips from 4 games, 7.9 GB, research-only. Held in reserve.
- `koppolusameer/yolo11n-basketball-court-keypoints`: 48 court keypoints on
  broadcast frames, AGPL-3.0. The route to two-versus-three via homography,
  alongside `abdullahtarek/basketball_analysis` (MIT, 18 keypoints, 28x15 m court).

Basketball-51 (Kaggle, 10,311 clips from 51 NBA broadcasts) remains the largest
source of labelled 2s and 3s if more data is needed.
