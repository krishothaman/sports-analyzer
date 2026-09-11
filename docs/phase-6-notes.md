# Phase 6 notes: making the working model better, without risking it

## The goal, and the rule that came with it

Phase 5 left a model that works when asked about a known moment and is weak at
three measured things: it is fragile to timing, half its field-goal calls are
false, and its video backbone never adapts to basketball. Phase 6 attacks those,
cheapest first.

The owner's condition: **the Phase 5 model must survive whatever is tried.** So
before any code:

- `git tag phase5-working` on the Phase 5 commit, and all work on a `phase6` branch
- the weights, manifest, hoop detections and feature cache the model depends on
  copied to `data/baseline/`, with their SHA-256 hashes
- the seed-0 baseline retrained from scratch into a scratch path: 69.62% goal
  accuracy and the identical confusion matrix, before and after every change to
  the training code
- every new path writes to new files. That includes one real trap:
  `ingest/hoop.py`'s `save_records` rewrites every row for the matches it
  touches, so cutting extra clips into `data/hoops.csv` would have wiped the
  original match01/02 detections. Extra manifests now get their own detections
  file, and `tests/test_jitter.py` checks that no new path can land on a
  protected file.
- a new model replaces the Phase 5 head only if it loses no goal class and
  improves precision, the spot check or the scan's false calls

## The baseline, with precision

Five seeds, held-out match03, recall and precision per goal class (Phase 5 only
ever recorded seed 0's precision):

| goal class | recall | precision |
|---|---|---|
| `field_goal` | 75.3 (72.6-76.7) | **49.9 (48.7-52.4)** |
| `free_throw` | 67.6 (62.0-72.0) | 60.1 (58.5-60.7) |
| `none` | 65.5 (63.5-68.2) | 88.2 (86.6-88.8) |
| goal accuracy | 68.33 (67.58-69.62) | |

Field-goal precision is the number most worth moving: one "field goal" in two
is wrong.

---

## A scorer that uses raw video

Every earlier number scored cached clips. `clips/evaluate.py` instead runs the
real `clips.predict.Predictor` on the match video, at every test-match
manifest moment. That is the only way to score anything that changes how
windows are read. Its unshifted row gives 204/293 (69.62%) with the same
confusion matrix as `clips.train` seed 0. It measures the shipped model and
nothing else.

## Averaging over nearby windows: tried, didn't help

Phase 5 saw one answer flip when a query moved by a single frame. So
`clips.predict --shifts -0.25 0 0.25` reads three windows around the moment and
averages their probabilities. On match03, raw video:

| | none | field_goal | free_throw | fg precision | right |
|---|---|---|---|---|---|
| one window (Phase 5) | 68.2 | **74.0** | 68.0 | **52.4** | **204/293** |
| averaged over +/-0.25 s | 68.8 | 68.5 | 68.0 | 51.5 | 201/293 |
| one window, double marks removed | 68.2 | **74.6** | 67.6 | **50.0** | **188/270** |
| averaged, double marks removed | 68.8 | 66.7 | 70.3 | 48.8 | 185/270 |

Averaging costs four field goals and doesn't improve precision. A window a
quarter second late shows less of the approach to the rim, and blending it in
dilutes the one window that was right. `--shifts` stays `0`.

## Training on shifted copies: tried, made free throws worse

Every training clip starts 1.5 s before its event, so the model has only ever
seen a play at one moment in its window. `ingest/jitter.py` copied each of the
192 training-match events at 1.0, 1.25, 1.75 and 2.0 s before (768 extra
clips, `data/clip_manifest_jitter.csv`). They were cut into close-ups with
their own detections file (`data/hoops_jitter.csv`, hoop found in 98%), cached
separately, and added to the training side only (`clips.train --extra jitter`).

Five seeds, same settings as the baseline:

| goal class | baseline recall | + shifted copies | precision (baseline -> shifted) |
|---|---|---|---|
| `field_goal` | 75.3 (72.6-76.7) | 76.7 (74.0-80.8) | 49.9 -> 49.5 |
| `free_throw` | 67.6 (62.0-72.0) | **53.6 (50.0-56.0)** | 60.1 -> 61.0 |
| `none` | 65.5 (63.5-68.2) | 68.6 (64.1-71.8) | 88.2 -> 86.0 |
| goal accuracy | 68.33 | 68.05 | |

Free-throw recall falls 14 points, well outside the seed spread, and
field-goal precision doesn't move. By the promotion rule it is not adopted,
and `--extra jitter` stays off.

---

## Unfreezing the last block: tried, didn't help

`models/backbone.split_video_backbone` cuts MViTv2-S into a frozen trunk (15 of
16 blocks) and a trainable tail (the last block and final norm, 7.1M weights).
The trunk's output is cached once (`tokens_mvit_v2_s_hoop.pt`, 807 x 393 x 768
in float16), so an epoch trains one block instead of running sixteen. Before
training anything: cached tokens through the untrained tail reproduce the
Phase 5 features at cosine 1.000000 on 41 clips (largest difference 0.0002,
from float16).

Settings were chosen without looking at match03: train on match01, watch
match02 every epoch (`clips.finetune --validate`). Against the same script with
the tail frozen (`--freeze-tail`, i.e. Phase 5's linear probe, same optimizer
and data):

| match01 -> match02, seeds 0 and 1 | goal accuracy | field-goal precision |
|---|---|---|
| frozen tail, head only (100 epochs) | 76.5-77.5% | 55-57% |
| tail learning at 1/100 of the head's rate (best epochs 20-30) | 74-76% | 50-54% |

Letting the block learn does slightly worse on the match it never saw. It
learns (training accuracy 62% -> 85%), but what it learns from 190 clips of one
broadcast doesn't carry to the next. So it wasn't run on match03 at all: there
was nothing to confirm, and running it anyway would only have spent a look at
the test set.
