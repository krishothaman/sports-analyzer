# Phase 2: video source, dataset and result

## Source footage

One full Olympic-qualifier basketball broadcast downloaded from YouTube with
`yt-dlp`. Deliberately a *full broadcast*, not a highlight reel: a highlight reel
is 100% game action and would teach the filter nothing about what to reject.

| | |
|---|---|
| resolution | 854 x 480 |
| frame rate | 25 fps |
| length | 1h 48m 21s (162,536 frames) |
| download | video-only stream, no audio, no ffmpeg merge |

Chosen at 480p because every frame ends up centre-cropped to 224 x 224 for
ResNet-18 anyway. Higher resolution costs disk and decode time and buys nothing.

The video itself is **not in this repo** and never will be: it is not ours to
redistribute, and `data/` is gitignored wholesale. To reproduce, download any
full broadcast and run the extraction below.

## Extraction

```
python -m frames.extract data/video/match01.mp4 --match-id match01 --every 12
```

Sampling every 12 seconds gives 542 frames (6,501s / 12). The interval is a
trade-off: closer together and consecutive frames are near-duplicates of the same
possession, which inflates the dataset without adding information.

## Labelling

520 of the 542 frames labelled by hand; 22 skipped as genuinely unreadable.

| raw label | count | folds to |
|---|---|---|
| `game` | 278 | `game` |
| `crowd` | 232 | `not_game` |
| `graphic` | 10 | `not_game` |

**The three-class plan became a two-class problem.** `graphic` came in at 10
frames -- a class that small cannot be learned or honestly measured. The merge is
applied at read time by `LABEL_MAP` in `frames/data.py`; the manifest still holds
all three raw labels, so a future broadcast with more adverts restores the third
class without any relabelling.

Merged, the split is 278 / 242 -- close enough to balanced that accuracy is a
meaningful number. (On a 90/10 split it would not be: always guessing the
majority class would score 90%.)

Rules used are in [label-guide.md](label-guide.md).

## Split

Chronological, never random. Train is the first 70% of each match by timestamp,
test the last 30%: train 0-75m, test 75-108m, with a 12-second gap at the
boundary and zero filename overlap.

Frames 12 seconds apart show the same possession. A random split would put one in
train and its near-twin in test, and the model would score well by *recognising a
moment it had already seen*. Critically, that failure **raises** the accuracy
number rather than lowering it, so it cannot be caught by noticing something looks
wrong -- which is why `tests/test_frames_data.py` asserts
`max(train timestamp) < min(test timestamp)` instead.

## Result

364 train / 156 test frames. 1,026 trainable parameters against ~11.2M frozen.
Loss 0.69 (the coin-flip baseline, -log 0.5) down to 0.058.

```
confusion matrix (rows = truth, columns = prediction)
                game  not_game
game              83         1
not_game           0        72

class      precision   recall   support
game           100.0%    98.8%        84
not_game        98.6%   100.0%        72

overall accuracy 155/156 = 99.36%
```

The one miss is `match01_00500.jpg`, an overhead camera looking straight down at
the rim -- a genuine game frame from an angle that appears nowhere in the training
half. That is the honest failure mode of a single-video dataset.

### The label-noise pass

The first training run scored 98.08% (three test errors). Running
`python -m frames.review` -- which surfaces frames where the trained model
confidently disagrees with the labeller -- showed **two of those three were
labelling slips, not model errors**. Correcting two rows of the manifest and
retraining gave 99.36%.

The model did not get better. The answer key did. At this accuracy a ~0.4% label
error rate accounts for a large share of the apparent remaining error, so the
measurement is now partly a measurement of the labelling.

The review tool is circular by construction (it was trained on the labels it is
checking) and this bounds what it can find: a one-off slip stands out against 500
consistent examples, but a rule applied *wrongly and consistently* becomes the
pattern the model learns, and it will agree every time. Slips: use the tool.
Systematic errors: use the label guide.

## Caveat carried into later phases

99.36% on one broadcast is not 99.36% in general. The strongest cue available in
this data is plausibly **wide shot vs close-up**, which correlates almost perfectly
with game vs not-game here but is not the same thing -- `00500` is exactly where
that shortcut breaks. Expect a real drop on broadcasts with unusual camera work,
and treat the first multi-match evaluation in Phase 3 as the honest number.

Also worth revisiting: `weights.transforms()` centre-crops to 224, discarding the
left-edge score bug. That overlay is a strong "this is live game footage" signal
and the model never sees it. Squashing instead of cropping is the first thing to
try if `game` recall degrades on new footage.
