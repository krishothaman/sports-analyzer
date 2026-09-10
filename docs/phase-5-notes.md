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

## Stage 4: pointing it at raw video

`clips/predict.py` runs the whole pipeline from a video file and a moment
(`--at 36:01`) or a stretch (`--from 33:55 --to 38:55`). Everything before it read
clips already cut and features already cached.

### It reproduces training exactly

20 match03 clips, run from the raw video and compared with the cached pipeline:
identical close-up pixels (mean difference 0.000), feature cosine 1.0000, 20/20
identical answers. The close-up cutting is shared code (`ingest/hoop.close_ups`),
and the JPEG compression the training crops went through is repeated in memory.

### Two things the first version got wrong

**The decision rule.** The first version added up each goal group's probabilities
and answered the biggest. `grouped_report`, which scored every number above, takes
the single most likely class and maps it to its group. These sound the same, but
three field-goal classes pooled together outvote `none` more often. On seed 0:

| rule | `field_goal` | `free_throw` | `none` |
|---|---|---|---|
| most likely class, then group (as scored) | 74.0% | 68.0% | 68.2% |
| largest group sum | 84.9% | 60.0% | 59.4% |

The consistency check above didn't catch it, because it used the same rule on
both sides. `goal_answer` now matches `grouped_report`, and a test checks the two
against each other on 200 random heads.

**Asking about footage it never trained on.** A scan meets player close-ups,
replays and broadcast graphics every few seconds. The model had never seen them:
`ingest/cut.py` ran Phase 2's live-play filter over the background clips and threw
them all out. The first scan duly called a player's face a free throw at 97%.
So the first shipped version ran the same filter on each window's centre frame
and skipped windows that weren't live play.

**That filter turned out to be the bigger problem.** Asked about the 8 known
scoring plays in the stretch below, it skipped 5 of them as "not live play". A
contact sheet of their start, centre and end frames shows 23 of 24 are ordinary
wide shots (the other is a bench close-up), called `not_game` at up to 92%.
Phase 2 trained the filter on match01 only, and it doesn't carry over to
match03's broadcaster. Its verdicts flip within a single 2-second window of
continuous live play.

| same 12 moments (8 plays, 4 background) | right |
|---|---|
| filter on | 4 / 12 (5 plays skipped) |
| **filter off** | **8 / 12**: field goals 3/5, free throws 3/3, background 2/4 |

The filter is now opt-in (`--live-filter`). A side effect worth knowing: the
same filter chose which background clips made it into every match's `none`
set, so the test set's `none` clips are the ones it happened to call live.

### The test set counts some plays twice

The scan turned up pairs of match03 event clips under a second apart with the
same label: 23 pairs among 132 events (match01 has 1, match02 has 5). They look
like the same basket marked twice, by hand and by the scoreboard reader, both
kept. Dropping one clip of each pair barely moves the seed-0 result: field goals
74.6% (47/63), free throws 67.6% (25/37), `none` unchanged. The goal result
stands, but the effective test set is 270 clips, not 293. Deduplicating the
manifest is left for whoever resumes the project.

### A five-minute scan: where the model stands

match03 33:55–38:55 holds 8 known scoring plays: three threes, two twos, and three
free throws. Scanned back to back in 2-second windows, each matched to a known
play within 2.5 s:

| version | plays found | right type | false calls |
|---|---|---|---|
| group sums, no filter (first try) | 7 / 8 | 6 | ~27 of 36 |
| group sums + live-play filter | 7 / 8 | 4 | 15 of 22 |
| scored rule + filter (first shipped) | 5 / 8 | 3 | 13 of 18 |
| **scored rule, no filter (as shipped)** | **7 / 8** | **5** | **28 of 35** |

The group-sum rows *look* good only because that rule says "field goal" more
often. It catches more and invents more, and it isn't the rule the results were
measured with.

Between the last two rows the filter is a straight trade. Without it, the scan
finds 4 of 5 field goals (up from 3) and the two plays the filter had thrown
out. But false calls double, to about four for every real play, several of them
free throws at 97-98%. So the filter was removing some genuine junk along with
the real plays. Free throws stay the weak spot either way: 1 of 3 is labelled a
free throw. The other two come back as field goals (at 37:05 the scan also calls
a free throw 1.4 s away, but the field-goal window is closer).

Neither setting gives a usable timeline. For asking about a moment, where the
filter skipped most real plays, off is clearly better, so off is the default.

Scanning is harder than the test set, for three reasons:

- **The base rate.** The test set is 58% `none`; a scan is almost all `none`.
  At 52% field-goal precision on the test set, a stream that is 95% background
  has to fill up with false calls.
- **Alignment.** Training put every event 1.5 s into its window. Back-to-back
  windows land wherever they land, so an event can sit at the edge of a window
  or be split between two. `--every 1` halves the gap and doubles the cost.
- **Nothing reliable says "this isn't live play".** A scan meets close-ups and
  replays the model never trained on. The one filter available was trained on
  another broadcast, and on this one it is wrong about as often as it is right.

## Where the project stops

Phase 5's goal is met on the test set: every goal class above 50% recall, in every
seed, on a broadcaster the model never trained on. Asked about a known moment,
it's right about 7 times in 10 (8 of 12 in a spot check on raw video). It is
**not** yet a timeline generator. On a continuous stretch it finds most plays,
but invents about four for every real one.

The project pauses here. The levers, cheapest first:

1. **Unfreeze the backbone's last block.** Minutes of training, no new data.
2. **More training data**, especially broadcast `none` near the rim (missed shots
   and possessions) to push field-goal precision up. Basketball-51 for made shots.
3. **Scan smarter:** overlapping windows (`--every 1`), and a live-play filter
   retrained on all three broadcasts, judged on several frames rather than one.
4. **Deduplicate the manifest** (above) and re-cut.
5. **Two versus three:** court keypoints and homography.
6. **The hybrid:** `ingest/scoreboard.py` knows exactly when points were scored.
   Let it say *when*, and the video say *what kind*.

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
