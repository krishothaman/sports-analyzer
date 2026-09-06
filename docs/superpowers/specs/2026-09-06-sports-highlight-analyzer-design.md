# Sports Highlight Analyzer — Design

**Date:** 2026-09-06
**Status:** Approved, not yet implemented

---

## 1. Goal

Point the system at a full basketball match video; get back a **tagged timeline** — a list of
timestamped, labelled events you can jump to.

```json
[
  {"t": "00:12:31", "event": "three_pointer", "confidence": 0.78},
  {"t": "00:14:02", "event": "dunk",          "confidence": 0.91}
]
```

Basketball is version 1. The architecture must let other sports be added later **without a rewrite**.

### Secondary goal, weighted equally

The project owner is learning ML from scratch. Understanding *why* each piece exists is part of the
deliverable, not a side effect. A working analyzer the owner cannot explain is a failed outcome.

---

## 2. Non-goals

- No other sports until basketball works end to end.
- No object detection, player tracking, or court homography in v1 (see §9, Deferred).
- No cloud training, no web deployment, no mobile.
- No real-time / live-stream analysis. Offline video files only.
- MNIST is a warm-up rung, not a deliverable. One or two sessions, then we leave it.

---

## 3. Constraints

| Constraint | Detail |
|---|---|
| Hardware | NVIDIA RTX 4060 Laptop, **8 GB VRAM**; i7-12650H, 10 cores; 15.7 GB RAM |
| Blocker | Installed PyTorch is the **CPU-only build** (`torch 2.14.0+cpu`) — must be replaced with a CUDA build |
| Time | No deadline. Steady, serious effort. Depth preferred over speed |
| Data | No pre-existing labelled data. Footage from public sources, labelled by hand |
| Collaboration | Owner runs all training/inference commands; Claude writes machinery and teaches first |

### Working agreement

- **Owner does:** all training runs, all data collection and labelling, all design calls.
- **Claude does:** scaffolding, model/pipeline code, tooling, debugging — always explained before use.
- **Hard rule:** the owner never runs a script they can't roughly describe. If that happens, the
  explanation was inadequate, not the owner.

---

## 4. Decisions

| Decision | Choice | Why |
|---|---|---|
| Output | Tagged timeline (timestamps + labels) | Useful on its own; a cut reel is editing work layered on top |
| Sport | Basketball first, architected for more | Data is the bottleneck; each extra sport multiplies it |
| Classes | 6 events + background: `two_pointer`, `three_pointer`, `dunk`, `free_throw`, `block`, `steal`, `none` | Owner's explicit choice, accepting the cost |
| Data source | Hybrid — public dataset first, own labelled clips to fill gaps | Model trains in week one instead of week four |
| Architecture | **A now, B later** — frozen pretrained backbone + temporal head, then a true video network | A is cheap, fast and maps to the concepts being taught; B is measured against it |
| Multi-sport | Shared backbone, one head per sport | Avoids catastrophic forgetting; adding a sport = one head + one config entry |

### Known-hard classes, accepted going in

- **`two_pointer` vs `three_pointer`** — distinguishing them properly requires knowing where the
  shooter stood, which means court-line detection and homography. Deferred. The model will infer it
  indirectly from shooter distance and ball flight, and **will** be unreliable near the arc. Accepted
  for v1.
- **`block` / `steal`** — brief, fast, visually similar to ordinary contact. Expect these to lag and
  to need targeted extra examples.

---

## 5. Architecture

### 5.1 Components

```
ingest/       download match video (yt-dlp), cut into clips
label/        keyboard-driven local clip labelling tool
data/         datasets, dataloaders, transforms, feature cache
models/
  backbone.py   the "eyes" — frozen pretrained feature extractor (swappable)
  head.py       the "rulebook" — small temporal classifier, one per sport
  registry.py   assembles backbone + head from a sport config
train/        training loop, metrics, confusion matrix
infer/        sliding-window inference over a full match, event merging
viewer/       timeline UI — click a timestamp, jump to that moment
docs/         label guide, this spec, notes
```

The multi-sport promise lives entirely in `registry.py` and the sport configs. Adding cricket touches
one new head file and one config entry; nothing else in the tree changes.

### 5.2 The eyes / rulebook split

```
                          ┌──> basketball head ──> "dunk at 4:12"
video ──> shared backbone ─┼──> football head   ──> "goal at 12:31"
           (frozen)        └──> cricket head    ──> "wicket at 45:02"
```

The backbone is a pretrained image network with its classification layer removed, producing a
**512-dimensional feature vector** per frame (ResNet-18 width). It is **frozen** in Phase A — its
weights never update. Only the head trains, which is ~1% of the parameters.

Freezing gives three things: training in minutes rather than hours, immunity to catastrophic
forgetting when sports are added, and the ability to cache features (§5.5).

### 5.3 Training data flow

```
raw videos → clips → owner labels them → manifest.csv (clip_path, label, match_id)
                                             │
                                             ▼ split by match_id into train / val / test
                                             ▼ run backbone ONCE, cache features to disk
                                             ▼ train head on cached features
                                             ▼ score: per-class precision/recall + confusion matrix
```

### 5.4 Inference data flow

```
match.mp4
  ▼ sample frames at 8 fps
  ▼ overlapping 2s windows (16 frames, stride 0.5s)
  ▼ backbone → 512 numbers per frame (computed once per frame, shared across windows)
  ▼ head → per-window class probabilities
  ▼ threshold, then merge contiguous runs, keep peak-confidence window as the event time
  ▼
timeline.json → viewer
```

**Why overlapping windows:** non-overlapping blocks slice events across boundaries, and neither half
looks like anything. A 0.5s stride means every moment is examined four times at four offsets, so any
event under 2 seconds sits fully inside at least one window. Neighbouring windows share most frames,
so frame features are computed once and indexed, not recomputed.

**Merging:** a real event fires several adjacent windows. Take the contiguous above-threshold run,
keep the peak as the timestamp, discard the rest. Six firings become one timeline entry.

### 5.5 Feature caching

The backbone is frozen, so a given clip's features can never change. Compute once, write to disk,
read thereafter.

| | Per epoch (~5,000 clips) | 50 epochs |
|---|---|---|
| Decode video + run backbone | ~4 min | ~3.5 h |
| Read cached arrays | ~3 s | ~2.5 min |

Storage: 16 frames × 512 floats × 4 bytes = 32 KB/clip → ~160 MB for 5,000 clips.

**Costs, accepted knowingly:**
- No random data augmentation while cached (features are frozen snapshots). Pre-augmented variants
  can be cached if needed.
- Caching dies at Phase 6 — fine-tuning changes the backbone's weights every epoch. Training slows
  from seconds to minutes. Expected, not a regression.

---

## 6. Data plan

### 6.1 Sources

| Source | Role | Caveat |
|---|---|---|
| Public basketball action dataset | Bootstrap — train something in week one | Classes and licence must be verified before use (§10) |
| **Full game footage** (yt-dlp) | Backbone of the dataset | 95% dead time; slow to label |
| Highlight reels (yt-dlp) | Top up rare classes only | Slow-mo, replays and graphics are unrepresentative |

**Distribution shift is the governing rule:** training data must resemble the footage the model will
run on. A model trained only on highlight reels has never seen a normal possession and will fire
constantly on a real game. Full games dominate the mix; reels are used surgically.

Footage is for personal learning use. Neither footage nor derived clips get redistributed.

### 6.2 Labelling

A local keyboard-driven tool: clip loops, keys `1`–`6` assign a class, `0` marks background,
backspace undoes, and it auto-advances. No mouse. Target throughput ~4–6 s/clip, i.e. 600–900/hour.

**Volume target:** ~300 examples per *event* class minimum → **~1,800 event clips** for v1.

`none` (background) is counted separately and is not part of that 1,800. It is abundant and nearly
free — any stretch of full-game footage with no event in it qualifies, so it can be sampled
automatically rather than hand-labelled. Expect the final dataset to be roughly 70–80% background by
volume, which is realistic and matches what inference will actually encounter. That imbalance is
handled at training time (weighted loss / sampling, §8), **not** by throwing background away —
discarding it would reintroduce the highlight-reel distribution shift described in §6.1.

**A written label guide precedes any labelling** (`docs/label-guide.md`) with exact, testable rules —
e.g. *"Block: defender contacts the ball on its way up. Contact after the ball peaks is not a block."*
Inconsistent labels cannot be fixed by more labels; two identical clips labelled differently cancel
to noise.

**Consistency check:** re-label 100 previously-labelled clips a week later. Below ~90% agreement with
your past self means the guide needs tightening, not the model.

### 6.3 Splits

Train / validation / test, **split by `match_id`, never by clip.**

Clips from one game sharing arena, lighting and jerseys across train and test produce a model that
learned "this arena" and a test score around 95% that collapses on new footage. A dishonest test
score is worse than none, because it lies to you.

Test set stays sealed. Decisions get made against validation. The moment tuning targets the test set,
it stops measuring anything.

---

## 7. Evaluation

**Accuracy alone is rejected as a metric.** With ~80% background clips, a model answering "nothing"
to everything scores 80% and is worthless.

Metrics used instead:

- **Per-class precision** — when it says "dunk", how often is it right? *(Don't cry wolf.)*
- **Per-class recall** — of all real dunks, how many did it find? *(Don't miss the wolf.)*
- **Confusion matrix** — not just *that* it's wrong but *how*, which converts directly into a data
  shopping list (e.g. blocks bleeding into `none` → go collect blocks).
- **Dumb baseline** — always predict the most common class. Any model that can't beat it has learned
  nothing, and usually indicates a broken pipeline rather than a weak model.

**Precision/recall balance:** lean toward recall for a highlight timeline. A few false entries the
viewer skips past cost less than a missed dunk.

---

## 8. Failure modes

| Symptom | Cause | Mitigation |
|---|---|---|
| Train accuracy climbs, validation flattens | Overfitting | More data, dropout, early stopping |
| Rare classes never predicted | Class imbalance | Weighted loss, oversampling |
| Strong on validation, weak on a new game | Leakage or too few distinct matches | Split by match; source more different games |
| Timeline full of junk entries | Threshold too low | Raise confidence cutoff, merge more aggressively |
| Fires during ads and replays | Non-game footage | Phase 2 frame classifier filters it out upstream |
| Everything is slow after Phase 6 | Feature cache disabled by fine-tuning | Expected. Reduce clip count per experiment |

---

## 9. Roadmap

| # | Phase | Deliverable | Owner runs | Claude builds | Size |
|---|---|---|---|---|---|
| 0 | Environment | GPU working, project skeleton | CUDA reinstall, GPU check | skeleton, configs | 1 evening |
| 1 | MNIST | A training loop the owner has written and run | training | model + loop, commented | 1–2 sessions |
| 2 | Frame classifier | Game-action vs crowd/replay/ad filter | label ~300 frames, train | backbone + caching code | ~1 week |
| 3 | Data pipeline | ~1,800 labelled clips + manifest | download, label | ingest, cutter, label tool | **the long one** |
| 4 | Clip classifier | Model naming events in a 2s clip | train, read confusion matrix | backbone/head/registry, metrics | ~2 weeks |
| 5 | Match timeline | **End-to-end working analyzer** | run on a chosen match | sliding-window infer, merge, viewer | ~1 week |
| 6 | Video model | Option B measured against A | run both, compare | X3D/VideoMAE swap, fine-tuning | ~1 week |
| 7 | Second sport | Architecture proven | pick sport, gather data | new head, config wiring | ~1 weekend |

**One implementation plan per phase.** This spec is deliberately too large for a single plan. Each
phase gets its own plan written when it starts, informed by what the previous phase actually produced
— planning Phase 4 before seeing Phase 3's real data would be guesswork.

**Phase 2 is deliberately dual-purpose:** it teaches transfer learning, frozen backbones and feature
caching on easy data, *and* produces a real pipeline component. Not a throwaway exercise.

**Phase 3 is the attrition risk.** Not difficult — tedious. If this project is abandoned, it will be
here. The bootstrap dataset and the keyboard-driven tool exist specifically to shorten it.

### Deferred (not v1)

- Object detection + tracking + court homography for definitive 2pt/3pt calls.
- Cut highlight reel output (video editing on top of the timeline).
- Player identification, possession stats, formations.

---

## 10. Open questions

1. **Which public dataset.** Candidate basketball action datasets from academic work need their
   actual class lists, sizes and licences verified before we depend on one. Resolve at Phase 3 start;
   do not promise specifics before checking.
2. **Backbone choice.** ResNet-18 assumed for its speed and 512-d output. Worth a quick comparison
   against a small EfficientNet or a CLIP image encoder at Phase 2, decided by measurement.
3. **Head architecture.** GRU vs small transformer vs 1D-conv over the frame sequence — decide at
   Phase 4 by measurement, not preference.
4. **Clip length.** 2s / 16 frames @ 8fps is the starting assumption. Free throws and steals may want
   different windows; revisit with real data.
5. **Git.** This directory is not currently a git repository. Should be initialised before Phase 0 so
   work is checkpointed.
