# Phase 4: the clip classifier, and what the first honest test showed

807 clips, 514 train / 293 test. Train on two Paris 2024 games, test on a FIBA
World Cup Qualifier. Both heads run identically; the only difference between them
is whether frame order is available.

## Headline: the honest test score is below the dumb baseline

| | mean-pool | GRU | dumb baseline |
|---|---|---|---|
| parameters | 3,591 | 111,431 | -- |
| train | 84.63% | 95.91% | -- |
| **test (cross-broadcast)** | **53.58%** | **53.92%** | **59.9%** |
| train-test gap | +32.1% | +42.0% | -- |

A model that always answered `none` would score better than either. Taken alone
this reads as total failure, and the two heads tying at ~53% reads as "frame
order does not help" -- which was very nearly the conclusion recorded here.

Both of those readings are wrong, and one diagnostic separates them.

## The diagnostic: can it learn at all, in-domain?

Same heads, same features, same loop -- but trained and tested inside the Paris
matches only, split chronologically:

| | mean-pool | GRU | dumb baseline |
|---|---|---|---|
| in-domain test | 47.10% (+3.9) | **60.65% (+17.4)** | 43.23% |

**This number is not a generalisation estimate and must never be quoted as one.**
It comes from the chronological-within-match split that Phase 3 exists to avoid:
train and test share arena, lighting and camera crew. It is a diagnostic, valid
only for the comparison it makes possible.

## What that comparison establishes

**1. The clips and the features do carry the signal.** In-domain the GRU clears
its baseline by 17.4 points. Something real is being learned. The cross-broadcast
score was hiding it, not reflecting it.

**2. Frame order matters, and by a wide margin.** The order-aware head beats the
order-blind one by **13.5 points** in-domain (47.1 -> 60.7). This resolves spec
section 10 question 3, and it vindicates Phase 3's central design choice: clips
rather than frames, a 2-second window, and per-broadcaster lag calibration.

Note how close this came to being decided backwards. On the cross-broadcast test
the two heads differ by 0.3 points and both sit under the baseline; the obvious
reading is "motion adds nothing, drop the sequence model". Domain shift had
swamped the effect being measured. **A tie between two treatments means nothing
if the experiment cannot detect a difference at all** -- the in-domain run is
what showed the experiment had power.

**3. The failure is domain shift, and it is specific.** The model learns
basketball on one production and cannot carry it to another. Not a head problem;
tuning the GRU cannot fix it.

Corroborating evidence from Phase 3: the Phase 2 frame filter, trained on Paris
2024 only, still classified FIBA footage correctly but its mean confidence fell
from ~0.96 to 0.87. The same shift, visible one phase earlier and much smaller,
because "is this basketball at all" is a far coarser question than "which of six
events is this".

## Also worth reading

- **`free_throw`: 90 clips, 50 in test, 0% recall.** Every one predicted `none`
  or `two_pointer`. It should be the easiest class -- static scene, player alone
  at the line, clock stopped -- so a total miss points at the free throws in the
  test match specifically, not at the class. Worth a look before more collection.
- **Mean-pool overfits too**, +32.1% gap on 3,591 parameters. Capacity is not the
  whole story; a model that small cannot memorise 514 clips. That gap is mostly
  the domain gap.
- **Mean-pool's test score peaks at epoch 20 (58.70%) and decays to 52.56%.**
  Early stopping would buy back ~6 points and still not clear the baseline.
- `dunk` (3), `block` (3) and `steal` (6) have too few test examples to read.
  Reported as unmeasurable rather than as zeros.

## Decision

**Do not tune the head, and do not collect more Paris 2024 footage.** Neither
addresses the measured failure. A bigger training set drawn from the same two
broadcasts leaves the transfer problem exactly where it is.

**Next: a fourth match from a different broadcaster, added to training.** With
two production styles in train, the model has to learn what is common to
basketball rather than what is common to Paris 2024. This is the smallest change
that attacks the thing actually measured.

The design limitation behind this was recorded when the split was built: match03
is both the only FIBA match and the entire test set, so the model never sees that
production style before being graded on it. That was known. Its cost is now
measured at roughly the difference between 60% and 54%.
