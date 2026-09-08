# Phase 3 pilot: first pass through the clip pipeline

Result of running the whole chain on ~22 minutes of `match01` (the Paris 2024
USA-Serbia semi-final broadcast already used for Phase 2).

**Verdict: the machinery works, the dataset does not yet.** Every stage behaved
as designed on real data. There are nowhere near enough events to train on.

---

## What was marked

70 marks over 02:01-22:05 of video. The first ~10 minutes were pre-game (player
entries, graphics, crowd), so this is roughly **12 minutes of actual basketball**.

| class | marks |
|---|---|
| `three_pointer` | 11 |
| `free_throw` | 7 |
| `two_pointer` | 5 |
| `dunk` | 3 |
| `steal` | 3 |
| `block` | 1 |
| `exclude` | 40 |
| **events** | **30** |

**2.5 events per minute of live play.** Better than the ~1.3/min assumed when
planning, so the full match should yield roughly 200 events.

The 40 excludes are inflated: the first session used `x` for crowd shots,
close-ups and graphics before the guide made clear that non-game footage needs no
key at all. The second session used it only for misses, as intended.

## What was cut

```
30 event clips + 41 background = 71 clips, 16 frames each at 8 fps
99 MB of JPEGs -> 2.3 MB feature cache (71 x 16 x 512)
```

### The Phase 2 filter earned its keep

**79 of 120 background candidates were rejected as not-game — 66%.** Almost all
of it was the pre-game stretch: entries, line-up graphics, crowd. None of it
entered the dataset labelled "basketball with nothing happening", and none of it
cost a keystroke. This was the entire justification for building Phase 2 as a
real pipeline component rather than a teaching exercise, and it is now measured.

### Clip timing is correctly calibrated

Spot-checked a dunk and a three-pointer frame by frame. **The action lands at
frames 9-11 of 16.** With `--pre 1.5` that puts the keypress at roughly `t-0.4s`
into the window, which matches the assumed human reaction delay almost exactly.
Run-up before, follow-through after. No adjustment needed.

## Bugs found by running it

**Background was being sampled from the entire 108-minute video** while only the
first 22 minutes had been reviewed. Every unmarked dunk in the unwatched 86
minutes would have become a `none` clip — label noise aimed squarely at the
rarest classes, and invisible in every metric, because those clips look exactly
like what they were mislabelled as.

Fixed: `mark.py` records the watch position on quit, and `cut.py` refuses to
sample background beyond it (`reviewed_until_for`, overridable with
`--reviewed-until`). Covered by `tests/test_ingest_cut.py`.

This is the second time in this project that the dangerous bug was one that would
have made the numbers look *better*. Both were caught by reasoning about the data
rather than by anything failing.

## Where the dataset stands

```
71 clips across 1 match
  two_pointer      5      block            1
  three_pointer   11      steal            3
  dunk             3      none            41
  free_throw       7

dumb baseline: always say 'none' -> 41/71 = 57.7%
```

**All six event classes are under 20 examples.** A class that thin produces a row
of the confusion matrix built from single digits — noise, not a result. Training
on this would produce a model that says `none` to everything and scores 58%.

The split fell back to chronological-within-match with the intended warning,
since one match cannot be split by match.

## Decision: continue marking, do not train yet

Not a pipeline problem. Every component did its job; there is simply not enough
data behind it.

**Next: finish marking `match01`.** ~86 minutes of video remain. At the observed
2.5 events/minute of live play that should give roughly 200 events total, which
puts `three_pointer`, `two_pointer` and `free_throw` in usable shape.

`dunk`, `block` and `steal` will still be thin, and that is structural rather
than fixable by finishing this match — spec section 6.1 resolves it with
highlight reels used surgically for rare classes. Worth noting that the NCAA
broadcast dataset (257 games) annotates dunks and steals but declined to annotate
blocks at all, which is independent evidence that `block` is the hardest class
here and will need targeted collection.

Then match02 and match03, because **the test score stays dishonest until there
are three matches to split by.**
