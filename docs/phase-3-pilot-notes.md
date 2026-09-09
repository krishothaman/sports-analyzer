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

---

# Update: the scoreboard reader, and what it changed

Written after wiring the broadcast score bug into the pipeline as a second,
automatic source of marks.

## The idea

Three of the six classes put points on the board. A broadcast displays those
points in a fixed font at a fixed position, so they can be read by template
matching -- no model, no training data, no labelling. Every score change is a
scoring play, and its type follows from the size of the jump: +1 free throw,
+2 two-pointer, +3 three-pointer.

## Result on two matches

| | match01 | match02 |
|---|---|---|
| readable score readings | 9,422 | 9,537 |
| scoring plays detected | 68 | 84 |
| points accounted for | 145 / 186 (78%) | 172 / 185 (93%) |
| final score tracked to | 95-91 | 87-98 |

Both final scores are exactly right (USA-Serbia semi-final, and the France-USA
gold medal game). That is the validation that matters: an error anywhere in the
chain compounds forward, so arriving at the true final score means the reader
did not drift.

The gap between 78% and 93% is a template fix, not a rule change -- see below.
The remaining shortfall is **missed** baskets, not invented ones, which is the
direction this component is deliberately biased in.

**match01 went from 30 event clips to 84 with no additional labelling.**

## Three bugs, all found by looking rather than by a test failing

**The score walks backwards through highlight recaps.** The broadcast cuts to
montages that replay earlier moments with the old score on screen. A reader that
trusts every number walks the score back down and then back up, inventing a
basket at every step: a naive pass claimed 186 points of scoring plays in a match
that finished 95-91. The fix is a rule, not a threshold -- *a live score never
goes down*, so any reading below the confirmed score is not the game.

**One example of the digit `9`.** The template set had between one and five
examples per digit, and `9` had one. match02 read `39` as `30`. Harvesting 24
more labelled glyphs from cells read by hand took match02 from unusable to
17/17 on known scoreboards, and lifted match01's coverage from 78% to 93% of
points. The lesson is that the reader's failures were silent refusals, not wrong
answers -- it was working as designed and still losing a fifth of the data.

**A dunk is also a two-pointer.** A hand-marked `dunk` and the reader's
`two_pointer` are the same basket, and the de-dup only compared labels, so both
survived: two clips of identical footage under conflicting labels, aimed at the
class with the fewest examples to spare. Now any hand-marked *scoring* label
suppresses the reader's mark. Blocks and steals deliberately do not suppress --
they are not baskets, and the fast break that follows a steal is a real separate
event.

## Division of labour this establishes

| class | source |
|---|---|
| `two_pointer`, `three_pointer`, `free_throw` | scoreboard, automatic |
| `dunk`, `block`, `steal` | hand only -- a dunk reads as a two-pointer, and blocks and steals change no score at all |
| `none` | auto-sampled, capped at the reviewed watch position |

This is the whole reason the reader was worth building: it removes the ~200
common events per match that are tedious and leaves only the ~20 rare ones,
which is where a human's attention was always the scarce resource.

## Dataset after the update

```
187 clips across 1 match
  two_pointer     29      block            1
  three_pointer   29      steal            3
  dunk             3      none           103
  free_throw      19

dumb baseline: always say 'none' -> 103/187 = 55.1%
```

`two_pointer` and `three_pointer` are now usable. `free_throw` is borderline at
19. `dunk`, `block` and `steal` remain thin, and no amount of scoreboard reading
will change that -- only the rare-class skim will.

Background is still capped at match01's 22 reviewed minutes, and the frame
filter rejected 233 of 336 candidates there (69%), because that stretch is
mostly pre-game. The cap is correct and should stay: sampling past the watch
position would turn every unmarked block into a `none` clip.

## Still open

- **match03 is a different broadcast** (FIBA World Cup Qualifier, JPN-QAT, 30fps).
  Different score bug, different font, and the bar animates -- a sponsor banner
  slides in and pushes the digits outward, so fixed crop coordinates do not hold.
  Needs a second layout and its own template set.
- **Torch is CPU-only** (`2.11.0+cpu`). Correctness is unaffected, speed is not.

---

# Update: two matches in

After the match02 skim (two quarters, rare classes only) and its cut.

```
446 clips across 2 matches
  two_pointer     63      block            3
  three_pointer   54      steal            8
  dunk            12      none           267
  free_throw      39

dumb baseline: always say 'none' -> 267/446 = 59.9%
```

| class | pilot | +match01 auto | +match02 | source |
|---|---|---|---|---|
| `two_pointer` | 5 | 29 | 63 | scoreboard |
| `three_pointer` | 11 | 29 | 54 | scoreboard |
| `free_throw` | 7 | 19 | 39 | scoreboard |
| `dunk` | 3 | 3 | 12 | hand only |
| `steal` | 3 | 3 | 8 | hand only |
| `block` | 1 | 1 | 3 | hand only |

**Three classes crossed into usable territory; the three without an automatic
source did not.** That is the same split every time, and it is not a coincidence:
`dunk`, `steal` and `block` grow only by what a person actually watched. Two
quarters of attention bought 16 clips. The scoreboard bought 195 across the same
two matches while nobody watched anything.

## Cross-checking hand marks against the scoreboard

A hand-marked dunk must coincide with a +2, and a block or steal must coincide
with nothing. Comparing the two sources catches mis-marks for free:

- 9 of 11 dunk marks landed within +/-1s of a detected +2, with offsets centred
  on zero -- confirming both the marks and the measured 2.86s graphic lag.
- 1 dunk had no rim action anywhere in an 11s window. A stray keypress; removed.
- 1 dunk had none in a 6s window. Also removed.
- 1 dunk matched a real basket the reader had **missed** (the graphic was hidden
  by the replay). The mark was right and the reader was wrong -- which is the
  expected direction, since the reader is biased towards missing.
- Every block and steal correctly matched no score change.

Two bad marks in 82 is a ~2.4% error rate overall, but 2 in 11 dunks is ~18% --
concentrated in the class least able to absorb it. Worth doing again per match.

## Graphic lag is measured per match, not assumed

match01 measured +2.68s over 11 marks; match02 measured +2.86s over 26. The
`DEFAULT_LAG` fallback of 2.7s is only for a match with no hand marks yet, and it
carries a warning, because a clip is two seconds long: a lag wrong by more than
that cuts footage the event does not appear in at all. It is not transferable
across producers -- match03 is a different broadcaster and needs its own.

## Still one match short

The split still falls back to chronological-within-each-match, so every match
appears on both sides and the score measures 'can it do this on a venue it has
already seen'. match03 is what makes it honest.
