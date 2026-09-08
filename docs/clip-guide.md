# Clip label guide (Phase 3)

Rules for marking events in a full match with `ingest/mark.py`. Read this **before** your
first marking session and keep it open beside the window.

Why a guide exists at all: two identical clips labelled differently cancel each other to
noise, and no amount of extra data repairs that. Inconsistent labels are the one dataset
problem you cannot fix later by collecting more. A model can only ever be as consistent as
the person who taught it.

---

## The keys

| key | class | key | class |
|---|---|---|---|
| `1` | `two_pointer` | `4` | `free_throw` |
| `2` | `three_pointer` | `5` | `block` |
| `3` | `dunk` | `6` | `steal` |
| | | `x` | **exclude** — see below |

There is no key for `none`. Background clips are sampled automatically from the stretches
you marked nothing in, so you never spend a keystroke on dead time.

---

## When to press

**Press the instant the outcome is visible.** Ball through the net. Defender's hand on the
ball. Ball settled in the stealer's hands.

Do not anticipate, and do not wait to admire it. Consistency of *timing* matters as much as
consistency of *class*, because the clip is cut from where you pressed.

You are naturally about 0.4s late — that is human reaction time and it is fine. The cutter
expects it: a mark at `t` cuts the window `[t - 1.5s, t + 0.5s]`, sitting mostly **behind**
your keypress, because the useful motion (the drive, the gather, the rise) happened before
the outcome you reacted to.

---

## The classes

### `1` two_pointer
A **made** field goal from inside the arc that is not a dunk: layup, jump shot, hook,
floater, tip-in, put-back.

### `2` three_pointer
A **made** shot with the shooter's feet **entirely behind the arc** at release.

Judge by the feet, not by how far away it looked. If the feet are hidden, or the shooter is
straddling the line, **press `x`, not a guess.** Distinguishing 2 from 3 properly needs
court-line geometry we deliberately did not build; the design accepts unreliability near the
arc, and a wrong label costs far more than a missing one.

### `3` dunk
Ball driven down **through** the rim with the hand at or above rim level.

Not a dunk: finger-roll, high layup off the glass, an alley-oop that is caught and laid in.
A rim-hang or a missed dunk that rattles out — `x`.

### `4` free_throw
A **made** free throw. Clock stopped, shooter alone at the line, nobody contesting.

### `5` block
A defender contacts the ball **on its way up**, while the shot is still rising.

Not a block: contact after the ball peaks (that is goaltending or a rebound), a hand in the
shooter's face with no contact on the ball, or a stripped dribble (that is a steal).

### `6` steal
Possession **actually transfers** from offence to defence — the defender comes away with it.

Not a steal: a deflection that goes out of bounds, a jump ball, a loose-ball scramble nobody
clearly wins. Possession must visibly change hands.

---

## `x` — exclude

`x` marks a moment as **"leave this out of the dataset entirely"**. Not an event, and not
available as background either. It punches a hole in the timeline.

Press `x` for:

- **A missed shot of any kind.** This is the most common use and the most important. A missed
  three looks almost identical to a made three. If you leave it unmarked it gets sampled as
  background, and you have taught the model that the same picture is both `three_pointer` and
  `none`. That contradiction is worse than either label alone.
- A shot you cannot classify (feet hidden near the arc, obscured by a player).
- A foul, a scramble, or anything chaotic you would not want in a highlight.
- Anything you are unsure about.

**When in doubt, press `x`.** It costs you nothing — the pilot does not need every event, it
needs consistent ones.

---

## Two rules that protect the dataset

**Never mark a replay or slow-motion.** It is the same event you already marked, shown again
at an unrepresentative speed and camera angle. Live play only. If you are not sure whether
you are watching live play or a replay, you are watching a replay — live basketball has a
running clock and a wide camera.

**Never mark twice for one moment.** If two classes apply, use this precedence, highest first:

```
dunk  >  block  >  steal  >  three_pointer  >  two_pointer  >  free_throw
```

A blocked dunk attempt is a `block`. A dunk after a steal is two separate moments a second
apart — mark both, at their own timestamps.

---

## Checking yourself

Spec §6.2 asks for a consistency check: **re-mark 100 previously-marked moments a week later
and compare.** Below about 90% agreement with your past self means this guide needs tightening,
not that the model needs more data.

If you find yourself hesitating on the same kind of moment repeatedly, that is a gap in this
document. Say so and we will add a rule rather than leaving it to your mood on the night.
