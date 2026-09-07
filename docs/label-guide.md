# Frame label guide — Phase 2

Three classes plus a skip. When a frame is ambiguous, ask what the frame is *about*, and
stay consistent — consistency matters more than any individual call, because the model
learns whatever pattern you apply, including a wrong one applied evenly.

## `g` — game

Live play seen from the broadcast camera. Court readable, players in position, the shape
of the game visible.

Includes inbounds, free throws, players moving up court, and dead-ball moments still shot
from the game camera.

## `c` — crowd

Anything shot away from live play. Audience, bench, coaches, huddles, referees conferring,
a player's face in close-up, celebration cutaways, the tunnel, pre-game footage of fans
outside the venue.

## `x` — graphic

Frames dominated by rendered content rather than the arena. Station idents, scoreboard and
stat cards, sponsor stings, title screens, black frames, transition wipes.

## `s` — skip

You genuinely cannot tell. Press it and move on.

This is not a fourth class and it is not laziness. A guessed label teaches the model
something false, and a guess that lands in the test set makes the accuracy number lie.
400 honest labels beat 542 with 80 guesses in them.

---

## Rules for the hard cases

These are the calls that actually come up in this broadcast. Decide once, apply every time.

**Score bug over live play → `game`.** Nearly every broadcast frame carries an overlay.
An overlay alone never makes a frame `graphic`.

**Tight close-up of a player during play → `crowd`.** If the court is not readable, the
frame carries no game information, which is exactly what the filter is for.

**Replay of live action → `game`.** Replays are deferred to Phase 4. For a single frame,
`game` is the honest call because that is what the pixels show. Do not try to identify
replays by eye — that is the job of a model that can see time.

**Loose-ball scramble shot tight, court floor visible → `game`.** The floor and multiple
players make it readable as play. Only go `crowd` when a single person fills the frame.

**Pre-game fan footage outside the venue → `crowd`.** Not the arena, not the game, but
it is people rather than rendered graphics.

**Advert board filling the shot → `game` if players are visible, otherwise `graphic`.**

---

## Target

At least **300 labelled frames**, and at least **40 in the smallest class**.

An Olympic broadcast has no commercial breaks, so `graphic` is the class most likely to
run short. If it lands under 40, the fix is not to relabel more loosely — it is to merge
`graphic` into a two-class problem (`game` vs `not-game`), which is what the Phase 5
pipeline actually consumes. That decision gets made from the count, not from a guess.
