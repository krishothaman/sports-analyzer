# Phase 3: Clip Data Pipeline

## Context

Phase 2 shipped a frame filter at 99.36% that answers *"is this basketball?"* about one still
image. It cannot answer *"what just happened?"* — a frozen frame of a player near the rim is
equally a dunk, a layup, a block or a rebound. That question needs **motion**, so the unit of data
becomes a **clip**: 2 seconds, 16 frames at 8 fps.

Phase 3 builds the machine that produces labelled clips. Phase 4 builds the model that reads them.
Spec §9 names Phase 3 the attrition risk — not hard, tedious — so every decision below is chosen to
cut the owner's keystrokes rather than to be elegant.

**Spec:** `docs/superpowers/specs/2026-09-06-sports-highlight-analyzer-design.md`

### Decisions taken this session

| Decision | Choice | Why |
|---|---|---|
| Labelling workflow | **Mark-then-cut** | Cut-then-label means judging ~3,250 clips per match, ~95% of them background. Marking spends attention only on the ~5% that matters. |
| First milestone | **Pilot on match01 only** | Prove the pipeline end-to-end through Phase 4 before committing hours. Poor accuracy is an acceptable pilot outcome; broken machinery discovered after 10 hours of labelling is not. |
| Bootstrap dataset | **None** (researched, rejected) | SpaceJam is MIT-licensed and 16-frame, but player-cropped with motion-primitive classes — the exact distribution shift spec §6.1 forbids. NCAA/Stanford is broadcast footage with a near-matching class list, but is a 112 GB YouTube scrape from 2016 with heavy link-rot and no stated licence. Kept as a scale-up option, not a dependency. |

**Resolves spec §10 open question 1.** Also worth recording: NCAA annotates every shot type, dunks
and steals, but has **no `block` class**. Independent confirmation of §4's "known-hard classes"
warning — expect `block` to be the weakest class.

### Global constraints

- Owner runs every marking, cutting, caching and training command. Claude writes and explains first.
  The owner never runs a script they cannot roughly describe.
- **No `Co-Authored-By` trailer on any commit.**
- Repo is public. Video, frames, clips, features and manifests stay gitignored (`data/` already is).
- PowerShell-compatible commands only.
- 8 GB VRAM, 15.7 GB RAM. Clip storage must stay in the hundreds of MB for the pilot.


### Amendment made during Task 1 — the `x` exclude key

Writing the label guide exposed a hole in the approved plan. The plan had six event keys and
nothing else, with everything unmarked becoming background. But a **missed** shot is visually
almost identical to a made one. Left unmarked it would be sampled as background, teaching the
model that the same picture is both `three_pointer` and `none` — a direct contradiction, and
worse than either label on its own.

So `mark.py` gains a seventh key, `x`, meaning *leave this moment out of the dataset entirely*:
not an event, and not available as background. It punches a hole in the timeline. It also
absorbs every "I cannot tell" case, which is what makes "when in doubt, press `x`" a rule the
owner can actually follow.

`x` marks are written to the events CSV like any other so they participate in the guard band,
and are dropped by `cut.py` rather than cut into clips.

---

## Architecture

```
ingest/mark.py       scrub match video, keys 1-6 mark an event   -> data/events/<match>.csv
ingest/cut.py        events.csv -> 2s clips + auto-sampled background, Phase 2 filtered
docs/clip-guide.md   exact testable rules for all 6 classes  (written BEFORE any marking)
clips/data.py        clip manifest -> split BY MATCH -> 16x512 feature cache -> DataLoaders
tests/test_ingest_cut.py, tests/test_clips_data.py
```

`clips/` deliberately mirrors `frames/` and `mnist/`: `data.py` means what it already means.
`models/backbone.py` is reused **unchanged** — that was the point of putting it outside `frames/`.

### Three design points that carry the phase

**1. The clip sits behind the mark, not on it.** The owner presses a key when they *see* the event,
which is already ~0.4s late, and the informative motion (the run-up, the approach) happened *before*.
So a mark at `t` cuts the window `[t - pre, t + post]` with `pre=1.5`, `post=0.5`. Both are CLI flags,
because this is a guess that wants tuning after looking at real cut clips.

**2. Seek approximately, cut exactly.** Phase 2 established that `cap.set()` seeking is unreliable
across codecs. That is fine for *marking* — being a few frames off while eyeballing a 2-second event
changes nothing — so `mark.py` uses `cap.set()` and stays responsive. `cut.py` needs exact frames, so
it uses the sequential `grab()`/`retrieve()` pattern from `frames/extract.py`, walking the video
**once** and emitting every clip in that single pass. Re-seeking per clip would take hours.

**3. Background is sampled, never marked.** Walk the timeline in 2s steps, discard any window within
a guard band (`--guard 4.0`) of a mark, and keep a random subset sized to `--bg-ratio` × the event
count. Each candidate's middle frame goes through **the Phase 2 frame classifier**; anything not
`game` is dropped, so adverts, crowd shots and graphics never enter the dataset as "basketball with
nothing happening". Event clips are **not** filtered — the owner saw them; a filter false-negative
would silently delete a real dunk.

### Class keys

| key | class | key | class |
|---|---|---|---|
| `1` | `two_pointer` | `4` | `free_throw` |
| `2` | `three_pointer` | `5` | `block` |
| `3` | `dunk` | `6` | `steal` |

`none` has no key. `u` undoes the last mark, `q` saves and quits. Marks are appended to a CSV after
every keypress (the Phase 2 crash-safety pattern from `frames/label.py`), and a mis-timed mark is one
editable row — nothing is baked in until `cut.py` runs.

---

## Tasks

### Task 1 — Clip label guide *(Claude writes, owner reviews)*

`docs/clip-guide.md`, written **before** any marking exists to be inconsistent. Spec §6.2: two
identical clips labelled differently cancel to noise, and no amount of extra data repairs it.

Must state testable rules for: dunk vs layup (ball driven down through the rim while the hand is at
or above it); block vs contest vs rebound (defender contacts the ball on its way *up*); steal vs
deflection out of bounds (possession must actually transfer); `two_pointer` vs `three_pointer` (call
it by the shooter's feet if visible, **otherwise do not mark it at all** — §4 accepts unreliability
near the arc, and an unmarked event costs far less than a wrong one); free throw (uncontested, from
the line, clock stopped).

Plus two rules that protect the dataset:
- **Never mark a replay or slow-motion.** It duplicates an event already marked, with unrepresentative
  footage. Live play only.
- **When unsure, do not mark.** There is no cost to a missed event during the pilot; there is a real
  cost to a wrong label, which becomes noise in training and a lie in the test set.

Commit before Task 2.

### Task 2 — `ingest/mark.py` *(Claude writes, owner runs)*

**Files:** create `ingest/__init__.py`, `ingest/mark.py`, `tests/test_ingest_mark.py`

An OpenCV window playing the match. Controls: `space` play/pause, `d`/`a` seek ±5s, `w`/`s` seek
±30s, `1`-`6` mark, `u` undo, `q` quit. Letter keys rather than arrows — arrow key codes differ
across platforms in OpenCV and would silently do nothing on Windows.

Playback is `cv2.waitKey(delay)` inside the read loop, so keys register mid-play without a separate
input thread. Header bar shows the running clock, per-class mark tallies and the key legend, matching
`frames/label.py`'s annotate style.

Appends to `data/events/<match_id>.csv` (`match_id, timestamp_sec, label`) after **every** keypress.
Reruns load existing marks so a session can be split across evenings.

Testable without a GUI: extract mark bookkeeping (append, undo, tally, load-existing) into pure
functions and test those; the OpenCV loop stays a thin shell.

### Task 3 — Mark match01 *(owner, ~30-45 min)*

```
python -m ingest.mark data/video/match01.mp4 --match-id match01
```

Target ~100-150 marks from one 108-minute game. Realistic per-game yield is roughly 60 two-pointers,
25 threes, 40 free throws, 8 steals, 5 dunks, 5 blocks — so `dunk` and `block` will be badly
under-supplied after one match. **That is expected and is not a reason to stop the pilot.** Spec §6.1
resolves it later with highlight reels used surgically for rare classes only.

### Task 4 — `ingest/cut.py` *(Claude writes, owner runs)*

**Files:** create `ingest/cut.py`, `tests/test_ingest_cut.py`; modify `frames/predict.py`

Reads every `data/events/*.csv`, plans the background sample, then makes **one sequential pass** over
the video emitting all clips.

Each clip is 16 JPEGs at `data/clips/<match_id>/<clip_id>/00.jpg`…`15.jpg`, resized to short side 256
so the existing 224 centre crop still works. JPEGs rather than a tensor for two reasons: the owner can
*look* at them, and the pending squash-vs-crop preprocessing experiment can be rerun without
re-cutting. Budget ~290 KB/clip → well under 200 MB for the pilot.

Writes `data/clip_manifest.csv`: `clip_id, match_id, start_sec, label, n_frames`. Clips truncated by
the start or end of the video are dropped rather than padded.

**Reuse, do not duplicate:** `frames/predict.py:classify()` currently takes file paths. Refactor its
inner loop into a function that accepts already-decoded images so `cut.py` can filter in-memory frames
through the Phase 2 model, with `classify()` calling it. The 20 existing tests must stay green.

Prints a composition table on exit — per-class counts, background count, how many background
candidates the frame filter rejected. That rejection count is the first real measurement of whether
Phase 2 is earning its keep.

### Task 5 — `clips/data.py` + feature cache *(Claude writes, owner runs)*

**Files:** create `clips/__init__.py`, `clips/data.py`, `tests/test_clips_data.py`

`CLASSES` = the 6 events + `none`. Loads the clip manifest, splits, caches features, returns
DataLoaders — the same shape as `frames/data.py`, including its stale-cache `SystemExit` message.

**Splitting is by `match_id`, never by clip** (spec §6.3) — the same leakage lesson as Phase 2, one
level up. Clips from one game share arena, lighting and jerseys; split by clip and the score measures
"can it recognise this arena", scoring ~95% and collapsing on new footage.

With a single pilot match that split is impossible. So `clips/data.py` ships **both**: `match_split()`
is the real one and is used whenever ≥3 matches exist; below that it falls back to a chronological
within-match split and **prints a loud warning that the resulting number is optimistic and not a
generalisation estimate**. The fallback must be impossible to use accidentally without seeing that.

Feature cache: each clip's 16 frames go through the frozen backbone as one batch → `[16, 512]`;
the cache is `[N, 16, 512]`. ~8 MB for the pilot, ~230 MB at full scale — matching spec §5.5. Valid
for the same reason as Phase 2's: the backbone is frozen, so a clip's features are permanent. Dies at
Phase 6.

### Task 6 — Pilot readout and go/no-go *(both)*

Owner runs the full chain; Claude reads the composition table against the **dumb baseline** (spec §7):
always predicting the most common class. Any Phase 4 model that cannot beat it has learned nothing.

Record in `docs/phase-3-pilot-notes.md`: marks per class, background kept vs filtered, clip count,
cache size, and which classes are too thin to measure. Then decide explicitly whether to scale to
matches 2-4 plus highlight reels, or fix the pipeline first.

---

## Verification

- `pytest tests/ -v` — the 20 existing tests stay green, plus new ones. The two that matter:
  - **no `match_id` appears in both train and test** (the leakage tripwire, spec §6.3)
  - **no background clip window falls within the guard band of any mark** — otherwise the model is
    taught that a dunk's run-up is `none`, which poisons the very class it should trigger
  - plus: window arithmetic (a mark at `t` yields 16 frames spanning `[t-1.5, t+0.5]`), undo/append
    bookkeeping, and every class name maps to an index
- `python -m ingest.mark ...` — window opens, plays, keys register during playback, `data/events/match01.csv`
  grows one row per keypress, quitting and reopening resumes.
- `python -m ingest.cut --match-id match01` — prints the composition table; spot-check 3 cut clips by
  eye and confirm the event is actually inside the 2 seconds. **If events land at the very end of the
  clip, `--pre`/`--post` need adjusting — this is the most likely thing to be wrong on the first run.**
- `python -m clips.data` — prints `cached N x 16 x 512 features`, and prints the single-match split
  warning.
- `git push` — the branch is currently 13 commits ahead of origin and has not been pushed this session.

## Out of scope

The clip **model** is Phase 4. This phase ends with a labelled, cached, honestly-split dataset and a
recorded decision about whether to scale it.
