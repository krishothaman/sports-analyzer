"""Read the broadcast score bug and turn score changes into event marks.

The broadcast already knows what happened. When a basket is made the number in
the corner goes up -- by 1, 2 or 3 -- and that is not an inference about the
video, it is the scoreboard. Reading it turns three of the six classes into
automatic labels with exact timing:

    +1 -> free_throw     +2 -> two_pointer     +3 -> three_pointer

What it cannot do, and why hand-marking does not go away:

  * block and steal do not change the score. They still need eyes.
  * a dunk is a two_pointer as far as the scoreboard is concerned. Separating
    them needs a human pass over the auto-found 2-point plays.
  * the geometry below is specific to one broadcast's graphics package. A
    different production needs the boxes re-measured (see LAYOUTS).

Reading the digits uses template matching rather than a general OCR engine.
The font is fixed, the position is fixed, and there are ten possible glyphs --
a general solution here would be more to install, more to go wrong, and no more
accurate.
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np

from ingest.mark import events_path, load_marks, save_marks

# Measured by hand on an 854x480 stream of the Paris 2024 basketball graphics.
# name_box is a large white rectangle that is only white when the bug is on
# screen, which is what tells us whether the numbers mean anything: the graphic
# vanishes during replays and close-ups, and the pixels behind it are crowd.
LAYOUTS = {
    "paris2024": {
        "frame_width": 854,
        "name_box": (39, 56, 95, 138),
        # Tight, and verified against seven hand-read scoreboards. Wider boxes
        # pull in the neighbouring cell and the graphic's own rules.
        "home": (40, 55, 145, 172),
        "away": (59, 71, 145, 172),
        "present_threshold": 130.0,
    },
}

POINTS_TO_LABEL = {1: "free_throw", 2: "two_pointer", 3: "three_pointer"}

# Machine marks live in their own file and never touch the hand-marked one.
# Learned the hard way: merging them in made the owner's work unrecoverable
# when the detector turned out to be over-firing, and a labelling tool must
# never be able to damage labels a human produced.
AUTO_SUFFIX = ".auto.csv"

TEMPLATES_PATH = os.path.join("ingest", "digit_templates.json")
GLYPH_SIZE = (12, 16)           # width, height a segmented digit is normalised to
DIGIT_THRESHOLD = 140           # grey level separating white digits from teal
# Measured, not guessed: across every digit in seven hand-read scoreboards the
# worst correct match scored 57. 70 clears that with margin while still refusing
# a glyph that matches nothing. Comparing greyscale beats comparing binarised
# masks here -- the binary version misreads a 6 as a 0.
MAX_GLYPH_DISTANCE = 70.0

# Shape guards, measured on this broadcast: every clean digit is 8-9 px tall and
# 3-6 px wide. Anything outside these bounds is a bright border bleeding in or
# two digits touching, and the whole reading is refused rather than guessed at.
# Skipping a frame costs nothing -- the score is sampled once a second and holds
# for many seconds -- while a bad read invents a basket that never happened.
# A row this full of white is the cell's border rule, not part of a digit.
BORDER_ROW_FILL = 0.75

DIGIT_HEIGHT = (7, 11)
DIGIT_WIDTH = (2, 8)

# How stale the previous reading may be before a score change stops being
# attributable to a single basket. Two free throws are roughly 20s apart, so a
# gap under this cannot hide one.
MAX_GAP = 8.0


def auto_events_path(match_id):
    return os.path.join(os.path.dirname(events_path(match_id)),
                        f"{match_id}{AUTO_SUFFIX}")


def load_layout(name, frame_width):
    """Return the layout, scaled if the video is not the width it was measured at."""
    layout = dict(LAYOUTS[name])
    scale = frame_width / layout["frame_width"]
    if abs(scale - 1.0) > 0.01:
        for key in ("name_box", "home", "away"):
            layout[key] = tuple(int(round(v * scale)) for v in layout[key])
    return layout


def crop(frame, box):
    y0, y1, x0, x1 = box
    return frame[y0:y1, x0:x1]


def is_present(frame, layout):
    """Is the score bug on screen at all?

    During replays and close-ups the graphic is hidden and those boxes show
    crowd. Reading them then would invent score changes out of noise.
    """
    box = cv2.cvtColor(crop(frame, layout["name_box"]), cv2.COLOR_BGR2GRAY)
    return float(box.mean()) > layout["present_threshold"]


def segment_digits(cell):
    """Split a score cell into normalised digit bitmaps, or None if it looks wrong.

    Returning None is the safe answer. The alternative -- handing a merged
    two-digit blob to the matcher -- produces a confident wrong digit, which
    becomes a score change, which becomes an event that never happened.
    """
    grey = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    mask = (grey > DIGIT_THRESHOLD).astype(np.uint8)

    # The cells are separated by a bright horizontal rule that bleeds into the
    # crop. Left in, it fuses the digits into one wide blob ("70" becomes a
    # single 27-pixel span) and the whole reading is thrown away. A digit never
    # spans the full width of the cell, so a nearly-solid row is always border,
    # never data -- blank those rows rather than cropping tighter, which clips
    # digits that sit a pixel low.
    solid = mask.mean(axis=1) > BORDER_ROW_FILL
    mask[solid] = 0

    lit_rows = np.where(mask.sum(axis=1) > 0)[0]
    if len(lit_rows) == 0:
        return None
    height = lit_rows[-1] - lit_rows[0] + 1
    if not DIGIT_HEIGHT[0] <= height <= DIGIT_HEIGHT[1]:
        return None

    spans, run = [], None
    columns = mask.sum(axis=0)
    for x, count in enumerate(columns):
        if count > 0 and run is None:
            run = x
        elif count == 0 and run is not None:
            spans.append((run, x))
            run = None
    if run is not None:
        spans.append((run, len(columns)))

    if not spans or any(not DIGIT_WIDTH[0] <= x1 - x0 <= DIGIT_WIDTH[1]
                        for x0, x1 in spans):
        return None

    glyphs = []
    for x0, x1 in spans:
        sub = mask[lit_rows[0]:lit_rows[-1] + 1, x0:x1]
        glyphs.append(cv2.resize(sub * 255, GLYPH_SIZE, interpolation=cv2.INTER_AREA))
    return glyphs


def load_templates(path=TEMPLATES_PATH):
    """{digit: [example, ...]} -- several bitmaps per digit, not one average.

    Averaging the examples together seemed tidier and was wrong. A '0' drawn on
    its own and a '0' as the second digit of '10' sit a fraction differently in
    the cell; blending them produced a blurred template that matched neither
    well enough, and the reader silently refused every standalone zero for the
    whole first quarter. Keeping the examples apart and taking the closest one
    costs nothing and tolerates that variation.
    """
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    return {digit: [np.array(rows, dtype=np.uint8) for rows in examples]
            for digit, examples in blob.items()}


def read_glyph(glyph, templates):
    """Best-matching digit, or None if nothing is close enough.

    Refusing to guess matters more than it looks. A wrong digit invents a score
    change, which invents an event that never happened -- and unlike a missed
    event, a fabricated one actively teaches the model something false.
    """
    best, best_distance = None, None
    for digit, examples in templates.items():
        for template in examples:
            distance = float(np.abs(glyph.astype(np.int16)
                                    - template.astype(np.int16)).mean())
            if best_distance is None or distance < best_distance:
                best, best_distance = digit, distance

    if best_distance is None or best_distance > MAX_GLYPH_DISTANCE:
        return None
    return best


def read_cell(cell, templates):
    """Read one score cell as an integer, or None if any digit is unreadable."""
    glyphs = segment_digits(cell)
    if glyphs is None or len(glyphs) > 3:
        return None

    text = ""
    for glyph in glyphs:
        digit = read_glyph(glyph, templates)
        if digit is None:
            return None
        text += digit
    return int(text)


def read_score(frame, layout, templates):
    """(home, away) or None when the bug is hidden or unreadable."""
    if not is_present(frame, layout):
        return None
    home = read_cell(crop(frame, layout["home"]), templates)
    away = read_cell(crop(frame, layout["away"]), templates)
    if home is None or away is None:
        return None
    return home, away


def scan(video_path, layout, templates, every=1.0, until=None, progress=True):
    """Walk the video once, returning [(seconds, home, away)] for readable frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    last = int(until * fps) if until else (total - 1 if total else None)
    step = max(1, int(round(fps * every)))

    readings = []
    index = 0
    while last is None or index <= last:
        if not cap.grab():
            break
        if index % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                score = read_score(frame, layout, templates)
                if score is not None:
                    readings.append((index / fps, score[0], score[1]))
            if progress and index % (step * 60) == 0:
                print(f"  scanned {index / fps / 60:.1f} min, {len(readings)} readable",
                      end="\r", flush=True)
        index += 1

    cap.release()
    return readings


def score_changes(readings, confirm=2, max_gap=MAX_GAP):
    """[(seconds, points)] for every confirmed increase in either team's score.

    Four guards, all of them there to avoid inventing events:

    * a new score must hold for `confirm` consecutive readings. A single
      misread digit would otherwise fire a change and then fire again when the
      real value came back.
    * only increases of 1 to 3 count. Anything else is a misread or a graphic
      correction, not a basket.
    * both teams changing at once is impossible, so it is dropped.
    * the old score must have been visible within `max_gap` seconds. This is
      the subtle one. The graphic hides during replays -- which is exactly what
      follows a basket -- and if it stays hidden across two free throws the
      score reappears +2. Marked naively that becomes a `two_pointer` clip of
      footage containing no two-pointer: a confident, plausible, false label,
      which is far worse than a missed event. When the gap is too long to rule
      that out, the new score is accepted as current and nothing is marked.
    """
    changes = []
    current, current_time = None, None
    pending, pending_count, pending_time = None, 0, None

    for seconds, home, away in readings:
        score = (home, away)

        if current is None:
            current, current_time = score, seconds
            continue

        # A live score never goes down. Anything below the confirmed score is
        # not the game: broadcasts cut to highlight recaps that replay earlier
        # moments with the old score on screen, and a naive reader walks that
        # score back up again, inventing a basket for every step. Ignoring
        # low readings outright skips recaps without needing to detect them.
        if home < current[0] or away < current[1]:
            continue

        if score == current:
            current_time = seconds
            pending, pending_count = None, 0
            continue

        if score != pending:
            pending, pending_count, pending_time = score, 1, seconds
        else:
            pending_count += 1

        if pending_count < confirm:
            continue

        delta_home = pending[0] - current[0]
        delta_away = pending[1] - current[1]
        moved = [d for d in (delta_home, delta_away) if d != 0]

        if (len(moved) == 1 and moved[0] in POINTS_TO_LABEL
                and pending_time - current_time <= max_gap):
            changes.append((pending_time, moved[0]))

        current, current_time = pending, pending_time
        pending, pending_count = None, 0

    return changes


def calibrate_lag(changes, manual_marks, window=6.0):
    """Median offset between a detected score change and the owner's own mark.

    The graphic updates a beat after the ball drops, and the scan only samples
    every `every` seconds. Rather than guessing that delay, measure it against
    marks made by a human watching the same footage.
    """
    manual = sorted((float(m["timestamp_sec"]), m["label"]) for m in manual_marks
                    if m["label"] in POINTS_TO_LABEL.values())
    if not manual:
        return None, 0

    offsets = []
    for detected, points in changes:
        label = POINTS_TO_LABEL[points]
        nearby = [t for t, name in manual if name == label and abs(t - detected) <= window]
        if nearby:
            closest = min(nearby, key=lambda t: abs(t - detected))
            offsets.append(detected - closest)

    if not offsets:
        return None, 0
    return float(np.median(offsets)), len(offsets)


def to_marks(changes, match_id, lag):
    return [{"match_id": match_id,
             "timestamp_sec": f"{max(0.0, seconds - lag):.2f}",
             "label": POINTS_TO_LABEL[points]}
            for seconds, points in changes]


def main():
    parser = argparse.ArgumentParser(description="Mark scoring plays from the score bug.")
    parser.add_argument("--match-id", default="match01")
    parser.add_argument("--video", default=None)
    parser.add_argument("--layout", default="paris2024", choices=sorted(LAYOUTS))
    parser.add_argument("--every", type=float, default=0.5,
                        help="seconds between score readings")
    parser.add_argument("--until", type=float, default=None,
                        help="stop after this many seconds of video")
    parser.add_argument("--lag", type=float, default=None,
                        help="seconds the graphic trails the basket "
                             "(default: measured against your own marks)")
    parser.add_argument("--confirm", type=int, default=4,
                        help="consecutive readings a new score must hold")
    parser.add_argument("--load-readings", default=None,
                        help="reuse saved readings instead of re-walking the video")
    parser.add_argument("--save-readings", default=None,
                        help="write raw score readings here for offline tuning")
    parser.add_argument("--load-readings", default=None,
                        help="reuse saved readings instead of re-walking the video")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be marked, write nothing")
    args = parser.parse_args()

    video = args.video or os.path.join("data", "video", f"{args.match_id}.mp4")
    if not os.path.exists(video):
        raise SystemExit(f"no video at {video}")

    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 854)
    cap.release()

    layout = load_layout(args.layout, width)
    templates = load_templates()

    # Scanning walks 162,000 frames and takes minutes. Saving the readings once
    # and reloading them makes the detection rules cheap to re-tune, which is
    # how the recap problem was found: the fix was a rule change, not a rescan.
    if args.load_readings:
        with open(args.load_readings, newline="", encoding="utf-8") as fh:
            readings = [(float(t), int(h), int(a)) for t, h, a in csv.reader(fh)]
        print(f"reusing {len(readings)} readings from {args.load_readings}")
    else:
        print(f"scanning {video} every {args.every}s")
        readings = scan(video, layout, templates, args.every, args.until)
        print(f"\n{len(readings)} readable score readings")

    if args.save_readings:
        with open(args.save_readings, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(readings)
        print(f"readings saved to {args.save_readings}")

    changes = score_changes(readings, confirm=args.confirm)
    counts = {}
    for _, points in changes:
        counts[POINTS_TO_LABEL[points]] = counts.get(POINTS_TO_LABEL[points], 0) + 1
    print(f"{len(changes)} scoring plays detected")
    for name in ("free_throw", "two_pointer", "three_pointer"):
        print(f"  {name:<16}{counts.get(name, 0):>5}")

    existing = load_marks(events_path(args.match_id))

    lag = args.lag
    if lag is None:
        measured, matched = calibrate_lag(changes, existing)
        if measured is None:
            lag = 1.0
            print("\nno overlapping manual marks -- assuming a 1.0s graphic lag")
        else:
            lag = measured
            print(f"\ngraphic lag measured against {matched} of your own marks: "
                  f"{lag:+.2f}s")

    if args.dry_run:
        print("\ndry run -- nothing written")
        return

    # Auto marks are written to their own file. The hand-marked file is read
    # only, so a bug in this detector can never cost the owner work they did
    # themselves. Marks the owner already made are skipped so one play does not
    # end up as two clips.
    auto = to_marks(changes, args.match_id, lag)
    hand_times = [(float(m["timestamp_sec"]), m["label"]) for m in existing]
    fresh = [m for m in auto
             if not any(name == m["label"] and abs(t - float(m["timestamp_sec"])) < 3.0
                        for t, name in hand_times)]

    save_marks(fresh, auto_events_path(args.match_id))
    print(f"\nwrote {len(fresh)} auto marks to {auto_events_path(args.match_id)}")
    print(f"({len(auto) - len(fresh)} were already marked by hand)")
    print(f"your own {len(existing)} marks in {events_path(args.match_id)} are untouched")
    print(f"\nnext: python -m ingest.cut --match-id {args.match_id}")


if __name__ == "__main__":
    main()
