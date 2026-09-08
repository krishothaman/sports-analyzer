"""Mark events in a match video by keypress -- the first half of mark-then-cut.

Nothing is cut here. This tool only records *when* something happened and *what*
it was, one row per keypress, into data/events/<match_id>.csv. ingest/cut.py
turns those rows into clips later.

Splitting it that way is what makes the labelling survivable. Cutting a 108-minute
match into 2-second clips and judging them one by one would mean ~3,250 decisions,
about 95% of them "nothing happening". Marking spends your attention only on the
5% that matters, and background clips get sampled automatically from the gaps.

It also makes mistakes cheap: a mis-timed mark is one editable row in a CSV, and
nothing is committed to disk as pixels until you run the cutter.

Rules for what to press live in docs/clip-guide.md. Read that first.
"""

import argparse
import csv
import os

import cv2

MARK_KEYS = {
    ord("1"): "two_pointer",
    ord("2"): "three_pointer",
    ord("3"): "dunk",
    ord("4"): "free_throw",
    ord("5"): "block",
    ord("6"): "steal",
    # Not an event and not background either -- a hole in the timeline. Misses,
    # unclassifiable shots, fouls, scrambles, anything you are unsure about.
    # Without this key a missed three would go unmarked, get sampled as
    # background, and teach the model that one picture is both three_pointer and
    # none. A contradiction is worse than either label alone.
    ord("x"): "exclude",
}

PLAY_PAUSE_KEY = 32                 # space
UNDO_KEY = ord("u")
QUIT_KEYS = {ord("q"), 27}          # q or Escape
NO_KEY = 255                        # waitKey returns -1 when nothing was pressed

# Letters, not arrow keys: OpenCV reports arrow key codes differently across
# platforms, so they would silently do nothing on Windows.
SEEK_KEYS = {ord("a"): -5.0, ord("d"): 5.0, ord("s"): -30.0, ord("w"): 30.0}

SLOWER_KEY = ord("[")
FASTER_KEY = ord("]")
SPEEDS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

EVENTS_ROOT = os.path.join("data", "events")
MARK_FIELDS = ["match_id", "timestamp_sec", "label"]

# A second press of the same key this soon after the last is a stutter, not a
# second event. Real repeats of one class inside three quarters of a second do
# not happen in basketball.
DUPLICATE_WINDOW = 0.75


def events_path(match_id):
    return os.path.join(EVENTS_ROOT, f"{match_id}.csv")


def position_path(match_id):
    return os.path.join(EVENTS_ROOT, f"{match_id}.pos")


def save_position(seconds, path):
    """Remember where you stopped watching, so a rerun does not rewind you.

    Kept separate from the marks: this is session convenience, the CSV is the
    dataset. Losing this file costs you nothing but a manual --start.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{seconds:.2f}\n")


def load_position(path):
    if not os.path.exists(path):
        return None
    try:
        return float(open(path, encoding="utf-8").read().strip())
    except ValueError:
        return None


def resume_at(marks, saved_position):
    """Where a rerun should pick up.

    The later of 'just before your last mark' and 'where you stopped watching'.
    Resuming from the last mark alone rewinds you through every stretch you
    watched without marking anything -- which is most of a broadcast.
    """
    from_marks = None
    if marks:
        from_marks = max(0.0, max(float(m["timestamp_sec"]) for m in marks) - 5.0)

    candidates = [c for c in (from_marks, saved_position) if c is not None]
    return max(candidates) if candidates else None


def load_marks(path):
    """Return the marks already recorded, in the order they were made."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def save_marks(marks, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MARK_FIELDS)
        writer.writeheader()
        writer.writerows(marks)


def add_mark(marks, match_id, timestamp_sec, label, window=DUPLICATE_WINDOW):
    """Append a mark unless it duplicates one just made.

    Returns (marks, accepted). Insertion order is preserved rather than sorting
    by time, so undo removes what you just added even if you had seeked
    backwards to add it. The cutter sorts when it needs to.
    """
    for mark in marks:
        if (mark["label"] == label
                and abs(float(mark["timestamp_sec"]) - timestamp_sec) < window):
            return marks, False

    return marks + [{
        "match_id": match_id,
        "timestamp_sec": f"{timestamp_sec:.2f}",
        "label": label,
    }], True


def tally(marks):
    counts = {}
    for mark in marks:
        counts[mark["label"]] = counts.get(mark["label"], 0) + 1
    return counts


def format_clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def summary_line(marks):
    """One-line tally, events first and exclude last, for the header bar."""
    counts = tally(marks)
    order = [name for name in MARK_KEYS.values() if name != "exclude"] + ["exclude"]
    shown = [f"{name[:4]} {counts[name]}" for name in order if name in counts]
    return "  ".join(shown) if shown else "no marks yet"


def annotate(image, top, middle, bottom):
    bar = 34
    canvas = cv2.copyMakeBorder(image, bar, bar * 2, 0, 0,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, top, (10, 23), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, middle, (10, canvas.shape[0] - 42), font, 0.55,
                (150, 255, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, bottom, (10, canvas.shape[0] - 13), font, 0.5,
                (120, 220, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Mark events in a match by keypress.")
    parser.add_argument("video", help="path to the match video")
    parser.add_argument("--match-id", default="match01")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="initial playback speed; adjust live with [ and ]")
    parser.add_argument("--start", type=float, default=None,
                        help="start this many seconds in (default: just before your last mark)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0.0

    path = events_path(args.match_id)
    marks = load_marks(path)

    print(f"{args.video} -- {format_clock(duration)}, {fps:.1f} fps")
    print(f"{len(marks)} marks already recorded in {path}")
    print("  1 two_pointer   2 three_pointer   3 dunk   4 free_throw")
    print("  5 block         6 steal           x exclude (miss / unsure / foul)")
    print("  space play-pause   a/d seek +-5s   s/w seek +-30s")
    print("  [ ] slower/faster  u undo          q save and quit")
    print("\nrules: docs/clip-guide.md -- press the instant you SEE the outcome\n")

    pos_path = position_path(args.match_id)
    start = args.start
    if start is None:
        start = resume_at(marks, load_position(pos_path))
    if start:
        # Seeking is approximate across codecs, which is fine here: being a few
        # frames off while eyeballing a 2-second event changes nothing. The
        # cutter, which does need exact frames, walks the video sequentially.
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        print(f"resuming at {format_clock(start)}")

    # Start at the nearest supported speed to whatever was asked for.
    speed_index = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - args.speed))

    playing = True
    frame = None
    seconds = 0.0
    flash = ""          # confirmation of the last mark, shown for a moment
    flash_left = 0

    while True:
        if playing or frame is None:
            ok, raw = cap.read()
            if not ok:
                break
            frame = raw
            # After a successful read, POS_FRAMES points at the NEXT frame, so
            # the one on screen is one earlier.
            seconds = max(0.0, (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps)

        speed = SPEEDS[speed_index]
        state = f"x{speed:g}" if playing else "PAUSED"
        top = (f"{format_clock(seconds)} / {format_clock(duration)}   {state}"
               f"   marks {len(marks)}")
        if flash_left > 0:
            top += f"   <- {flash}"
            flash_left -= 1

        legend = ("1 2pt  2 3pt  3 dunk  4 ft  5 blk  6 stl  x skip | "
                  "space  a/d 5s  s/w 30s  [] speed  u undo  q quit")
        cv2.imshow("mark", annotate(frame, top, summary_line(marks), legend))

        # While playing, the wait is what sets the frame rate. While paused we
        # poll often enough to feel responsive without spinning the CPU.
        delay = max(1, int(1000.0 / (fps * speed))) if playing else 30
        key = cv2.waitKey(delay) & 0xFF

        if key == NO_KEY:
            continue

        if key in QUIT_KEYS:
            break

        if key == PLAY_PAUSE_KEY:
            playing = not playing
            continue

        if key in SEEK_KEYS:
            limit = duration - 1.0 if duration else seconds
            seconds = min(max(0.0, seconds + SEEK_KEYS[key]), limit)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
            frame = None            # force a fresh read even while paused
            continue

        if key == FASTER_KEY:
            speed_index = min(len(SPEEDS) - 1, speed_index + 1)
            continue

        if key == SLOWER_KEY:
            speed_index = max(0, speed_index - 1)
            continue

        if key == UNDO_KEY:
            if marks:
                removed = marks[-1]
                marks = marks[:-1]
                save_marks(marks, path)
                flash = f"undid {removed['label']}"
                flash_left = 20
            continue

        if key in MARK_KEYS:
            label = MARK_KEYS[key]
            marks, accepted = add_mark(marks, args.match_id, seconds, label)
            if accepted:
                # Written after every accepted keypress. A crash, a closed window
                # or a dead battery costs you nothing you had already marked.
                save_marks(marks, path)
                flash = f"{label} @ {format_clock(seconds)}"
            else:
                flash = f"ignored double {label}"
            flash_left = 20
            continue

    cap.release()
    cv2.destroyAllWindows()
    save_position(seconds, pos_path)

    counts = tally(marks)
    print(f"\n{len(marks)} marks saved to {path}")
    for name in MARK_KEYS.values():
        if name in counts:
            print(f"  {name:<15}{counts[name]:>4}")

    events = sum(count for name, count in counts.items() if name != "exclude")
    print(f"\n{events} event marks, {counts.get('exclude', 0)} excluded")
    if events:
        print(f"next: python -m ingest.cut --match-id {args.match_id}")


if __name__ == "__main__":
    main()
