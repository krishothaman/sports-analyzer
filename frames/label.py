"""Keyboard-driven frame labelling.

Opens each unlabelled frame in a window. One keypress assigns a class and
advances. Every label is written to disk immediately, so quitting halfway loses
nothing and rerunning resumes exactly where you stopped.

There is a deliberate skip key. A guessed label is worse than no label: it
teaches the model something false, and if the guess lands in the test set it
makes the accuracy number lie. When a frame is genuinely unreadable, skip it.
"""

import argparse
import csv
import os

import cv2

LABELS = {
    ord("g"): "game",
    ord("c"): "crowd",
    ord("x"): "graphic",
}
SKIP_KEY = ord("s")
UNDO_KEY = ord("u")
QUIT_KEYS = {ord("q"), 27}          # q or Escape

MANIFEST = os.path.join("data", "manifest.csv")
MANIFEST_FIELDS = ["match_id", "filename", "timestamp_sec", "label"]
FRAMES_ROOT = os.path.join("data", "frames")


def load_manifest(path=MANIFEST):
    """Return {(match_id, filename): row} for everything already labelled."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["match_id"], r["filename"]): r for r in csv.DictReader(fh)}


def save_manifest(rows, path=MANIFEST):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)


def read_index(match_dir):
    with open(os.path.join(match_dir, "index.csv"), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def counts_by_label(rows):
    tally = {}
    for row in rows.values():
        tally[row["label"]] = tally.get(row["label"], 0) + 1
    return tally


def annotate(image, top, bottom):
    """Draw the progress line above the frame and the key legend below it."""
    bar = 34
    canvas = cv2.copyMakeBorder(image, bar, bar, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, top, (10, 23), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, bottom, (10, canvas.shape[0] - 11), font, 0.6,
                (120, 220, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Label sampled frames by keypress.")
    parser.add_argument("--match-id", default="match01")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many new labels (0 = no limit)")
    args = parser.parse_args()

    match_dir = os.path.join(FRAMES_ROOT, args.match_id)
    rows = load_manifest()
    entries = read_index(match_dir)

    todo = [e for e in entries if (args.match_id, e["filename"]) not in rows]

    print(f"{len(rows)} already labelled, {len(todo)} to go")
    print("  g = game     c = crowd     x = graphic")
    print("  s = skip (cannot tell)     u = undo     q = save and quit")

    if not todo:
        print(f"\nnothing left to label. totals: {counts_by_label(rows)}")
        return

    history = []          # keys added this session, so undo knows what to remove
    done = 0
    skipped = 0

    for position, entry in enumerate(todo, start=1):
        if args.limit and done >= args.limit:
            break

        image = cv2.imread(os.path.join(match_dir, entry["filename"]))
        if image is None:
            continue

        # Shrink oversized frames so the window fits on screen. 854x480 already
        # does, so this normally does nothing -- it is here for other sources.
        if image.shape[0] > 720:
            scale = 720 / image.shape[0]
            image = cv2.resize(image, None, fx=scale, fy=scale)

        seconds = float(entry["timestamp_sec"])
        clock = f"{int(seconds) // 60:>3}m{int(seconds) % 60:02d}s"
        top = f"{position}/{len(todo)}   t={clock}   labelled={done}  skipped={skipped}"
        bottom = "g game    c crowd    x graphic    s skip    u undo    q quit"

        cv2.imshow("label", annotate(image, top, bottom))

        # waitKey(0) blocks until a key is pressed. The & 0xFF masks off high bits
        # some platforms set, leaving a plain ASCII code.
        key = cv2.waitKey(0) & 0xFF

        if key in QUIT_KEYS:
            break

        if key == SKIP_KEY:
            skipped += 1
            continue

        if key == UNDO_KEY:
            # Undo rewrites the whole manifest rather than tracking a diff. At a
            # few hundred rows that is instant, and simple beats clever here.
            if history:
                rows.pop(history.pop(), None)
                save_manifest(rows)
                done -= 1
            continue

        if key not in LABELS:
            continue        # any other key: ignore it and stay on this frame

        record_key = (args.match_id, entry["filename"])
        rows[record_key] = {
            "match_id": args.match_id,
            "filename": entry["filename"],
            "timestamp_sec": entry["timestamp_sec"],
            "label": LABELS[key],
        }
        history.append(record_key)
        # Written after every single keypress. Closing the window, a crash, or
        # pulling the plug costs you the current frame and nothing else.
        save_manifest(rows)
        done += 1

    cv2.destroyAllWindows()

    tally = counts_by_label(rows)
    print(f"\n{len(rows)} labelled total ({done} this session, {skipped} skipped)")
    print(f"  {tally}")

    if tally:
        smallest = min(tally.values())
        if smallest < 40:
            weakest = min(tally, key=tally.get)
            print(f"\n  '{weakest}' has only {smallest} examples. Under ~40 a class is hard")
            print("  to learn and its row of the confusion matrix will be noise.")


if __name__ == "__main__":
    main()
