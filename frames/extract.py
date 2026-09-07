"""Sample still frames from a match video.

One frame every few seconds is plenty for a frame classifier: consecutive frames
of a 30fps video are near-identical, so sampling densely would just mean labelling
the same picture over and over.
"""

import argparse
import csv
import os

import cv2

DEFAULT_EVERY_SECONDS = 3.0


def extract(video_path, out_dir, match_id, every_seconds=DEFAULT_EVERY_SECONDS):
    """Write one JPEG every `every_seconds` of video. Returns the number saved."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    # Some containers report no fps at all, so fall back rather than divide by zero.
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * every_seconds)))

    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.csv")

    saved = 0
    frame_index = 0

    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "timestamp_sec"])

        while True:
            # grab() advances to the next frame without fully decoding it, which is
            # far cheaper than read(). We only pay for a full decode via retrieve()
            # on the frames we actually keep -- roughly 1 in 90 of them.
            if not cap.grab():
                break

            if frame_index % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    name = f"{match_id}_{saved:05d}.jpg"
                    cv2.imwrite(os.path.join(out_dir, name), frame)
                    # The timestamp is kept because the train/test split is
                    # chronological -- see frames/data.py.
                    writer.writerow([name, f"{frame_index / fps:.2f}"])
                    saved += 1

            frame_index += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Sample frames from a match video.")
    parser.add_argument("video", help="path to the video file")
    parser.add_argument("--match-id", default="match01")
    parser.add_argument("--every", type=float, default=DEFAULT_EVERY_SECONDS,
                        help="seconds between sampled frames")
    args = parser.parse_args()

    out_dir = os.path.join("data", "frames", args.match_id)
    saved = extract(args.video, out_dir, args.match_id, args.every)
    print(f"saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
