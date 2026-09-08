"""Find and fix labelling mistakes using the trained model as a second opinion.

Runs the classifier over every frame you labelled and surfaces the ones where it
confidently disagrees with you, most confident first. Those are where labelling
slips concentrate: at ~98% accuracy the model and the labeller are about equally
reliable, so a confident disagreement is roughly as likely to be your error as
its own.

This is deliberately circular -- the model was trained on these very labels. That
limits what it can catch. A one-off slip stands out against 500 consistent
examples and gets surfaced. A rule you applied wrongly but *consistently* becomes
the pattern the model learned, and it will agree with you every time. Use this
for slips; use the label guide for systematic errors.
"""

import argparse
import os

import cv2
import torch

from frames.data import CLASS_TO_INDEX, CLASSES, LABEL_MAP, load_cache, load_manifest
from frames.label import LABELS, MANIFEST, load_manifest as load_raw, save_manifest
from frames.model import FrameHead

FRAMES_ROOT = os.path.join("data", "frames")
KEEP_KEY = ord("k")
QUIT_KEYS = {ord("q"), 27}


def disagreements(min_confidence):
    """Return [(row, model_class, confidence)] sorted most-confident first."""
    rows = load_manifest()          # labels already folded to CLASSES
    lookup = load_cache()

    head = FrameHead(num_classes=len(CLASSES))
    head.load_state_dict(torch.load("frame_head.pt", map_location="cpu"))
    head.eval()

    features = torch.stack([lookup[(r["match_id"], r["filename"])] for r in rows])
    with torch.no_grad():
        probs = torch.softmax(head(features), dim=1)
        confidence, predicted = probs.max(dim=1)

    found = []
    for i, row in enumerate(rows):
        if predicted[i].item() != CLASS_TO_INDEX[row["label"]]:
            conf = confidence[i].item()
            if conf >= min_confidence:
                found.append((row, CLASSES[predicted[i].item()], conf))

    found.sort(key=lambda item: item[2], reverse=True)
    return found, len(rows)


def main():
    parser = argparse.ArgumentParser(description="Review frames the model disagrees with.")
    parser.add_argument("--min-confidence", type=float, default=0.90,
                        help="only show disagreements at least this confident")
    parser.add_argument("--list", action="store_true",
                        help="just print them, do not open the review window")
    args = parser.parse_args()

    found, total = disagreements(args.min_confidence)

    print(f"{len(found)} confident disagreements out of {total} labelled frames "
          f"({len(found) / total:.1%})")

    if not found:
        print("nothing above the confidence threshold -- labels and model agree.")
        return

    if args.list:
        print(f"\n{'frame':<28}{'you said':>12}{'model says':>12}{'conf':>8}")
        for row, model_class, conf in found:
            print(f"{row['filename']:<28}{row['label']:>12}{model_class:>12}{conf:>7.1%}")
        return

    print("  g = game    c = crowd    x = graphic    k = keep mine    q = quit")

    raw = load_raw(MANIFEST)        # the untouched three-class labels on disk
    changed = 0

    for position, (row, model_class, conf) in enumerate(found, start=1):
        path = os.path.join(FRAMES_ROOT, row["match_id"], row["filename"])
        image = cv2.imread(path)
        if image is None:
            continue

        key_of = (row["match_id"], row["filename"])
        mine = raw[key_of]["label"]

        bar = 34
        canvas = cv2.copyMakeBorder(image, bar, bar, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"{position}/{len(found)}   you: {mine}   model: {model_class} ({conf:.0%})",
                    (10, 23), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "g game   c crowd   x graphic   k keep mine   q quit",
                    (10, canvas.shape[0] - 11), font, 0.6, (120, 220, 255), 1, cv2.LINE_AA)
        cv2.imshow("review", canvas)

        key = cv2.waitKey(0) & 0xFF

        if key in QUIT_KEYS:
            break
        if key == KEEP_KEY or key not in LABELS:
            continue

        # Writes the raw three-class label, not the merged one, so the manifest
        # stays the original record and the merge remains a read-time decision.
        raw[key_of]["label"] = LABELS[key]
        save_manifest(raw, MANIFEST)
        changed += 1

    cv2.destroyAllWindows()
    print(f"\n{changed} labels corrected")
    if changed:
        print("labels changed -- rebuild nothing, but retrain:")
        print("  python -m frames.train")
        print("(the feature cache is unaffected: the pixels did not change, only the labels)")


if __name__ == "__main__":
    main()
