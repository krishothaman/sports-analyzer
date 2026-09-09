"""Ask the pretrained weights what they already recognise, before training anything.

Every phase so far has started by training something and then reading a number.
This does the opposite, and it costs about a minute.

The backbone arrives knowing Kinetics-400: four hundred classes of human action,
four of which are `dribbling basketball`, `dunking basketball`, `playing
basketball` and `shooting basketball`. It has seen thousands of dunks. We own
eighteen. So before spending a training run on `dunk`, it is worth asking the
question directly -- do our dunk clips already light up Kinetics class 107?

Three outcomes, and each one changes what to do next:

  * dunk clips score high on 107 and other clips do not -> `dunk` is nearly
    free, and the 18-clip problem was never the obstacle
  * everything scores high on 107 -> the model sees "basketball" but not the
    specific action, so the class has to be learned rather than borrowed
  * nothing scores on any basketball class -> our clips look unlike Kinetics
    footage, which would be a preprocessing problem and not a data problem

None of those is visible in an accuracy number, and finding out costs one script
instead of one training run.

    python -m clips.probe_kinetics
    python -m clips.probe_kinetics --crop squash
"""

import argparse
from collections import Counter

import torch

from clips.data import CLASSES, load_clip_tensor, load_clips
from models.backbone import CROP_MODES, build_video_backbone, kinetics_categories

# The four Kinetics-400 classes that are about basketball. Verified against the
# weights' own category list in tests/test_backbone.py -- if these indices ever
# drift, this script would silently report the wrong classes.
DUNKING = 107
SHOOTING = 296
BASKETBALL = {99: "dribbling", DUNKING: "dunking", 220: "playing",
              SHOOTING: "shooting"}

# Enough of the common classes to be representative, while keeping every clip of
# the rare ones -- there are only 18 dunks and 7 blocks, and the whole point is
# to look at those.
PER_CLASS = 80


def sample(rows, per_class):
    """At most `per_class` clips of each label, in manifest order."""
    seen, kept = Counter(), []
    for row in rows:
        if seen[row["label"]] < per_class:
            seen[row["label"]] += 1
            kept.append(row)
    return kept


def kinetics_probabilities(rows, backbone, crop, device, batch_size=4):
    """[N, 400] softmax probabilities from the untouched pretrained classifier."""
    model, transform, _ = build_video_backbone(backbone, device, crop,
                                               keep_classifier=True)
    chunks = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        clips = torch.stack([transform(load_clip_tensor(row)) for row in batch])

        with torch.no_grad():
            logits = model(clips.to(device))
        # Softmax so the numbers are comparable across clips. Raw logits are not
        # -- a confident clip and an uncertain one can share a logit value.
        chunks.append(torch.softmax(logits, dim=1).cpu())

        seen = min(start + batch_size, len(rows))
        print(f"  probing: {seen}/{len(rows)} clips", end="\r", flush=True)

    print()
    return torch.cat(chunks)


def report(rows, probs, categories):
    """One row per label of ours: what Kinetics thinks these clips are."""
    header = f"{'our label':<16}"
    for name in BASKETBALL.values():
        header += f"{'P(' + name + ')':>14}"
    header += "   most common Kinetics guess"
    print("\n" + header)
    print("-" * len(header))

    for label in CLASSES:
        index = [i for i, row in enumerate(rows) if row["label"] == label]
        if not index:
            continue

        rows_probs = probs[index]
        line = f"{label:<16}"
        for kinetics_class in BASKETBALL:
            line += f"{rows_probs[:, kinetics_class].mean().item():>14.3f}"

        top1 = Counter(rows_probs.argmax(dim=1).tolist()).most_common(1)[0]
        share = top1[1] / len(index)
        line += f"   {categories[top1[0]]} ({share:.0%} of {len(index)})"
        print(line)


def verdict(rows, probs):
    """State plainly whether the dunk shortcut is available."""
    dunk_rows = [i for i, row in enumerate(rows) if row["label"] == "dunk"]
    other_rows = [i for i, row in enumerate(rows) if row["label"] != "dunk"]

    if not dunk_rows:
        print("\nno dunk clips in the manifest -- nothing to probe.")
        return

    ours = probs[dunk_rows, DUNKING].mean().item()
    theirs = probs[other_rows, DUNKING].mean().item()

    print(f"\nP(dunking basketball) on our {len(dunk_rows)} dunk clips: {ours:.3f}")
    print(f"                       on the other {len(other_rows)} clips: {theirs:.3f}")

    if ours > theirs * 2:
        print("\n-> the pretrained weights already separate dunks from everything"
              "\n   else. `dunk` may be close to free despite only 18 examples.")
    elif ours > theirs:
        print("\n-> a real but weak signal. Worth keeping dunk as a class, but it"
              "\n   will have to be learned rather than borrowed.")
    else:
        print("\n-> no usable signal. Kinetics recognises basketball here, but not"
              "\n   this particular action -- 18 clips will not be enough, and the"
              "\n   honest move is to report dunk as unmeasurable.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mvit_v2_s")
    parser.add_argument("--crop", default="center", choices=list(CROP_MODES))
    parser.add_argument("--per-class", type=int, default=PER_CLASS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = sample(load_clips(), args.per_class)
    if not rows:
        raise SystemExit("no clips in the manifest -- run: python -m ingest.cut")

    print(f"probing {len(rows)} clips with {args.backbone} "
          f"({args.crop} crop) on {device}")
    print("the classifier is the ORIGINAL Kinetics head -- nothing here is trained.\n")

    probs = kinetics_probabilities(rows, args.backbone, args.crop, device)
    report(rows, probs, kinetics_categories(args.backbone))
    verdict(rows, probs)


if __name__ == "__main__":
    main()
