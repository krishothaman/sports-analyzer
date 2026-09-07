"""Manifest -> chronological split -> cached features -> DataLoaders.

The backbone is frozen, so a given frame's 512 numbers can never change. We
compute them once, write them to disk, and every training run afterwards reads
arrays instead of decoding JPEGs. Training drops from minutes to under a second.
"""

import csv
import os

import torch
from PIL import Image

from models.backbone import FEATURE_DIM, build_backbone

# Two classes, not three. The manifest holds three raw labels, but 'graphic'
# came in at 10 examples out of 520 -- with a 70/30 split that is about three
# frames in the test set, so its recall would be measured on three numbers and
# mean nothing.
#
# The merge happens here, at read time, and the manifest on disk keeps its
# original labels. If a later match supplies enough 'graphic' frames, this map
# changes and nothing needs relabelling. Never destroy raw data to suit the
# model you happen to be training today.
#
# 'game' vs 'not_game' is also exactly the question Phase 5 asks of this filter.
CLASSES = ["game", "not_game"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

LABEL_MAP = {
    "game": "game",
    "crowd": "not_game",
    "graphic": "not_game",
}

MANIFEST = os.path.join("data", "manifest.csv")
CACHE = os.path.join("data", "features", "frames.pt")
FRAMES_ROOT = os.path.join("data", "frames")


def load_manifest(path=MANIFEST):
    """Read the manifest and fold the raw labels down to CLASSES."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = []
        for row in csv.DictReader(fh):
            mapped = LABEL_MAP.get(row["label"])
            if mapped is None:
                continue        # an unknown label is skipped, never guessed at
            rows.append({**row, "label": mapped})
    return rows


def chronological_split(rows, train_frac=0.7):
    """Split each match by time: earliest frames train, latest frames test.

    A random split would leak. Frames twelve seconds apart show the same
    possession, the same players, the same camera angle -- near-duplicates. Put
    one in train and its neighbour in test and the model scores well by
    recognising a moment it has already seen, which measures memory, not
    understanding. Cutting on time keeps a shared moment on one side.

    This is the single-video stand-in for the match-level splitting that Phase 3
    uses once there are several matches.
    """
    by_match = {}
    for row in rows:
        by_match.setdefault(row["match_id"], []).append(row)

    train, test = [], []
    for match_id in sorted(by_match):
        ordered = sorted(by_match[match_id], key=lambda r: float(r["timestamp_sec"]))
        cut = int(len(ordered) * train_frac)
        train.extend(ordered[:cut])
        test.extend(ordered[cut:])
    return train, test


def build_cache(rows, device, batch_size=32, cache_path=CACHE):
    """Run every frame through the backbone once and save the results."""
    model, preprocess = build_backbone(device)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    keys, chunks = [], []

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = []
        for row in batch:
            path = os.path.join(FRAMES_ROOT, row["match_id"], row["filename"])
            # convert("RGB") guards against greyscale or palettised JPEGs, which
            # would arrive with the wrong number of channels.
            images.append(preprocess(Image.open(path).convert("RGB")))
            keys.append((row["match_id"], row["filename"]))

        stacked = torch.stack(images).to(device)
        with torch.no_grad():
            # .cpu() matters: keeping 520 GPU tensors alive would hold VRAM for
            # data we only ever read back.
            chunks.append(model(stacked).cpu())

        seen = min(start + batch_size, len(rows))
        print(f"  features: {seen}/{len(rows)}", end="\r", flush=True)

    features = torch.cat(chunks)
    torch.save({"keys": keys, "features": features}, cache_path)
    print(f"\ncached {features.shape[0]} x {features.shape[1]} features to {cache_path}")
    return features


def load_cache(cache_path=CACHE):
    if not os.path.exists(cache_path):
        raise SystemExit(f"no feature cache at {cache_path} -- run: python -m frames.data")
    blob = torch.load(cache_path, map_location="cpu")
    return {key: blob["features"][i] for i, key in enumerate(blob["keys"])}


def _dataset(rows, lookup):
    missing = [r["filename"] for r in rows if (r["match_id"], r["filename"]) not in lookup]
    if missing:
        raise SystemExit(
            f"{len(missing)} labelled frames are not in the feature cache "
            f"(first: {missing[0]}). The cache is stale -- rerun: python -m frames.data"
        )
    features = torch.stack([lookup[(r["match_id"], r["filename"])] for r in rows])
    labels = torch.tensor([CLASS_TO_INDEX[r["label"]] for r in rows])
    return torch.utils.data.TensorDataset(features, labels)


def get_loaders(batch_size=32, train_frac=0.7):
    """Return (train_loader, test_loader) over cached features."""
    rows = load_manifest()
    lookup = load_cache()
    train_rows, test_rows = chronological_split(rows, train_frac)

    train_loader = torch.utils.data.DataLoader(
        _dataset(train_rows, lookup), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        _dataset(test_rows, lookup), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def main():
    """Build the feature cache. Run after labelling, and again after adding labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest()

    tally = {}
    for row in rows:
        tally[row["label"]] = tally.get(row["label"], 0) + 1
    print(f"{len(rows)} labelled frames {tally}")
    print(f"extracting {FEATURE_DIM}-d features on {device}")

    build_cache(rows, device)


if __name__ == "__main__":
    main()
