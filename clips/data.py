"""Clip manifest -> match-level split -> cached features -> DataLoaders.

The same shape as frames/data.py one level up: the unit is a 2-second clip
instead of a still frame, so a cached example is [16, 512] rather than [512].
The backbone is still frozen, so a clip's features are still permanent and still
worth computing once.
"""

import os

import torch
from PIL import Image

from ingest.cut import (BACKGROUND_LABEL, CLIP_FRAMES, CLIPS_ROOT,
                        load_clip_manifest)
from models.backbone import FEATURE_DIM, build_backbone

# Six events plus background. Unlike Phase 2 there is no merging: every one of
# these is a class the timeline is supposed to emit.
CLASSES = ["two_pointer", "three_pointer", "dunk", "free_throw",
           "block", "steal", BACKGROUND_LABEL]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

CACHE = os.path.join("data", "features", "clips.pt")

# Below this many matches a match-level split is impossible, so we fall back --
# loudly. Three is the minimum that leaves distinct matches on both sides.
MIN_MATCHES_FOR_MATCH_SPLIT = 3


def load_clips(path=None):
    rows = load_clip_manifest(path) if path else load_clip_manifest()
    return [row for row in rows if row["label"] in CLASS_TO_INDEX]


def match_split(rows, train_frac=0.7):
    """Split by match_id. Whole matches go to one side or the other.

    Clips from one game share the arena, the lighting, the jerseys and the
    camera operator. Split by clip and the model can score well by recognising
    the venue rather than the basketball -- around 95% on the test set, then
    collapse on new footage. A dishonest test score is worse than no test score,
    because it stops you looking for the problem.
    """
    match_ids = sorted({row["match_id"] for row in rows})
    cut = max(1, int(round(len(match_ids) * train_frac)))
    train_ids = set(match_ids[:cut])

    train = [row for row in rows if row["match_id"] in train_ids]
    test = [row for row in rows if row["match_id"] not in train_ids]
    return train, test


def chronological_split(rows, train_frac=0.7):
    """Fallback for a single match: earliest clips train, latest clips test.

    Weaker than a match split -- it still shares arena and lighting across the
    boundary -- but it at least stops the same possession appearing on both
    sides. Identical in spirit to Phase 2's split.
    """
    by_match = {}
    for row in rows:
        by_match.setdefault(row["match_id"], []).append(row)

    train, test = [], []
    for match_id in sorted(by_match):
        ordered = sorted(by_match[match_id], key=lambda r: float(r["start_sec"]))
        cut = int(len(ordered) * train_frac)
        train.extend(ordered[:cut])
        test.extend(ordered[cut:])
    return train, test


def split_clips(rows, train_frac=0.7, quiet=False):
    """Match-level split where possible, chronological where not.

    The warning is not decoration. A number produced by the fallback measures
    'can it do this on the match it trained on', which is a much easier question
    than the one the project is actually asking.
    """
    match_count = len({row["match_id"] for row in rows})

    if match_count >= MIN_MATCHES_FOR_MATCH_SPLIT:
        return match_split(rows, train_frac)

    if not quiet:
        print(f"\n  !! only {match_count} match(es) -- falling back to a "
              f"chronological split within the match.")
        print("  !! the resulting score is NOT a generalisation estimate: train and")
        print("  !! test share one arena, one lighting rig and one camera crew.")
        print(f"  !! it becomes honest at {MIN_MATCHES_FOR_MATCH_SPLIT} matches.\n")

    return chronological_split(rows, train_frac)


def clip_frame_paths(row):
    directory = os.path.join(CLIPS_ROOT, row["match_id"], row["clip_id"])
    return [os.path.join(directory, f"{i:02d}.jpg") for i in range(CLIP_FRAMES)]


def build_cache(rows, device, clips_per_batch=4, cache_path=CACHE):
    """Run every clip's 16 frames through the backbone once and save the result.

    Frames are batched across clips so the GPU sees 64 images at a time rather
    than 16 -- the backbone is the expensive part and it likes a full batch.
    """
    model, preprocess = build_backbone(device)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    keys, chunks = [], []

    for start in range(0, len(rows), clips_per_batch):
        batch = rows[start:start + clips_per_batch]
        images = []
        for row in batch:
            for path in clip_frame_paths(row):
                images.append(preprocess(Image.open(path).convert("RGB")))
            keys.append(row["clip_id"])

        stacked = torch.stack(images).to(device)
        with torch.no_grad():
            features = model(stacked).cpu()          # [clips * 16, 512]
        # Back to one row per clip: [clips, 16, 512]. The head needs the frames
        # kept in order -- that ordering is the only thing separating a clip
        # from a bag of unrelated pictures.
        chunks.append(features.view(len(batch), CLIP_FRAMES, FEATURE_DIM))

        seen = min(start + clips_per_batch, len(rows))
        print(f"  features: {seen}/{len(rows)} clips", end="\r", flush=True)

    features = torch.cat(chunks)
    torch.save({"keys": keys, "features": features}, cache_path)
    print(f"\ncached {features.shape[0]} x {features.shape[1]} x {features.shape[2]} "
          f"features to {cache_path}")
    return features


def load_cache(cache_path=CACHE):
    if not os.path.exists(cache_path):
        raise SystemExit(f"no clip feature cache at {cache_path} -- run: python -m clips.data")
    blob = torch.load(cache_path, map_location="cpu")
    return {key: blob["features"][i] for i, key in enumerate(blob["keys"])}


def _dataset(rows, lookup):
    missing = [r["clip_id"] for r in rows if r["clip_id"] not in lookup]
    if missing:
        raise SystemExit(
            f"{len(missing)} clips are not in the feature cache (first: {missing[0]}). "
            f"The cache is stale -- rerun: python -m clips.data"
        )
    features = torch.stack([lookup[r["clip_id"]] for r in rows])
    labels = torch.tensor([CLASS_TO_INDEX[r["label"]] for r in rows])
    return torch.utils.data.TensorDataset(features, labels)


def get_loaders(batch_size=16, train_frac=0.7):
    rows = load_clips()
    lookup = load_cache()
    train_rows, test_rows = split_clips(rows, train_frac)

    train_loader = torch.utils.data.DataLoader(
        _dataset(train_rows, lookup), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        _dataset(test_rows, lookup), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def composition(rows):
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def report(rows):
    """Print what the dataset holds and what a model has to beat."""
    counts = composition(rows)
    print(f"{len(rows)} clips across {len({r['match_id'] for r in rows})} match(es)")
    for name in CLASSES:
        print(f"  {name:<16}{counts.get(name, 0):>5}")

    # Spec section 7: always predict the most common class. Any model that
    # cannot beat this has learned nothing, and usually means a broken pipeline
    # rather than a weak model.
    if counts:
        biggest = max(counts.values())
        common = max(counts, key=counts.get)
        print(f"\ndumb baseline: always say '{common}' -> "
              f"{biggest}/{len(rows)} = {biggest / len(rows):.1%}")

    thin = [name for name in CLASSES if 0 < counts.get(name, 0) < 20]
    if thin:
        print(f"\ntoo thin to measure ({', '.join(thin)}): under ~20 examples a class's"
              "\nrow of the confusion matrix is noise, not a result.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_clips()
    if not rows:
        raise SystemExit("no clips in the manifest -- run: python -m ingest.cut")

    report(rows)
    train_rows, test_rows = split_clips(rows)
    print(f"split: {len(train_rows)} train / {len(test_rows)} test")
    print(f"extracting {FEATURE_DIM}-d features for {CLIP_FRAMES} frames per clip "
          f"on {device}")

    build_cache(rows, device)


if __name__ == "__main__":
    main()
