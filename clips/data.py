"""Clip manifest -> match-level split -> cached features -> DataLoaders.

The same shape as frames/data.py one level up: the unit is a 2-second clip
instead of a still frame. The backbone is still frozen, so a clip's features are
still permanent and still worth computing once.

What a cached example looks like depends on which backbone made it:

    resnet18   [16, 512]  16 separate per-frame descriptions (Phase 2-4)
    mvit_v2_s  [768]      one description of the clip as a whole (Phase 5)

Hence one cache file per backbone-and-crop combination, and a recorded backbone
name inside each -- the vectors are not interchangeable, and nothing downstream
can tell them apart by looking.
"""

import argparse
import os

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from ingest.cut import (BACKGROUND_LABEL, CLIP_FRAMES, CLIPS_ROOT,
                        load_clip_manifest)
from models.backbone import (CROP_MODES, FEATURE_DIM, VIDEO_BACKBONES,
                             build_backbone, build_video_backbone)

# Six events plus background. Unlike Phase 2 there is no merging: every one of
# these is a class the timeline is supposed to emit.
CLASSES = ["two_pointer", "three_pointer", "dunk", "free_throw",
           "block", "steal", BACKGROUND_LABEL]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}

# Phase 4's cache. Kept under its original name so the resnet18 baseline -- and
# the 53.92% it produced -- can still be reproduced bit for bit.
CACHE = os.path.join("data", "features", "clips.pt")

# Phase 2 and 3's per-frame backbone, still selectable so the old result stays
# reproducible. Everything else in VIDEO_BACKBONES reads the clip as a clip.
FRAME_BACKBONE = "resnet18"
BACKBONES = [FRAME_BACKBONE, *sorted(VIDEO_BACKBONES)]

# Below this many matches a match-level split is impossible, so we fall back --
# loudly. Three is the minimum that leaves distinct matches on both sides.
MIN_MATCHES_FOR_MATCH_SPLIT = 3


def load_clips(path=None, fold=()):
    """Every usable clip, with the labels in `fold` relabelled as background.

    Folding is for classes too thin to learn: 7 blocks and 15 steals cannot be
    measured, and training on them only teaches the head to spend probability
    on answers it will almost never be right about. Relabelling them `none`
    rather than dropping them keeps the clips -- they are genuinely "not a
    scoring play", which is the truth about them for this model.
    """
    rows = load_clip_manifest(path) if path else load_clip_manifest()
    rows = [row for row in rows if row["label"] in CLASS_TO_INDEX]
    return [{**row, "label": BACKGROUND_LABEL} if row["label"] in fold else row
            for row in rows]


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
              f"chronological split within each match.")
        print("  !! the resulting score is NOT a generalisation estimate: every match")
        print("  !! appears on BOTH sides of the split, so train and test share the")
        print("  !! same arenas, lighting rigs and camera crews. The score measures")
        print("  !! 'can it do this on footage it has already seen the venue of'.")
        print(f"  !! it becomes honest at {MIN_MATCHES_FOR_MATCH_SPLIT} matches.\n")

    return chronological_split(rows, train_frac)


def clip_frame_paths(row):
    directory = os.path.join(CLIPS_ROOT, row["match_id"], row["clip_id"])
    return [os.path.join(directory, f"{i:02d}.jpg") for i in range(CLIP_FRAMES)]


def cache_path(backbone, crop):
    """Where one backbone-and-crop combination's features live.

    Separate files per combination, because the vectors are not interchangeable:
    resnet18 gives [16, 512] per clip and mvit_v2_s gives [768]. One shared
    filename would mean whichever cache was built last silently decides what
    every later run trains on.
    """
    if backbone == FRAME_BACKBONE and crop == "center":
        return CACHE
    return os.path.join("data", "features", f"clips_{backbone}_{crop}.pt")


def load_clip_tensor(row):
    """The 16 stored JPEGs as one [16, 3, H, W] uint8 tensor.

    That layout is exactly what torchvision's video preprocessing consumes, and
    it returns [3, 16, 224, 224] -- what the model wants. Stored frames are
    455x256, because ingest/cut.py writes them at SHORT_SIDE = 256.
    """
    return torch.stack([pil_to_tensor(Image.open(path).convert("RGB"))
                        for path in clip_frame_paths(row)])


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


def build_video_cache(rows, backbone, crop, device, clips_per_batch=4):
    """Run every clip through a video backbone once and save the result.

    One clip in, ONE vector out -- not sixteen. That single shape change is the
    whole of Phase 5. The temporal reasoning that MeanPoolHead and GRUHead had
    to do on top of frozen per-frame features now happens inside the backbone,
    which was trained on 400 classes of action to do exactly that.

    Still worth caching for exactly the Phase 2 reason: the backbone is frozen,
    so a clip's features never change, and computing them twice is waste. The
    result is also far smaller -- 768 numbers per clip instead of 16 x 512.
    """
    model, transform, feature_dim = build_video_backbone(backbone, device, crop)
    path = cache_path(backbone, crop)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    keys, chunks = [], []

    for start in range(0, len(rows), clips_per_batch):
        batch = rows[start:start + clips_per_batch]
        # transform returns [3, 16, 224, 224] per clip; stacking gives the
        # [B, C, T, H, W] the model expects.
        clips = torch.stack([transform(load_clip_tensor(row)) for row in batch])
        keys.extend(row["clip_id"] for row in batch)

        with torch.no_grad():
            chunks.append(model(clips.to(device)).cpu())

        seen = min(start + clips_per_batch, len(rows))
        print(f"  features: {seen}/{len(rows)} clips", end="\r", flush=True)

    features = torch.cat(chunks)
    torch.save({"keys": keys, "features": features, "backbone": backbone,
                "crop": crop, "feature_dim": feature_dim}, path)
    print(f"\ncached {features.shape[0]} x {features.shape[1]} features "
          f"({backbone}, {crop} crop) to {path}")
    return features


def load_cache(backbone, crop="center"):
    """Return ({clip_id: features}, feature_dim), refusing a mismatched cache.

    The check is not paranoia. Training on features from the wrong backbone
    raises no error if the dimensions happen to line up -- it just produces a
    number, and the number is wrong. This project has already been bitten once
    by reading a stale file and reporting what it said.
    """
    path = cache_path(backbone, crop)
    if not os.path.exists(path):
        raise SystemExit(f"no feature cache at {path} -- run: "
                         f"python -m clips.data --backbone {backbone} --crop {crop}")

    blob = torch.load(path, map_location="cpu")
    # Phase 4's blob predates these keys, and could only ever have been resnet18.
    stored = (blob.get("backbone", FRAME_BACKBONE), blob.get("crop", "center"))

    if stored != (backbone, crop):
        raise SystemExit(
            f"{path} holds {stored[0]}/{stored[1]} features, but "
            f"{backbone}/{crop} was asked for -- rebuild with:\n"
            f"  python -m clips.data --backbone {backbone} --crop {crop}")

    features = blob["features"]
    feature_dim = blob.get("feature_dim", features.shape[-1])
    return {key: features[i] for i, key in enumerate(blob["keys"])}, feature_dim


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


def get_loaders(backbone, crop="center", batch_size=16, train_frac=0.7, fold=()):
    rows = load_clips(fold=fold)
    lookup, feature_dim = load_cache(backbone, crop)
    train_rows, test_rows = split_clips(rows, train_frac)

    train_loader = torch.utils.data.DataLoader(
        _dataset(train_rows, lookup), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        _dataset(test_rows, lookup), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, feature_dim


def composition(rows):
    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def class_weights(rows):
    """One weight per class, inversely proportional to how common it is.

    `none` outnumbers `block` by roughly a hundred to one. Left alone, the
    cheapest way for the optimizer to reduce the loss is to answer `none` to
    everything: that scores about 58%, beats nothing, and learns nothing. The
    gradient from four block clips simply cannot outvote the gradient from four
    hundred background ones.

    Weighting makes one block mistake cost as much as a hundred background
    mistakes, so the rare classes are worth paying attention to. It buys
    attention, not examples -- a class with four training clips is still a class
    with four training clips, and this cannot conjure the ones that are missing.

    Computed on the training rows only. Deriving weights from the whole dataset
    would feed the test set's class balance into training, which is a small
    leak, but leaks in this project have a habit of flattering the result.
    """
    counts = composition(rows)
    total = sum(counts.values())
    # Absent classes get weight 0: they contribute no gradient because they
    # contribute no examples, and 1/0 is not a number.
    return torch.tensor(
        [total / (len(CLASSES) * counts[name]) if counts.get(name) else 0.0
         for name in CLASSES],
        dtype=torch.float,
    )


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mvit_v2_s", choices=BACKBONES,
                        help="resnet18 sees one frame at a time (Phase 2-4); "
                             "the rest read the clip as a clip")
    parser.add_argument("--crop", default="center", choices=list(CROP_MODES),
                        help="center keeps the middle 224 of 455 pixels; "
                             "squash keeps the whole frame, aspect and all")
    args = parser.parse_args()

    if args.backbone == FRAME_BACKBONE and args.crop != "center":
        raise SystemExit(f"--crop applies to the video backbones only; "
                         f"{FRAME_BACKBONE} uses its own ImageNet preprocessing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_clips()
    if not rows:
        raise SystemExit("no clips in the manifest -- run: python -m ingest.cut")

    report(rows)
    train_rows, test_rows = split_clips(rows)
    print(f"split: {len(train_rows)} train / {len(test_rows)} test")

    if args.backbone == FRAME_BACKBONE:
        print(f"extracting {FEATURE_DIM}-d features for {CLIP_FRAMES} frames "
              f"per clip on {device}")
        build_cache(rows, device)
        return

    dim = VIDEO_BACKBONES[args.backbone][3]
    print(f"extracting one {dim}-d feature per clip with {args.backbone} "
          f"({args.crop} crop) on {device}")
    build_video_cache(rows, args.backbone, args.crop, device)


if __name__ == "__main__":
    main()
