"""Let the backbone's last block learn: the first time Phase 5's eyes are not frozen.

Phase 5 trained one linear layer, about 5k weights, on MViTv2-S features that
were fixed forever. The network was trained on YouTube videos and has never
seen a broadcast basketball close-up. Letting its last block adjust (roughly
7M weights) gives it a chance to reshape its description around rims and
balls, not just re-weigh a description built for something else.

With far more weights than clips, this can memorise the training set, so:

  * only the last block learns; the trunk stays frozen and its output is
    cached once (models/backbone.split_video_backbone)
  * the tail learns 100x slower than the new head
  * settings are chosen on the TRAINING matches: train on match01, watch
    match02 (--validate). match03 is scored once, at the end, with the
    epochs chosen there. Picking epochs by watching the test score would
    make the test score a tuning target instead of a measurement.

    python -m clips.finetune --build                      # token cache, once
    python -m clips.finetune --build --extra jitter       # and for extra clips
    python -m clips.finetune --validate [--extra jitter]  # pick --epochs
    python -m clips.finetune --epochs N --seed 0 [--extra jitter]
"""

import argparse
import os

import torch
import torch.nn as nn

from clips.data import (CLASS_TO_INDEX, CLASSES, EXTRA_MANIFESTS, VIEWS,
                        add_extra_rows, class_weights, clip_frame_paths,
                        load_clip_tensor, load_clips, split_clips)
from clips.model import build_head
from clips.train import GOAL_GROUPS
from models.backbone import split_video_backbone
from models.metrics import confusion_matrix, group_matrix, grouped_report, print_report

BACKBONE = "mvit_v2_s"
VIEW = "hoop"
FOLD = ("block", "steal")
TAIL_LR = 1e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16


def token_cache_path(extra=None):
    suffix = f"_{extra}" if extra else ""
    return os.path.join("data", "features", f"tokens_{BACKBONE}_{VIEW}{suffix}.pt")


def finetune_checkpoint_path(extra=()):
    """Never the shipped head's name, whatever the extras."""
    suffix = "".join(f"_{name}" for name in extra)
    return f"clip_head_{BACKBONE}_{VIEW}_finetune{suffix}.pt"


def build_tokens(rows, device, extra=None):
    """Run every clip through the frozen trunk once; save its output tokens.

    Stored as float16 to halve the size (~0.6 MB a clip). The tail reads them
    back as float32.
    """
    root, mode = VIEWS[VIEW]
    rows = [row for row in rows if os.path.exists(clip_frame_paths(row, root)[-1])]
    trunk, _, transform, _ = split_video_backbone(BACKBONE, device, mode)

    keys, chunks, thw = [], [], None
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        clips = torch.stack([transform(load_clip_tensor(row, root)) for row in batch])
        with torch.no_grad():
            tokens, thw = trunk(clips.to(device))
        chunks.append(tokens.half().cpu())
        keys.extend(row["clip_id"] for row in batch)
        print(f"  tokens: {min(start + 4, len(rows))}/{len(rows)} clips", end="\r", flush=True)

    tokens = torch.cat(chunks)
    path = token_cache_path(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"keys": keys, "tokens": tokens, "thw": thw, "backbone": BACKBONE,
                "crop": VIEW, "extra": extra}, path)
    print(f"\ncached {tuple(tokens.shape)} tokens (grid {thw}) to {path}")


def load_tokens(extra=None):
    path = token_cache_path(extra)
    if not os.path.exists(path):
        flag = f" --extra {extra}" if extra else ""
        raise SystemExit(f"no token cache at {path} -- run: python -m clips.finetune --build{flag}")
    blob = torch.load(path, map_location="cpu")
    if (blob["backbone"], blob["crop"], blob.get("extra")) != (BACKBONE, VIEW, extra):
        raise SystemExit(f"{path} does not hold {BACKBONE}/{VIEW}/{extra} tokens")
    return {key: i for i, key in enumerate(blob["keys"])}, blob["tokens"], tuple(blob["thw"])


class FineTuned(nn.Module):
    """The trainable tail, then Phase 5's one linear layer."""

    def __init__(self, tail, thw, num_classes, feature_dim):
        super().__init__()
        self.tail = tail
        self.thw = thw
        self.head = build_head("clip", num_classes, feature_dim)

    def forward(self, tokens):
        return self.head(self.tail(tokens.float(), self.thw))


def dataset(rows, sources):
    """TensorDataset of (tokens, label) for rows found in any of `sources`."""
    picked, labels = [], []
    for row in rows:
        for index, tokens in sources:
            if row["clip_id"] in index:
                picked.append(tokens[index[row["clip_id"]]])
                labels.append(CLASS_TO_INDEX[row["label"]])
                break
        else:
            raise SystemExit(f"{row['clip_id']} is in no token cache -- rebuild with --build")
    return torch.utils.data.TensorDataset(torch.stack(picked), torch.tensor(labels))


def goal_line(matrix):
    names = list(GOAL_GROUPS)
    cells = []
    for i, name in enumerate(names):
        support, called, hits = matrix[i, :].sum().item(), matrix[:, i].sum().item(), matrix[i, i].item()
        cells.append(f"{name} {hits / support if support else 0:.1%}"
                     f"/{hits / called if called else 0:.0%}")
    total = matrix.sum().item()
    return "  ".join(cells) + f"  acc {matrix.diagonal().sum().item() / total:.2%}"


def predictions(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for tokens, labels in loader:
            preds.append(model(tokens.to(device)).argmax(dim=1).cpu())
            targets.append(labels)
    return torch.cat(preds), torch.cat(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="build the token cache")
    parser.add_argument("--validate", action="store_true",
                        help="train on the first training match, report the second every epoch")
    parser.add_argument("--extra", nargs="*", default=[], choices=sorted(EXTRA_MANIFESTS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, metavar="PATH")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.build:
        if len(args.extra) > 1:
            raise SystemExit("build one extra cache at a time")
        extra = args.extra[0] if args.extra else None
        rows = load_clips(EXTRA_MANIFESTS[extra]) if extra else load_clips()
        build_tokens(rows, device, extra)
        return

    if args.seed is not None:
        torch.manual_seed(args.seed)

    train_rows, test_rows = split_clips(load_clips(fold=FOLD), quiet=True)
    sources = [load_tokens()[:2]]
    extra_rows = []
    for name in args.extra:
        rows = load_clips(EXTRA_MANIFESTS[name], fold=FOLD)
        index, tokens, _ = load_tokens(name)
        sources.append((index, tokens))
        extra_rows += [row for row in rows if row["clip_id"] in index]
    thw = load_tokens()[2]

    if args.validate:
        matches = sorted({row["match_id"] for row in train_rows})
        watch = matches[-1]
        fit = [row for row in train_rows if row["match_id"] != watch]
        fit = add_extra_rows(fit, test_rows, [r for r in extra_rows if r["match_id"] != watch])
        evaluate_rows, label = [r for r in train_rows if r["match_id"] == watch], watch
    else:
        fit = add_extra_rows(train_rows, test_rows, extra_rows)
        evaluate_rows, label = test_rows, "test"

    print(f"fit on {len(fit)} clips; {label}: {len(evaluate_rows)} clips")
    fit_loader = torch.utils.data.DataLoader(dataset(fit, sources), batch_size=BATCH_SIZE,
                                             shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset(evaluate_rows, sources),
                                              batch_size=BATCH_SIZE)

    _, tail, _, feature_dim = split_video_backbone(BACKBONE, device, VIEWS[VIEW][1])
    model = FineTuned(tail, thw, len(CLASSES), feature_dim).to(device)
    trainable = sum(p.numel() for p in tail.parameters())
    print(f"trainable: tail {trainable:,} + head "
          f"{sum(p.numel() for p in model.head.parameters()):,} weights")

    criterion = nn.CrossEntropyLoss(weight=class_weights(fit).to(device))
    optimizer = torch.optim.AdamW([{"params": model.tail.parameters(), "lr": TAIL_LR},
                                   {"params": model.head.parameters(), "lr": HEAD_LR}],
                                  weight_decay=WEIGHT_DECAY)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for tokens, labels in fit_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(tokens), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()

        preds, targets = predictions(model, fit_loader, device)
        train_acc = (preds == targets).float().mean().item()
        line = f"epoch {epoch:>3}  loss {running / len(fit_loader):.4f}  train {train_acc:.1%}"
        # Every epoch on the watched training match -- that is what it is for.
        # Never on the test match: its score is read once, below.
        if args.validate:
            preds, targets = predictions(model, eval_loader, device)
            line += "   " + watch + ": " + goal_line(
                group_matrix(preds, targets, CLASSES, GOAL_GROUPS))
        print(line)

    if args.validate:
        print("\npick the epoch where the watched match peaks, then run without --validate")
        return

    preds, targets = predictions(model, eval_loader, device)
    print_report(confusion_matrix(preds, targets, len(CLASSES)), CLASSES)
    grouped_report(preds, targets, CLASSES, GOAL_GROUPS,
                   "collapsed to the goal: none / field_goal / free_throw")

    path = args.out or finetune_checkpoint_path(args.extra)
    torch.save({"state_dict": model.head.state_dict(), "tail_state_dict": tail.state_dict(),
                "head": "clip", "backbone": BACKBONE, "crop": [VIEW], "extra": args.extra,
                "feature_dim": feature_dim, "classes": CLASSES}, path)
    print(f"\nsaved weights to {path}")


if __name__ == "__main__":
    main()
