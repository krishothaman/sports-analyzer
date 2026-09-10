"""Train a clip head on cached [16, 512] features.

The five steps in train_one_epoch are character-for-character the ones from
mnist/train.py and frames/train.py. Digits, then frames, now clips: the loop has
never cared what it is looking at, which is the whole argument for splitting the
frozen "eyes" from the learned "rulebook".

What is new here exists because of something in this dataset rather than because
of anything about training in general:

  * `--backbone` and `--head`, so the Phase 4 result (resnet18 + gru) and the
    Phase 5 one (mvit_v2_s + clip) are produced by the same code and differ only
    in what did the looking. A comparison run through two different scripts is
    not a comparison.
  * class weighting, because `none` outnumbers `block` about a hundred to one
  * train accuracy printed beside test accuracy, because these heads have more
    parameters than this dataset has clips and could simply memorise it

Phase 4 asked whether frame order matters, and answered yes -- worth 13.5 points
in-domain. Phase 5 asks the follow-up: if order matters, is a 111k-parameter GRU
on ImageNet features the right way to read it, or should the backbone that was
trained on 400 classes of action do that job instead?
"""

import argparse

import torch
import torch.nn as nn

from clips.data import (BACKBONES, CLASSES, FRAME_BACKBONE, VIEWS,
                        class_weights, get_loaders, load_clips, report,
                        split_clips)
from clips.model import HEADS, SEQUENCE_HEADS, build_head
from models.metrics import (coarse_report, confusion_matrix, grouped_report,
                            print_report, thin_classes)

EPOCHS = 60
LEARNING_RATE = 1e-3
# Small heads on a few hundred cached vectors overfit readily. This is the
# cheapest regulariser available and costs nothing to try.
WEIGHT_DECAY = 1e-4

# The goal, as agreed: find the made baskets, and tell them from free throws and
# from nothing. Two-versus-three and dunks are refinements inside `field_goal`;
# blocks and steals are not scoring plays at all.
GOAL_GROUPS = {
    "none": ["none", "block", "steal"],
    "field_goal": ["two_pointer", "three_pointer", "dunk"],
    "free_throw": ["free_throw"],
}


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one full pass over the training data. Return the average loss."""
    model.train()

    running_loss = 0.0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        # ---- THE FIVE STEPS ----
        # 1. clear last batch's gradients. PyTorch ADDS into .grad rather than
        #    overwriting, so without this every batch's corrections pile onto
        #    the previous ones. No error, no warning, just worse learning.
        optimizer.zero_grad()

        # 2. forward pass: [B, 16, 512] in, one score per class out
        outputs = model(features)

        # 3. how wrong were we? one number
        loss = criterion(outputs, labels)

        # 4. work backwards to find, for every weight, which direction would
        #    reduce that number. This only computes -- it changes nothing.
        loss.backward()

        # 5. actually edit the weights using those directions
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def collect_predictions(model, loader, device):
    """Return (predictions, true labels) over a whole loader."""
    model.eval()

    preds, targets = [], []

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features.to(device))
            preds.append(outputs.argmax(dim=1).cpu())
            targets.append(labels)

    return torch.cat(preds), torch.cat(targets)


def accuracy(model, loader, device):
    preds, targets = collect_predictions(model, loader, device)
    return (preds == targets).float().mean().item()


def check_pairing(head, backbone):
    """Refuse head/backbone combinations whose tensor shapes cannot line up.

    A sequence head needs [B, 16, D]; a clip head needs [B, D]. Getting this
    wrong usually raises a shape error, but not always -- and a run that trains
    on the wrong thing and still prints a number is the failure mode this
    project keeps having to guard against.
    """
    sequence_cache = backbone == FRAME_BACKBONE

    if head in SEQUENCE_HEADS and not sequence_cache:
        raise SystemExit(
            f"--head {head} reads 16 per-frame vectors, but {backbone} returns "
            f"one vector for the whole clip.\n"
            f"  use: --backbone {backbone} --head clip")

    if head not in SEQUENCE_HEADS and sequence_cache:
        raise SystemExit(
            f"--head {head} expects one vector per clip, but {FRAME_BACKBONE} "
            f"caches 16 of them.\n"
            f"  use: --backbone {FRAME_BACKBONE} --head gru")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", default="clip", choices=sorted(HEADS),
                        help="meanpool/gru read a 16-frame sequence "
                             "(resnet18 only); clip reads one vector")
    parser.add_argument("--backbone", default="mvit_v2_s", choices=BACKBONES,
                        help="which cached features to train on")
    parser.add_argument("--crop", nargs="+", default=["center"], choices=list(VIEWS),
                        help="one or more views; several are concatenated, "
                             "e.g. --crop squash hoop")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=None,
                        help="fix the random init and shuffling, so a run can "
                             "be repeated -- and so several can be averaged")
    parser.add_argument("--fold", nargs="*", default=[], metavar="CLASS",
                        help="relabel these classes as none, e.g. --fold block steal")
    args = parser.parse_args()

    check_pairing(args.head, args.backbone)
    if args.backbone == FRAME_BACKBONE and args.crop != ["center"]:
        raise SystemExit(f"{FRAME_BACKBONE} has one cache, built with its own "
                         f"preprocessing -- drop --crop")
    unknown = [name for name in args.fold if name not in CLASSES or name == "none"]
    if unknown:
        raise SystemExit(f"cannot fold {unknown} -- choose event classes from {CLASSES}")

    # One run on 293 test clips moves by a clip or two on random init alone,
    # which is the size of the differences being compared. Seeding makes a run
    # repeatable; averaging several seeds is what makes a comparison real.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    rows = load_clips(fold=tuple(args.fold))
    report(rows)

    train_rows, _ = split_clips(rows, quiet=True)
    views = "+".join(args.crop)
    train_loader, test_loader, feature_dim = get_loaders(
        args.backbone, tuple(args.crop), fold=tuple(args.fold))
    print(f"\nbackbone: {args.backbone} ({views} view, {feature_dim}-d)"
          f"   head: {args.head}")
    print(f"{len(train_loader.dataset)} train / {len(test_loader.dataset)} test clips")

    model = build_head(args.head, num_classes=len(CLASSES),
                       feature_dim=feature_dim).to(device)
    trainable = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,}  "
          f"(the {args.backbone} backbone is frozen and not counted)")

    weights = class_weights(train_rows).to(device)
    print("class weights: " + "  ".join(
        f"{name}={w:.1f}" for name, w in zip(CLASSES, weights.tolist())) + "\n")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch % 10 == 0 or epoch == args.epochs:
            train_acc = accuracy(model, train_loader, device)
            test_acc = accuracy(model, test_loader, device)
            # The gap is the thing to watch. Train climbing while test stalls
            # means the head is memorising these particular clips, and more
            # epochs will make that worse rather than better.
            print(f"epoch {epoch:>3}/{args.epochs}  loss {avg_loss:.4f}  "
                  f"train {train_acc:.2%}  test {test_acc:.2%}  "
                  f"gap {train_acc - test_acc:+.1%}")

    preds, targets = collect_predictions(model, test_loader, device)
    matrix = confusion_matrix(preds, targets, len(CLASSES))
    print_report(matrix, CLASSES)

    thin = thin_classes(matrix, CLASSES)
    if thin:
        print(f"\nrows built on too few test examples to read ({', '.join(thin)}) --"
              "\na recall computed from a handful of clips is a coin landing, not a"
              "\nmeasurement. Do not quote these numbers.")

    coarse_report(preds, targets, CLASSES)
    grouped_report(preds, targets, CLASSES, GOAL_GROUPS,
                   "collapsed to the goal: none / field_goal / free_throw")

    # Named for what produced it. Two runs differing only in --crop are two
    # different models, and overwriting one with the other loses the comparison.
    path = f"clip_head_{args.backbone}_{views}_{args.head}.pt"
    torch.save({"state_dict": model.state_dict(), "head": args.head,
                "backbone": args.backbone, "crop": args.crop,
                "feature_dim": feature_dim, "classes": CLASSES}, path)
    print(f"\nsaved weights to {path}")


if __name__ == "__main__":
    main()
