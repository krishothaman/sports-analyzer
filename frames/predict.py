"""Classify frames with the trained filter.

Unlike training, this path has no cache: a new frame has never been through the
backbone, so both halves run -- backbone for the 512 numbers, head for the answer.
This is the shape Phase 5 uses over a whole match.
"""

import argparse
import glob
import os

import torch
from PIL import Image

from frames.data import CLASSES
from frames.model import FrameHead
from models.backbone import build_backbone


def load_filter(device):
    """Assemble the trained frame filter: frozen eyes plus the learned rulebook.

    Returned as a bundle so a caller running many batches -- ingest/cut.py sifting
    background candidates -- pays the model construction cost once instead of
    rebuilding an 11M-parameter backbone per batch.
    """
    backbone, preprocess = build_backbone(device)

    head = FrameHead(num_classes=len(CLASSES)).to(device)
    head.load_state_dict(torch.load("frame_head.pt", map_location=device))
    head.eval()

    return backbone, preprocess, head


def classify_images(images, device, bundle=None, batch_size=32):
    """Classify already-decoded PIL images.

    The pixels-in entry point. ingest/cut.py has frames in memory straight from
    the video decoder and never writes the rejects to disk, so it cannot use the
    path-based version below.
    """
    backbone, preprocess, head = bundle or load_filter(device)

    preds, confs = [], []

    for start in range(0, len(images), batch_size):
        batch = torch.stack(
            [preprocess(image) for image in images[start:start + batch_size]]
        ).to(device)

        with torch.no_grad():
            features = backbone(batch)           # [N, 512] -- the eyes
            logits = head(features)              # [N, 2]   -- the rulebook
            # Softmax purely so the number is human-readable. Training never
            # needed it; CrossEntropyLoss applied it internally.
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)

        preds.append(predicted.cpu())
        confs.append(confidence.cpu())

    if not preds:
        return torch.empty(0, dtype=torch.long), torch.empty(0)
    return torch.cat(preds), torch.cat(confs)


def classify(paths, device, batch_size=32):
    """Classify frames on disk. Loads in batches so a huge glob stays in memory."""
    bundle = load_filter(device)

    preds, confs = [], []

    for start in range(0, len(paths), batch_size):
        images = [Image.open(p).convert("RGB")
                  for p in paths[start:start + batch_size]]
        batch_preds, batch_confs = classify_images(images, device, bundle, batch_size)
        preds.append(batch_preds)
        confs.append(batch_confs)

    return torch.cat(preds), torch.cat(confs)


def main():
    parser = argparse.ArgumentParser(description="Classify frames as game / not_game.")
    parser.add_argument("pattern", help="a JPEG path or a glob, quoted")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--unsure", action="store_true",
                        help="show only the least confident calls -- the frames worth your eyes")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no files matched {args.pattern}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"classifying {len(paths)} frames on {device}")
    predicted, confidence = classify(paths, device)

    order = range(len(paths))
    if args.unsure:
        # Sorting by confidence surfaces the frames the model found hardest.
        # Those are where labelling mistakes and real weaknesses both hide --
        # reviewing 20 of these is worth far more than reviewing 20 at random.
        order = sorted(order, key=lambda i: confidence[i].item())
    order = list(order)[:args.limit]

    print(f"\n{'frame':<28}{'prediction':>12}{'conf':>8}")
    for i in order:
        print(f"{os.path.basename(paths[i]):<28}"
              f"{CLASSES[predicted[i].item()]:>12}{confidence[i].item():>7.1%}")

    kept = sum(1 for p in predicted if CLASSES[p.item()] == "game")
    print(f"\n{kept}/{len(paths)} frames pass the game-action filter "
          f"({kept / len(paths):.0%} of the broadcast is live play)")


if __name__ == "__main__":
    main()
