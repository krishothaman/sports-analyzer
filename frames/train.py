"""Train the frame classifier head on cached features.

The five steps are identical to mnist/train.py. Only the data changed: 512-number
descriptions of basketball frames instead of 28x28 pixel grids. That is the point
of the eyes/rulebook split -- the training loop does not care what it is looking at.
"""

import torch
import torch.nn as nn

from frames.data import CLASSES, get_loaders
from frames.model import FrameHead

# 30 where MNIST used 3. An epoch here is ~12 batches of 364 training frames
# rather than 938 batches of 60,000 images, so a single pass barely moves the
# weights. Thirty passes over cached vectors still finishes in about a second.
EPOCHS = 30
LEARNING_RATE = 1e-3


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one full pass over the training data. Return the average loss."""
    model.train()

    running_loss = 0.0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        # ---- THE FIVE STEPS ----
        # 1. clear the gradients left over from the last batch. PyTorch ADDS into
        #    .grad rather than overwriting it, so without this every batch's
        #    corrections pile on top of the previous ones. No error, no warning,
        #    just a model that learns badly.
        optimizer.zero_grad()

        # 2. forward pass: features in, one score per class out
        outputs = model(features)

        # 3. how wrong were we? one number
        loss = criterion(outputs, labels)

        # 4. work backwards and compute, for every weight, which direction would
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


def confusion_matrix(preds, targets, num_classes):
    """matrix[true][predicted] = count.

    Accuracy compresses every kind of mistake into one number. This does not: it
    shows which class gets confused for which, which is what tells you what to do
    next. 'not_game' called 'game' and 'game' called 'not_game' are different
    problems with different fixes.
    """
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true, pred in zip(targets, preds):
        matrix[true.item(), pred.item()] += 1
    return matrix


def print_report(matrix, classes):
    width = max(len(name) for name in classes) + 2

    print("\nconfusion matrix (rows = truth, columns = prediction)")
    print(" " * width + "".join(f"{name:>10}" for name in classes))
    for i, name in enumerate(classes):
        counts = "".join(f"{matrix[i, j].item():>10}" for j in range(len(classes)))
        print(f"{name:<{width}}{counts}")

    print(f"\n{'class':<{width}}{'precision':>11}{'recall':>9}{'support':>9}")
    for i, name in enumerate(classes):
        # support:   how many frames really were this class
        # recall:    of those, how many did we catch
        # precision: when we said this class, how often were we right
        support = matrix[i, :].sum().item()
        predicted_as = matrix[:, i].sum().item()
        hits = matrix[i, i].item()
        recall = hits / support if support else 0.0
        precision = hits / predicted_as if predicted_as else 0.0
        print(f"{name:<{width}}{precision:>10.1%}{recall:>9.1%}{support:>9}")

    total = matrix.sum().item()
    correct = matrix.diagonal().sum().item()
    print(f"\noverall accuracy {correct}/{total} = {correct / total:.2%}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, test_loader = get_loaders()
    print(f"{len(train_loader.dataset)} train / {len(test_loader.dataset)} test frames")

    model = FrameHead(num_classes=len(CLASSES)).to(device)
    trainable = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable:,}  (backbone: 11,000,000+, all frozen)\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch % 5 == 0 or epoch == EPOCHS:
            preds, targets = collect_predictions(model, test_loader, device)
            acc = (preds == targets).float().mean().item()
            print(f"epoch {epoch:>3}/{EPOCHS}  loss {avg_loss:.4f}  test accuracy {acc:.2%}")

    preds, targets = collect_predictions(model, test_loader, device)
    print_report(confusion_matrix(preds, targets, len(CLASSES)), CLASSES)

    torch.save(model.state_dict(), "frame_head.pt")
    print("\nsaved weights to frame_head.pt")


if __name__ == "__main__":
    main()
