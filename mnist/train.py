"""Train the MNIST CNN.

The five steps inside train_one_epoch() are the whole of supervised learning.
Every model in this project -- including the basketball one -- uses this same
loop. Only the data and the network change.
"""

import torch
import torch.nn as nn

from mnist.data import get_loaders
from mnist.model import MnistCNN

EPOCHS = 3
LEARNING_RATE = 1e-3


def evaluate(model, loader, device):
    """Return accuracy on a loader, as a fraction between 0.0 and 1.0."""
    # Modules have a train mode and an eval mode. Ours behaves identically in
    # both, but switching is the habit that prevents a real bug later.
    model.eval()

    correct = 0
    total = 0

    # Normally every tensor records the operations done to it, so PyTorch can
    # trace backwards later and work out how to improve. That recording is how
    # learning happens -- and it costs time and memory. We are only scoring
    # here, never learning, so no_grad switches the recording off.
    with torch.no_grad():
        for images, labels in loader:
            # Model and data must sit on the same device or nothing can talk.
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)             # [B, 10] scores
            predicted = outputs.argmax(dim=1)   # POSITION of the top score = the guess

            # (predicted == labels) gives [True, False, ...]; .sum() counts the
            # Trues; .item() unwraps the result from its one-element tensor.
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one full pass over the training data. Return the average loss."""
    model.train()

    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # ---- THE FIVE STEPS ----

        # 1. Clear the gradients left over from the last batch. PyTorch ADDS
        #    into .grad rather than overwriting it, so skipping this steers
        #    the model by a ghost of every batch seen so far. No error, no
        #    warning -- just a model that learns badly.
        optimizer.zero_grad()

        # 2. Forward pass: make predictions.
        outputs = model(images)

        # 3. Measure how wrong they were, as a single number.
        loss = criterion(outputs, labels)

        # 4. Backward pass: walk the recorded history in reverse and fill in,
        #    for every weight in the network, which direction would reduce
        #    that number. This is backpropagation.
        loss.backward()

        # 5. Read those directions and actually move the weights.
        #    backward() decides; step() acts.
        optimizer.step()

        # .item() matters here: adding the raw tensor would keep every batch's
        # recorded history alive in memory for the whole epoch.
        running_loss += loss.item()

    return running_loss / len(loader)


def main():
    # Use the GPU if it is available, otherwise fall back to CPU. MNIST is small
    # enough that CPU works fine -- this line matters much more in later phases.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, test_loader = get_loaders()

    model = MnistCNN().to(device)

    # The loss function: how wrong were we? CrossEntropyLoss is the standard
    # choice for "pick one of N categories". It punishes confident wrong
    # answers far more harshly than uncertain ones, and applies softmax
    # internally -- which is why the model must not apply it too.
    criterion = nn.CrossEntropyLoss()

    # The optimizer owns the "nudge every weight" step. Adam adapts its step
    # size per-parameter, which makes it forgiving about the learning rate --
    # a good default before you have intuition for tuning it.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}/{EPOCHS}  loss {avg_loss:.4f}  test accuracy {test_acc:.2%}")

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("saved weights to mnist_cnn.pt")


if __name__ == "__main__":
    main()
