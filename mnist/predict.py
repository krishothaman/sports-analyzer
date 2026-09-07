"""Load the trained model and predict on a handful of test images.

This is what the whole phase was for: watching a network you built read
handwriting it has never seen.

Kept separate from train.py because training and inference are different jobs.
Training needs the full dataset, an optimizer and gradients. Inference needs
only the saved weights -- which is why this file has no optimizer, no loss
function, and no loop over epochs.
"""

import torch

from mnist.data import get_loaders
from mnist.model import MnistCNN

NUM_SAMPLES = 10


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MnistCNN().to(device)

    # A state_dict is just a dictionary of the learned weights. The architecture
    # comes from MnistCNN() above -- the file holds only the numbers, which is
    # why you need both the class definition and the saved file to reload a model.
    # map_location=device means "load these numbers onto whichever device we are
    # using", so weights saved on a GPU still load on a CPU-only machine.
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))

    # eval() flips the module into scoring mode. This model has no dropout or
    # batchnorm so it changes nothing today, but forgetting it on a model that
    # does have them is a classic silent bug -- so it is a habit worth forming.
    model.eval()

    _, test_loader = get_loaders()

    # iter() turns the loader into something you can pull from; next() pulls one
    # batch. The test loader batches 1000 at a time, so this hands us 1000 images
    # and we slice off the first NUM_SAMPLES.
    images, labels = next(iter(test_loader))
    images, labels = images[:NUM_SAMPLES].to(device), labels[:NUM_SAMPLES].to(device)

    with torch.no_grad():
        logits = model(images)          # [NUM_SAMPLES, 10] raw scores

        # Softmax turns the raw scores into probabilities summing to 1, purely
        # so the confidence number is human-readable. Training never needed this
        # -- CrossEntropyLoss did it internally.
        probs = torch.softmax(logits, dim=1)

        # max() returns two things: the highest value, and where it was found.
        # The value is the confidence; the position is the predicted digit.
        confidence, predicted = probs.max(dim=1)

    print(f"{'true':>5} {'pred':>5} {'conf':>7}   result")
    for true, pred, conf in zip(labels, predicted, confidence):
        mark = "ok" if true == pred else "WRONG"
        print(f"{true.item():>5} {pred.item():>5} {conf.item():>6.1%}   {mark}")

    correct = (predicted == labels).sum().item()
    print(f"\n{correct}/{NUM_SAMPLES} correct on this sample")


if __name__ == "__main__":
    main()
