# Phase 0 + Phase 1 Implementation Plan — Environment and MNIST

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the GPU actually doing work, establish repo hygiene, and have the owner write and run a
complete CNN training loop on MNIST reaching ≥98% test accuracy.

**Architecture:** Two independent slices. Phase 0 is environment and hygiene — a `.gitignore` before
any large files exist, a CUDA-enabled PyTorch replacing the CPU-only build, and a verification script
the owner runs to confirm the GPU is visible. Phase 1 is a small from-scratch CNN on MNIST, split
into four focused modules (data, model, train, predict) so each concept is isolated and testable.
Nothing from Phase 1 is reused later — its output is the owner's understanding of the training loop.

**Tech Stack:** Python 3.x, PyTorch (CUDA build), torchvision, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-sports-highlight-analyzer-design.md`

## Global Constraints

Copied from the spec. Every task inherits these.

- **The owner runs all training and inference commands.** Claude writes and explains; the owner
  executes. Do not run training scripts on the owner's behalf.
- **The owner never runs a script they cannot roughly describe.** If they can't, the explanation was
  inadequate — re-explain before proceeding.
- **Teaching precedes execution.** Explain what a file does before handing over the command that runs it.
- **Hardware:** NVIDIA RTX 4060 Laptop, 8 GB VRAM. Installed PyTorch is `torch 2.14.0+cpu` and MUST
  be replaced with a CUDA build.
- **Phase 1 success criterion:** ≥98% MNIST test accuracy, training in under ~2 minutes. Below that
  means something is broken, not undertuned.
- **MNIST is a warm-up.** One or two sessions. Do not extend, optimize, or elaborate on it.
- **YAGNI on scaffolding.** Create only the directories a current phase needs. The full tree in the
  spec is a destination, not a day-one skeleton.

---

## File Structure

| File | Responsibility |
|---|---|
| `.gitignore` | Keep venv, datasets, checkpoints, and cached features out of git |
| `requirements.txt` | Non-torch dependencies (torch is installed from a CUDA index, not from here) |
| `scripts/check_env.py` | Report Python/torch versions, CUDA availability, GPU name and VRAM |
| `mnist/data.py` | MNIST download, normalization, DataLoaders |
| `mnist/model.py` | The CNN definition, nothing else |
| `mnist/train.py` | The training loop and evaluation |
| `mnist/predict.py` | Load saved weights, predict on sample images |
| `tests/test_mnist_data.py` | Batch shapes, label ranges, split sizes |
| `tests/test_mnist_model.py` | Forward-pass shape contract |

**On testing in ML:** tests here cover the *deterministic plumbing* — tensor shapes, dataset sizes,
label ranges. These catch the overwhelming majority of real bugs and run in under a second. They
deliberately do **not** assert "the model learns"; that is verified empirically by the ≥98%
checkpoint, because a test that trains a model is too slow and too flaky to be useful.

---

## Task 0: Decide where data lives (blocking, owner's call)

**No code.** This must be settled before anything downloads, and it is the owner's decision.

The project sits in `C:\Users\krish\OneDrive\Documents\Ai sports analyzer`. OneDrive will attempt to
sync everything in it. MNIST is ~55 MB and harmless; Phase 3 footage and cached features will be tens
of gigabytes and will not be harmless.

Present these options and get an explicit answer:

| Option | Effect |
|---|---|
| **A. Exclude the project folder from OneDrive sync** | Code stays where it is, nothing syncs. Loses OneDrive backup of the code — but git is the real backup anyway. |
| **B. Keep code in OneDrive, put data outside it** | e.g. `C:\ml-data\sports-analyzer\`. Code stays backed up; bulk data never touches OneDrive. Requires a configurable data path from day one. |
| **C. Move the whole project out of OneDrive** | Cleanest. e.g. `C:\projects\sports-analyzer`. Costs a one-time move and re-clone of the git repo. |

**Recommendation: B.** It keeps the code backed up in two places, costs one config value, and the
configurable data path is something the project needs by Phase 3 regardless.

- [ ] **Step 1: Present the three options and get an explicit choice from the owner.**
- [ ] **Step 2: Record the decision** as a one-line note appended to §10 of the spec, so the reasoning
      survives.

---

## Task 1: Repo hygiene and CUDA environment

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `scripts/check_env.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: a verified CUDA-enabled PyTorch install; `scripts/check_env.py` runnable at any later
  point to re-verify the environment.

- [ ] **Step 1: Create `.gitignore`**

This comes first, before anything large exists. Retroactively removing a committed 3 GB venv from git
history is genuinely painful; preventing it costs nothing.

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/

# Virtual environment
venv/
.venv/

# Datasets, footage, model weights, cached features
data/
clips/
footage/
features/
checkpoints/
*.mp4
*.mkv
*.webm
*.pt
*.pth
*.npy

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/
```

Note `*.pt` — this ignores trained model weights. Model files are build outputs, not source: they are
large, they are regenerable by re-running training, and they change every run. Git is for the code
that produces them.

- [ ] **Step 2: Create `requirements.txt`**

```
pytest>=8.0
```

Deliberately minimal. `torch` and `torchvision` are **not** listed here because they must be installed
from PyTorch's own CUDA package index — a plain `pip install torch` gives the CPU build, which is
exactly the problem being fixed. The install command is documented in Step 4.

- [ ] **Step 3: Create `scripts/check_env.py`**

```python
"""Environment check: is PyTorch installed correctly and can it see the GPU?

Run this any time something seems off. It answers, in order:
  1. Which Python and which PyTorch build am I running?
  2. Was this PyTorch compiled with CUDA support at all?
  3. Can it actually find and talk to the GPU right now?

A "+cpu" build will always report CUDA unavailable no matter how good the
GPU is -- the support simply is not compiled in.
"""

import sys

import torch


def main():
    print(f"Python:      {sys.version.split()[0]}")
    print(f"PyTorch:     {torch.__version__}")

    # torch.version.cuda is the CUDA version this build was COMPILED against.
    # It is None for a CPU-only build -- that alone tells you the wrong wheel
    # is installed, before we even look for a GPU.
    print(f"Built w/ CUDA: {torch.version.cuda or 'NO -- this is a CPU-only build'}")

    available = torch.cuda.is_available()
    print(f"CUDA usable: {available}")

    if not available:
        print("\nGPU not usable. Either the CPU-only wheel is installed,")
        print("or the NVIDIA driver is not visible to PyTorch.")
        print("See Step 4 of the Phase 0 plan for the reinstall command.")
        return 1

    # Only reachable when a GPU is genuinely available.
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:         {props.name}")
    print(f"VRAM:        {props.total_memory / 1024**3:.1f} GB")

    # A real computation on the device. is_available() can be true while an
    # actual operation still fails (driver mismatch), so we prove it works
    # rather than trusting the flag.
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()  # GPU work is async; wait for it before claiming success
    print(f"Test matmul: OK (result sum {y.sum().item():.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: OWNER RUNS — replace the CPU-only PyTorch with a CUDA build**

Explain before handing this over:

> PyTorch ships as several different builds. The one currently installed
> (`torch 2.14.0+cpu`) has no GPU support compiled into it at all, so it will
> never use the 4060 regardless of drivers. We uninstall it and install the
> build compiled against CUDA — NVIDIA's platform for running general
> computation on a graphics card.

Uninstall first; pip will not reliably swap builds in place:

```bash
pip uninstall -y torch torchvision
```

Then install the CUDA build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**If that index 404s or resolves to the wrong version**, get the exact command from
<https://pytorch.org/get-started/locally/> — select Stable / Windows / Pip / Python / CUDA. That page
is the source of truth and always current; the URL above is the most likely correct one, not a
guarantee.

Then install the test dependency:

```bash
pip install -r requirements.txt
```

- [ ] **Step 5: OWNER RUNS — verify**

```bash
python scripts/check_env.py
```

Expected output:

```
Python:      3.x.x
PyTorch:     2.x.x+cu128
Built w/ CUDA: 12.8
CUDA usable: True
GPU:         NVIDIA GeForce RTX 4060 Laptop GPU
VRAM:        8.0 GB
Test matmul: OK (result sum ...)
```

The critical lines are `CUDA usable: True` and a torch version ending in `+cu...` rather than `+cpu`.
Do not proceed past this step until both are correct — every later phase depends on it.

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt scripts/check_env.py
git commit -m "Phase 0: gitignore, requirements, environment check script"
```

---

## Task 2: MNIST data loading

**Files:**
- Create: `mnist/data.py`
- Test: `tests/test_mnist_data.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `get_loaders(batch_size: int = 64, test_batch_size: int = 1000) -> tuple[DataLoader, DataLoader]`
  returning `(train_loader, test_loader)`. Each yields `(images, labels)` where `images` is a float
  tensor of shape `[B, 1, 28, 28]` and `labels` is an int64 tensor of shape `[B]` with values 0–9.
  Used by Tasks 4 and 5.

- [ ] **Step 1: Write the failing test**

`tests/test_mnist_data.py`:

```python
"""Tests for MNIST data loading.

These check the SHAPE CONTRACT -- that data comes out in the form the model
expects. Shape mismatches are the single most common bug in ML code, and
they are cheap to catch here instead of three layers deep in a training run.
"""

import torch

from mnist.data import get_loaders


def test_train_batch_has_expected_shape():
    train_loader, _ = get_loaders(batch_size=8)
    images, labels = next(iter(train_loader))

    # [batch, channels, height, width] -- 1 channel because MNIST is grayscale
    assert images.shape == (8, 1, 28, 28)
    assert labels.shape == (8,)


def test_labels_are_digits_zero_to_nine():
    train_loader, _ = get_loaders(batch_size=64)
    _, labels = next(iter(train_loader))

    assert labels.dtype == torch.int64
    assert labels.min() >= 0
    assert labels.max() <= 9


def test_images_are_normalized_not_raw_pixels():
    """Raw pixels are 0-255 ints. After ToTensor they are 0-1 floats, and
    after Normalize they straddle zero. Seeing negative values proves the
    normalization step actually ran -- forgetting it is a classic bug that
    makes training mysteriously bad rather than obviously broken."""
    train_loader, _ = get_loaders(batch_size=64)
    images, _ = next(iter(train_loader))

    assert images.dtype == torch.float32
    assert images.min() < 0, "expected normalized values, got un-normalized"
    assert images.max() < 5, "values look like raw pixels, not normalized"


def test_dataset_sizes_are_the_known_mnist_split():
    """MNIST is a fixed, standard dataset: exactly 60k train, 10k test.
    Wrong counts mean a corrupt or partial download."""
    train_loader, test_loader = get_loaders(batch_size=64, test_batch_size=1000)

    assert len(train_loader.dataset) == 60_000
    assert len(test_loader.dataset) == 10_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mnist_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mnist'`

- [ ] **Step 3: Write the implementation**

`mnist/data.py`:

```python
"""MNIST data loading.

Kept separate from the model so the two concerns stay independent: this file
knows about datasets and never about network architecture.
"""

import torch
from torchvision import datasets, transforms

# Every image passes through this pipeline before the network sees it.
transform = transforms.Compose([
    # PIL image (0-255 ints) -> float tensor scaled to 0.0-1.0, shape [1, 28, 28].
    # The leading 1 is the channel dimension: MNIST is grayscale so there is one
    # channel. A colour video frame would have three.
    transforms.ToTensor(),

    # Shift and scale so values are centred near zero instead of near 0.13.
    # These two constants are the actual mean and standard deviation of the
    # MNIST training set. Networks train faster and more stably on inputs that
    # are small and centred -- it keeps gradients in a sane range.
    transforms.Normalize((0.1307,), (0.3081,)),
])


def get_loaders(batch_size=64, test_batch_size=1000):
    """Download MNIST (once) and wrap it in DataLoaders.

    Returns (train_loader, test_loader).
    """
    # download=True only downloads if ./data does not already have the files.
    # train=True is the 60,000 images we learn from; train=False is the 10,000
    # held-out images we never train on, used purely to detect overfitting.
    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # shuffle=True on training data matters. If the network saw all the 0s, then
    # all the 1s, each update would be biased toward whichever digit is on
    # screen. Shuffling makes every batch a fair sample of the whole dataset.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    # The test set is not shuffled -- order is irrelevant when only scoring.
    # It uses a larger batch because no gradients are computed, so it is cheap.
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False
    )

    return train_loader, test_loader
```

- [ ] **Step 4: OWNER RUNS — verify the tests pass**

```bash
pytest tests/test_mnist_data.py -v
```

Expected: 4 passed. The first run also downloads ~55 MB of MNIST into `./data`, so it takes a few
seconds longer than later runs.

- [ ] **Step 5: Commit**

```bash
git add mnist/data.py tests/test_mnist_data.py
git commit -m "Phase 1: MNIST data loading with shape contract tests"
```

---

## Task 3: The CNN

**Files:**
- Create: `mnist/model.py`
- Test: `tests/test_mnist_model.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (the model is independent of the data module).
- Produces: `class MnistCNN(torch.nn.Module)` with `forward(x: Tensor[B, 1, 28, 28]) -> Tensor[B, 10]`.
  Output is raw **logits**, not probabilities. Used by Tasks 4 and 5.

- [ ] **Step 1: Write the failing test**

`tests/test_mnist_model.py`:

```python
"""Tests for the MNIST CNN.

These verify the model's shape contract using random noise as input. We are
not testing that it predicts correctly -- an untrained network predicts
nothing useful. We are testing that data of the right shape goes in and data
of the right shape comes out, which is what actually breaks in practice.
"""

import torch

from mnist.model import MnistCNN


def test_forward_pass_returns_one_score_per_digit():
    model = MnistCNN()
    batch = torch.randn(4, 1, 28, 28)  # 4 fake images

    out = model(batch)

    # 10 numbers per image: one score per digit 0-9
    assert out.shape == (4, 10)


def test_output_is_logits_not_probabilities():
    """The model returns raw scores, NOT probabilities. This matters: PyTorch's
    CrossEntropyLoss applies softmax internally, so applying it here too would
    do it twice and quietly cripple training. Logits are unbounded, so a batch
    of random input should produce at least one negative value."""
    model = MnistCNN()
    out = model(torch.randn(32, 1, 28, 28))

    assert out.min() < 0, "outputs look like probabilities; they should be raw logits"


def test_gradients_reach_the_first_layer():
    """Runs one backward pass and checks a gradient actually arrived at the
    earliest layer. If the network is wired wrong -- a detached tensor, a
    broken connection -- gradients silently stop flowing and the model never
    learns, with no error message. This catches that."""
    model = MnistCNN()
    out = model(torch.randn(2, 1, 28, 28))
    out.sum().backward()

    assert model.conv1.weight.grad is not None
    assert model.conv1.weight.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mnist_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mnist.model'`

- [ ] **Step 3: Write the implementation**

`mnist/model.py`:

```python
"""The MNIST CNN.

Deliberately small and readable. Every layer's output shape is written in a
comment, because shape arithmetic is where beginners get stuck -- and where
the bugs live.
"""

import torch.nn as nn


class MnistCNN(nn.Module):
    """A small convolutional network: 28x28 grayscale image -> 10 digit scores."""

    def __init__(self):
        super().__init__()

        # Conv2d(in_channels, out_channels, kernel_size, padding)
        # 1 input channel (grayscale) -> 16 output channels. "16 channels" means
        # 16 different 3x3 filters, each learning to detect a different pattern.
        # padding=1 keeps the output the same width/height as the input.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # ReLU: replace every negative number with 0. This is the ONLY nonlinear
        # step, and without it stacked conv layers collapse mathematically into
        # a single conv layer -- the network could then only learn straight-line
        # relationships and would fail on anything interesting.
        self.relu = nn.ReLU()

        # MaxPool: take the largest value in each 2x2 block, halving width and
        # height. Cuts computation and makes the network tolerant to small shifts.
        self.pool = nn.MaxPool2d(2)

        # After two conv+pool rounds: 32 channels of 7x7 -> 32*7*7 = 1568 numbers.
        # Those become the input to ordinary dense layers that do the final vote.
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # x arrives as [B, 1, 28, 28]
        x = self.pool(self.relu(self.conv1(x)))   # -> [B, 16, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))   # -> [B, 32,  7,  7]

        # Flatten every image's feature maps into one long vector, keeping the
        # batch dimension. start_dim=1 means "flatten everything except dim 0".
        x = x.flatten(start_dim=1)                # -> [B, 1568]

        x = self.relu(self.fc1(x))                # -> [B, 64]
        x = self.fc2(x)                           # -> [B, 10]

        # Raw logits, deliberately. CrossEntropyLoss applies softmax itself;
        # doing it here as well would apply it twice and break training.
        return x
```

- [ ] **Step 4: OWNER RUNS — verify the tests pass**

```bash
pytest tests/test_mnist_model.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add mnist/model.py tests/test_mnist_model.py
git commit -m "Phase 1: MNIST CNN with shape and gradient-flow tests"
```

---

## Task 4: The training loop

**Files:**
- Create: `mnist/train.py`

**Interfaces:**
- Consumes: `mnist.data.get_loaders`, `mnist.model.MnistCNN`.
- Produces: a saved weights file `mnist_cnn.pt` at the repo root, loadable by Task 5 via
  `MnistCNN().load_state_dict(torch.load("mnist_cnn.pt"))`.

**This is the centrepiece of Phase 1.** Explain the five-step loop before the owner runs anything.

- [ ] **Step 1: Write the implementation**

`mnist/train.py`:

```python
"""Train the MNIST CNN.

The five steps inside the inner loop below are the whole of supervised
learning. Every model in this project -- including the basketball one --
uses this same loop. Only the data and the network change.
"""

import torch
import torch.nn as nn

from mnist.data import get_loaders
from mnist.model import MnistCNN

EPOCHS = 3
LEARNING_RATE = 1e-3


def evaluate(model, loader, device):
    """Return accuracy on a loader, as a fraction between 0 and 1."""
    model.eval()  # switches layers like dropout/batchnorm to inference behaviour

    correct = 0
    total = 0

    # no_grad tells PyTorch to stop tracking operations for backpropagation.
    # We are only scoring here, never learning, so the bookkeeping is pure
    # waste -- this makes evaluation notably faster and lighter on memory.
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)             # [B, 10] scores
            predicted = outputs.argmax(dim=1)   # index of the highest score = the guess

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    # Use the GPU if it is available, otherwise fall back to CPU. MNIST is small
    # enough that CPU works fine -- this line matters much more in later phases.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, test_loader = get_loaders()

    model = MnistCNN().to(device)

    # The loss function: how wrong were we? CrossEntropyLoss is the standard
    # choice for "pick one of N categories". It punishes confident wrong
    # answers far more harshly than uncertain ones.
    criterion = nn.CrossEntropyLoss()

    # The optimizer owns the "nudge every knob" step. Adam adapts its step size
    # per-parameter, which makes it forgiving about the learning rate -- a good
    # default when you do not yet have intuition for tuning it.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()  # back to training behaviour
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # ---- THE FIVE STEPS ----

            # 1. Clear last batch's gradients. PyTorch ACCUMULATES gradients by
            #    default, so forgetting this silently mixes every batch together
            #    and training degrades for no visible reason.
            optimizer.zero_grad()

            # 2. Forward: make predictions.
            outputs = model(images)

            # 3. Measure how wrong we were.
            loss = criterion(outputs, labels)

            # 4. Backward: work out, for every single weight, which direction
            #    would reduce the loss. This is backpropagation.
            loss.backward()

            # 5. Step: move every weight a small distance in that direction.
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}/{EPOCHS}  loss {avg_loss:.4f}  test accuracy {test_acc:.2%}")

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("saved weights to mnist_cnn.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Explain the loop to the owner before they run it**

Walk through the five numbered steps in `main()`. Confirm they can describe what each does in their
own words. Per the global constraints, do not hand over the run command until they can.

Emphasise `optimizer.zero_grad()` specifically: omitting it produces no error, no crash, and no
warning — just a model that trains badly. It is the most common silent bug in beginner PyTorch code.

- [ ] **Step 3: OWNER RUNS — train the model**

```bash
python -m mnist.train
```

Expected output shape:

```
Training on: cuda
epoch 1/3  loss 0.19xx  test accuracy 97.xx%
epoch 2/3  loss 0.05xx  test accuracy 98.xx%
epoch 3/3  loss 0.03xx  test accuracy 98.xx%
saved weights to mnist_cnn.pt
```

**Success criterion: ≥98% test accuracy.** Loss must fall every epoch and accuracy must rise.

If accuracy sits near 10%, the model is guessing at random — that is a wiring bug, not a tuning
problem. Check in this order: is `optimizer.zero_grad()` present, is `loss.backward()` present, is
softmax being applied twice.

- [ ] **Step 4: Commit**

`mnist_cnn.pt` is excluded by `.gitignore` (it is a build output, regenerable by re-running training).

```bash
git add mnist/train.py
git commit -m "Phase 1: MNIST training loop"
```

---

## Task 5: Predict on real images

**Files:**
- Create: `mnist/predict.py`

**Interfaces:**
- Consumes: `mnist.data.get_loaders`, `mnist.model.MnistCNN`, and the `mnist_cnn.pt` file produced by
  Task 4.
- Produces: nothing later tasks depend on. This is the payoff step.

- [ ] **Step 1: Write the implementation**

`mnist/predict.py`:

```python
"""Load the trained model and predict on a handful of test images.

This is what the whole phase was for: watching a network you built read
handwriting it has never seen.
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
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()

    _, test_loader = get_loaders()
    images, labels = next(iter(test_loader))
    images, labels = images[:NUM_SAMPLES].to(device), labels[:NUM_SAMPLES].to(device)

    with torch.no_grad():
        logits = model(images)

        # Softmax turns the raw scores into probabilities summing to 1, purely
        # so the confidence number is human-readable. Training never needed this.
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(dim=1)

    print(f"{'true':>5} {'pred':>5} {'conf':>7}   result")
    for true, pred, conf in zip(labels, predicted, confidence):
        mark = "ok" if true == pred else "WRONG"
        print(f"{true.item():>5} {pred.item():>5} {conf.item():>6.1%}   {mark}")

    correct = (predicted == labels).sum().item()
    print(f"\n{correct}/{NUM_SAMPLES} correct on this sample")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: OWNER RUNS — see it work**

```bash
python -m mnist.predict
```

Expected: 9 or 10 of 10 correct, most confidences above 99%. An occasional miss on a genuinely
ambiguous digit is normal and expected at 98% accuracy — it is not a bug.

- [ ] **Step 3: Commit**

```bash
git add mnist/predict.py
git commit -m "Phase 1: MNIST prediction script"
```

- [ ] **Step 4: Close out Phase 1**

Confirm the owner can answer these without looking. If any answer is shaky, re-explain rather than
moving on — Phase 2 assumes all of it:

1. What are the five steps of the training loop?
2. What is an epoch, and what is a batch?
3. Why is there a test set that we never train on?
4. What does the loss number mean, and why should it go down?
5. Why is `optimizer.zero_grad()` there?

Then stop. Per the global constraints, MNIST does not get extended or optimized. Phase 2 begins.

---

## Self-Review

**Spec coverage.** This plan covers spec §9 Phase 0 (environment, skeleton) and Phase 1 (MNIST
training loop) in full, plus the `.gitignore` gap noted at §10.5 and the OneDrive data-location
question that §3's constraints imply but do not resolve. Phases 2–7 are deliberately out of scope —
per spec §9, each phase gets its own plan written when it starts.

**Placeholder scan.** No TBDs, no "add error handling", no "similar to Task N". Every code step
contains complete, runnable code. The one intentionally open item is Task 0, which is a decision for
the owner rather than a placeholder, and it blocks nothing in Tasks 1–5 (MNIST's 55 MB is harmless
under any of the three options).

**Type consistency.** `get_loaders(batch_size, test_batch_size)` returns `(train_loader, test_loader)`
in Tasks 2, 4 and 5 consistently. `MnistCNN` is the class name in Tasks 3, 4 and 5. `mnist_cnn.pt` is
the weights filename in Tasks 4 and 5 and is covered by the `*.pt` rule in Task 1's `.gitignore`.
Model output is logits in every task that touches it, with softmax applied only in `predict.py` for
display.

**One deliberate deviation from the skill's default.** Tasks 4 and 5 have no unit tests. Training and
inference scripts are verified empirically instead, against the ≥98% criterion. A test that trains a
model is slow and flaky, and would provide less signal than the accuracy number the owner reads
directly. Deterministic logic — shapes, label ranges, dataset sizes, gradient flow — is tested in
Tasks 2 and 3, where tests are fast and meaningful.
