"""How results get read. Shared by every phase that trains anything.

Accuracy is one number, and one number cannot tell you what to do next. A
confusion matrix can: it shows *which* class got mistaken for *which*, and
"dunk called two_pointer" and "two_pointer called none" are different problems
with different fixes. Everything here exists to make the mistakes legible.
"""

import torch


def confusion_matrix(preds, targets, num_classes):
    """matrix[true][predicted] = count."""
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
        # support:   how many really were this class
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


def coarse_report(preds, targets, classes, background="none"):
    """Collapse every event class into one and report `event` vs `background`.

    This is the number the product actually rests on. A timeline that says
    "something happened at 14:07" and gets the label wrong is still useful --
    you can click it and look. A timeline that misses the moment entirely is
    not. So a model scoring 45% across seven classes but 90% at telling
    basketball-happening from nothing-happening is a working foundation for
    Phase 5, and the seven-class number on its own would hide that completely.

    It also separates the two ways the fine-grained score can be bad: failing
    to find events at all, versus finding them and naming them wrong. Those
    have opposite fixes -- more data for the former, better features or a
    longer window for the latter.
    """
    background_index = classes.index(background)

    # 0 = background, 1 = an event of some kind.
    binary_preds = (preds != background_index).long()
    binary_targets = (targets != background_index).long()

    matrix = confusion_matrix(binary_preds, binary_targets, 2)
    print("\n--- collapsed to event vs nothing (what Phase 5 needs) ---")
    print_report(matrix, [background, "event"])
    return matrix


def thin_classes(matrix, classes, minimum=20):
    """Classes with too few test examples for their row to mean anything.

    A recall of 33% computed from three examples is not a measurement, it is a
    coin landing. Naming these explicitly stops a number being quoted that was
    never real -- the failure this project keeps meeting is the one that looks
    like a result.
    """
    return [name for i, name in enumerate(classes)
            if 0 < matrix[i, :].sum().item() < minimum]
