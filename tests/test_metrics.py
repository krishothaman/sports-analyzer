"""Contracts for the grouped report -- the one that measures the project's goal."""

import pytest
import torch

from clips.data import CLASSES
from models.metrics import grouped_report

TOY = ["a", "b", "c", "none"]
TOY_GROUPS = {"fg": ["a", "b"], "c": ["c"], "none": ["none"]}


def test_a_mix_up_inside_a_group_counts_as_right(capsys):
    # A three-pointer called a two-pointer is still a made field goal found.
    preds = torch.tensor([1, 0, 2, 3])
    targets = torch.tensor([0, 0, 2, 3])
    matrix = grouped_report(preds, targets, TOY, TOY_GROUPS, "toy")
    assert matrix.diagonal().sum().item() == 4


def test_a_mix_up_across_groups_still_counts_as_wrong(capsys):
    preds = torch.tensor([3])                 # said none
    targets = torch.tensor([0])               # was a field goal
    matrix = grouped_report(preds, targets, TOY, TOY_GROUPS, "toy")
    assert matrix.diagonal().sum().item() == 0


def test_every_class_must_land_in_exactly_one_group():
    # A class left out would crash on indexing at best; a class in two groups
    # would be counted as whichever came first, silently.
    with pytest.raises(ValueError):
        grouped_report(torch.tensor([0]), torch.tensor([0]), TOY,
                       {"fg": ["a", "b"], "none": ["none"]}, "missing c")
    with pytest.raises(ValueError):
        grouped_report(torch.tensor([0]), torch.tensor([0]), TOY,
                       {"fg": ["a", "b", "c"], "c": ["c"], "none": ["none"]}, "c twice")


def test_the_goal_groups_cover_every_real_class_exactly_once():
    from clips.train import GOAL_GROUPS
    members = [name for group in GOAL_GROUPS.values() for name in group]
    assert sorted(members) == sorted(CLASSES)
