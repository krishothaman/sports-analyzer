"""The two clip heads, and the experiment they exist to settle.

Phase 2's head read one frame: 512 numbers in, a class out. A clip is 16 frames,
so the input is now [16, 512] and something has to turn that sequence into a
single answer. *How* it does that is the open question spec section 10 leaves
for this phase, to be decided "by measurement, not preference".

So there are two heads here, and they disagree about one thing only:

    MeanPoolHead  averages the 16 frames together  -- order is destroyed
    GRUHead       reads the 16 frames in sequence  -- order is available

That single difference is the point. Everything Phase 3 built assumes motion
carries information: clips instead of frames, a 2-second window, timing
calibrated against each broadcaster's graphic lag to a tenth of a second. None
of that has ever been tested. If a head that cannot tell forwards from backwards
scores as well as one that can, then the frames are being classified on
appearance alone and the motion was never used.

That would not be a failure of the GRU. It would mean one of:

  * the backbone's per-frame features already separate these classes, and a
    still of a ball leaving a shooter's hands is simply enough, or
  * 2 seconds at 8fps is the wrong window -- spec section 10 question 4 flags
    exactly this as revisitable with real data.

Neither is visible in an accuracy number. Both change what to build next, which
is why the baseline is built into the phase rather than bolted on if results
disappoint.
"""

import torch.nn as nn

from models.backbone import FEATURE_DIM


class MeanPoolHead(nn.Module):
    """Average the 16 frames into one vector, then a single linear layer.

    Deliberately the dumbest thing that could work, and a strict control: the
    mean of a set is the same whatever order you add it in, so this head is
    mathematically incapable of noticing that the ball went up before it went
    through. Whatever it scores was achieved on appearance alone.

    It is also the direct descendant of Phase 2's FrameHead -- one linear layer
    on frozen features, a linear probe. Same reasoning applies: if a probe can
    separate the classes, the features already contain the answer.
    """

    def __init__(self, num_classes, feature_dim=FEATURE_DIM):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # x: [B, 16, 512] -> mean over dim=1 collapses the frames -> [B, 512]
        return self.fc(x.mean(dim=1))


class GRUHead(nn.Module):
    """Read the frames in order, classify from what the last step remembers.

    A GRU walks the sequence one frame at a time, carrying a hidden state it
    updates at each step and choosing what to keep and what to forget. After 16
    steps that state is a summary of the whole clip *as a sequence* -- which is
    the thing mean-pooling throws away.

    Sized small on purpose. At `hidden=64` this is roughly 30x MeanPoolHead's
    parameters against a few hundred training clips, so it is quite capable of
    memorising the training set instead of learning anything, and would then
    report a fine training accuracy while being useless. That is why clips/train.py
    prints train and test accuracy side by side, and why dropout is on: a gap
    between those two numbers is the tell.
    """

    def __init__(self, num_classes, feature_dim=FEATURE_DIM, hidden=64, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # batch_first means x is [B, 16, 512] rather than [16, B, 512].
        # outputs holds the state after every frame; we want only the last one,
        # which is `hidden` -- shaped [layers, B, hidden], hence hidden[-1].
        _, hidden = self.gru(x)
        return self.fc(self.drop(hidden[-1]))


HEADS = {"meanpool": MeanPoolHead, "gru": GRUHead}


def build_head(name, num_classes):
    if name not in HEADS:
        raise SystemExit(f"unknown head '{name}' -- choose from {sorted(HEADS)}")
    return HEADS[name](num_classes=num_classes)
