"""The 'eyes': frozen, pretrained networks that turn pictures into numbers.

Two kinds live here, and the difference between them is the whole of Phase 5.

`build_backbone` is Phase 2's: ResNet-18 trained on ImageNet, which looks at ONE
STILL PICTURE at a time and knows nothing about motion. Feed it a clip and you
get 16 unrelated descriptions; whether the ball went up before it went through
has to be reconstructed afterwards by a GRU. Phase 4 measured what that costs.

`build_video_backbone` is the replacement: a network trained on Kinetics-400,
400 classes of *action*, which reads all 16 frames at once and was built from
the start to understand movement. It also happens to know what a basketball is,
which ResNet-18 never did -- Kinetics-400 contains `dunking basketball`,
`shooting basketball`, `playing basketball` and `dribbling basketball`.

Nothing here is specific to basketball or to this project. The sport-specific
knowledge still lives in the head.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import (MC3_18_Weights, MViT_V2_S_Weights,
                                      Swin3D_T_Weights, mc3_18, mvit_v2_s,
                                      swin3d_t)
from torchvision.transforms import functional as F

# ResNet-18's penultimate layer is 512 wide. Swapping to a different backbone
# means changing this constant along with build_backbone.
FEATURE_DIM = 512


def build_backbone(device):
    """Return (frozen feature extractor, matching preprocessing transform)."""
    # DEFAULT is torchvision's best available ImageNet checkpoint for this
    # architecture: weights learned from 1.2 million labelled photographs. That
    # training is the thing we are borrowing -- it is why 520 basketball frames
    # is enough to build a classifier, when learning to see from scratch would
    # need hundreds of thousands.
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # resnet18 normally ends in Linear(512 -> 1000) to name an ImageNet class.
    # We do not want a class name -- "basketball" is not one of the 1000, and
    # "crowd of people" is not the question we are asking. We want the 512
    # numbers feeding that final layer: the description the network built before
    # committing to an answer. Identity is a layer that returns its input
    # unchanged, so replacing fc with it hands us those 512 numbers directly.
    model.fc = nn.Identity()

    # Freeze. Without this, gradients from the head would flow back into the
    # backbone and the optimizer would edit weights that took days of GPU time
    # to learn -- using 364 training frames to do it. The result would be worse
    # eyes, not better ones.
    for param in model.parameters():
        param.requires_grad = False

    # eval() matters here in a way it did not for MnistCNN. ResNet-18 contains
    # batchnorm layers, which in train mode update running statistics from
    # whatever data passes through. That would quietly change a model we just
    # declared frozen, with no error and no warning.
    model.eval()
    model.to(device)

    # The weights carry their own preprocessing: resize the short side to 256,
    # centre-crop 224, scale to 0-1, normalise by ImageNet's mean and std.
    # Using anything else feeds the network inputs shaped unlike its training
    # data, and accuracy drops for no visible reason.
    return model, weights.transforms()


# ---------------------------------------------------------------------------
# Phase 5: video backbones
# ---------------------------------------------------------------------------

# name -> (constructor, Kinetics-400 weights, classifier attribute, feature dim)
#
# All three take exactly [B, 3, 16, 224, 224] -- sixteen frames -- which is what
# ingest/cut.py has written since Phase 3 (CLIP_FRAMES = 16 at SAMPLE_FPS = 8).
# MViTv2-S's published evaluation recipe samples at 7.5 fps against our 8. The
# clip format needs no change: Phase 3's work carries over whole.
VIDEO_BACKBONES = {
    # 34.5M params, 80.76% top-1 on Kinetics-400. The default -- the best
    # accuracy that fits comfortably in 8GB.
    "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.KINETICS400_V1, "head", 768),
    # 28.2M params, 77.72%. A different family (shifted-window attention rather
    # than multiscale pooling), useful as a second opinion.
    "swin3d_t": (swin3d_t, Swin3D_T_Weights.KINETICS400_V1, "head", 768),
    # 11.7M params, 63.96%. The cheap one, for when a run has to be quick.
    "mc3_18": (mc3_18, MC3_18_Weights.KINETICS400_V1, "fc", 512),
}

CROP_MODES = ("center", "squash")


class SquashVideo(nn.Module):
    """Resize the whole frame to 224x224 instead of cropping the middle out.

    The stock torchvision preset resizes the short side to 256 and then takes a
    224 centre crop. Our stored frames are 455x256 (ingest/cut.py resizes the
    short side to SHORT_SIDE = 256), so that crop keeps the middle 224 of 455
    pixels -- **49% of the width** -- and discards both sidelines.

    For Kinetics that is the right call: a YouTube clip of someone dunking has
    the dunker in the middle of frame. For a broadcast wide shot it may not be,
    because during a shot the hoop sits near the left or right edge. A model
    that never sees the basket cannot tell a shot going in from one coming back
    out -- and every result this project has reported since Phase 2 was computed
    through that crop.

    Squashing distorts the aspect ratio, which is its own mismatch against what
    the weights were trained on. Which cost is worse is not a thing to reason
    about. It is a number, so clips/data.py builds both caches and we read it.
    """

    def __init__(self, size, mean, std, interpolation):
        super().__init__()
        self.size = list(size)
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, vid):
        # Mirrors torchvision's VideoClassification.forward exactly, except that
        # resize-then-centre-crop becomes one resize to a square. Same contract:
        # [T, C, H, W] uint8 in, [C, T, H, W] float out.
        squeeze = False
        if vid.ndim < 5:
            vid = vid.unsqueeze(dim=0)
            squeeze = True

        n, t, c, h, w = vid.shape
        vid = vid.view(-1, c, h, w)
        # antialias=False copies the stock preset. It is not the better setting
        # in general, but holding it fixed keeps this a clean experiment: the
        # only thing differing between the two caches is whether the sidelines
        # survive.
        vid = F.resize(vid, self.size, interpolation=self.interpolation,
                       antialias=False)
        vid = F.convert_image_dtype(vid, torch.float)
        vid = F.normalize(vid, mean=self.mean, std=self.std)
        vid = vid.view(n, t, c, self.size[0], self.size[1])
        vid = vid.permute(0, 2, 1, 3, 4)        # [N,T,C,H,W] -> [N,C,T,H,W]

        return vid.squeeze(dim=0) if squeeze else vid


def video_transform(weights, crop="center"):
    """Preprocessing for a video backbone, in one of two crop modes."""
    if crop not in CROP_MODES:
        raise SystemExit(f"unknown crop '{crop}' -- choose from {list(CROP_MODES)}")

    preset = weights.transforms()
    if crop == "center":
        return preset

    return SquashVideo(preset.crop_size, preset.mean, preset.std,
                       preset.interpolation)


def build_video_backbone(name, device, crop="center", keep_classifier=False):
    """Return (frozen video model, matching transform, feature dimension).

    The same three moves as build_backbone above -- strip the classifier, freeze
    every parameter, switch to eval mode -- applied to a network that reads all
    16 frames at once instead of one frame at a time.

    With keep_classifier=True the original 400-way Kinetics head is left on and
    the returned dimension is 400. That is not useful for training, but it lets
    clips/probe_kinetics.py ask the pretrained weights what they already
    recognise in our clips before we train anything of our own.
    """
    if name not in VIDEO_BACKBONES:
        raise SystemExit(f"unknown video backbone '{name}' -- "
                         f"choose from {sorted(VIDEO_BACKBONES)}")

    constructor, weights, classifier_attr, feature_dim = VIDEO_BACKBONES[name]
    model = constructor(weights=weights)

    if keep_classifier:
        feature_dim = len(weights.meta["categories"])
    else:
        # Same move as `model.fc = nn.Identity()` in build_backbone: we want the
        # description the network built, not its guess at a Kinetics label.
        setattr(model, classifier_attr, nn.Identity())

    # Freeze, for exactly Phase 2's reason -- a few hundred training clips must
    # not be allowed to edit weights that took days of GPU time to learn.
    for param in model.parameters():
        param.requires_grad = False

    # eval() matters as much here as it did for ResNet-18's batchnorm. These
    # models carry dropout and normalisation layers that behave differently in
    # train mode, which would make cached features non-reproducible -- and a
    # feature cache is only valid while the backbone is deterministic.
    model.eval()
    model.to(device)

    return model, video_transform(weights, crop), feature_dim


# ---------------------------------------------------------------------------
# Phase 6: letting the last block learn
# ---------------------------------------------------------------------------

class VideoTrunk(nn.Module):
    """An MViT up to, not including, its last block. Frozen, so its output can be cached."""

    def __init__(self, model):
        super().__init__()
        self.conv_proj = model.conv_proj
        self.pos_encoding = model.pos_encoding
        self.blocks = nn.ModuleList(model.blocks[:-1])

    def forward(self, x):
        # The first half of torchvision's MViT.forward, line for line, stopping
        # one block early. Returns the tokens and the (T, H, W) grid they sit
        # on, which the last block needs to pool them.
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        return x, thw


class VideoTail(nn.Module):
    """An MViT's last block and final norm: the part Phase 6 lets learn.

    Trunk then tail computes exactly what the whole model does, so a tail that
    has not been trained yet gives back Phase 5's features unchanged.
    """

    def __init__(self, model):
        super().__init__()
        self.block = model.blocks[-1]
        self.norm = model.norm

    def forward(self, tokens, thw):
        x, _ = self.block(tokens, thw)
        # The class token, which is what MViT.forward hands its classifier.
        return self.norm(x)[:, 0]


def split_video_backbone(name, device, crop="center", pretrained=True):
    """Return (frozen trunk, trainable tail, matching transform, feature dimension).

    Why split rather than unfreeze the whole network: 34.5M weights against
    about 1,500 training clips would be memorised, not learned. The last block
    is about a fifth of that, it is the most task-specific part of the network,
    and with the trunk frozen its output can be cached once, so each training
    epoch runs one block instead of sixteen.
    """
    if name != "mvit_v2_s":
        raise SystemExit(f"splitting is written for mvit_v2_s's blocks, not {name}")

    constructor, weights, classifier_attr, feature_dim = VIDEO_BACKBONES[name]
    model = constructor(weights=weights if pretrained else None)
    setattr(model, classifier_attr, nn.Identity())
    trunk, tail = VideoTrunk(model), VideoTail(model)

    for param in trunk.parameters():
        param.requires_grad = False
    trunk.eval().to(device)
    tail.eval().to(device)
    return trunk, tail, video_transform(weights, crop), feature_dim


def kinetics_categories(name):
    """The 400 Kinetics class names, in the order the classifier emits them."""
    return list(VIDEO_BACKBONES[name][1].meta["categories"])
