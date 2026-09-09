"""Contracts for the frozen backbone.

All three of these guard silent failures. None of them would raise an error in
normal use -- they would just make the model quietly worse, which is the hardest
kind of bug to notice.
"""

import torch
from torchvision.models.video import mc3_18, mvit_v2_s, swin3d_t

from models.backbone import (CROP_MODES, FEATURE_DIM, VIDEO_BACKBONES,
                             build_backbone, build_video_backbone,
                             kinetics_categories, video_transform)

# The contract tests below that need real weights use mc3_18 -- it is the
# smallest of the three, and the contract is identical for all of them. The
# registry test covers the other two without downloading anything.
CHEAPEST = "mc3_18"

# What ingest/cut.py actually writes: 16 frames, short side 256, so 455x256.
STORED_CLIP_SHAPE = (16, 3, 256, 455)


def test_backbone_turns_an_image_into_512_numbers():
    model, _ = build_backbone(torch.device("cpu"))
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, FEATURE_DIM)


def test_backbone_is_frozen():
    # The whole Phase 2 premise is that only the head learns. If any backbone
    # parameter could accumulate gradients, training would drift the eyes using
    # a few hundred frames -- and nothing would report it.
    model, _ = build_backbone(torch.device("cpu"))
    assert all(not p.requires_grad for p in model.parameters())


def test_backbone_is_in_eval_mode():
    # ResNet-18 has batchnorm. In train mode it updates running statistics from
    # our frames, which changes a model we declared frozen.
    model, _ = build_backbone(torch.device("cpu"))
    assert not model.training


def test_same_image_always_gives_the_same_features():
    # Feature caching is only valid if the backbone is deterministic. If this
    # ever fails, cached features and freshly computed ones would disagree and
    # training would learn from one while inference used the other.
    model, _ = build_backbone(torch.device("cpu"))
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(model(image), model(image))


# ---------------------------------------------------------------------------
# Phase 5: the video backbone
# ---------------------------------------------------------------------------


def test_registry_feature_dims_match_the_real_classifiers():
    # The registry hard-codes a feature dimension per backbone, and clips/data.py
    # writes that number into the cache blob. If it were ever wrong, the cache
    # would be built with one dimension and the head sized for another -- a
    # shape error at best, and silently wrong features at worst.
    #
    # Constructed with weights=None, so this checks all three architectures
    # without downloading a single checkpoint.
    builders = {"mvit_v2_s": mvit_v2_s, "swin3d_t": swin3d_t, "mc3_18": mc3_18}

    for name, (_, _, attr, declared) in VIDEO_BACKBONES.items():
        model = builders[name](weights=None)
        classifier = getattr(model, attr)
        # mvit's head is Sequential(Dropout, Linear); the others are bare Linear.
        linear = classifier[-1] if isinstance(classifier, torch.nn.Sequential) \
            else classifier
        assert linear.in_features == declared, name


def test_video_backbone_turns_sixteen_frames_into_one_vector():
    # The shape that separates this from Phase 2. ResNet-18 gives 16 separate
    # descriptions and leaves the sequencing to a GRU; this reads the clip as a
    # clip and returns a single answer.
    declared = VIDEO_BACKBONES[CHEAPEST][3]
    model, _, feature_dim = build_video_backbone(CHEAPEST, torch.device("cpu"))
    assert feature_dim == declared

    with torch.no_grad():
        out = model(torch.randn(2, 3, 16, 224, 224))
    assert out.shape == (2, feature_dim)


def test_video_backbone_is_frozen_and_in_eval_mode():
    # Both for the same reasons as Phase 2's backbone, and both invisible when
    # broken: gradients would quietly drift the eyes, and train-mode dropout
    # would make the cached features irreproducible.
    model, _, _ = build_video_backbone(CHEAPEST, torch.device("cpu"))
    assert all(not p.requires_grad for p in model.parameters())
    assert not model.training


def test_keeping_the_classifier_gives_four_hundred_kinetics_logits():
    # What clips/probe_kinetics.py relies on: the pretrained head left intact,
    # so we can ask the weights what they already recognise before training.
    model, _, feature_dim = build_video_backbone(
        CHEAPEST, torch.device("cpu"), keep_classifier=True)
    assert feature_dim == 400

    with torch.no_grad():
        assert model(torch.randn(1, 3, 16, 224, 224)).shape == (1, 400)


def test_kinetics_knows_about_basketball():
    # The reason a Kinetics-pretrained backbone is worth more here than an
    # ImageNet one, and the whole basis of the zero-shot dunk probe. If these
    # class names ever move, probe_kinetics.py is reading the wrong logits.
    names = kinetics_categories("mvit_v2_s")
    assert len(names) == 400
    assert names[107] == "dunking basketball"
    assert names[296] == "shooting basketball"


def test_both_crop_modes_produce_the_shape_the_model_wants():
    # [T, C, H, W] uint8 in, [C, T, H, W] float out -- the squash transform is a
    # drop-in for the stock preset or it is not usable at all.
    weights = VIDEO_BACKBONES["mvit_v2_s"][1]
    clip = torch.randint(0, 255, STORED_CLIP_SHAPE, dtype=torch.uint8)

    for crop in CROP_MODES:
        out = video_transform(weights, crop)(clip)
        assert out.shape == (3, 16, 224, 224), crop
        assert out.dtype == torch.float32, crop


def test_the_centre_crop_really_does_throw_the_sidelines_away():
    # The claim Phase 5 is built on, tested rather than asserted.
    #
    # A clip that is black everywhere except a bright stripe down the far left
    # edge -- where a broadcast camera puts the hoop during a shot. The centre
    # crop keeps columns 115..339 of 455, so it cannot see the stripe at all and
    # every pixel it returns is identical. Squashing the full frame keeps it.
    weights = VIDEO_BACKBONES["mvit_v2_s"][1]
    clip = torch.zeros(STORED_CLIP_SHAPE, dtype=torch.uint8)
    clip[:, :, :, :40] = 255

    centred = video_transform(weights, "center")(clip)
    squashed = video_transform(weights, "squash")(clip)

    assert centred.std().item() == 0.0, "the centre crop saw something it should not"
    assert squashed.std().item() > 0.0, "the squash lost the sideline"


def test_unknown_names_stop_the_run():
    # A typo must not fall back to a default, or a result gets filed under the
    # wrong backbone -- the same reason build_head refuses unknown head names.
    for bad in (lambda: build_video_backbone("videomae", torch.device("cpu")),
                lambda: video_transform(VIDEO_BACKBONES["mc3_18"][1], "letterbox")):
        try:
            bad()
        except SystemExit:
            continue
        raise AssertionError("an unknown name should stop the run")
