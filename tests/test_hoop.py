"""Contracts for the hoop close-up geometry.

The detector itself is not tested here -- it is someone else's model, and the
spike checked it by eye on real frames. What is ours is the geometry around it:
where the crop goes when the hoop is found, what happens between detections,
and what happens at the edge of the frame. Those fail silently, as crops of the
wrong thing, and nothing downstream would notice.
"""

import pytest

from ingest.hoop import CROP_SIDE, HOOP_ROW, crop_box, track

POSITIONS = (0, 5, 10, 15)


def test_the_track_blends_in_a_straight_line_between_detections():
    # The camera pans across two seconds; the frames we did not run the
    # detector on must follow the hoop rather than snap to one position.
    centres = track([(0.0, 0.0), None, None, (150.0, 30.0)], POSITIONS, 16)
    assert centres[0] == (0.0, 0.0)
    assert centres[15] == (150.0, 30.0)
    assert centres[5] == pytest.approx((50.0, 10.0))


def test_the_track_holds_the_nearest_detection_past_either_end():
    centres = track([None, (10.0, 20.0), None, None], POSITIONS, 16)
    assert centres[0] == (10.0, 20.0)
    assert centres[15] == (10.0, 20.0)


def test_a_clip_that_never_shows_a_hoop_has_no_track():
    # write_crops falls back to the whole frame on None. Returning a made-up
    # position instead would crop an arbitrary patch and call it a hoop.
    assert track([None, None, None, None], POSITIONS, 16) is None


def test_every_crop_is_the_full_size_and_inside_the_frame():
    # Including hoops right at the corners, which is where broadcast cameras
    # put them. A crop that ran off the frame would be padded or truncated.
    for centre in [(0, 0), (854, 480), (10, 470), (844, 5), (427, 240)]:
        x0, y0, x1, y1 = crop_box(centre, 854, 480)
        assert x1 - x0 == y1 - y0 == CROP_SIDE
        assert 0 <= x0 and x1 <= 854 and 0 <= y0 and y1 <= 480


def test_the_rim_sits_a_third_of_the_way_down_when_there_is_room():
    # Room above for the ball's arc, room below for the finish.
    x0, y0, _, _ = crop_box((427, 200), 854, 480)
    assert 427 - x0 == CROP_SIDE / 2
    assert 200 - y0 == round(CROP_SIDE * HOOP_ROW)


def test_a_frame_smaller_than_the_crop_is_cropped_to_fit():
    x0, y0, x1, y1 = crop_box((50, 50), 200, 150)
    assert (x1 - x0, y1 - y0) == (150, 150)


@pytest.mark.parametrize("found", [True, False])
def test_every_close_up_is_the_same_square_whether_or_not_a_hoop_was_found(found):
    # Training and prediction both go through close_ups. Whatever it hands
    # back is what the backbone sees, so the size must never depend on the
    # detector's luck.
    from PIL import Image

    from ingest.hoop import close_ups

    frames = [Image.new("RGB", (854, 480)) for _ in range(16)]
    centres = [(700.0, 100.0)] * 16 if found else None
    views = close_ups(frames, centres)
    assert len(views) == 16
    assert all(view.size == (CROP_SIDE, CROP_SIDE) for view in views)


def test_a_close_up_is_cut_around_the_hoop_not_from_the_middle():
    # A white square where the hoop is, black everywhere else: the close-up
    # must contain the white square.
    from PIL import Image, ImageDraw

    from ingest.hoop import close_ups

    frame = Image.new("RGB", (854, 480))
    ImageDraw.Draw(frame).rectangle((690, 90, 710, 110), fill="white")
    view = close_ups([frame], [(700.0, 100.0)])[0]
    assert view.convert("L").getextrema()[1] == 255
