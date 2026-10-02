"""Unit tests for jisr/features.py (the rules written in docs/features.md)."""
import numpy as np

from jisr.features import (FEATURE_SIZE, LEFT_FLAG, LEFT_HAND_START, NUM_FRAMES, POSE_START,
                           RIGHT_FLAG, RIGHT_HAND_START, assign_hands, clip_to_features,
                           fill_missing_pose, median, resample, trim_range)


def make_pose(shift_x=0.0, shift_y=0.0, scale=1.0):
    """A simple upright body. Signer's left is on the picture's right side."""
    base = np.array([
        [0.6, 0.4],   # left shoulder
        [0.4, 0.4],   # right shoulder
        [0.65, 0.6],  # left elbow
        [0.35, 0.6],  # right elbow
        [0.7, 0.5],   # left wrist
        [0.3, 0.8],   # right wrist
    ])
    return (base - 0.5) * scale + 0.5 + np.array([shift_x, shift_y])


def make_hand(wrist_x, wrist_y, scale=1.0):
    """A fake hand: 21 points going up from the wrist."""
    hand = np.zeros((21, 3))
    for k in range(21):
        hand[k] = (wrist_x, wrist_y - 0.005 * k * scale, 0.001 * k * scale)
    return hand


def make_clip(num_frames=10, shift_x=0.0, shift_y=0.0, scale=1.0):
    frames = []
    for _ in range(num_frames):
        pose = make_pose(shift_x, shift_y, scale)
        hand = make_hand(pose[4][0], pose[4][1], scale)     # a hand at the left wrist
        frames.append({"pose": pose, "hands": [hand]})
    return frames


def test_output_shape():
    features = clip_to_features(make_clip(), aspect=1.0)
    assert features.shape == (NUM_FRAMES, FEATURE_SIZE)
    assert features.dtype == np.float32


def test_position_does_not_matter():
    a = clip_to_features(make_clip(), aspect=1.0)
    b = clip_to_features(make_clip(shift_x=0.1, shift_y=-0.05), aspect=1.0)
    assert np.allclose(a, b, atol=1e-5)


def test_distance_from_camera_does_not_matter():
    a = clip_to_features(make_clip(), aspect=1.0)
    b = clip_to_features(make_clip(scale=0.5), aspect=1.0)
    assert np.allclose(a, b, atol=1e-5)


def test_shoulders_are_one_unit_apart():
    features = clip_to_features(make_clip(), aspect=1.0)
    left_shoulder = features[0, POSE_START:POSE_START + 2]
    right_shoulder = features[0, POSE_START + 2:POSE_START + 4]
    assert np.isclose(np.linalg.norm(left_shoulder - right_shoulder), 1.0, atol=1e-5)
    assert np.allclose(left_shoulder + right_shoulder, 0.0, atol=1e-5)   # midpoint is the origin


def test_missing_hand_is_zeros_with_flag():
    features = clip_to_features(make_clip(), aspect=1.0)
    # The clip only has a left hand.
    assert np.all(features[:, LEFT_FLAG] == 1.0)
    assert np.all(features[:, RIGHT_FLAG] == 0.0)
    assert np.all(features[:, RIGHT_HAND_START:RIGHT_HAND_START + 63] == 0.0)
    assert np.any(features[:, LEFT_HAND_START:LEFT_HAND_START + 63] != 0.0)


def test_hand_goes_to_nearest_wrist():
    pose = make_pose()
    near_left = make_hand(pose[4][0], pose[4][1])
    near_right = make_hand(pose[5][0], pose[5][1])
    assert assign_hands([near_left], pose)[0] is near_left
    assert assign_hands([near_right], pose)[1] is near_right
    # Two hands, given in the "wrong" order, are still put in the correct slots.
    left, right = assign_hands([near_right, near_left], pose)
    assert left is near_left and right is near_right
    assert assign_hands([], pose) == (None, None)


def test_clip_without_any_hand_is_dropped():
    frames = [{"pose": make_pose(), "hands": []} for _ in range(5)]
    assert clip_to_features(frames, aspect=1.0) is None


def test_clip_without_any_pose_is_dropped():
    frames = [{"pose": None, "hands": [make_hand(0.5, 0.5)]} for _ in range(5)]
    assert clip_to_features(frames, aspect=1.0) is None


def test_missing_pose_is_filled_from_neighbours():
    a, b = make_pose(), make_pose(shift_x=0.1)
    filled = fill_missing_pose([None, a, None, b, None])
    assert filled[0] is a          # nothing earlier, so the nearest later pose
    assert filled[2] is a          # nearest earlier pose
    assert filled[4] is b
    assert fill_missing_pose([None, None]) is None


def test_resample_picks_evenly_spaced_frames():
    vectors = np.arange(63, dtype=np.float64).reshape(63, 1)
    picked = resample(vectors)
    assert picked.shape == (NUM_FRAMES, 1)
    assert picked[0, 0] == 0 and picked[-1, 0] == 62
    assert list(picked[:3, 0]) == [0, 2, 4]
    # A short clip repeats frames; a single frame is repeated 32 times.
    assert resample(np.ones((1, 4))).shape == (NUM_FRAMES, 4)
    short = resample(np.arange(4, dtype=np.float64).reshape(4, 1))
    assert short[0, 0] == 0 and short[-1, 0] == 3


def wrist_poses(wrist_y_values):
    """Poses whose left wrist follows the given heights; everything else stays still."""
    return [np.array([[0.6, 0.4], [0.4, 0.4], [0, 0], [0, 0], [0.7, y], [0.3, 0.9]])
            for y in wrist_y_values]


def test_trim_cuts_off_fast_raise_and_lowering():
    # The wrist rises fast for 3 frames, stays, then drops fast for 2 frames.
    wrist_y = [0.9, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.8]
    times = [t * 40 for t in range(len(wrist_y))]
    trim = {"trim_speed": 2.0, "trim_max_ms": 500}
    assert trim_range(wrist_poses(wrist_y), times, 0.2, trim) == (3, 8)


def test_trim_leaves_a_still_clip_alone():
    times = [t * 40 for t in range(11)]
    trim = {"trim_speed": 2.0, "trim_max_ms": 500}
    assert trim_range(wrist_poses([0.6] * 11), times, 0.2, trim) == (0, 10)


def test_trim_stops_after_the_time_limit():
    wrist_y = [2 - t * 0.05 for t in range(30)]        # keeps rising for the whole clip
    times = [t * 40 for t in range(30)]
    first, _ = trim_range(wrist_poses(wrist_y), times, 0.2, {"trim_speed": 2.0, "trim_max_ms": 200})
    assert first == 5                                   # 200 ms = 5 frames


def test_trim_keeps_everything_if_too_little_would_remain():
    wrist_y = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]            # rising all the time, only 6 frames
    times = [t * 40 for t in range(6)]
    assert trim_range(wrist_poses(wrist_y), times, 0.2, {"trim_speed": 2.0, "trim_max_ms": 500}) == (0, 5)


def test_median():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5


def test_wide_picture_uses_same_unit_for_x_and_y():
    # The same scene in a square picture and in a picture that is twice as wide:
    # in the wide one every x fraction is halved.
    square = make_clip()
    wide = []
    for f in square:
        pose = f["pose"].copy()
        pose[:, 0] /= 2
        hand = f["hands"][0].copy()
        hand[:, 0] /= 2
        hand[:, 2] /= 2
        wide.append({"pose": pose, "hands": [hand]})
    a = clip_to_features(square, aspect=1.0)
    b = clip_to_features(wide, aspect=2.0)
    assert np.allclose(a, b, atol=1e-5)
