"""Unit tests for jisr/augment.py and jisr/metrics.py."""
import numpy as np

from jisr.augment import augment, change_speed, jitter, mirror, rotate, scale
from jisr.features import (FEATURE_SIZE, LEFT_FLAG, LEFT_HAND_START, NUM_FRAMES, POSE_START,
                           RIGHT_FLAG, RIGHT_HAND_START)
from jisr.metrics import app_metrics, fit_temperature, softmax, word_accuracy

SETTINGS = {"mirror_chance": 0.5, "max_rotation_degrees": 12, "max_scale_change": 0.15,
            "max_time_cut": 0.15, "drop_frame_chance": 0.1, "drop_hand_chance": 0.05, "jitter": 0.02}


def make_clip(seed=0):
    """A random clip that only has a left hand."""
    rng = np.random.default_rng(seed)
    clip = rng.normal(size=(NUM_FRAMES, FEATURE_SIZE)).astype(np.float32)
    clip[:, RIGHT_HAND_START:RIGHT_HAND_START + 63] = 0
    clip[:, LEFT_FLAG] = 1
    clip[:, RIGHT_FLAG] = 0
    return clip


def test_mirror_twice_gives_the_original():
    clip = make_clip()
    assert np.allclose(mirror(mirror(clip)), clip)


def test_mirror_moves_the_left_hand_to_the_right_slot():
    clip = make_clip()
    mirrored = mirror(clip)
    assert np.all(mirrored[:, LEFT_FLAG] == 0) and np.all(mirrored[:, RIGHT_FLAG] == 1)
    assert np.all(mirrored[:, LEFT_HAND_START:LEFT_HAND_START + 63] == 0)
    # x changes sign, y and z stay
    assert np.allclose(mirrored[:, RIGHT_HAND_START], -clip[:, LEFT_HAND_START])
    assert np.allclose(mirrored[:, RIGHT_HAND_START + 1], clip[:, LEFT_HAND_START + 1])
    # the left shoulder becomes the right shoulder
    assert np.allclose(mirrored[:, POSE_START + 2], -clip[:, POSE_START])
    assert np.allclose(mirrored[:, POSE_START + 3], clip[:, POSE_START + 1])


def test_rotate_and_scale_keep_a_missing_hand_at_zero():
    clip = make_clip()
    for changed in (rotate(clip, 10), scale(clip, 1.1), jitter(clip, 0.05, np.random.default_rng(1))):
        assert np.all(changed[:, RIGHT_HAND_START:RIGHT_HAND_START + 63] == 0)
        assert np.all(changed[:, RIGHT_FLAG] == 0) and np.all(changed[:, LEFT_FLAG] == 1)


def test_rotation_keeps_distances():
    clip = make_clip()
    rotated = rotate(clip, 12)
    before = np.hypot(clip[:, POSE_START], clip[:, POSE_START + 1])
    after = np.hypot(rotated[:, POSE_START], rotated[:, POSE_START + 1])
    assert np.allclose(before, after, atol=1e-5)


def test_change_speed_keeps_the_shape_and_frame_order():
    clip = np.tile(np.arange(NUM_FRAMES, dtype=np.float32)[:, None], (1, FEATURE_SIZE))
    changed = change_speed(clip, np.random.default_rng(3))
    assert changed.shape == clip.shape
    assert np.all(np.diff(changed[:, 0]) >= 0)


def test_augment_output_is_valid():
    rng = np.random.default_rng(5)
    for seed in range(20):
        out = augment(make_clip(seed), rng, SETTINGS)
        assert out.shape == (NUM_FRAMES, FEATURE_SIZE) and out.dtype == np.float32
        assert np.isfinite(out).all()
        assert set(np.unique(out[:, [LEFT_FLAG, RIGHT_FLAG]])) <= {0.0, 1.0}


def test_word_accuracy_ignores_the_other_class():
    # 3 classes: word 0, word 1, other (id 2)
    scores = np.array([[5.0, 1.0, 9.0],    # true word 0: "other" is highest, but among words 0 wins
                       [1.0, 5.0, 0.0],    # true word 1: correct
                       [0.0, 0.0, 5.0]])   # an "other" clip: not counted
    y = np.array([0, 1, 2])
    result = word_accuracy(scores, y, other_id=2)
    assert result["clips"] == 2 and result["top1"] == 1.0


def test_app_metrics_reject_low_confidence_and_other():
    probabilities = np.array([[0.9, 0.05, 0.05],   # word 0, confident and correct -> accepted
                              [0.4, 0.35, 0.25],   # word 1, low confidence -> rejected
                              [0.1, 0.1, 0.8]])    # unknown sign, model says other -> rejected
    y = np.array([0, 1, 2])
    result = app_metrics(probabilities, y, other_id=2, threshold=0.5)
    assert result["acceptance_rate"] == 0.5
    assert result["accuracy_on_accepted"] == 1.0
    assert result["unknown_sign_rejection_rate"] == 1.0


def test_temperature_cools_down_an_overconfident_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=600)
    scores = rng.normal(size=(600, 3))
    scores[np.arange(600), y] += 1.0          # a weak but real signal
    assert fit_temperature(scores * 10, y) > 2.0
    assert np.allclose(softmax(scores).sum(axis=1), 1.0)
