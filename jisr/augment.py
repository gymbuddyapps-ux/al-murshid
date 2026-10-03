"""Data augmentation: make slightly changed copies of training clips.

Why: we only have a few signers. By showing the model each clip a little rotated,
bigger or smaller, mirrored, faster or slower, it learns the sign itself instead of
memorizing one person's exact movement. This reduces overfitting (being good on the
training clips but bad on new people).

A clip here is a numpy array of shape (32, 140), see docs/features.md.
Augmentation is used only while training, never when testing.
"""
import numpy as np

from jisr.features import (FEATURE_SIZE, LEFT_FLAG, LEFT_HAND_START, NUM_FRAMES, POSE_START,
                           RIGHT_FLAG, RIGHT_HAND_START)

HAND_SIZE = 63      # 21 landmarks x 3 numbers
POSE_SIZE = 18      # 9 landmarks x 2 numbers

# Positions of all x values and all y values inside the 140-number vector.
X_INDEXES = ([LEFT_HAND_START + 3 * k for k in range(21)] +
             [RIGHT_HAND_START + 3 * k for k in range(21)] +
             [POSE_START + 2 * k for k in range(9)])
Y_INDEXES = [i + 1 for i in X_INDEXES]
Z_INDEXES = ([LEFT_HAND_START + 3 * k + 2 for k in range(21)] +
             [RIGHT_HAND_START + 3 * k + 2 for k in range(21)])


def mirror(clip):
    """Left/right mirror: what a left-handed signer would look like.

    x changes sign, the two hands swap places, and each left body point swaps with its right one.
    """
    out = clip.copy()
    out[:, X_INDEXES] *= -1
    # swap the two hands (and their "hand is present" flags)
    left = out[:, LEFT_HAND_START:LEFT_HAND_START + HAND_SIZE].copy()
    out[:, LEFT_HAND_START:LEFT_HAND_START + HAND_SIZE] = out[:, RIGHT_HAND_START:RIGHT_HAND_START + HAND_SIZE]
    out[:, RIGHT_HAND_START:RIGHT_HAND_START + HAND_SIZE] = left
    out[:, [LEFT_FLAG, RIGHT_FLAG]] = out[:, [RIGHT_FLAG, LEFT_FLAG]]
    # swap left and right ear, shoulder, elbow, wrist (after the nose, pose points come in left/right pairs)
    pairs_start = POSE_START + 2                                           # skip the nose (x, y)
    pose = out[:, pairs_start:POSE_START + POSE_SIZE].reshape(-1, 4, 2, 2)  # pair, side, (x, y)
    out[:, pairs_start:POSE_START + POSE_SIZE] = pose[:, :, ::-1, :].reshape(-1, POSE_SIZE - 2)
    return out


def rotate(clip, degrees):
    """Tilt the whole picture a little, as if the phone was not held straight."""
    angle = np.deg2rad(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    out = clip.copy()
    x, y = clip[:, X_INDEXES], clip[:, Y_INDEXES]
    out[:, X_INDEXES] = cos * x - sin * y
    out[:, Y_INDEXES] = sin * x + cos * y
    return out


def scale(clip, factor):
    """Make the movement a little bigger or smaller (people have different arm lengths)."""
    out = clip.copy()
    out[:, X_INDEXES + Y_INDEXES + Z_INDEXES] *= factor
    return out


def jitter(clip, amount, rng):
    """Add tiny random noise, like the small errors MediaPipe makes."""
    out = clip.copy()
    noise = rng.normal(0, amount, size=clip.shape).astype(clip.dtype)
    noise[:, [LEFT_FLAG, RIGHT_FLAG]] = 0
    # a missing hand must stay all zeros
    noise[:, LEFT_HAND_START:LEFT_HAND_START + HAND_SIZE] *= clip[:, [LEFT_FLAG]]
    noise[:, RIGHT_HAND_START:RIGHT_HAND_START + HAND_SIZE] *= clip[:, [RIGHT_FLAG]]
    return out + noise


def change_speed(clip, rng, max_cut=0.15):
    """Cut a little off the start and/or end and stretch the rest back to 32 frames.

    This imitates signing faster or slower, and a clip that was cut a bit early or late.
    """
    start = rng.uniform(0, max_cut) * (NUM_FRAMES - 1)
    end = (NUM_FRAMES - 1) - rng.uniform(0, max_cut) * (NUM_FRAMES - 1)
    positions = np.linspace(start, end, NUM_FRAMES)
    return clip[np.floor(positions + 0.5).astype(int)]


def drop_frames(clip, rng, chance=0.1):
    """Repeat the previous frame in a few random places, like a camera that skips frames."""
    out = clip.copy()
    for t in range(1, NUM_FRAMES):
        if rng.random() < chance:
            out[t] = out[t - 1]
    return out


def drop_hand(clip, rng, chance=0.05):
    """Make a hand disappear in a few random frames, like a missed detection."""
    out = clip.copy()
    for start, flag in ((LEFT_HAND_START, LEFT_FLAG), (RIGHT_HAND_START, RIGHT_FLAG)):
        missed = rng.random(NUM_FRAMES) < chance
        out[missed, start:start + HAND_SIZE] = 0
        out[missed, flag] = 0
    return out


def augment(clip, rng, settings):
    """Apply a random mix of all the changes above to one clip."""
    assert clip.shape == (NUM_FRAMES, FEATURE_SIZE)
    if rng.random() < settings["mirror_chance"]:
        clip = mirror(clip)
    clip = rotate(clip, rng.uniform(-settings["max_rotation_degrees"], settings["max_rotation_degrees"]))
    clip = scale(clip, rng.uniform(1 - settings["max_scale_change"], 1 + settings["max_scale_change"]))
    clip = change_speed(clip, rng, settings["max_time_cut"])
    clip = drop_frames(clip, rng, settings["drop_frame_chance"])
    clip = drop_hand(clip, rng, settings["drop_hand_chance"])
    clip = jitter(clip, settings["jitter"], rng)
    return clip.astype(np.float32)
