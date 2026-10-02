"""Turn the landmarks of one clip into the model's input (32 frames x 140 numbers).

This file follows docs/features.md step by step. The browser version
(web/src/features.js) must do exactly the same thing.

It only needs numpy, so it can be tested without MediaPipe.
"""
import math

import numpy as np

NUM_FRAMES = 32          # every clip is resampled to this many frames
HAND_POINTS = 21         # landmarks per hand
POSE_POINTS = 6          # shoulders, elbows, wrists
FEATURE_SIZE = 2 * HAND_POINTS * 3 + POSE_POINTS * 2 + 2   # = 140
MIN_FRAMES_AFTER_TRIM = 4   # trimming never leaves fewer frames than this

# Positions inside our 6-point pose list (see the table in docs/features.md)
LEFT_SHOULDER, RIGHT_SHOULDER = 0, 1
LEFT_WRIST, RIGHT_WRIST = 4, 5

# Where each part lives inside the 140-number vector
LEFT_HAND_START = 0
RIGHT_HAND_START = 63
POSE_START = 126
LEFT_FLAG = 138
RIGHT_FLAG = 139


def median(values):
    """Middle value of a list (average of the two middle values for an even count)."""
    ordered = sorted(values)
    n = len(ordered)
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def fill_missing_pose(poses):
    """Step 2. `poses` is a list with one entry per frame: a (6, 2) array or None.

    Returns a list with no None, or None if the clip has no pose at all.
    """
    if all(p is None for p in poses):
        return None
    filled = list(poses)
    # Copy forward: a missing frame takes the nearest earlier pose.
    for i in range(1, len(filled)):
        if filled[i] is None:
            filled[i] = filled[i - 1]
    # Frames at the start that are still missing take the nearest later pose.
    for i in range(len(filled) - 2, -1, -1):
        if filled[i] is None:
            filled[i] = filled[i + 1]
    return filled


def assign_hands(hands, pose):
    """Step 5. Decide which detected hand is the left one and which is the right one.

    `hands` is a list of 0, 1 or 2 arrays of shape (21, 3), already in X, Y, Z units.
    `pose` is the (6, 2) pose of the same frame.
    Returns (left_hand, right_hand); each is an array or None.
    """
    def dist(hand, wrist_index):
        dx = hand[0][0] - pose[wrist_index][0]
        dy = hand[0][1] - pose[wrist_index][1]
        return math.sqrt(dx * dx + dy * dy)

    if len(hands) == 0:
        return None, None
    if len(hands) == 1:
        if dist(hands[0], LEFT_WRIST) <= dist(hands[0], RIGHT_WRIST):
            return hands[0], None
        return None, hands[0]
    a, b = hands[0], hands[1]
    if dist(a, LEFT_WRIST) + dist(b, RIGHT_WRIST) <= dist(a, RIGHT_WRIST) + dist(b, LEFT_WRIST):
        return a, b
    return b, a


def shoulder_width(pose):
    """Distance between the two shoulders of one pose."""
    return math.sqrt((pose[LEFT_SHOULDER][0] - pose[RIGHT_SHOULDER][0]) ** 2 +
                     (pose[LEFT_SHOULDER][1] - pose[RIGHT_SHOULDER][1]) ** 2)


def trim_range(poses, times, width, trim):
    """Step 3. Find the part of the clip that is left after cutting off the raising
    of the hand at the start and the lowering at the end.

    `poses` have no gaps, `times` are in milliseconds, `width` is the shoulder width,
    `trim` holds "trim_speed" (shoulder widths per second) and "trim_max_ms".
    Returns (first, last): the indexes of the first and last frame to keep.
    """
    total = len(poses)

    def speed_up(t):
        """How fast the fastest wrist moved UP between frame t-1 and frame t."""
        seconds = (times[t] - times[t - 1]) / 1000
        if seconds <= 0:
            return 0.0
        left = poses[t - 1][LEFT_WRIST][1] - poses[t][LEFT_WRIST][1]
        right = poses[t - 1][RIGHT_WRIST][1] - poses[t][RIGHT_WRIST][1]
        return max(left, right) / width / seconds

    def speed_down(t):
        """How fast the fastest wrist moved DOWN between frame t-1 and frame t."""
        seconds = (times[t] - times[t - 1]) / 1000
        if seconds <= 0:
            return 0.0
        left = poses[t][LEFT_WRIST][1] - poses[t - 1][LEFT_WRIST][1]
        right = poses[t][RIGHT_WRIST][1] - poses[t - 1][RIGHT_WRIST][1]
        return max(left, right) / width / seconds

    first = 0
    while (first < total - 1
           and times[first + 1] - times[0] <= trim["trim_max_ms"]
           and speed_up(first + 1) > trim["trim_speed"]):
        first += 1

    last = total - 1
    while (last > first
           and times[total - 1] - times[last - 1] <= trim["trim_max_ms"]
           and speed_down(last) > trim["trim_speed"]):
        last -= 1

    if last - first + 1 < MIN_FRAMES_AFTER_TRIM:
        return 0, total - 1          # too little would be left: keep everything
    return first, last


def clip_to_features(frames, aspect, trim=None):
    """Build the (32, 140) feature array of one clip.

    `frames` is a list with one dict per video frame:
        {"pose": (6, 2) array of x, y as given by MediaPipe, or None,
         "hands": list of 0-2 arrays of shape (21, 3) as given by MediaPipe,
         "time": time of the frame in milliseconds (only needed when trimming)}
    `aspect` is picture width / picture height.
    `trim` is the content of config/segmenter.json, or None to skip step 3.

    Returns None if the clip must be dropped (no pose, or no hand in any frame).
    """
    if len(frames) == 0:
        return None

    # Step 1: same unit for x and y.
    poses = []
    for f in frames:
        if f["pose"] is None:
            poses.append(None)
        else:
            p = np.array(f["pose"], dtype=np.float64)
            p[:, 0] *= aspect
            poses.append(p)
    all_hands = []
    for f in frames:
        scaled = []
        for h in f["hands"]:
            h = np.array(h, dtype=np.float64)
            h[:, 0] *= aspect
            h[:, 2] *= aspect
            scaled.append(h)
        all_hands.append(scaled)

    # Step 2: fill frames without a pose.
    poses = fill_missing_pose(poses)
    if poses is None:
        return None

    # Step 3: cut off the raising and the lowering of the hand.
    if trim is not None:
        width = median([shoulder_width(p) for p in poses])
        if width < 1e-6:
            return None
        first, last = trim_range(poses, [f["time"] for f in frames], width, trim)
        poses = poses[first:last + 1]
        all_hands = all_hands[first:last + 1]

    if all(len(h) == 0 for h in all_hands):
        return None

    # Step 4: one body reference for the whole clip.
    mid_x = median([(p[LEFT_SHOULDER][0] + p[RIGHT_SHOULDER][0]) / 2 for p in poses])
    mid_y = median([(p[LEFT_SHOULDER][1] + p[RIGHT_SHOULDER][1]) / 2 for p in poses])
    width = median([shoulder_width(p) for p in poses])
    if width < 1e-6:
        return None

    # Steps 5 and 6: one 140-number vector per frame.
    vectors = np.zeros((len(poses), FEATURE_SIZE), dtype=np.float64)
    for t in range(len(poses)):
        left, right = assign_hands(all_hands[t], poses[t])
        for hand, start, flag in ((left, LEFT_HAND_START, LEFT_FLAG),
                                  (right, RIGHT_HAND_START, RIGHT_FLAG)):
            if hand is None:
                continue
            for k in range(HAND_POINTS):
                vectors[t, start + 3 * k + 0] = (hand[k][0] - mid_x) / width
                vectors[t, start + 3 * k + 1] = (hand[k][1] - mid_y) / width
                vectors[t, start + 3 * k + 2] = hand[k][2] / width
            vectors[t, flag] = 1.0
        for k in range(POSE_POINTS):
            vectors[t, POSE_START + 2 * k + 0] = (poses[t][k][0] - mid_x) / width
            vectors[t, POSE_START + 2 * k + 1] = (poses[t][k][1] - mid_y) / width

    # Step 7: exactly 32 frames.
    return resample(vectors).astype(np.float32)


def resample(vectors, num_frames=NUM_FRAMES):
    """Step 7. Pick `num_frames` frames spread evenly over the clip (no blending)."""
    total = len(vectors)
    picked = []
    for i in range(num_frames):
        if total == 1:
            source = 0
        else:
            source = int(math.floor(i * (total - 1) / (num_frames - 1) + 0.5))
        picked.append(vectors[source])
    return np.stack(picked)
