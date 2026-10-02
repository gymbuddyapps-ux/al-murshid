"""Run MediaPipe (hand + pose) on the pictures of one clip.

Used by scripts/extract_landmarks.py (training data) and by the test tools, so that
every Python path finds landmarks in exactly the same way.
"""
import json
from pathlib import Path

import cv2
import numpy as np

EXTRACTION = json.loads(Path("config/extraction.json").read_text(encoding="utf-8"))
FRAME_MS = 1000 // EXTRACTION["dataset_fps"]      # 40 ms between frames at 25 frames per second

# MediaPipe pose indexes for: left shoulder, right shoulder, left elbow, right elbow,
# left wrist, right wrist (see docs/features.md).
POSE_INDEXES = [11, 12, 13, 14, 15, 16]


def make_landmarkers():
    """Create the hand and pose detectors with the settings from config/extraction.json."""
    import mediapipe as mp  # imported here so each worker process loads its own copy
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=EXTRACTION["hand_model"]),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=EXTRACTION["num_hands"],
        min_hand_detection_confidence=EXTRACTION["min_hand_detection_confidence"],
        min_hand_presence_confidence=EXTRACTION["min_hand_presence_confidence"],
        min_tracking_confidence=EXTRACTION["min_tracking_confidence"]))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=EXTRACTION["pose_model"]),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=EXTRACTION["num_poses"],
        min_pose_detection_confidence=EXTRACTION["min_pose_detection_confidence"],
        min_pose_presence_confidence=EXTRACTION["min_pose_presence_confidence"],
        min_tracking_confidence=EXTRACTION["min_tracking_confidence"]))
    return mp, hand, pose


def detect_clip(images):
    """Find the landmarks in every picture of one clip.

    `images` is a list of pictures in OpenCV's BGR format (None for an unreadable picture).
    Returns (pose, hands, hand_count):
        pose        (T, 6, 2)      x, y of shoulders, elbows, wrists; NaN where no pose was found
        hands       (T, 2, 21, 3)  up to two hands in MediaPipe's order; NaN where missing
        hand_count  (T,)           how many hands MediaPipe found in each frame
    """
    # New detectors for every clip, so tracking never leaks from one clip into the next.
    mp, hand_landmarker, pose_landmarker = make_landmarkers()

    total = len(images)
    pose = np.full((total, 6, 2), np.nan, dtype=np.float32)
    hands = np.full((total, 2, 21, 3), np.nan, dtype=np.float32)
    hand_count = np.zeros(total, dtype=np.int8)

    for t, bgr in enumerate(images):
        if bgr is None:
            continue
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        timestamp = t * FRAME_MS

        pose_result = pose_landmarker.detect_for_video(image, timestamp)
        if pose_result.pose_landmarks:
            body = pose_result.pose_landmarks[0]
            for k, index in enumerate(POSE_INDEXES):
                pose[t, k] = (body[index].x, body[index].y)

        hand_result = hand_landmarker.detect_for_video(image, timestamp)
        found = hand_result.hand_landmarks[:2]
        hand_count[t] = len(found)
        for h, landmarks in enumerate(found):
            for k, point in enumerate(landmarks):
                hands[t, h, k] = (point.x, point.y, point.z)

    hand_landmarker.close()
    pose_landmarker.close()
    return pose, hands, hand_count


def read_image(path):
    """Read a picture file. (np.fromfile + imdecode works with any characters in the path.)"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def to_frames(pose, hands, hand_count):
    """Turn the arrays from detect_clip() into the list of frames that jisr/features.py expects."""
    frames = []
    for t in range(len(hand_count)):
        frames.append({
            "pose": None if np.isnan(pose[t]).any() else pose[t],
            "hands": [hands[t, h] for h in range(int(hand_count[t]))],
            "time": t * FRAME_MS,
        })
    return frames
