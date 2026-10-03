"""Create test data that checks the JavaScript feature code against the Python one.

It invents random landmark clips (including missing poses, missing hands, one hand,
two hands, wide and tall pictures), runs jisr/features.py on them, and saves both
the inputs and the expected outputs to web/tests/fixtures/feature_parity.json.

The JavaScript test (web/tests/features.test.js) feeds the same inputs to
web/src/features.js and compares the outputs.

Usage:  python scripts/make_parity_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jisr.features import clip_to_features  # noqa: E402

OUT = Path("web/tests/fixtures/feature_parity.json")


def random_clip(rng, num_frames, pose_missing_chance, hand_chances, wrist_step):
    """Invent a clip. `wrist_step` is how far the wrists may move between two frames:
    small steps give slow wrists (little trimming), large steps give fast wrists."""
    frames = []
    time = 0.0
    pose_now = rng.uniform(0.3, 0.7, size=(9, 2))
    for _ in range(num_frames):
        pose_now = pose_now + rng.uniform(-wrist_step, wrist_step, size=(9, 2))
        pose = None
        if rng.random() >= pose_missing_chance:
            pose = pose_now.round(6).tolist()
        hands = []
        for chance in hand_chances:
            if rng.random() < chance:
                hands.append(rng.uniform(0.0, 1.0, size=(21, 3)).round(6).tolist())
        frames.append({"pose": pose, "hands": hands, "time": round(time, 3)})
        time += float(rng.uniform(25, 60))          # uneven frame times, like a real camera
    return frames


def main():
    rng = np.random.default_rng(7)
    trim = {"trim_speed": 2.0, "trim_max_ms": 500}
    settings = [
        # (frames, aspect, chance a pose is missing, chance of hand A and B, wrist step, trim)
        (40, 1.0, 0.0, (1.0, 0.0), 0.01, None),
        (15, 1.0, 0.2, (0.8, 0.5), 0.01, None),
        (64, 0.5625, 0.1, (1.0, 1.0), 0.01, trim),     # tall phone picture (9:16)
        (33, 1.7778, 0.5, (0.6, 0.6), 0.03, trim),     # wide picture (16:9)
        (1, 1.0, 0.0, (1.0, 0.0), 0.01, trim),         # a single frame
        (5, 0.75, 0.0, (0.0, 0.0), 0.01, trim),        # no hands at all -> dropped
        (5, 0.75, 1.0, (1.0, 1.0), 0.01, trim),        # no pose at all  -> dropped
        (100, 1.3333, 0.3, (0.9, 0.4), 0.05, trim),    # fast wrists: trimming happens
        (50, 1.0, 0.0, (1.0, 1.0), 0.08, trim),
        (6, 1.0, 0.0, (1.0, 0.0), 0.2, trim),          # short and fast: too little left, keep all
    ]
    cases = []
    for num_frames, aspect, pose_missing, hand_chances, wrist_step, case_trim in settings:
        frames = random_clip(rng, num_frames, pose_missing, hand_chances, wrist_step)
        features = clip_to_features(frames, aspect, case_trim)
        cases.append({
            "aspect": aspect,
            "trim": case_trim,
            "frames": frames,
            "expected": None if features is None else features.reshape(-1).astype(float).round(6).tolist(),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    kept = sum(1 for c in cases if c["expected"] is not None)
    print(f"wrote {len(cases)} cases ({kept} with features, {len(cases) - kept} dropped) -> {OUT}")


if __name__ == "__main__":
    main()
