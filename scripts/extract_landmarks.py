"""Run MediaPipe on every clip and save the raw landmarks (Phase 1, part 1).

Input:   data/raw/karsl/frames/<sign id>/<sample name>/*.jpg
Output:  data/landmarks/<sign id>/<sample name>.npz     (one small file per clip)

Each output file holds, for a clip of T frames:
    pose        (T, 6, 2)      x, y of shoulders, elbows, wrists; NaN where no pose was found
    hands       (T, 2, 21, 3)  up to two hands in MediaPipe's order; NaN where missing
    hand_count  (T,)           how many hands MediaPipe found in each frame
    width, height              picture size in pixels

The slow part (MediaPipe) is done once. A clip whose output file already exists is
skipped, so the script can be stopped and started again ("resume").
The features themselves are built from these files by scripts/build_features.py.

Usage:  python scripts/extract_landmarks.py [--workers 6]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jisr.detect import detect_clip, read_image  # noqa: E402
from jisr.vocab import load_vocab, other_sign_ids  # noqa: E402

FRAMES_DIR = Path("data/raw/karsl/frames")
OUT_DIR = Path("data/landmarks")


def extract_clip(clip_dir):
    """Process one clip folder. Returns a short status string."""
    clip_dir = Path(clip_dir)
    out_path = OUT_DIR / clip_dir.parent.name / (clip_dir.name + ".npz")
    if out_path.exists():
        return "cached"

    images = [read_image(p) for p in sorted(clip_dir.glob("*.jpg"))]
    images = [img for img in images if img is not None]
    if not images:
        return "empty"

    pose, hands, hand_count = detect_clip(images)
    height, width = images[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, pose=pose, hands=hands, hand_count=hand_count,
                        width=width, height=height)
    return "done"


def choose_clips():
    """All clips of the vocabulary signs, plus a few clips per signer of each "other" sign."""
    vocab = load_vocab()
    other_ids = {f"{i:04d}" for i in other_sign_ids(vocab)}
    per_signer = vocab.get("other_class", {}).get("clips_per_sign_per_signer", 0)

    chosen = []
    for sign_dir in sorted(p for p in FRAMES_DIR.glob("*") if p.is_dir()):
        clips = sorted(p for p in sign_dir.glob("*") if p.is_dir())
        if sign_dir.name in other_ids:
            # Sample names look like 03_01_0092_(date)_c : the second part is the signer.
            taken = {}
            for clip in clips:
                signer = clip.name.split("_")[1]
                if taken.get(signer, 0) < per_signer:
                    taken[signer] = taken.get(signer, 0) + 1
                    chosen.append(str(clip))
        else:
            chosen.extend(str(clip) for clip in clips)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    clip_dirs = choose_clips()
    print(f"{len(clip_dirs)} clips to process with {args.workers} workers")

    counts = {}
    with Pool(args.workers) as pool:
        for i, status in enumerate(pool.imap_unordered(extract_clip, clip_dirs, chunksize=4), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 200 == 0:
                print(f"  {i}/{len(clip_dirs)} {counts}", flush=True)
    print("finished:", counts)


if __name__ == "__main__":
    main()
