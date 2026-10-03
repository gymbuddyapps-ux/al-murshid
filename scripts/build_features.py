"""Turn the raw landmarks into model inputs (Phase 1, part 2).

Input:   data/landmarks_v2/<sign id>/<sample name>.npz     (from scripts/extract_landmarks.py)
Output:  data/features/karsl_v2.npz
             X        (N, 32, 146)  features, see docs/features.md
             y        (N,)          class id from config/vocab.json; the last id is "other sign"
             signer   (N,)          signer id (1, 2 or 3)
             sign_id  (N,)          KArSL sign id of the clip
             sample   (N,)          sample name, to trace a row back to its clip
         reports/extraction_report.json   how many clips were kept and dropped per class

Usage:  python scripts/build_features.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jisr.detect import to_frames  # noqa: E402
from jisr.features import clip_to_features, fill_missing_pose, median, shoulder_width, trim_range  # noqa: E402
from jisr.vocab import class_sign_ids, load_vocab, other_sign_ids  # noqa: E402

LANDMARKS_DIR = Path("data/landmarks_v2")
OUT_PATH = Path("data/features/karsl_v2.npz")
REPORT_PATH = Path("reports/extraction_report.json")

SEGMENTER = json.loads(Path("config/segmenter.json").read_text(encoding="utf-8"))


def load_clip(path):
    """Read one landmark file and return (frames, aspect) in the form features.py expects."""
    data = np.load(path)
    frames = to_frames(data["pose"], data["hands"], data["hand_count"])
    aspect = float(data["width"]) / float(data["height"]) if data["height"] > 0 else 1.0
    return frames, aspect


def clip_files(sign_id, is_other, vocab):
    """The landmark files of one sign. For "other" signs only the first few clips per signer
    are used (the same rule as scripts/extract_landmarks.py), so the class stays balanced."""
    files = sorted((LANDMARKS_DIR / f"{sign_id:04d}").glob("*.npz"))
    if not is_other:
        return files
    limit = vocab["other_class"]["clips_per_sign_per_signer"]
    taken, chosen = {}, []
    for path in files:
        signer = path.stem.split("_")[1]
        if taken.get(signer, 0) < limit:
            taken[signer] = taken.get(signer, 0) + 1
            chosen.append(path)
    return chosen


def drop_reason(frames):
    """Explain why features.py returned None for this clip."""
    if all(f["pose"] is None for f in frames):
        return "no_pose"
    return "no_hands"


def frames_trimmed(frames, aspect):
    """How many frames step 3 (trimming) removes from this clip. Only for the report."""
    poses = []
    for f in frames:
        if f["pose"] is None:
            poses.append(None)
        else:
            p = np.array(f["pose"], dtype=np.float64)
            p[:, 0] *= aspect
            poses.append(p)
    poses = fill_missing_pose(poses)
    if poses is None:
        return 0
    width = median([shoulder_width(p) for p in poses])
    if width < 1e-6:
        return 0
    first, last = trim_range(poses, [f["time"] for f in frames], width, SEGMENTER)
    return len(frames) - (last - first + 1)


def main():
    vocab = load_vocab()
    other_id = len(vocab["classes"])            # "other sign" gets the id after the last word

    # (class id, name for the report, KArSL sign id) for every sign folder we use
    sources = [(c["id"], c["arabic"], i) for c in vocab["classes"] for i in class_sign_ids(c)]
    sources += [(other_id, vocab["other_class"]["label"], i) for i in other_sign_ids(vocab)]

    X, y, signer, sign_ids, sample = [], [], [], [], []
    report = {"classes": {}, "total_clips": 0, "total_kept": 0, "total_dropped": 0}
    total_frames = total_hand_frames = total_trimmed = 0

    for class_id, name, sign_id in sources:
        stats = report["classes"].setdefault(str(class_id), {
            "label": name, "clips": 0, "kept": 0, "dropped_no_hands": 0, "dropped_no_pose": 0,
            "kept_per_signer": {}, "frames": 0, "frames_with_a_hand": 0, "frames_trimmed": 0})

        for path in clip_files(sign_id, class_id == other_id, vocab):
            frames, aspect = load_clip(path)
            stats["clips"] += 1
            stats["frames"] += len(frames)
            stats["frames_with_a_hand"] += sum(1 for f in frames if len(f["hands"]) > 0)
            stats["frames_trimmed"] += frames_trimmed(frames, aspect)

            features = clip_to_features(frames, aspect, SEGMENTER)
            if features is None:
                stats["dropped_" + drop_reason(frames)] += 1
                continue

            # Sample names look like 03_01_0092_(date)_c : group, SIGNER, sign id, recording time.
            signer_id = int(path.stem.split("_")[1])
            X.append(features)
            y.append(class_id)
            signer.append(signer_id)
            sign_ids.append(sign_id)
            sample.append(path.stem)
            stats["kept"] += 1
            key = str(signer_id)
            stats["kept_per_signer"][key] = stats["kept_per_signer"].get(key, 0) + 1

    for class_id, stats in report["classes"].items():
        frames = max(stats["frames"], 1)
        stats["frames_with_a_hand_percent"] = round(100 * stats["frames_with_a_hand"] / frames, 1)
        stats["frames_trimmed_percent"] = round(100 * stats["frames_trimmed"] / frames, 1)
        report["total_clips"] += stats["clips"]
        report["total_kept"] += stats["kept"]
        total_frames += stats["frames"]
        total_hand_frames += stats["frames_with_a_hand"]
        total_trimmed += stats["frames_trimmed"]
        print(f"{class_id:>2} clips={stats['clips']:4d} kept={stats['kept']:4d} "
              f"hand-frames={stats['frames_with_a_hand_percent']:5.1f}% "
              f"trimmed={stats['frames_trimmed_percent']:4.1f}%  {stats['kept_per_signer']}")

    report["total_dropped"] = report["total_clips"] - report["total_kept"]
    report["frames_with_a_hand_percent"] = round(100 * total_hand_frames / max(total_frames, 1), 1)
    report["frames_trimmed_percent"] = round(100 * total_trimmed / max(total_frames, 1), 1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, X=np.stack(X), y=np.array(y, dtype=np.int64),
                        signer=np.array(signer, dtype=np.int64),
                        sign_id=np.array(sign_ids, dtype=np.int64), sample=np.array(sample))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"kept {report['total_kept']} of {report['total_clips']} clips "
          f"(trimmed {report['frames_trimmed_percent']}% of frames) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
