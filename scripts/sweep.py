"""Try a few training settings on the VALIDATION signers only, and print a table.

The test signer is never used here. The best setting is then written into
config/training.json, and scripts/train.py + scripts/evaluate.py run once.

Usage:  python scripts/sweep.py
Output: reports/sweep.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jisr.metrics import word_accuracy  # noqa: E402
from jisr.training import load_config, load_data, predict_scores, split_masks, train_model  # noqa: E402
from jisr.vocab import class_labels, load_vocab  # noqa: E402

STRONGER_AUGMENTATION = {"mirror_chance": 0.5, "max_rotation_degrees": 15, "max_scale_change": 0.2,
                         "max_time_cut": 0.2, "drop_frame_chance": 0.15, "drop_hand_chance": 0.08, "jitter": 0.03}

# Each entry: a name and the settings that differ from config/training.json.
CANDIDATES = [
    ("current", {}),
    ("wider", {"hidden": [96, 128, 192], "dropout": 0.4}),
    ("longer_stronger_aug", {"max_epochs": 120, "augmentation": STRONGER_AUGMENTATION}),
]


def main():
    base = load_config()
    vocab = load_vocab()
    num_classes = len(class_labels(vocab))
    other_id = num_classes - 1
    data = load_data()
    results = []
    for name, changes in CANDIDATES:
        config = {**base, **changes}
        folds = []
        started = time.time()
        for val_signer in base["dev_signers"]:
            train_signers = [s for s in base["dev_signers"] if s != val_signer]
            train_mask, val_mask = split_masks(data, train_signers, val_signer, "validation")
            model, _ = train_model(data["X"][train_mask], data["y"][train_mask], num_classes, config,
                                   config["max_epochs"], config["seed"])
            folds.append(word_accuracy(predict_scores(model, data["X"][val_mask]), data["y"][val_mask], other_id))
        row = {"name": name, "changes": changes,
               "top1": float(np.mean([f["top1"] for f in folds])),
               "top3": float(np.mean([f["top3"] for f in folds])),
               "seconds": round(time.time() - started)}
        results.append(row)
        print(f"{name:22s} validation top-1 {row['top1']:.3f}  top-3 {row['top3']:.3f}  ({row['seconds']} s)",
              flush=True)
        Path("reports/sweep.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
