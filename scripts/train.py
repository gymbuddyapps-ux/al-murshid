"""Train the sign model (Phase 2).

What happens, in order:
 1. Validation rounds. The two development signers take turns: train on one, check on the
    other. This tells us how well the settings in config/training.json work on a person
    the model has not seen. The simple baseline (nearest neighbour + DTW) is checked the
    same way, and the better of the two is kept.
 2. The final model is trained on both development signers together.
 3. The final model is saved for Python (models/jisr_model.pt) and for the browser
    (models/jisr_model.onnx).

The test signer is never touched here. It is only used by scripts/evaluate.py.

Outputs:
    models/jisr_model.pt, models/jisr_model.onnx
    reports/training_report.json        validation numbers, model choice, settings used
    reports/validation_scores.npz       the model's validation outputs (for calibration)

Usage:  python scripts/train.py [--skip-baseline]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jisr.metrics import word_accuracy  # noqa: E402
from jisr.model import export_onnx  # noqa: E402
from jisr.training import (knn_dtw_predict, load_config, load_data, predict_scores,  # noqa: E402
                           split_masks, train_model)
from jisr.vocab import class_labels, load_vocab  # noqa: E402


def onnx_check(model, clips, onnx_path):
    """Run the saved ONNX file with ONNX Runtime and compare with PyTorch on a few clips.
    Returns the largest difference between the two sets of scores (should be about 0)."""
    import onnxruntime

    session = onnxruntime.InferenceSession(onnx_path)
    # The ONNX model takes one clip at a time, exactly like the browser.
    from_onnx = np.concatenate([session.run(None, {"features": clip[None]})[0] for clip in clips])
    from_torch = predict_scores(model, clips)
    return float(np.abs(from_onnx - from_torch).max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline", action="store_true",
                        help="do not run the slow nearest-neighbour baseline")
    parser.add_argument("--skip-validation", action="store_true",
                        help="reuse reports/validation_scores.npz and only train the final model")
    parser.add_argument("--final-signers", default="dev", choices=["dev", "all"],
                        help="dev: train the final model on the development signers (the model that is "
                             "evaluated). all: also use the test signer, for the model shipped in the app "
                             "(its accuracy on new signers cannot be measured; see docs/REPORT.md)")
    args = parser.parse_args()

    config = load_config()
    vocab = load_vocab()
    labels = class_labels(vocab)
    num_classes = len(labels)
    other_id = num_classes - 1
    data = load_data()
    dev_signers = config["dev_signers"]
    started = time.time()

    report = {"config": config, "labels": labels, "folds": []}
    validation_scores, validation_y, validation_signer = [], [], []

    if args.skip_validation:
        report = json.loads(Path("reports/training_report.json").read_text(encoding="utf-8"))
        dev_signers = []          # no validation rounds this time

    # ---- 1. Validation rounds ----
    for val_signer in dev_signers:
        train_signers = [s for s in dev_signers if s != val_signer]
        train_mask, val_mask = split_masks(data, train_signers, val_signer, "validation")
        X_train, y_train = data["X"][train_mask], data["y"][train_mask]
        X_val, y_val = data["X"][val_mask], data["y"][val_mask]

        model, history = train_model(X_train, y_train, num_classes, config, config["max_epochs"],
                                     config["seed"], X_val, y_val)
        scores = predict_scores(model, X_val)
        fold = {
            "train_signers": train_signers,
            "validation_signer": val_signer,
            "train_clips": int(train_mask.sum()),
            "validation_clips": int(val_mask.sum()),
            "cnn": word_accuracy(scores, y_val, other_id),
            "cnn_accuracy_all_classes_per_epoch": [round(a, 4) for a in history],
        }

        if not args.skip_baseline:
            # The baseline only knows the vocabulary words (no "other" class), compared on word clips.
            words_train, words_val = y_train != other_id, y_val != other_id
            predicted = knn_dtw_predict(X_train[words_train], y_train[words_train], X_val[words_val])
            fold["knn_dtw"] = {"clips": int(words_val.sum()),
                               "top1": float((predicted == y_val[words_val]).mean())}

        report["folds"].append(fold)
        validation_scores.append(scores)
        validation_y.append(y_val)
        validation_signer.append(np.full(len(y_val), val_signer))
        print(f"validate on signer {val_signer}: CNN top-1 {fold['cnn']['top1']:.3f} "
              f"top-3 {fold['cnn']['top3']:.3f}"
              + (f" | kNN-DTW top-1 {fold['knn_dtw']['top1']:.3f}" if "knn_dtw" in fold else ""),
              flush=True)

    if not args.skip_validation:
        report["validation_mean"] = {
            "cnn_top1": float(np.mean([f["cnn"]["top1"] for f in report["folds"]])),
            "cnn_top3": float(np.mean([f["cnn"]["top3"] for f in report["folds"]])),
        }
        if not args.skip_baseline:
            report["validation_mean"]["knn_dtw_top1"] = float(
                np.mean([f["knn_dtw"]["top1"] for f in report["folds"]]))
            cnn_wins = report["validation_mean"]["cnn_top1"] >= report["validation_mean"]["knn_dtw_top1"]
            report["chosen_model"] = "cnn" if cnn_wins else "knn_dtw"
        else:
            report["chosen_model"] = "cnn"
        print("validation mean:", report["validation_mean"], "-> chosen:", report["chosen_model"])

        Path("reports").mkdir(exist_ok=True)
        np.savez_compressed("reports/validation_scores.npz",
                            scores=np.concatenate(validation_scores), y=np.concatenate(validation_y),
                            signer=np.concatenate(validation_signer))

    # ---- 2. Final model: on the development signers (evaluated) or on all signers (shipped) ----
    final_signers = config["dev_signers"] + ([config["test_signer"]] if args.final_signers == "all" else [])
    train_mask, _ = split_masks(data, final_signers, config["test_signer"], "test")
    model, _ = train_model(data["X"][train_mask], data["y"][train_mask], num_classes, config,
                           config["max_epochs"], config["seed"])
    key = "final_model" if args.final_signers == "dev" else "shipped_model"
    report[key] = {
        "train_signers": final_signers,
        "train_clips": int(train_mask.sum()),
        "epochs": config["max_epochs"],
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }

    # ---- 3. Save ----
    # jisr_model_measured.* is the evaluated model (development signers only).
    # jisr_model.* is what the app ships: the same file, or the all-signer model when --final-signers all.
    Path("models").mkdir(exist_ok=True)
    stem = "models/jisr_model" if args.final_signers == "all" else "models/jisr_model_measured"
    torch.save(model.state_dict(), stem + ".pt")
    export_onnx(model, stem + ".onnx")
    if args.final_signers == "dev":
        shutil.copy(stem + ".pt", "models/jisr_model.pt")
        shutil.copy(stem + ".onnx", "models/jisr_model.onnx")
        report.pop("shipped_model", None)
    report[key]["onnx_bytes"] = Path(stem + ".onnx").stat().st_size
    report[key]["onnx_max_difference"] = onnx_check(model, data["X"][train_mask][:64], stem + ".onnx")
    report["seconds"] = round(time.time() - started)
    Path("reports/training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {stem}.onnx ({report[key]['onnx_bytes']} bytes), {report[key]['train_clips']} training clips "
          f"from signers {final_signers}, {report['seconds']} s")


if __name__ == "__main__":
    main()
