"""Honest evaluation on the held-out signer (Phase 3).

What happens, in order:
 1. Calibration on VALIDATION data only: fit the temperature and choose the rejection
    threshold from reports/validation_scores.npz (made by scripts/train.py).
 2. Test: run the final model once on the test signer, a person who was never used for
    training or for any choice.
 3. Write the numbers and figures:
       reports/metrics.json                 all numbers
       docs/figures/confusion_matrix.png    which signs get mixed up
       docs/figures/confidence.png          confidence of correct vs wrong guesses
       models/labels.json                   labels + temperature + threshold for the app

Usage:  python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jisr.metrics import (app_decisions, app_metrics, choose_threshold, confusion_matrix,  # noqa: E402
                          fit_temperature, per_class_report, softmax, word_accuracy)
from jisr.model import SignCNN  # noqa: E402
from jisr.training import load_config, load_data, predict_scores, split_masks  # noqa: E402
from jisr.vocab import class_labels, load_vocab  # noqa: E402

TARGET_TOP1 = 0.90
TARGET_TOP3 = 0.97


def english_labels(vocab):
    """English names for the figures (matplotlib does not draw Arabic letters joined correctly)."""
    return [c["english"] for c in vocab["classes"]] + ["other sign"]


def plot_confusion(matrix, names, path):
    figure, axis = plt.subplots(figsize=(11, 10))
    axis.imshow(matrix, cmap="Greens")
    axis.set_xticks(range(len(names)), names, rotation=90)
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("What the model answered")
    axis.set_ylabel("The true sign")
    axis.set_title("Confusion matrix, held-out signer (number of clips)")
    for i in range(len(names)):
        for j in range(len(names)):
            if matrix[i, j]:
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                axis.text(j, i, matrix[i, j], ha="center", va="center", color=color, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_confidence(confidence, correct, threshold, path):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 21)
    axis.hist(confidence[correct], bins=bins, alpha=0.75, color="#1f6b4a", label="correct guesses")
    axis.hist(confidence[~correct], bins=bins, alpha=0.75, color="#8e1b1f", label="wrong guesses")
    axis.axvline(threshold, color="black", linestyle="--", label=f"rejection threshold = {threshold:.2f}")
    axis.set_xlabel("Confidence of the best guess (after calibration)")
    axis.set_ylabel("Number of clips")
    axis.set_title("Confidence on the held-out signer, vocabulary clips")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main():
    config = load_config()
    vocab = load_vocab()
    labels = class_labels(vocab)
    num_classes = len(labels)
    other_id = num_classes - 1

    # ---- 1. Calibration, on validation data only ----
    validation = np.load("reports/validation_scores.npz")
    temperature = fit_temperature(validation["scores"], validation["y"])
    validation_probabilities = softmax(validation["scores"], temperature)
    threshold, threshold_table = choose_threshold(validation_probabilities, validation["y"], other_id)

    # ---- 2. Test on the held-out signer ----
    data = load_data()
    _, test_mask = split_masks(data, config["dev_signers"], config["test_signer"], "test")
    X_test, y_test = data["X"][test_mask], data["y"][test_mask]

    model = SignCNN(num_classes, hidden=tuple(config["hidden"]), dropout=config["dropout"])
    # The measured model: trained on the development signers only (see scripts/train.py).
    model.load_state_dict(torch.load("models/jisr_model_measured.pt"))
    scores = predict_scores(model, X_test)
    probabilities = softmax(scores, temperature)

    words = word_accuracy(scores, y_test, other_id)
    app = app_metrics(probabilities, y_test, other_id, threshold)

    accepted, best_word, confidence = app_decisions(probabilities, other_id, threshold)
    is_word = y_test != other_id
    correct = best_word == y_test

    metrics = {
        "protocol": {
            "train_signers": config["dev_signers"],
            "test_signer": config["test_signer"],
            "note": "The test signer was not used for training, model choice, temperature or threshold.",
        },
        "test": {
            "vocabulary_clips": words["clips"],
            "top1_accuracy": words["top1"],
            "top3_accuracy": words["top3"],
            "target_top1": TARGET_TOP1,
            "target_top3": TARGET_TOP3,
            "targets_met": bool(words["top1"] >= TARGET_TOP1 and words["top3"] >= TARGET_TOP3),
        },
        "calibration": {
            "temperature": temperature,
            "threshold": threshold,
            "chosen_on": "validation data (development signers only)",
            "validation_threshold_table": threshold_table,
        },
        "test_with_rejection": app,
        "confidence": {
            "mean_when_correct": float(confidence[is_word & correct].mean()),
            "mean_when_wrong": float(confidence[is_word & ~correct].mean()) if (is_word & ~correct).any() else None,
        },
        "per_class": per_class_report(scores, y_test, other_id, labels, probabilities, threshold),
        "confusion_matrix": confusion_matrix(scores, y_test, num_classes).tolist(),
        "labels": labels,
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 3. Figures and the app's label file ----
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    plot_confusion(np.array(metrics["confusion_matrix"]), english_labels(vocab), "docs/figures/confusion_matrix.png")
    plot_confidence(confidence[is_word], correct[is_word], threshold, "docs/figures/confidence.png")

    Path("models/labels.json").write_text(json.dumps({
        "labels": labels, "other_index": other_id,
        "temperature": temperature, "threshold": threshold,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"held-out signer {config['test_signer']}: top-1 {words['top1']:.4f}  top-3 {words['top3']:.4f} "
          f"on {words['clips']} vocabulary clips (targets met: {metrics['test']['targets_met']})")
    print(f"temperature {temperature}, threshold {threshold:.2f}: acceptance {app['acceptance_rate']:.3f}, "
          f"accuracy on accepted {app['accuracy_on_accepted']:.3f}, "
          f"unknown signs rejected {app['unknown_sign_rejection_rate']}")


if __name__ == "__main__":
    main()
