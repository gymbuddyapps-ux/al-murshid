"""Shared pieces for training and evaluation: loading data, splitting by signer,
training one model, and the simple baseline (nearest neighbour with dynamic time warping).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from jisr.augment import augment
from jisr.model import SignCNN
from jisr.vocab import load_vocab

FEATURES_PATH = Path("data/features/karsl_v1.npz")
TRAINING_CONFIG = Path("config/training.json")


def load_config():
    return json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))


def load_data():
    """Load all features. Returns a dict of numpy arrays: X, y, signer, sign_id, sample."""
    data = np.load(FEATURES_PATH)
    return {key: data[key] for key in data.files}


def split_masks(data, train_signers, eval_signer, eval_part):
    """Decide which clips are used for training and which for checking.

    Vocabulary words: split only by signer.
    "Other sign" class: split by signer AND by sign, so the checked clips are of signs
    the model never saw during training:
        training uses the "train" other-signs,
        validation uses the "validation" other-signs,
        the final test uses the "test" other-signs.

    eval_part is "validation" or "test".
    Returns (train_mask, eval_mask): two true/false arrays with one entry per clip.
    """
    vocab = load_vocab()
    other = vocab["other_class"]
    other_id = len(vocab["classes"])

    is_other = data["y"] == other_id
    in_train_signers = np.isin(data["signer"], train_signers)
    in_eval_signer = data["signer"] == eval_signer

    other_train_signs = np.isin(data["sign_id"], other["train_sign_ids"])
    other_eval_signs = np.isin(data["sign_id"], other[eval_part + "_sign_ids"])

    train_mask = in_train_signers & (~is_other | other_train_signs)
    eval_mask = in_eval_signer & (~is_other | other_eval_signs)
    return train_mask, eval_mask


def set_seed(seed):
    """Fix all random numbers so that a run can be repeated exactly."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def predict_scores(model, X):
    """Run the model on clips and return its raw scores (one row per clip)."""
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy()


def train_model(X_train, y_train, num_classes, config, epochs, seed, X_val=None, y_val=None):
    """Train one model for a fixed number of epochs.

    An epoch is one pass over all training clips. Every clip is randomly changed
    (augmented) each time it is shown.
    If validation clips are given, the validation accuracy after every epoch is returned too.
    Returns (model, list of validation accuracies).
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = SignCNN(num_classes, hidden=tuple(config["hidden"]), dropout=config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    # The learning rate goes down smoothly to zero over the run.
    steps_per_epoch = int(np.ceil(len(X_train) / config["batch_size"]))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["learning_rate"], total_steps=epochs * steps_per_epoch)
    loss_function = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    history = []
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), config["batch_size"]):
            batch = order[start:start + config["batch_size"]]
            if len(batch) < 2:
                continue                                  # batch norm needs at least 2 clips
            clips = np.stack([augment(X_train[i], rng, config["augmentation"]) for i in batch])
            scores = model(torch.from_numpy(clips))
            loss = loss_function(scores, torch.from_numpy(y_train[batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        if X_val is not None:
            predictions = predict_scores(model, X_val).argmax(axis=1)
            history.append(float((predictions == y_val).mean()))
    return model, history


# ---------- Baseline: nearest neighbour with dynamic time warping (DTW) ----------

def dtw_distances(clip, references):
    """DTW distance between one clip and many reference clips at once.

    DTW compares two movements even if one is faster than the other: it finds the best
    way to line up the frames of the two clips, then adds up how different the
    lined-up frames are.

    clip: (T, D); references: (N, T, D). Returns N distances.
    """
    steps = clip.shape[0]
    # cost[n, i, j] = how different frame i of the clip is from frame j of reference n
    cost = np.linalg.norm(clip[None, :, None, :] - references[:, None, :, :], axis=3)
    total = np.full((len(references), steps + 1, steps + 1), np.inf)
    total[:, 0, 0] = 0
    for i in range(1, steps + 1):
        for j in range(1, steps + 1):
            best_before = np.minimum(np.minimum(total[:, i - 1, j], total[:, i, j - 1]),
                                     total[:, i - 1, j - 1])
            total[:, i, j] = cost[:, i - 1, j - 1] + best_before
    return total[:, steps, steps]


def knn_dtw_predict(X_train, y_train, X_eval, neighbours=1, every=2):
    """Baseline classifier: give each clip the label of its most similar training clip.

    To keep it fast, only every second frame is used (16 frames instead of 32).
    """
    references = X_train[:, ::every].astype(np.float32)
    predictions = []
    for clip in X_eval[:, ::every].astype(np.float32):
        distances = dtw_distances(clip, references)
        nearest = np.argsort(distances)[:neighbours]
        votes = np.bincount(y_train[nearest])
        predictions.append(int(votes.argmax()))
    return np.array(predictions)
