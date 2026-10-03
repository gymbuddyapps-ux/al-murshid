"""How we measure the model. Used by scripts/train.py and scripts/evaluate.py.

Words used here:
- scores:        the model's raw output, one number per class, for each clip.
- probabilities: scores turned into numbers between 0 and 1 that add up to 1 (softmax).
- confidence:    the probability of the model's best guess.
- top-1 accuracy: how often the best guess is the correct word.
- top-3 accuracy: how often the correct word is among the 3 best guesses
                  (the app shows 3 candidates, so this is what the user experiences).
"""
import numpy as np


def softmax(scores, temperature=1.0):
    """Turn scores into probabilities. A temperature above 1 makes the model less sure."""
    scaled = scores / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exps = np.exp(scaled)
    return exps / exps.sum(axis=1, keepdims=True)


def word_ranking(scores, other_id):
    """For each clip, the word classes ordered from most to least likely (the "other" class is left out,
    exactly like the app does when it shows candidates)."""
    word_scores = np.delete(scores, other_id, axis=1)
    return np.argsort(-word_scores, axis=1)


def word_accuracy(scores, y, other_id):
    """Top-1 and top-3 accuracy on clips of vocabulary words."""
    is_word = y != other_id
    ranking = word_ranking(scores[is_word], other_id)
    truth = y[is_word]
    top1 = float((ranking[:, 0] == truth).mean())
    top3 = float((ranking[:, :3] == truth[:, None]).any(axis=1).mean())
    return {"clips": int(is_word.sum()), "top1": top1, "top3": top3}


def app_decisions(probabilities, other_id, threshold):
    """What the app does with each clip.

    Returns (accepted, best_word, confidence):
      accepted[i]   True if the app shows candidates, False if it says "unclear, repeat the sign"
      best_word[i]  the most likely vocabulary word
      confidence[i] the probability of that word
    """
    word_probabilities = np.delete(probabilities, other_id, axis=1)
    best_word = word_probabilities.argmax(axis=1)
    confidence = word_probabilities.max(axis=1)
    model_says_other = probabilities.argmax(axis=1) == other_id
    accepted = ~model_says_other & (confidence >= threshold)
    return accepted, best_word, confidence


def app_metrics(probabilities, y, other_id, threshold):
    """Acceptance rate, accuracy on accepted clips, and how often unknown signs are rejected."""
    accepted, best_word, _ = app_decisions(probabilities, other_id, threshold)
    is_word = y != other_id
    correct = best_word == y

    word_accepted = accepted & is_word
    result = {
        "threshold": float(threshold),
        "word_clips": int(is_word.sum()),
        "acceptance_rate": float(accepted[is_word].mean()),
        "accuracy_on_accepted": float(correct[word_accepted].mean()) if word_accepted.any() else 0.0,
        "accepted_and_correct_rate": float((word_accepted & correct).sum() / max(is_word.sum(), 1)),
        "unknown_sign_clips": int((~is_word).sum()),
        "unknown_sign_rejection_rate": float((~accepted[~is_word]).mean()) if (~is_word).any() else None,
    }
    return result


def fit_temperature(scores, y):
    """Temperature scaling: find the one number that makes the confidences most honest.

    We try many temperatures and keep the one with the lowest "negative log likelihood":
    a score that punishes being confident and wrong.
    """
    best_temperature, best_loss = 1.0, np.inf
    for temperature in np.arange(0.5, 5.01, 0.05):
        probabilities = softmax(scores, temperature)
        loss = -np.log(probabilities[np.arange(len(y)), y] + 1e-12).mean()
        if loss < best_loss:
            best_temperature, best_loss = float(round(temperature, 2)), float(loss)
    return best_temperature


def choose_threshold(probabilities, y, other_id):
    """Pick the rejection threshold on validation data.

    For every candidate threshold we compute a balance of two things we want:
      - vocabulary clips that are accepted AND correct,
      - unknown-sign clips that are rejected.
    We keep the threshold with the best average of the two.
    Returns (threshold, table of all candidates).
    """
    table = []
    for threshold in np.arange(0.0, 0.96, 0.05):
        m = app_metrics(probabilities, y, other_id, threshold)
        m["balance"] = (m["accepted_and_correct_rate"] + m["unknown_sign_rejection_rate"]) / 2
        table.append(m)
    best = max(table, key=lambda m: m["balance"])
    return best["threshold"], table


def per_class_report(scores, y, other_id, labels, probabilities=None, threshold=None):
    """Precision and recall for each class, using the model's single best guess over all classes.

    recall:    of all clips that really are this class, how many did the model find?
    precision: of all clips the model called this class, how many really are?
    If probabilities and a threshold are given, two app-level numbers are added for each word:
    in_top3:   how often the word was among the 3 candidates (what the user can tap),
    accepted:  how often the app showed candidates instead of "unclear".
    """
    predicted = scores.argmax(axis=1)
    if probabilities is not None:
        accepted_all, _, _ = app_decisions(probabilities, other_id, threshold)
        ranking = word_ranking(scores, other_id)
    rows = []
    for class_id, label in enumerate(labels):
        truth = y == class_id
        guess = predicted == class_id
        hit = int((truth & guess).sum())
        row = {
            "class_id": class_id,
            "label": label,
            "clips": int(truth.sum()),
            "precision": hit / int(guess.sum()) if guess.any() else None,
            "recall": hit / int(truth.sum()) if truth.any() else None,
        }
        if probabilities is not None and class_id != other_id and truth.any():
            row["accepted"] = float(accepted_all[truth].mean())
            row["in_top3"] = float((ranking[truth, :3] == class_id).any(axis=1).mean())
            row["accepted_and_in_top3"] = float((accepted_all[truth] & (ranking[truth, :3] == class_id).any(axis=1)).mean())
        rows.append(row)
    return rows


def confusion_matrix(scores, y, num_classes):
    """matrix[true class][predicted class] = number of clips."""
    predicted = scores.argmax(axis=1)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for truth, guess in zip(y, predicted):
        matrix[truth, guess] += 1
    return matrix
