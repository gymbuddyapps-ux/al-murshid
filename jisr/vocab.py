"""Small helpers for reading config/vocab.json."""
import json
from pathlib import Path

VOCAB_PATH = Path("config/vocab.json")


def load_vocab():
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def other_sign_ids(vocab):
    """All KArSL sign ids used for the "other sign" class (training, validation and test)."""
    other = vocab.get("other_class", {})
    return (other.get("train_sign_ids", []) + other.get("validation_sign_ids", []) +
            other.get("test_sign_ids", []))


def class_labels(vocab):
    """The label of every model output, in order. The last one is the "other sign" class."""
    return [c["arabic"] for c in vocab["classes"]] + [vocab["other_class"]["label"]]
