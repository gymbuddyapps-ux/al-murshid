"""Pick the KArSL signs used for the "other sign" class and save them in config/vocab.json.

Why an "other" class: KArSL clips contain only the sign itself (no hands-down part),
so there is nothing to build a "no sign" class from. Instead the model learns one extra
class made of signs that are NOT in our vocabulary. When a user makes an unknown gesture,
the model can then answer "other", and the app asks to repeat instead of guessing a word.

60 signs are picked at random (fixed seed, so the choice is repeatable):
  - 40 are used for training,
  - 10 different ones are used only for validation (choosing settings),
  - 10 different ones are used only for the final test,
so we can check that the model also rejects signs it has never seen.

Usage:  python scripts/choose_other_signs.py
"""
import json
import random
from pathlib import Path

VOCAB = Path("config/vocab.json")
SEED = 2026
# Signs that may look like a vocabulary word (same topic, similar label); never used as "other".
EXCLUDED = {134, 496, 498, 499}      # sick, pharmacist, nurse, orderly


def main():
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    used = {c["karsl_sign_id"] for c in vocab["classes"]}
    pool = [i for i in range(1, 503) if i not in used and i not in EXCLUDED]

    picked = random.Random(SEED).sample(pool, 60)
    vocab["other_class"] = {
        "label": "إشارة أخرى",
        "clips_per_sign_per_signer": 5,
        "train_sign_ids": sorted(picked[:40]),
        "validation_sign_ids": sorted(picked[40:50]),
        "test_sign_ids": sorted(picked[50:]),
    }
    VOCAB.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for part in ("train_sign_ids", "validation_sign_ids", "test_sign_ids"):
        print(part, vocab["other_class"][part])


if __name__ == "__main__":
    main()
