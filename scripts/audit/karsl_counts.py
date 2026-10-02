"""Count KArSL samples per sign and per signer.

Inputs (both downloaded from Kaggle mirrors of KArSL, see docs/vocab_audit.md):
  data/raw/karsl/labels_mirror/KARSL-502_Labels.xlsx   official label sheet (SignID, Arabic, English)
  data/raw/karsl/probe/metadata.json                   one record per sample (sign_id, signer_id, num_frames)

Output:
  reports/audit/karsl_counts.json   one row per sign, used to write docs/vocab_audit.md

Usage:  python scripts/audit/karsl_counts.py
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

LABELS = Path("data/raw/karsl/labels_mirror/KARSL-502_Labels.xlsx")
METADATA = Path("data/raw/karsl/probe/metadata.json")
OUT = Path("reports/audit/karsl_counts.json")

# Chapters of the KArSL dictionary. The official site lists the chapter names;
# the SignID ranges below were read off the label sheet (where each chapter's
# words visibly start and end).
CHAPTERS = [
    (1, 31, "Numbers"),
    (32, 70, "Letters"),
    (71, 159, "Health"),
    (160, 191, "Common verbs"),
    (192, 223, "Family"),
    (224, 272, "Characteristics"),
    (273, 288, "Directions and places"),
    (289, 298, "Social relationships"),
    (299, 355, "In house"),
    (356, 458, "Religion"),
    (459, 502, "Jobs and professions"),
]


def chapter_of(sign_id):
    for first, last, name in CHAPTERS:
        if first <= sign_id <= last:
            return name
    return "?"


def main():
    labels = pd.read_excel(LABELS)
    samples = json.loads(METADATA.read_text(encoding="utf-8"))

    # videos[sign_id][signer_id] = number of samples
    videos = defaultdict(lambda: defaultdict(int))
    frames = defaultdict(list)
    for s in samples:
        videos[s["sign_id"]][s["signer_id"]] += 1
        frames[s["sign_id"]].append(s["num_frames"])

    rows = []
    for row in labels.itertuples(index=False):
        sign_id = int(row[0])
        per_signer = dict(sorted(videos[sign_id].items()))
        n_frames = sorted(frames[sign_id])
        rows.append({
            "sign_id": sign_id,
            "arabic": str(row[1]).strip(),
            "english": str(row[2]).strip(),
            "chapter": chapter_of(sign_id),
            "videos": sum(per_signer.values()),
            "signers": len(per_signer),
            "videos_per_signer": per_signer,
            "median_frames": n_frames[len(n_frames) // 2] if n_frames else 0,
        })

    summary = {
        "total_signs": len(rows),
        "total_videos": sum(r["videos"] for r in rows),
        "signer_ids": sorted({s["signer_id"] for s in samples}),
        "min_videos_per_sign": min(r["videos"] for r in rows),
        "signs_without_all_3_signers": [r["sign_id"] for r in rows if r["signers"] < 3],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "signs": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
