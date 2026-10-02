"""List every file of a Kaggle dataset and count videos per class folder.

Usage:  python scripts/audit/kaggle_list_classes.py owner/dataset-slug

Only the file LIST is requested (no video is downloaded).
The result is saved to reports/audit/<slug>_files.json so that
docs/vocab_audit.md can be written from a real script output.

Credentials are read by the kaggle library itself from ~/.kaggle
(never printed, never stored in this project).
"""
import json
import sys
from collections import Counter
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_all_files(api, dataset):
    """Ask Kaggle for the file list page by page until there are no more pages."""
    files = []
    token = None
    while True:
        page = api.dataset_list_files(dataset, page_token=token, page_size=200)
        files.extend(page.files)
        token = page.next_page_token
        if not token:
            return files


def main():
    dataset = sys.argv[1]
    api = KaggleApi()
    api.authenticate()

    files = list_all_files(api, dataset)

    # The class name is the folder that directly contains the video.
    videos_per_folder = Counter()
    other_files = []
    total_bytes = 0
    for f in files:
        path = Path(f.name)
        total_bytes += f.total_bytes or 0
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            videos_per_folder[path.parent.as_posix()] += 1
        else:
            other_files.append(f.name)

    result = {
        "dataset": dataset,
        "total_files": len(files),
        "total_bytes": total_bytes,
        "videos_per_folder": dict(sorted(videos_per_folder.items())),
        "non_video_files": other_files[:200],
        "all_file_names": [f.name for f in files],
    }
    out = Path("reports/audit") / (dataset.replace("/", "__") + "_files.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{dataset}: {len(files)} files, {total_bytes / 1e9:.2f} GB, "
          f"{len(videos_per_folder)} folders with videos -> {out}")


if __name__ == "__main__":
    main()
