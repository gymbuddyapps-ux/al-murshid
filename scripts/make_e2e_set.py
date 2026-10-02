"""Prepare the data for the two browser tests (Phase 5). Uses held-out test-signer clips only.

1. Feature parity test  (web/e2e/parity.spec.js)
   For a few clips, save the exact pictures (as PNG, so Python and the browser see the
   same pixels), the features Python computes from them, and the model's scores.
       data/e2e/parity/<sample>/<frame>.png
       data/e2e/parity.json

2. End-to-end camera test  (web/e2e/camera.spec.js)
   Build videos that Chrome can play as a fake camera. Each video is a row of test clips
   with a short grey pause between them (the pause is the "hands down" moment).
   For every clip, save what the Python pipeline predicts for the same pictures.
       data/e2e/batches/batch_<n>.y4m
       data/e2e/e2e.json

Usage:  python scripts/make_e2e_set.py [--parity-per-class 2] [--e2e-per-class 10] [--batch-size 10]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jisr.detect import FRAME_MS, detect_clip, read_image, to_frames  # noqa: E402
from jisr.features import clip_to_features  # noqa: E402
from jisr.metrics import app_decisions, softmax, word_ranking  # noqa: E402
from jisr.model import SignCNN  # noqa: E402
from jisr.training import load_config, load_data, split_masks  # noqa: E402
from jisr.vocab import class_labels, load_vocab  # noqa: E402

FRAMES_DIR = Path("data/raw/karsl/frames")
OUT = Path("data/e2e")
SEGMENTER = json.loads(Path("config/segmenter.json").read_text(encoding="utf-8"))

# The fake camera picture: 640 x 480, with the square KArSL picture in the middle.
WIDTH, HEIGHT = 640, 480
GREY = 128
LEAD_IN_FRAMES = 50       # 2 seconds of grey before the first clip
PAUSE_FRAMES = 30         # 1.2 seconds of grey between clips


def clip_images(sign_id, sample):
    folder = FRAMES_DIR / f"{sign_id:04d}" / sample
    return [read_image(p) for p in sorted(folder.glob("*.jpg"))]


def letterbox(image):
    """Put the square picture in the middle of a 640 x 480 grey picture."""
    canvas = np.full((HEIGHT, WIDTH, 3), GREY, dtype=np.uint8)
    square = cv2.resize(image, (HEIGHT, HEIGHT), interpolation=cv2.INTER_LINEAR)
    left = (WIDTH - HEIGHT) // 2
    canvas[:, left:left + HEIGHT] = square
    return canvas


def through_video(image):
    """Convert a picture to the video's color format and back, so Python sees what the camera shows."""
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    return yuv, cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


def predict(model, images, aspect, info):
    """The whole Python pipeline for one clip: landmarks -> features -> scores -> app decision."""
    features = clip_to_features(to_frames(*detect_clip(images)), aspect, SEGMENTER)
    if features is None:
        return None, None, {"kind": "unclear", "top3": []}
    with torch.no_grad():
        scores = model(torch.from_numpy(features[None])).numpy()
    probabilities = softmax(scores, info["temperature"])
    accepted, _, _ = app_decisions(probabilities, info["other_index"], info["threshold"])
    ranking = word_ranking(scores, info["other_index"])[0][:3]
    decision = {"kind": "ok" if accepted[0] else "unclear",
                "top3": [info["labels"][i] for i in ranking]}
    return features, scores[0], decision


def pick(data, mask, per_class, skip=0):
    """Pick clips evenly: for each class, `per_class` clips after skipping the first `skip`."""
    chosen = []
    for class_id in sorted(set(data["y"][mask])):
        rows = np.flatnonzero(mask & (data["y"] == class_id))
        rows = rows[np.argsort(data["sample"][rows])]
        chosen.extend(rows[skip:skip + per_class])
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-per-class", type=int, default=2)
    parser.add_argument("--e2e-per-class", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    config = load_config()
    vocab = load_vocab()
    labels = class_labels(vocab)
    info = json.loads(Path("models/labels.json").read_text(encoding="utf-8"))
    model = SignCNN(len(labels), hidden=tuple(config["hidden"]), dropout=config["dropout"])
    model.load_state_dict(torch.load("models/jisr_model.pt"))
    model.eval()

    data = load_data()
    _, test_mask = split_masks(data, config["dev_signers"], config["test_signer"], "test")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "batches").mkdir(parents=True)

    # ---- 1. Feature parity cases ----
    parity = []
    for row in pick(data, test_mask, args.parity_per_class):
        sample, sign_id = str(data["sample"][row]), int(data["sign_id"][row])
        images = clip_images(sign_id, sample)
        folder = OUT / "parity" / sample
        folder.mkdir(parents=True)
        for i, image in enumerate(images):
            cv2.imencode(".png", image)[1].tofile(str(folder / f"{i:04d}.png"))
        height, width = images[0].shape[:2]
        features, scores, decision = predict(model, images, width / height, info)
        # The stored training features must be exactly what we just recomputed.
        assert np.allclose(features, data["X"][row], atol=1e-6), sample
        parity.append({
            "sample": sample, "class_id": int(data["y"][row]), "frames": len(images),
            "frame_ms": FRAME_MS, "folder": str(folder.resolve()),
            "features": np.round(features.reshape(-1).astype(float), 6).tolist(),
            "scores": np.round(scores.astype(float), 5).tolist(),
            "python": decision,
        })
    (OUT / "parity.json").write_text(json.dumps(parity, ensure_ascii=False), encoding="utf-8")
    print(f"parity: {len(parity)} clips")

    # ---- 2. Fake-camera videos ----
    grey_yuv, _ = through_video(np.full((HEIGHT, WIDTH, 3), GREY, dtype=np.uint8))
    rows = pick(data, test_mask, args.e2e_per_class, skip=args.parity_per_class)
    rng = np.random.default_rng(0)
    rng.shuffle(rows)                       # mix the classes inside each video

    batches = []
    for b in range(0, len(rows), args.batch_size):
        path = OUT / "batches" / f"batch_{b // args.batch_size:03d}.y4m"
        clips = []
        with open(path, "wb") as video:
            video.write(f"YUV4MPEG2 W{WIDTH} H{HEIGHT} F{1000 // FRAME_MS}:1 Ip A1:1 C420jpeg\n".encode())
            frame_number = 0

            def write(yuv):
                nonlocal frame_number
                video.write(b"FRAME\n" + yuv.tobytes())
                frame_number += 1

            for _ in range(LEAD_IN_FRAMES):
                write(grey_yuv)
            for row in rows[b:b + args.batch_size]:
                sample, sign_id = str(data["sample"][row]), int(data["sign_id"][row])
                start = frame_number
                seen_by_python = []
                for image in clip_images(sign_id, sample):
                    yuv, back = through_video(letterbox(image))
                    write(yuv)
                    seen_by_python.append(back)
                _, _, decision = predict(model, seen_by_python, WIDTH / HEIGHT, info)
                clips.append({
                    "sample": sample, "class_id": int(data["y"][row]),
                    "true_label": labels[int(data["y"][row])],
                    "start_ms": start * FRAME_MS, "end_ms": frame_number * FRAME_MS,
                    "python": decision,
                })
                for _ in range(PAUSE_FRAMES):
                    write(grey_yuv)
            for _ in range(LEAD_IN_FRAMES):
                write(grey_yuv)
        batches.append({"video": str(path.resolve()), "duration_ms": frame_number * FRAME_MS, "clips": clips})
        print(f"{path.name}: {len(clips)} clips, {frame_number * FRAME_MS / 1000:.0f} s", flush=True)

    (OUT / "e2e.json").write_text(json.dumps(batches, ensure_ascii=False), encoding="utf-8")
    print(f"e2e: {sum(len(b['clips']) for b in batches)} clips in {len(batches)} videos")


if __name__ == "__main__":
    main()
