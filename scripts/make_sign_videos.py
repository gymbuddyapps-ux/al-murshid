"""Make the small videos for the app's learning page (one per word and signer).

Input:  data/raw/karsl/frames/<sign id>/<sample>/*.jpg   (KArSL frames)
Output: web/public/signs/<class id>_<signer>.mp4          (short, silent, looping in the page)
        web/public/signs/signs.json                        (what the learning page shows)

Needs ffmpeg on the PATH. Each video is the longest clip of the word by that signer,
played at 12.5 frames per second (half speed, easier to copy) at 256 x 256 pixels.

Usage:  python scripts/make_sign_videos.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jisr.vocab import class_sign_ids, load_vocab  # noqa: E402

FRAMES = Path("data/raw/karsl/frames")
OUT = Path("web/public/signs")
SIGNERS = ["01", "02"]          # two different people per word (the model knows both well)


def longest_clip(sign_id, signer):
    clips = [p for p in (FRAMES / f"{sign_id:04d}").iterdir() if p.name.split("_")[1] == signer]
    return max(clips, key=lambda p: len(list(p.glob("*.jpg"))))


def make_video(clip_dir, target):
    """Copy the frames to numbered files (ffmpeg wants that) and encode them."""
    frames = sorted(clip_dir.glob("*.jpg"))
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            shutil.copy(frame, Path(tmp) / f"{i:04d}.jpg")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", "12.5", "-i", str(Path(tmp) / "%04d.jpg"),
            "-vf", "scale=256:256", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-movflags", "+faststart",
            str(target),
        ], check=True)
    return len(frames)


def main():
    vocab = load_vocab()
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for c in vocab["classes"]:
        sign_id = class_sign_ids(c)[0]
        videos = []
        for signer in SIGNERS:
            name = f"{c['id']:02d}_{signer}.mp4"
            frames = make_video(longest_clip(sign_id, signer), OUT / name)
            videos.append({"file": name, "signer": int(signer), "frames": frames})
        entries.append({"id": c["id"], "arabic": c["arabic"], "english": c["english"],
                        "karsl_sign_ids": class_sign_ids(c), "videos": videos})
        print(c["arabic"], [v["frames"] for v in videos], flush=True)

    (OUT / "signs.json").write_text(json.dumps({
        "source": "KArSL: Arabic Sign Language Database (Sidig, Luqman, Mahmoud, Mohandes, 2021)",
        "signs": entries,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.glob("*.mp4"))
    print(f"{len(entries)} words, {2 * len(entries)} videos, {total / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
