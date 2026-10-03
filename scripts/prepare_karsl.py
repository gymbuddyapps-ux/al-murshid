"""Unpack only the chosen signs from the big KArSL zip.

The zip holds every sign as folders of JPG frames:
    <signer>/<signer>/<train|test>/<sign id>/<sample name>/<sample name>_<frame number>.jpg

This script copies just the signs listed in config/vocab.json to
    data/raw/karsl/frames/<sign id>/<sample name>/*.jpg
and writes reports/audit/karsl_downloaded_counts.json (real counts from the files).

The zip is read from the start, one file after another, so the script also works
while the 25 GB download is still running. It remembers how far it got
(data/raw/karsl/prepare_state.json); run it again later to continue from there.

Usage:  python scripts/prepare_karsl.py
"""
import json
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jisr.vocab import load_vocab, other_sign_ids, vocab_sign_ids  # noqa: E402

ZIP_PATH = Path("data/raw/karsl/mirror_502/karsl-502.zip")
OUT_DIR = Path("data/raw/karsl/frames")
STATE = Path("data/raw/karsl/prepare_state.json")
REPORT = Path("reports/audit/karsl_downloaded_counts.json")

LOCAL_HEADER = b"PK\x03\x04"      # every file inside a zip starts with these 4 bytes


def signer_of(sample_name):
    """Sample names look like 03_01_0092_(date)_c : group, SIGNER, sign id, recording time."""
    return sample_name.split("_")[1]


def read_entry_header(f):
    """Read the small header in front of one file inside the zip.

    Returns (name, method, compressed_size) or None when there are no more files
    (or the rest has not been downloaded yet).
    """
    head = f.read(30)
    if len(head) < 30 or head[:4] != LOCAL_HEADER:
        return None
    _version, flags, method, _time, _date, _crc, csize, usize, name_len, extra_len = \
        struct.unpack("<HHHHHIIIHH", head[4:])
    name = f.read(name_len).decode("utf-8")
    extra = f.read(extra_len)
    if len(extra) < extra_len:
        return None
    if csize == 0xFFFFFFFF:            # very large file: the real size is in the extra field
        i = 0
        while i < len(extra):
            tag, size = struct.unpack("<HH", extra[i:i + 4])
            if tag == 1:
                usize, csize = struct.unpack("<QQ", extra[i + 4:i + 20])
            i += 4 + size
    if flags & 8 and csize == 0:
        raise RuntimeError("this zip stores sizes after the data; use zipfile instead")
    return name, method, csize


def main():
    vocab = load_vocab()
    wanted = {f"{i:04d}" for i in vocab_sign_ids(vocab)}
    # Also the signs of the "other sign" class (see scripts/choose_other_signs.py).
    wanted |= {f"{i:04d}" for i in other_sign_ids(vocab)}

    state =json.loads(STATE.read_text()) if STATE.exists() else {"offset": 0, "wanted": []}
    if sorted(wanted) != state["wanted"]:
        state = {"offset": 0, "wanted": sorted(wanted)}      # vocabulary changed: start again

    available = ZIP_PATH.stat().st_size
    copied = 0
    with open(ZIP_PATH, "rb") as f:
        f.seek(state["offset"])
        while True:
            start = f.tell()
            entry = read_entry_header(f)
            if entry is None:
                break
            name, method, csize = entry
            if f.tell() + csize > available:      # this file is not fully downloaded yet
                break
            parts = name.split("/")
            # The last three parts are: sign id / sample name / frame file
            if name.lower().endswith(".jpg") and len(parts) >= 3 and parts[-3] in wanted:
                data = f.read(csize)
                if method == 8:                    # 8 = "deflate" compression
                    data = zlib.decompress(data, -15)
                # Safety check: the signer in the sample name must match the zip's top folder.
                if signer_of(parts[-2]) != parts[0]:
                    state["signer_mismatches"] = state.get("signer_mismatches", 0) + 1
                target = OUT_DIR / parts[-3] / parts[-2] / parts[-1]
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    copied += 1
            else:
                f.seek(csize, 1)                   # skip a file we do not need
            state["offset"] = f.tell()
        stopped_at = start

    STATE.write_text(json.dumps(state))

    # Count what is really on disk now.
    signs = {}
    for sign_id in sorted(wanted):
        samples = [p.name for p in (OUT_DIR / sign_id).glob("*") if p.is_dir()]
        signers = sorted({signer_of(s) for s in samples})
        signs[sign_id] = {
            "samples": len(samples),
            "signers": signers,
            "samples_per_signer": {s: sum(1 for x in samples if signer_of(x) == s) for s in signers},
        }
    report = {"zip": str(ZIP_PATH), "zip_bytes_read": stopped_at, "zip_bytes_available": available,
              "signer_mismatches": state.get("signer_mismatches", 0), "signs": signs}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    total = sum(s["samples"] for s in signs.values())
    print(f"copied {copied} new frames; {total} samples of {len(wanted)} signs on disk; "
          f"read {stopped_at / 1e9:.2f} of {available / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
