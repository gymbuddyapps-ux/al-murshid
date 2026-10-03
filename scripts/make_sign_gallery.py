"""Make a local page to learn the 20 signs: one looping animation per word and signer,
built from the KArSL frames on disk (dataset material, not committed to git).

Output: data/sign_gallery/index.html  (open it in a browser on the laptop)

Usage:  python scripts/make_sign_gallery.py
"""
import html
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jisr.vocab import load_vocab  # noqa: E402

FRAMES = Path("data/raw/karsl/frames")
OUT = Path("data/sign_gallery")
SIGNERS = ["01", "03"]       # two different people per word
SIZE = 320                   # pixels; frames are 256, scaled up a little


def clip_for(sign_id, signer):
    """The longest clip of this sign by this signer (longer clips show the movement best)."""
    clips = [p for p in (FRAMES / f"{sign_id:04d}").iterdir() if p.name.split("_")[1] == signer]
    return max(clips, key=lambda p: len(list(p.glob("*.jpg"))))


def make_gif(clip_dir, target):
    frames = [Image.open(p).convert("RGB").resize((SIZE, SIZE)) for p in sorted(clip_dir.glob("*.jpg"))]
    frames += [frames[-1]] * 12           # hold the last frame for half a second
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=40, loop=0)
    return len(frames) - 12


def main():
    vocab = load_vocab()
    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for c in vocab["classes"]:
        gifs = []
        for signer in SIGNERS:
            clip = clip_for(c["karsl_sign_id"], signer)
            name = f"{c['id']:02d}_{signer}.gif"
            n = make_gif(clip, OUT / name)
            gifs.append(f'<figure><img src="{name}" alt=""><figcaption>الموقّع {int(signer)} · {n} إطاراً</figcaption></figure>')
        cards.append(f"""<section class="card">
  <h2>{html.escape(c['arabic'])} <small>{html.escape(c['english'])}</small></h2>
  <div class="row">{''.join(gifs)}</div>
</section>""")

    page = f"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>تعلّم إشارات جسر</title>
<style>
body{{font-family:system-ui,sans-serif;background:#fafaf8;color:#2b211c;margin:0;padding:1rem}}
h1{{color:#8e1b1f}} p{{max-width:60ch}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(680px,1fr));gap:1rem}}
.card{{background:#fff;border:1px solid #e3e1dc;border-radius:12px;padding:1rem}}
.card h2{{margin:0 0 .5rem;font-size:1.75rem}} .card small{{color:#6a5d55;font-size:1rem;margin-inline-start:.5rem}}
.row{{display:flex;gap:1rem;flex-wrap:wrap}} figure{{margin:0}} img{{width:{SIZE}px;height:{SIZE}px;border-radius:8px;display:block}}
figcaption{{color:#6a5d55;font-size:.9rem;margin-top:.25rem}}
</style></head><body>
<h1>تعلّم إشارات جسر</h1>
<p>كل كلمة معروضة مرتين من مقاطع KArSL الأصلية بموقّعَين مختلفَين. قلّد الحركة أمام كاميرا الهاتف: ارفع يدك، أدِّ الإشارة، ثم أنزل يدك.
ملاحظة: الصور قد تظهر معكوسة بالنسبة إليك، فاستخدم اليد التي تبدو طبيعية لك؛ النموذج تدرّب على الاتجاهين.</p>
<p>المادة من قاعدة بيانات KArSL (Sidig, Luqman, Mahmoud, Mohandes, 2021) للاستخدام المحلي في الاختبار فقط.</p>
<div class="grid">{''.join(cards)}</div></body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} with {len(cards)} words")


if __name__ == "__main__":
    main()
