"""Draw the app icons (PNG) from the same star shape as web/public/icons/logo.svg.

Usage:  python scripts/make_icons.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path("web/public/icons")
RED = (200, 16, 46)       # #c8102e, Omani red
WHITE = (255, 255, 255)


def draw_icon(size, logo_fraction):
    """White eight-pointed star on a red square. `logo_fraction` is how much of the icon the logo fills."""
    image = Image.new("RGB", (size, size), RED)
    draw = ImageDraw.Draw(image)
    scale = size * logo_fraction / 64
    offset = (size - 64 * scale) / 2

    def at(point):
        return (offset + point[0] * scale, offset + point[1] * scale)

    width = max(2, round(4 * scale))
    star = [(32, 6), (37, 27), (58, 32), (37, 37), (32, 58), (27, 37), (6, 32), (27, 27), (32, 6)]
    draw.line([at(p) for p in star], fill=WHITE, width=width, joint="curve")
    for a, b in [((14, 14), (29, 29)), ((50, 14), (35, 29)), ((50, 50), (35, 35)), ((14, 50), (29, 35))]:
        draw.line([at(a), at(b)], fill=WHITE, width=width)
    cx, cy = at((32, 32))
    r = 3.5 * scale
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=WHITE)
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw_icon(192, 0.8).save(OUT / "icon-192.png")
    draw_icon(512, 0.8).save(OUT / "icon-512.png")
    # "Maskable" icons may be cropped to a circle by Android, so the logo is smaller.
    draw_icon(512, 0.55).save(OUT / "icon-maskable-512.png")
    print("icons written to", OUT)


if __name__ == "__main__":
    main()
