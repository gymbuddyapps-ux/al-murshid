"""Draw the app icons (PNG) from the same bridge shape as web/public/icons/logo.svg.

Usage:  python scripts/make_icons.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path("web/public/icons")
RED = (142, 27, 31)       # #8E1B1F
WHITE = (255, 255, 255)


def bridge_points(steps=40):
    """The arch of the logo: the curve 'M12 50 C12 28 52 28 52 50' on a 64x64 grid."""
    p0, p1, p2, p3 = (12, 50), (12, 28), (52, 28), (52, 50)
    points = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        points.append((x, y))
    return points


def draw_icon(size, logo_fraction):
    """White bridge on a red square. `logo_fraction` is how much of the icon the logo fills."""
    image = Image.new("RGB", (size, size), RED)
    draw = ImageDraw.Draw(image)
    scale = size * logo_fraction / 64
    offset = (size - 64 * scale) / 2

    def at(point):
        return (offset + point[0] * scale, offset + point[1] * scale)

    width = max(2, round(5 * scale))
    lines = [
        [(6, 22), (58, 22)],          # deck
        [(12, 22), (12, 50)],         # left pier
        [(52, 22), (52, 50)],         # right pier
        bridge_points(),              # arch
    ]
    for line in lines:
        draw.line([at(p) for p in line], fill=WHITE, width=width, joint="curve")
        for p in (line[0], line[-1]):                      # round line ends
            x, y = at(p)
            draw.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=WHITE)
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
