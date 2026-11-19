"""Kampong's logo: a 3x3 grid of HDB windows with one lit, the flat Kampong checked.  python scripts/make_logo.py

Writes web/icons/logo.svg (any size), icon-192/512, the maskable icon, apple-touch-icon and favicon-32/16.
One colour family: brand blue #1a88d4 tile, windows in white at low opacity, the lit window solid white.
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "web/icons"
BLUE, LIT = (26, 136, 212), (1, 2)  # lit window: middle row, right column


def svg():
    cells = []
    for r in range(3):
        for c in range(3):
            x, y = 22 + c * 20, 22 + r * 20
            op = "1" if (r, c) == LIT else ".32"
            cells.append(f'<rect x="{x}" y="{y}" width="16" height="16" rx="2.5" fill="#fff" fill-opacity="{op}"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" role="img" aria-label="Kampong">'
            '<rect width="100" height="100" rx="22" fill="#1a88d4"/>' + "".join(cells) + "</svg>\n")


def png(size, pad=0.0, round_=True):
    """pad: share of the tile kept clear around the grid (maskable icons need a safe zone)."""
    s = size * 4  # draw large, then shrink, for smooth edges
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * .22) if round_ else 0, fill=BLUE + (255,))
    inner = s * (1 - 2 * pad)
    off = s * pad
    k = inner / 100
    for r in range(3):
        for c in range(3):
            x, y = off + (22 + c * 20) * k, off + (22 + r * 20) * k
            a = 255 if (r, c) == LIT else 82
            layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
            ImageDraw.Draw(layer).rounded_rectangle([x, y, x + 16 * k, y + 16 * k], radius=2.5 * k, fill=(255, 255, 255, a))
            im = Image.alpha_composite(im, layer)
    return im.resize((size, size), Image.LANCZOS)


def small(size):
    """16 and 32 px: chunkier windows and no rounding blur, so the grid still reads."""
    im = Image.new("RGBA", (size, size), BLUE + (255,))
    d = ImageDraw.Draw(im)
    cell, gap = (4, 1) if size == 16 else (7, 2)
    start = (size - (3 * cell + 2 * gap)) // 2
    for r in range(3):
        for c in range(3):
            x, y = start + c * (cell + gap), start + r * (cell + gap)
            d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(255, 255, 255, 255) if (r, c) == LIT else (118, 184, 229, 255))
    return im


if __name__ == "__main__":
    (OUT / "logo.svg").write_text(svg(), encoding="utf-8")
    png(192).save(OUT / "icon-192.png")
    png(512).save(OUT / "icon-512.png")
    png(512, pad=0.1, round_=False).save(OUT / "icon-maskable-512.png")
    png(180, round_=False).convert("RGB").save(OUT / "apple-touch-icon.png")
    small(32).save(OUT / "favicon-32.png")
    small(16).save(OUT / "favicon-16.png")
    print("logo written to", OUT)
