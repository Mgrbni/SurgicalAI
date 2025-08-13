from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw


def make_img(path: Path, seed: int) -> None:
    w = h = 512
    img = Image.new("RGB", (w, h), (235, 235, 235))
    d = ImageDraw.Draw(img)
    r = 80 + (seed * 13) % 60
    x0 = 120 + (seed * 17) % 80
    y0 = 140 + (seed * 31) % 80
    d.ellipse((x0, y0, x0 + r, y0 + r), fill=(60, 60, 60))
    img.save(path, format="JPEG", quality=92)


def main() -> None:
    out = Path(__file__).parent
    os.makedirs(out, exist_ok=True)
    for i in range(10):
        make_img(out / f"g{i:02d}.jpg", i + 1)


if __name__ == "__main__":
    main()
