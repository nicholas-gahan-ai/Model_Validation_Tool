"""Build the transparent pi icon set from assets/logo_icon_src.jpg (maroon on white, no alpha).

  python tools/make_icons.py
writes assets/logo_icon.png (512), logo_icon_256/64/32.png and logo_icon.ico (16, 24, 32, 48, 64, 256).
"""
from pathlib import Path

import numpy as np
from PIL import Image

A = Path(__file__).resolve().parent.parent / "assets"
LO, HI = 18.0, 90.0  # distance-to-white ramp (0..255 units): soft edge, no white halo


def main():
    src = np.asarray(Image.open(A / "logo_icon_src.jpg").convert("RGB")).astype(np.float32)
    dist = (255.0 - src).max(axis=2)                       # how far from white
    alpha = np.clip((dist - LO) / (HI - LO), 0, 1)
    # the logo is one flat colour: paint every pixel with the median ink colour and keep only the alpha,
    # so JPEG noise on the anti-aliased edge cannot leave a light halo on dark backgrounds
    ink = np.median(src[alpha > 0.95], axis=0)
    rgb = np.broadcast_to(ink, src.shape)
    rgba = np.dstack([rgb, alpha * 255]).astype(np.uint8)
    ys, xs = np.where(alpha > 0.05)
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = rgba.shape[:2]
    side = int(max(h, w) * 1.08)
    sq = np.zeros((side, side, 4), np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = rgba
    img = Image.fromarray(sq, "RGBA")
    img.resize((512, 512), Image.LANCZOS).save(A / "logo_icon.png")
    for s in (256, 64, 32):
        img.resize((s, s), Image.LANCZOS).save(A / f"logo_icon_{s}.png")
    img.resize((256, 256), Image.LANCZOS).save(A / "logo_icon.ico",
                                                sizes=[(s, s) for s in (16, 24, 32, 48, 64, 256)])
    print("wrote", ", ".join(p.name for p in sorted(A.glob("logo_icon*")) if p.name != "logo_icon_src.jpg"))


if __name__ == "__main__":
    main()
