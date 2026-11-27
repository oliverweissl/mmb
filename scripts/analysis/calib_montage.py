#!/usr/bin/env python
"""Montage of calibration minimal crossings (image-only): codec reconstruction | minimal crossing, per run.
    python scripts/analysis/calib_montage.py r1 --out docs/figs/calib/r1_image_crossings.png [--arm a7_image_only]"""
import argparse, glob, json
from pathlib import Path
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("round"); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--arm", default="a7_image_only"); ap.add_argument("--max", type=int, default=12); ap.add_argument("--thumb", type=int, default=200)
a = ap.parse_args()
runs = sorted(glob.glob(f"runs/CALIB/{a.round}/*/*/{a.arm}/s0/*/min_cross.png"))[: a.max]
t = a.thumb
cv = Image.new("RGB", (4 * t + 30, ((len(runs) + 1) // 2) * (t + 14)), "white"); d = ImageDraw.Draw(cv)
for i, p in enumerate(runs):
    rd = Path(p).parent
    m = json.loads((rd / "min_cross.json").read_text())
    x, y = (i % 2) * (2 * t + 30), (i // 2) * (t + 14)
    cv.paste(Image.open(rd / "anchor_recon.png").convert("RGB").resize((t, t)), (x, y + 14))
    cv.paste(Image.open(p).convert("RGB").resize((t, t)), (x + t, y + 14))
    d.text((x + 2, y), f"{rd.parts[3]}/{rd.parts[4]} {m.get('n_edits')} edits", fill="black")
a.out.parent.mkdir(parents=True, exist_ok=True); cv.save(a.out); print(a.out, len(runs))
