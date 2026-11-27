#!/usr/bin/env python
"""Visual check of the region operator's strength ladder (CPU or GPU, no SUT).

For each seed image: the codec reconstruction, then one panel per ladder level in which the same N regions
(fixed, spread over the image) are moved to their rank-r neighbour (r = level's rank) under the chosen KNN.
Rows: images; columns: levels. Also writes per-level LPIPS to the reconstruction and the share of dead codes.

    python scripts/analysis/ladder_demo.py --cells cells/imagenet_calib.yaml --n-img 3 --metric l2_live \
        --levels 1,2,3,4,6,8,12,16,24,32,48,64,96,128 --regions 6 --block 4 --out docs/figs/calib/ladder_imagenet.png
"""
import argparse, json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--n-img", type=int, default=3)
    ap.add_argument("--metric", default="l2_live")
    ap.add_argument("--levels", default="1,2,3,4,6,8,12,16,24,32,48,64,96,128")
    ap.add_argument("--regions", type=int, default=6)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thumb", type=int, default=160)
    a = ap.parse_args()
    from mmb.manipulator.image.manipulator import ImageConfig, ImageManipulator, apply_region_genotype
    from mmb.manipulator.image.types import PatchSelection
    from mmb.manipulator.image_factory import build_image_backend
    from mmb.validity.proxies import ProxyModels
    levels = tuple(int(x) for x in a.levels.split(","))
    man = build_image_backend(image_config=ImageConfig(region_block=a.block, region_levels=levels, knn_metric=a.metric),
                              device=a.device)
    codec, knn = man._codec, man._knn
    live = np.linalg.norm(codec.codebook, axis=1) > 1e-3
    items = yaml.safe_load(a.cells.read_text())["seeds"]["slot_items"]["items"][: a.n_img]
    models = ProxyModels(local_files_only=True)
    rows, stats = [], []
    rng = np.random.default_rng(0)
    for it in items:
        img = Image.open(Path(it["image"]).expanduser()).convert("RGB")
        grid = codec.encode(codec.preprocess(img) if hasattr(codec, "preprocess") else img)
        H, W = grid.indices.shape
        b = a.block
        anchors = np.array([(r, c) for r in range(0, H, b) for c in range(0, W, b)], dtype=np.intp)
        sel = PatchSelection(positions=anchors, candidates=tuple(np.arange(1, len(levels) + 1) for _ in anchors),
                             original_codes=np.zeros(len(anchors), dtype=np.int64))
        pick = rng.choice(len(anchors), size=min(a.regions, len(anchors)), replace=False)
        rec = codec.decode(grid)
        panels = [rec]
        for k in range(1, len(levels) + 1):
            g = np.zeros(len(anchors), dtype=np.int64); g[pick] = k
            gg = apply_region_genotype(grid, sel, g, knn, b, levels)
            im = codec.decode(gg)
            ch = gg.indices != grid.indices
            stats.append(dict(image=Path(it["image"]).name, level=k, rank=levels[k - 1],
                              lpips=float(models.lpips(rec, im)),
                              dead_share=float((~live[gg.indices[ch]]).mean()) if ch.any() else 0.0))
            panels.append(im)
        rows.append(panels)
    t = a.thumb
    canvas = Image.new("RGB", (t * (len(levels) + 1), (t + 14) * len(rows) + 14), "white")
    d = ImageDraw.Draw(canvas)
    for j, lab in enumerate(["recon"] + [f"L{k} r{levels[k-1]}" for k in range(1, len(levels) + 1)]):
        d.text((j * t + 3, 1), lab, fill="black")
    for i, panels in enumerate(rows):
        for j, im in enumerate(panels):
            canvas.paste(im.resize((t, t)), (j * t, 14 + i * (t + 14)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out)
    a.out.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    by = {}
    for s in stats:
        by.setdefault(s["level"], []).append(s["lpips"])
    print(a.metric, {k: round(float(np.mean(v)), 3) for k, v in by.items()})
    print("dead share", round(float(np.mean([s["dead_share"] for s in stats])), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
