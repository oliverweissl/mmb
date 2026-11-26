#!/usr/bin/env python
"""CPU demo of the dead-codeword effect of the region operator on one stored run.

Re-renders the run's minimal crossing (min_cross.json genotype, image genes) from the seed image with the operator as
run (cosine KNN over all 16,384 codewords) and with a candidate fix (L2 KNN over the live codewords only), counts the
replacement codes that are dead (norm <= 1e-3), and writes a side-by-side PNG:
codec reconstruction | stored min_cross.png | re-rendered as run | re-rendered with live-L2 KNN.

    python scripts/analysis/dead_code_demo.py <run_dir> --out docs/figs/dead_code_demo.png
"""
import argparse, json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from mmb.manipulator.image.manipulator import apply_region_genotype


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from mmb.runners.run_boundary_test import load_config
    from mmb.manipulator.image_factory import build_image_backend
    st = json.loads((a.run / "stats.json").read_text())
    m = st["run_meta"]
    cfgp = next(Path("configs/generated/MAIN").glob(f"*__s{m['rng_seed']}__{m['task']}__{m['sut']}__{m['arm']}.yaml"))
    cfg = yaml.safe_load(cfgp.read_text()); cfg["device"] = "cpu"
    exp = load_config(cfg)
    man = build_image_backend(image_config=exp.image, device="cpu")
    codec, knn = man._codec, man._knn
    cb = codec.codebook
    live = np.linalg.norm(cb, axis=1) > 1e-3
    L = np.nonzero(live)[0]
    d2 = ((cb[:, None, :] - cb[None, L, :]) ** 2).sum(-1)          # every code -> live codes by L2
    order = L[np.argsort(d2, axis=1)]
    w = len(L) - 1                                                   # ranks above the live count clip to the farthest live code
    pad = np.stack([o[o != i][:w] for i, o in enumerate(order)]).astype(np.int64)
    img = Image.open(a.run / "origin.png").convert("RGB")
    grid = codec.encode(codec.preprocess(img) if hasattr(codec, "preprocess") else img)
    mc = json.loads((a.run / "min_cross.json").read_text())
    g = np.asarray(mc["genotype"], dtype=np.int64)[:64]
    b = exp.image.region_block
    H, W = grid.indices.shape
    from mmb.manipulator.image.types import PatchSelection
    anchors = np.array([(r, c) for r in range(0, H, b) for c in range(0, W, b)], dtype=np.intp)
    sel = PatchSelection(positions=anchors, candidates=tuple(np.arange(1, 15) for _ in anchors), original_codes=np.zeros(len(anchors), dtype=np.int64))
    levels = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
    g_run = apply_region_genotype(grid, sel, g, knn, b, levels)
    g_fix = apply_region_genotype(grid, sel, g, pad, b, levels)
    changed = g_run.indices != grid.indices
    dead_run = float((~live[g_run.indices[changed]]).mean()) if changed.any() else 0.0
    ims = [codec.decode(grid), Image.open(a.run / "min_cross.png").convert("RGB"), codec.decode(g_run), codec.decode(g_fix)]
    ims = [im.resize((256, 256)) for im in ims]
    canvas = Image.new("RGB", (256 * 4 + 30, 256), "white")
    for i, im in enumerate(ims):
        canvas.paste(im, (i * 266, 0))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out)
    diff = float(np.abs(np.asarray(ims[1], float) - np.asarray(ims[2], float)).mean())
    res = {"run": str(a.run), "levels": [int(x) for x in g[g > 0]], "n_regions": int((g > 0).sum()),
           "codes_changed": int(changed.sum()), "dead_share_of_replacements": dead_run,
           "rerender_vs_stored_mean_abs_diff": diff}
    print(json.dumps(res, indent=1))
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
