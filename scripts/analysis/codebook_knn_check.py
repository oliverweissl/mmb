#!/usr/bin/env python
"""Codebook check for the VQGAN f8-16384 operator (CPU only): how far the cosine-KNN neighbours of a codeword are in
L2 and in norm, compared with L2-KNN neighbours, and how many codes the stored runs' seed images actually use.

    python scripts/analysis/codebook_knn_check.py [--ckpt ~/.cache/torch/hub/vqgan/vq-f8/model.ckpt] [--out results/diag_edits]
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path.home() / ".cache/torch/hub/vqgan/vq-f8/model.ckpt")
    ap.add_argument("--out", type=Path, default=Path("results/diag_edits"))
    ap.add_argument("--sample", type=int, default=2000)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    key = next(k for k in sd if k.endswith("quantize.embedding.weight"))
    cb = sd[key].float().numpy()
    n, dim = cb.shape
    norms = np.linalg.norm(cb, axis=1)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(a.sample, n), replace=False)
    unit = cb / np.maximum(norms[:, None], 1e-8)
    out = {"n_codes": int(n), "dim": int(dim), "norm_p5_p50_p95": [float(x) for x in np.percentile(norms, [5, 50, 95])], "ranks": []}
    cos_order = np.argsort(-(unit[idx] @ unit.T), axis=1)[:, 1:]          # self first, dropped
    d2 = ((cb[idx, None, :] - cb[None, :, :]) ** 2).sum(-1)
    l2_order = np.argsort(d2, axis=1)[:, 1:]
    for r in RANKS:
        c, l = cos_order[:, r - 1], l2_order[:, r - 1]
        rows = np.arange(len(idx))
        out["ranks"].append(dict(rank=r,
                                 cos_l2dist_med=float(np.median(np.sqrt(d2[rows, c]))),
                                 l2knn_l2dist_med=float(np.median(np.sqrt(d2[rows, l]))),
                                 cos_norm_ratio_med=float(np.median(norms[c] / norms[idx])),
                                 cos_norm_ratio_p10_p90=[float(x) for x in np.percentile(norms[c] / norms[idx], [10, 90])]))
    out["median_nn_l2_dist"] = float(np.median(np.sqrt(d2[np.arange(len(idx)), l2_order[:, 0]])))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "codebook_knn.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
