"""Cost of a crossing: how cheap is the cheapest boundary input a run found?

The first crossing of a random search is cheap to find but not cheap to apply (many edits). The
multi-objective arms optimise exactly that second quantity, so the comparison needs outcomes beyond
"crossed within b evaluations". All of them are computed from trace.parquet / stats.json, per run.

Outcomes (a crossing evaluation = strict sign change against the anchor, as in collect.py):

edit counts (available for every arm; a gene is an edit when it is non-zero, image genes first, then
text genes, split by stats image_dim / text_dim):
  min_edits_cross     fewest active genes (image + text) among all crossing evaluations;
                      censored at image_dim + text_dim when the run never crossed
  edits_img_at_min    image / text genes of that cheapest crossing (NaN when never crossed)
  edits_txt_at_min
  hv_edits_cross      hypervolume (fraction of the reference box, in [0, 1]) of the crossing evaluations in
                      (image edits, text edits), both minimised, reference point (image_dim + 1, text_dim + 1);
                      0 when never crossed
  min_txt_chardist_cross  smallest character edit distance between decoded text and the seed question
                      among crossing evaluations (NaN when never crossed or no text)

magnitudes (only when the trace logs fitness_MatrixDistance_* / fitness_TextDist; NaN otherwise, e.g. for
single-objective arms that do not log them):
  best_img_mag_cross / best_txt_mag_cross   smallest image RMS / text cosine distance among crossings
  hv_mag_cross        hypervolume fraction over (image RMS, text cosine), reference point
                      (IMG_MAG_REF, TXT_MAG_REF) fixed in this file, never data-dependent

All reference points are constants so values are comparable across runs, arms and SUTs.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

IMG_MAG_REF = 0.1      # image RMS; observed maximum in smoke traces is about 0.073
TXT_MAG_REF = 1.0      # cosine distance between sentence embeddings, at most 1 for near-orthogonal

NAN = float("nan")


def hypervolume_2d(points: np.ndarray, ref: tuple[float, float]) -> float:
    """Area dominated by *points* (both coordinates minimised) inside the box [0, ref]; NaNs and points
    outside the box are ignored. Exact sweep, O(n log n)."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    p = p[~np.isnan(p).any(axis=1)]
    p = p[(p[:, 0] < ref[0]) & (p[:, 1] < ref[1])]
    if not len(p):
        return 0.0
    p = p[np.argsort(p[:, 0], kind="stable")]
    area, best_y = 0.0, ref[1]
    for i in range(len(p)):
        best_y = min(best_y, p[i, 1])
        x_next = p[i + 1, 0] if i + 1 < len(p) else ref[0]
        area += (x_next - p[i, 0]) * (ref[1] - best_y)
    return float(area)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cost_of_crossing(
    crossing: np.ndarray,
    genotypes: Sequence[Sequence[int]],
    image_dim: int,
    text_dim: int,
    *,
    img_mag: np.ndarray | None = None,
    txt_mag: np.ndarray | None = None,
    texts: Sequence[str] | None = None,
    seed_text: str | None = None,
) -> dict[str, Any]:
    """Cost-of-crossing columns for one run; see the module docstring."""
    crossing = np.asarray(crossing, dtype=bool)
    n_max = int(image_dim + text_dim)
    out: dict[str, Any] = {
        "min_edits_cross": float(n_max), "edits_img_at_min": NAN, "edits_txt_at_min": NAN,
        "hv_edits_cross": 0.0, "min_txt_chardist_cross": NAN,
        "best_img_mag_cross": NAN, "best_txt_mag_cross": NAN, "hv_mag_cross": NAN,
    }
    if not len(crossing):
        return out
    has_img_mag = img_mag is not None and not np.all(np.isnan(np.asarray(img_mag, float)))
    has_txt_mag = txt_mag is not None and not np.all(np.isnan(np.asarray(txt_mag, float)))
    if has_img_mag and has_txt_mag:
        out["hv_mag_cross"] = 0.0
    if not crossing.any():
        return out
    idx = np.flatnonzero(crossing)
    gt = np.vstack([np.asarray(genotypes[i]) for i in idx])
    e_img = np.count_nonzero(gt[:, :image_dim], axis=1)
    e_txt = np.count_nonzero(gt[:, image_dim:image_dim + text_dim], axis=1)
    tot = e_img + e_txt
    k = int(np.argmin(tot))                        # first of the cheapest crossings
    out["min_edits_cross"] = float(tot[k])
    out["edits_img_at_min"] = float(e_img[k])
    out["edits_txt_at_min"] = float(e_txt[k])
    box = (float(image_dim) + 1.0, float(text_dim) + 1.0)          # a crossing with 0 edits dominates all of it
    pts = np.column_stack([e_img, e_txt]).astype(float)
    out["hv_edits_cross"] = hypervolume_2d(pts, box) / (box[0] * box[1])
    if has_img_mag:
        out["best_img_mag_cross"] = float(np.nanmin(np.asarray(img_mag, float)[idx]))
    if has_txt_mag:
        out["best_txt_mag_cross"] = float(np.nanmin(np.asarray(txt_mag, float)[idx]))
    if has_img_mag and has_txt_mag:
        m = np.column_stack([np.asarray(img_mag, float)[idx], np.asarray(txt_mag, float)[idx]])
        out["hv_mag_cross"] = hypervolume_2d(m, (IMG_MAG_REF, TXT_MAG_REF)) / (IMG_MAG_REF * TXT_MAG_REF)
    if texts is not None and seed_text:
        out["min_txt_chardist_cross"] = float(min(levenshtein(str(texts[i]), seed_text) for i in idx))
    return out
