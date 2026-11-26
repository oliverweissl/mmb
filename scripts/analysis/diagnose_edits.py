#!/usr/bin/env python
"""Diagnose the image edits and the channel balance of minimal crossings on STORED runs (CPU only, no model calls).

    python scripts/analysis/diagnose_edits.py runs/MAIN --results results/MAIN.csv --validity results/validity_MAIN.csv \
        --out results/diag_edits [--arms a0_directed,a1_random_tiered,a5_mimicry_objective,a9_deepjanus] [--seed 0]

A) per edited region of each confirmed minimal crossing (min_cross.json genotype; image genes first, 64 regions of
   4x4 codes = 32x32 px, row-major): level (KNN rank = 2^(level-1)), colour shift of the region mean in CIELAB
   (dE76), chroma change, sharpness ratio (variance of the Laplacian of L*, edited / codec reconstruction) and an
   olive/green flag (edited hue 70-140 deg with chroma < 45 while the reconstruction's hue is outside that band).
   Reference = anchor_recon.png (codec reconstruction, as in the validity proxies).
   Trace view: per run, does ANY recorded crossing (batched scores, unconfirmed) use only levels <= L, and with how
   few edits.
B) image vs text edits of the minimal crossings and of generation 0.
C) curves over seed inputs: share with a confirmed minimal crossing of <= k edits, <= LPIPS x, <= k edits with all
   levels <= L; per-seed-input dominance of MMBA against each method.
Writes <out>/regions.csv, runs.csv, trace_levels.csv, curves.csv, dominance.csv and summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from scipy.ndimage import laplace
from skimage.color import rgb2lab

N_IMG, BLOCK, PX = 64, 4, 8          # 64 region genes, 4x4 codes per region, 8 px per code (f8)
LEVEL_CAPS = (4, 6, 8, 10)           # KNN rank <= 8, 32, 128, 512


def _lab(p: Path) -> np.ndarray:
    return rgb2lab(np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0)


def _hue_chroma(lab_mean):
    a, b = lab_mean[1], lab_mean[2]
    return (np.degrees(np.arctan2(b, a)) % 360.0), float(np.hypot(a, b))


def region_rows(run: Path, geno: np.ndarray, ctx: dict) -> list[dict]:
    rec, edi = _lab(run / "anchor_recon.png"), _lab(run / "min_cross.png")
    pos = ctx["image_patch_positions"]
    rows = []
    for i in np.nonzero(geno[:N_IMG])[0]:
        r0, c0 = pos[i]
        sl = (slice(r0 * PX, (r0 + BLOCK) * PX), slice(c0 * PX, (c0 + BLOCK) * PX))
        m0, m1 = rec[sl].reshape(-1, 3).mean(0), edi[sl].reshape(-1, 3).mean(0)
        h0, c0_ = _hue_chroma(m0)
        h1, c1 = _hue_chroma(m1)
        v0, v1 = laplace(rec[sl][..., 0]).var(), laplace(edi[sl][..., 0]).var()
        olive = (70 <= h1 <= 140) and c1 < 45 and not (70 <= h0 <= 140)
        rows.append(dict(region=int(i), level=int(geno[i]), rank=int(2 ** (int(geno[i]) - 1)),
                         dE=float(np.linalg.norm(m1 - m0)), dChroma=c1 - c0_, sharp_ratio=float(v1 / max(v0, 1e-6)),
                         olive=bool(olive), L0=float(m0[0]), L1=float(m1[0])))
    return rows


def trace_levels(run: Path, stats: dict) -> dict:
    t = pq.read_table(run / "trace.parquet", columns=["genotype", "logprobs"]).to_pandas()
    ia, ib = stats["target_classes"]
    s0 = np.sign(stats["anchor_g"]) or 1.0
    G = np.stack(t["genotype"].to_numpy())
    g = np.array([lp[ia] - lp[ib] for lp in t["logprobs"]])
    cross = s0 * g < 0
    out = {"n_trace_cross": int(cross.sum())}
    lv = G[:, :N_IMG].max(axis=1)
    ne = (G != 0).sum(axis=1)
    for L in LEVEL_CAPS:
        ok = cross & (lv <= L)
        out[f"any_le{L}"] = bool(ok.any())
        out[f"min_edits_le{L}"] = int(ne[ok].min()) if ok.any() else np.nan
    out["min_edits_any"] = int(ne[cross].min()) if cross.any() else np.nan
    # generation 0 (first pop rows): image vs text active genes
    g0 = G[:30]
    out["gen0_img_active"] = float((g0[:, :N_IMG] != 0).sum(1).mean())
    out["gen0_txt_active"] = float((g0[:, N_IMG:] != 0).sum(1).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--validity", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="a0_directed,a1_random_tiered,a5_mimicry_objective,a9_deepjanus")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    arms = a.arms.split(",")
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.results)
    df = df[df["arm"].isin(arms) & (df["rng_seed"] == a.seed) & df["sut"].isin(["qwen", "internvl", "ministral"])]
    v = pd.read_csv(a.validity)
    vmin = v[v["view"] == "min"].drop_duplicates("run_dir").set_index("run_dir")
    regs, runs, trs = [], [], []
    for _, r in df.iterrows():
        run = Path(r["run_dir"])
        stats = json.loads((run / "stats.json").read_text())
        ctx = json.loads((run / "context.json").read_text())
        base = dict(arm=r["arm"], task=r["task"], sut=r["sut"], cell_id=r["cell_id"], run_dir=str(run))
        tr = trace_levels(run, stats)
        trs.append({**base, **tr})
        mc = run / "min_cross.json"
        crossed = str(r["crossed"]).lower() in ("true", "1")
        if not (mc.exists() and crossed):
            runs.append({**base, "has_min": False})
            continue
        m = json.loads(mc.read_text())
        geno = np.asarray(m["genotype"], dtype=np.int64)
        rr = region_rows(run, geno, ctx)
        for x in rr:
            regs.append({**base, **x})
        lv = [x["level"] for x in rr]
        vm = vmin.loc[str(run)] if str(run) in vmin.index else None
        runs.append({**base, "has_min": True, "n_edits": int((geno != 0).sum()), "n_img": int((geno[:N_IMG] != 0).sum()),
                     "n_txt": int((geno[N_IMG:] != 0).sum()), "max_level": max(lv) if lv else 0,
                     "mean_level": float(np.mean(lv)) if lv else np.nan,
                     "lpips": float(vm["img_lpips"]) if vm is not None else np.nan,
                     "vis_char": float(vm["vis_char_dist"]) if vm is not None else np.nan,
                     "n_off_dE15": int(sum(x["dE"] > 15 for x in rr)), "n_olive": int(sum(x["olive"] for x in rr))})
    R, U, T = pd.DataFrame(regs), pd.DataFrame(runs), pd.DataFrame(trs)
    R.to_csv(a.out / "regions.csv", index=False); U.to_csv(a.out / "runs.csv", index=False); T.to_csv(a.out / "trace_levels.csv", index=False)

    # C) curves over seed inputs (each run = one seed input x model x task; no crossing = never counted)
    U2 = U.copy()
    curves = []
    for arm, g in U2.groupby("arm"):
        n = len(g)
        h = g[g["has_min"]]
        for k in range(1, 41):
            curves.append(dict(arm=arm, view="edits_le_k", x=k, share=float((h["n_edits"] <= k).sum() / n)))
            for L in (6, 8):
                curves.append(dict(arm=arm, view=f"edits_le_k_levels_le{L}", x=k,
                                   share=float(((h["n_edits"] <= k) & (h["max_level"] <= L)).sum() / n)))
        for x in np.round(np.arange(0.02, 0.62, 0.02), 2):
            curves.append(dict(arm=arm, view="lpips_le_x", x=float(x), share=float((h["lpips"] <= x).sum() / n)))
    C = pd.DataFrame(curves); C.to_csv(a.out / "curves.csv", index=False)

    # dominance per seed input (task, sut, cell): MMBA vs each other arm
    key = ["task", "sut", "cell_id"]
    ref = U2[U2["arm"] == "a0_directed"].set_index(key)
    dom = []
    for arm in [x for x in arms if x != "a0_directed"]:
        o = U2[U2["arm"] == arm].set_index(key)
        j = ref.join(o, lsuffix="_ref", rsuffix="_arm", how="inner")
        for metric in ("n_edits", "lpips"):
            res = {"better": 0, "equal": 0, "worse": 0, "neither": 0}
            for _, x in j.iterrows():
                cr, ca = bool(x["has_min_ref"]), bool(x["has_min_arm"])
                if not cr and not ca:
                    res["neither"] += 1
                elif cr and not ca:
                    res["better"] += 1
                elif ca and not cr:
                    res["worse"] += 1
                else:
                    d = float(x[f"{metric}_ref"]) - float(x[f"{metric}_arm"])
                    res["better" if d < 0 else "worse" if d > 0 else "equal"] += 1
            dom.append(dict(arm=arm, metric=metric, n=len(j), **res))
    D = pd.DataFrame(dom); D.to_csv(a.out / "dominance.csv", index=False)

    summ = {}
    for arm, g in R.groupby("arm"):
        summ[arm] = dict(regions=len(g), level_mean=float(g["level"].mean()),
                         share_level_ge10=float((g["level"] >= 10).mean()), share_level_le6=float((g["level"] <= 6).mean()),
                         dE_median=float(g["dE"].median()), share_dE_gt15=float((g["dE"] > 15).mean()),
                         share_dE_gt25=float((g["dE"] > 25).mean()), share_olive=float(g["olive"].mean()),
                         sharp_ratio_median=float(g["sharp_ratio"].median()), share_blur_lt05=float((g["sharp_ratio"] < 0.5).mean()))
    lvl = R.groupby("level").agg(n=("dE", "size"), dE_median=("dE", "median"), share_dE_gt15=("dE", lambda s: float((s > 15).mean())),
                                 share_olive=("olive", "mean"), sharp_median=("sharp_ratio", "median")).reset_index()
    lvl.to_csv(a.out / "by_level.csv", index=False)
    runs_s = U[U["has_min"]].groupby("arm").agg(n=("n_edits", "size"), edits_med=("n_edits", "median"), img_med=("n_img", "median"),
                                                txt_med=("n_txt", "median"), share_img_only=("n_txt", lambda s: float((s == 0).mean())),
                                                share_txt_only=("n_img", lambda s: float((s == 0).mean())),
                                                img_share_of_edits=("n_img", "sum"), max_level_med=("max_level", "median"),
                                                share_any_olive=("n_olive", lambda s: float((s > 0).mean())),
                                                share_any_dE15=("n_off_dE15", lambda s: float((s > 0).mean())),
                                                lpips_med=("lpips", "median")).reset_index()
    tot = U[U["has_min"]].groupby("arm")["n_edits"].sum()
    runs_s["img_share_of_edits"] = runs_s.apply(lambda x: x["img_share_of_edits"] / tot[x["arm"]], axis=1)
    runs_s.to_csv(a.out / "runs_summary.csv", index=False)
    trs_s = T.groupby("arm").agg(**{f"any_le{L}": (f"any_le{L}", "mean") for L in LEVEL_CAPS},
                                 **{f"min_edits_le{L}_med": (f"min_edits_le{L}", "median") for L in LEVEL_CAPS},
                                 min_edits_any_med=("min_edits_any", "median"), gen0_img=("gen0_img_active", "mean"),
                                 gen0_txt=("gen0_txt_active", "mean")).reset_index()
    trs_s.to_csv(a.out / "trace_summary.csv", index=False)
    (a.out / "summary.json").write_text(json.dumps(summ, indent=2))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ).T.round(3).to_string()); print(lvl.round(3).to_string()); print(runs_s.round(3).to_string())
    print(trs_s.round(3).to_string()); print(D.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
