"""RQ3: join the ratings returned by raters with manifest.csv and summarise validity, agreement and proxy calibration.

Usage::

    python -m mmb.validity.analyze_ratings study/rq3/manifest.csv ratings_*.json [ratings_*.csv] \
        --out results/validity_human.csv [--agreement-out ...] [--calibration-out ...]

Ratings files are the page's JSON downloads ({rater_id, rows: [...]}) or its CSV downloads; duplicate
(rater, item) submissions keep the last one.
Per rating, valid = the content answer matches the seed input (imagenet: "yes", the image still shows the seed
class; pope: the answer equals the ground truth) AND the question is judged still the same ("yes").
Per item the validity is the mean over its raters; a group's share is the mean over its items, with a 95%
percentile bootstrap that resamples items with all their ratings (item-clustered; 10,000 resamples, seed 42).
Groups: (stratum, task, arm); controls (unedited codec reconstructions) have an empty arm.
Agreement: Krippendorff's alpha (nominal) and Gwet's AC1 over items with >= 2 ratings, for the valid label and
for each of the two questions, overall and per stratum x task x arm.
Calibration: per edited item, the human validity share against the proxies of manifest.csv (img_lpips,
vis_char_dist; see mmb.validity.calibrate): Spearman rho and AUC for "share >= 0.5", with bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmb.analysis import stats as st
from mmb.validity import calibrate as cal

CANT = ("__cant",)
ANSWER_LEVELS = ("yes", "no", "__cant")


def rating_valid(row) -> bool:
    a = str(row["answer_content"]).lower()
    ok = a == str(row["gt"]).lower() if row["task"] == "pope" else a == "yes"
    return ok and str(row["answer_same_question"]).lower() == "yes"


def read_ratings(p: Path) -> pd.DataFrame:
    p = Path(p)
    if p.suffix.lower() == ".json":
        j = json.loads(p.read_text())
        rows = j.get("rows", j) if isinstance(j, dict) else j
        return pd.DataFrame(rows).astype(str)
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def load(key: Path, ratings: list[Path]) -> pd.DataFrame:
    k = pd.read_csv(key, dtype=str, keep_default_na=False)
    r = pd.concat([read_ratings(p) for p in ratings], ignore_index=True)
    r = r.drop_duplicates(["rater_id", "item_id"], keep="last")
    d = r.merge(k, on="item_id", how="inner", validate="many_to_one")
    d["valid"] = d.apply(rating_valid, axis=1)
    d["cant"] = d["answer_content"].isin(CANT) | d["answer_same_question"].isin(CANT)
    return d


def summarize(d: pd.DataFrame, n_boot: int = st.N_BOOT, by=("stratum", "task", "arm")) -> pd.DataFrame:
    rows = []
    for gk, g in d.groupby(list(by), dropna=False):
        per_item = g.groupby("item_id").agg(valid=("valid", "mean"), cant=("cant", "mean"), n=("rater_id", "nunique"))
        v = per_item["valid"].to_numpy(float)
        pt, lo, hi = st.bootstrap_stat(v, stat=np.mean, n_boot=n_boot)
        rows.append({**dict(zip(by, gk if isinstance(gk, tuple) else (gk,))), "items": len(per_item),
                     "ratings": len(g), "raters": g["rater_id"].nunique(), "valid_share": pt, "lo": lo, "hi": hi,
                     "cant_share": float(per_item["cant"].mean())})
    return pd.DataFrame(rows)


# --- agreement (ported from the paper repo, analysis/hs01_reanalysis.py) ---------------------------------------
def coincidence_alpha(units, kind="nominal", levels=None):
    """Krippendorff alpha from a list of per-unit rating lists (handles missing; units with <2 ratings dropped)."""
    units = [u for u in units if len(u) >= 2]
    if not units:
        return np.nan
    vals = sorted({v for u in units for v in u}) if levels is None else list(levels)
    idx = {v: i for i, v in enumerate(vals)}
    n = len(vals)
    o = np.zeros((n, n))
    for u in units:
        m = len(u)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[idx[u[a]], idx[u[b]]] += 1.0 / (m - 1)
    nc = o.sum(1)
    N = nc.sum()
    if N <= 1:
        return np.nan
    if kind == "nominal":
        delta = 1 - np.eye(n)
    else:  # interval
        v = np.array(vals, float)
        delta = (v[:, None] - v[None, :]) ** 2
    do = (o * delta).sum()
    de = (np.outer(nc, nc) * delta).sum() / (N - 1)
    return 1 - do / de if de > 0 else np.nan


def gwet_ac1(units, levels):
    """Gwet AC1 for multiple raters with missing ratings (items with >=2 ratings)."""
    units = [u for u in units if len(u) >= 2]
    if not units:
        return np.nan
    q = len(levels)
    pa, pi = [], np.zeros(q)
    for u in units:
        n = len(u)
        cnt = np.array([sum(1 for x in u if x == c) for c in levels], float)
        pa.append((cnt * (cnt - 1)).sum() / (n * (n - 1)))
        pi += cnt / n
    pa = float(np.mean(pa))
    pi /= len(units)
    pe = (pi * (1 - pi)).sum() / (q - 1)
    return (pa - pe) / (1 - pe) if pe < 1 else np.nan


def agreement(d: pd.DataFrame, by=("stratum", "task", "arm")) -> pd.DataFrame:
    """Alpha and AC1 per rated variable, overall (group 'all') and per group."""
    targets = {"valid": (d["valid"].astype(bool), (False, True)),
               "answer_content": (d["answer_content"].str.lower(), ANSWER_LEVELS),
               "answer_same_question": (d["answer_same_question"].str.lower(), ANSWER_LEVELS)}
    groups = [("all", d)] + [("/".join(map(str, k if isinstance(k, tuple) else (k,))), g)
                             for k, g in d.groupby(list(by), dropna=False)]
    rows = []
    for name, g in groups:
        for var, (col, levels) in targets.items():
            units = [list(v) for _, v in col.loc[g.index].groupby(g["item_id"])]
            units = [u for u in units if len(u) >= 2]
            rows.append(dict(group=name, variable=var, items=len(units),
                             krippendorff_alpha=coincidence_alpha(units, levels=levels),
                             gwet_ac1=gwet_ac1(units, levels)))
    return pd.DataFrame(rows)


def proxy_calibration(d: pd.DataFrame, threshold: float = 0.5, n_boot: int = cal.N_BOOT) -> pd.DataFrame:
    """Human validity share of each edited item against its proxies (columns of manifest.csv)."""
    e = d[d["stratum"] == "edited"]
    if e.empty:
        return pd.DataFrame()
    per = e.groupby("item_id").agg(validity=("valid", "mean"),
                                   **{c: (c, "first") for c in ("img_lpips", "vis_char_dist") if c in e})
    for c in ("img_lpips", "vis_char_dist"):
        if c in per:
            per[c] = pd.to_numeric(per[c], errors="coerce")
    return cal.calibrate(per.reset_index(), threshold, n_boot)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("key", type=Path, help="manifest.csv written by mmb.validity.rating_set (the private key)")
    ap.add_argument("ratings", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("results/validity_human.csv"))
    ap.add_argument("--agreement-out", type=Path, default=Path("results/validity_human_agreement.csv"))
    ap.add_argument("--calibration-out", type=Path, default=Path("results/validity_human_calibration.csv"))
    ap.add_argument("--n-boot", type=int, default=st.N_BOOT)
    a = ap.parse_args(argv)
    d = load(a.key, a.ratings)
    outs = {a.out: summarize(d, a.n_boot), a.agreement_out: agreement(d),
            a.calibration_out: proxy_calibration(d, n_boot=a.n_boot)}
    for p, t in outs.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(p, index=False)
        print(f"== {p}\n{t.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
