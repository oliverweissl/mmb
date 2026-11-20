"""Calibrate validity proxies against human validity labels (HS-01).

Usage::

    python -m mmb.validity.calibrate --proxies results/validity_hs01.csv \\
        --human <paper repo>/analysis/out/hs01_pair_item_validity_seed.csv --out results/validity_calibration.csv

NEEDS the HS-01 *stimuli*: score them first (``python -m mmb.validity.score <hs01_bundle> ...``) with
one directory per item, named by its ``item_id``, holding origin.png, first_cross.png, first_cross.json
and stats.json. The per-item human validity shares alone (the paper repo's analysis/out) are not enough.

For every proxy: Spearman rho with the item's human validity share and AUC of the proxy for
"human validity share >= threshold" (default 0.5), with percentile-bootstrap 95% CIs (10,000 resamples,
seed 42, resampling items), the proxy direction (distances are negated so that higher = more valid),
and the threshold that maximises Youden's J. Items are few (26 boundary pair items in HS-01), so the
intervals are wide; report them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

#: proxy column -> sign making "higher = more valid" (distances are negated)
PROXIES = {
    "img_clip_cos": 1, "img_dino_cos": 1, "img_lpips": -1, "img_ssim": 1,
    "txt_minilm_cos": 1, "txt_word_edit": -1, "question_keyword_sim": 1, "question_content_kept": 1,
}
N_BOOT = 10_000
BOOT_SEED = 42


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative (ties count half); NaN if one class is empty."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = sps.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def youden_threshold(score: np.ndarray, label: np.ndarray) -> float:
    score, label = np.asarray(score, float), np.asarray(label, bool)
    best, thr = -1.0, float("nan")
    for t in np.unique(score):
        pred = score >= t
        tpr = (pred & label).sum() / max(label.sum(), 1)
        fpr = (pred & ~label).sum() / max((~label).sum(), 1)
        if tpr - fpr > best:
            best, thr = tpr - fpr, float(t)
    return thr


def calibrate(joined: pd.DataFrame, threshold: float = 0.5, n_boot: int = N_BOOT) -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED)
    y = joined["validity"].to_numpy(float)
    lab = y >= threshold
    rows = []
    for col, sign in PROXIES.items():
        if col not in joined or joined[col].notna().sum() < 3:
            continue
        ok = joined[col].notna().to_numpy()
        x = sign * joined.loc[ok, col].to_numpy(float)
        yy, ll = y[ok], lab[ok]
        rho = float(sps.spearmanr(x, yy).statistic) if len(set(x)) > 1 else float("nan")
        a = auc(x, ll)
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        br = np.array([sps.spearmanr(x[i], yy[i]).statistic if len(set(x[i])) > 1 else np.nan for i in idx])
        ba = np.array([auc(x[i], ll[i]) for i in idx])
        r_lo, r_hi = np.nanpercentile(br, [2.5, 97.5]) if np.isfinite(br).any() else (np.nan, np.nan)
        a_lo, a_hi = np.nanpercentile(ba, [2.5, 97.5]) if np.isfinite(ba).any() else (np.nan, np.nan)
        rows.append(dict(proxy=col, direction="higher=valid" if sign > 0 else "negated distance", n_items=int(ok.sum()),
                         n_valid=int(ll.sum()), spearman=rho, rho_lo=r_lo, rho_hi=r_hi,
                         auc=a, auc_lo=a_lo, auc_hi=a_hi,
                         youden_threshold=sign * youden_threshold(x, ll)))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--proxies", type=Path, required=True, help="CSV from mmb.validity.score (item_id column)")
    ap.add_argument("--human", type=Path, required=True,
                    help="per-item human validity CSV (item_id, validity), e.g. hs01_pair_item_validity_seed.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="item counts as valid at human share >= this")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    joined = pd.read_csv(args.proxies).merge(pd.read_csv(args.human)[["item_id", "validity"]], on="item_id", how="inner")
    if joined.empty:
        raise SystemExit("no item_id in common: the proxies must be scored on the HS-01 stimuli, one directory per item_id")
    out = calibrate(joined, args.threshold, args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
