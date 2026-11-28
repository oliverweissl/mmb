"""Hardness: how the nominal margin and the language prior relate to the effort a search needs (Discussion).

Per arm (MMBA and the two baselines), pooled over SUTs and tasks, the Spearman rank correlation of
  |anchor_g|  the margin of the unedited input (how confidently the model decides), and
  |prior_g|   the margin of the content-free input (gray image, same prompt: the strength of the language prior)
with the calls to the first crossing (crossing runs) and the magnitude of the smallest crossing found within the
budget (min_cross.json). Intervals: percentile bootstrap over runs. The correlations rank-order runs; they are not
predictions.

Usage::

    python -m mmb.analysis.hardness results/MAIN.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from mmb.analysis.stats import BOOT_SEED, load_results

#: arm id -> macro name
ARMS = {"a0_directed": "Directed", "a5_mimicry_objective": "Mimicry", "a9_deepjanus": "Janus"}
PREDICTORS = (("anchor_g", "Anchor"), ("prior_g", "Prior"))
OUTCOMES = (("first_cross_eval", "Calls"), ("magnitude", "Mag"))


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(("true", "1", "1.0"))


def frame(df: pd.DataFrame) -> pd.DataFrame:
    """Crossing runs of the ARMS with |anchor_g|, |prior_g|; magnitude kept only when within the budget."""
    d = df[df["arm"].isin(ARMS) & df["crossed"].astype(bool)].copy()
    for c in ("anchor_g", "prior_g"):
        d[c] = pd.to_numeric(d[c], errors="coerce").abs() if c in d else np.nan
    mag = pd.to_numeric(d["magnitude"], errors="coerce") if "magnitude" in d else pd.Series(np.nan, index=d.index)
    if "min_within_budget" in d:
        mag = mag.where(_truthy(d["min_within_budget"]))
    d["magnitude"] = mag
    return d


def rho(x: np.ndarray, y: np.ndarray, n_boot: int = 2000):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 8:
        return np.nan, np.nan, np.nan, len(x)
    r = float(sps.spearmanr(x, y)[0])
    rng = np.random.default_rng(BOOT_SEED)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        b.append(sps.spearmanr(x[i], y[i])[0])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return r, float(lo), float(hi), len(x)


def correlations(d: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for arm, aname in ARMS.items():
        g = d[d["arm"] == arm]
        if g.empty:
            continue
        for pc, pn in PREDICTORS:
            for oc, on in OUTCOMES:
                r, lo, hi, n = rho(g[pc].to_numpy(float), pd.to_numeric(g[oc], errors="coerce").to_numpy(float), n_boot)
                rows.append(dict(arm=arm, arm_name=aname, pred=pn, outcome=on, rho=r, lo=lo, hi=hi, n=n))
    return pd.DataFrame(rows)


def add_numbers(num, c: pd.DataFrame) -> None:
    """Macros EHard<Pred><Outcome><Arm>{Rho,Lo,Hi,N}."""
    for r in c.itertuples():
        if np.isnan(r.rho):
            continue
        tail = [r.pred, r.outcome, r.arm_name]
        what = "calls to the first crossing" if r.outcome == "Calls" else "magnitude of the smallest crossing"
        d = f"Spearman rho of |{r.pred.lower()} margin| with {what}, {r.arm}, pooled over SUTs and tasks"
        num.add(["EHard", "Rho"] + tail, f"{r.rho:.2f}", d)
        num.add(["EHard", "Lo"] + tail, f"{r.lo:.2f}", "bootstrap 95% lower bound; " + d)
        num.add(["EHard", "Hi"] + tail, f"{r.hi:.2f}", "bootstrap 95% upper bound; " + d)
        num.add(["EHard", "N"] + tail, str(int(r.n)), "runs; " + d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    a = ap.parse_args(argv)
    print(correlations(frame(load_results([Path(a.results)]))).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
