"""Paired comparisons with cell-cluster bootstrap, per-stratum tests and a mixed model.

Why: runs of different arms share cells (and, pooled over SUTs, the same cells again), so a run-level
interval or test treats dependent pairs as independent. Here

* a *pair* is the two runs of an arm and the reference with the same (cell_id, rng_seed);
* a *cell difference* is the mean of the pair differences inside one cell (S seeds -> one number);
* the interval of a paired difference is a percentile bootstrap over cells (10,000 resamples, seed 42):
  mean cell difference for binary outcomes (a difference of shares), median cell difference otherwise;
* the p-values stay run-level and exact as pre-committed (McNemar for crossing, Wilcoxon signed-rank for
  the rest), Holm-adjusted across arms within a metric. Primary tests are per (task, SUT) stratum;
  pooled tests are secondary.

Metrics (columns of collect.py): crossed, first_cross_eval, floor (log10 nats), and the smallest crossing found within
the budget (min_cross.json): magnitude (log10), mag_image, mag_text, min_text_d (MiniLM distance of the edited
question), min_eval_idx (calls to it). The smallest-crossing metrics are NaN for runs without an in-budget smallest
crossing and are compared on pairs in which both runs have one (smaller is better).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from mmb.analysis import stats as st
from mmb.analysis import tablefmt as tf

logger = logging.getLogger(__name__)

#: name -> (column, kind, transform, direction); direction -1: smaller is better, +1: larger is better
METRICS = {
    "crossed": ("crossed", "binary", None, +1),
    "first_cross_eval": ("first_cross_eval", "num", None, -1),
    "floor": ("floor_nats", "num", "log10", -1),
    "magnitude": ("magnitude", "num", "log10", -1),          # smallest in-budget crossing (L2 of the block magnitudes)
    "mag_image": ("mag_image", "num", None, -1),
    "mag_text": ("mag_text", "num", None, -1),
    "text_d": ("min_text_d", "num", None, -1),
    "min_eval_idx": ("min_eval_idx", "num", None, -1),      # calls to the smallest crossing
    "wall": ("wall_s", "num", None, -1),
}
#: metrics of the smallest crossing: defined only for runs whose smallest crossing lies within the budget
CROSS_ONLY = ("magnitude", "mag_image", "mag_text", "text_d", "min_eval_idx")
MIN_PAIRS_CROSS_ONLY = 3
DEFAULT_METRICS = ("crossed", "first_cross_eval", "floor", "magnitude", "mag_image", "mag_text")
CLUSTER = "cluster_id"


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(("true", "1", "1.0"))


def _values(s: pd.Series, kind: str, transform: str | None) -> np.ndarray:
    v = s.astype(float).to_numpy()
    if transform == "log10":
        v = np.log10(np.clip(v, 1e-300, None))
    return v


def pair_frame(df: pd.DataFrame, ref: str, arm: str, metric: str) -> pd.DataFrame:
    """Matched pairs for one metric: cell, cluster, ref value, arm value (NaN pairs dropped)."""
    col, kind, tr, _ = METRICS[metric]
    if col not in df:
        return pd.DataFrame(columns=["cell_id", "cluster", "ref", "arm"])
    keys = st.PAIR_KEYS
    parts = []
    for name, a in (("ref", ref), ("arm", arm)):
        s = df.loc[df["arm"] == a]
        if metric in CROSS_ONLY:
            ok = _truthy(s["min_within_budget"]) if "min_within_budget" in s else s["crossed"].astype(bool)
            s = s.assign(**{col: pd.to_numeric(s[col], errors="coerce").where(ok)})
        if s.duplicated(keys).any():
            raise ValueError(f"arm {a!r} has several runs per (cell_id, rng_seed)")
        cl = s[CLUSTER] if CLUSTER in s else s["cell_id"]
        parts.append(pd.DataFrame({**{k: s[k].to_numpy() for k in keys}, "cluster": cl.to_numpy(),
                                   name: _values(s[col], kind, tr)}))
    m = parts[0].merge(parts[1].drop(columns="cluster"), on=keys, how="inner")
    return m.dropna(subset=["ref", "arm"]).reset_index(drop=True)


def cell_differences(pf: pd.DataFrame) -> np.ndarray:
    """Mean pair difference (arm - ref) per cluster."""
    if pf.empty:
        return np.zeros(0)
    return (pf["arm"] - pf["ref"]).groupby(pf["cluster"]).mean().to_numpy()


def cluster_ci(diffs: np.ndarray, kind: str, n_boot: int = st.N_BOOT, seed: int = st.BOOT_SEED):
    """(point, lo, hi) of the mean (binary) or median (numeric) cell difference, bootstrap over cells."""
    if len(diffs) == 0:
        return (float("nan"),) * 3
    stat = np.mean if kind == "binary" else np.median
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boots = stat(diffs[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(stat(diffs)), float(lo), float(hi)


def paired_table(df: pd.DataFrame, ref: str, metrics: Sequence[str] = DEFAULT_METRICS,
                 arms: Sequence[str] | None = None, n_boot: int = st.N_BOOT,
                 min_pairs: int = 6) -> pd.DataFrame:
    """One row per (arm, metric): pairs, values, cell-cluster CI of the difference, exact test, Holm.

    Arms with fewer than `min_pairs` matched (cell, seed) pairs (an incomplete layer) are left out, so a
    degenerate one-pair comparison is never printed."""
    if ref not in set(df["arm"]):
        raise ValueError(f"reference arm {ref!r} not in data")
    arms = [a for a in (arms or sorted(set(df["arm"]))) if a != ref]
    rows = []
    for metric in metrics:
        col, kind, tr, direction = METRICS[metric]
        for arm in arms:
            pf = pair_frame(df, ref, arm, metric)
            n = len(pf)
            if n < (min(min_pairs, MIN_PAIRS_CROSS_ONLY) if metric in CROSS_ONLY else min_pairs):
                continue
            point, lo, hi = cluster_ci(cell_differences(pf), kind, n_boot)
            if kind == "binary":
                mc = st.mcnemar_exact(pf["ref"].to_numpy() > 0.5, pf["arm"].to_numpy() > 0.5)
                p, eff, eff_name = mc["p"], mc["diff"], "share diff"
                extra = dict(b_ref_only=mc["b_ref_only"], c_arm_only=mc["c_arm_only"])
                ref_v, arm_v = float(pf["ref"].mean()), float(pf["arm"].mean())
            else:
                w = st.wilcoxon_paired(pf["ref"].to_numpy(), pf["arm"].to_numpy())
                p, eff, eff_name, extra = w["p"], w["r_rb"], "rank-biserial r", dict(n_nonzero=w["n_nonzero"])
                ref_v, arm_v = float(np.median(pf["ref"])), float(np.median(pf["arm"]))
            rows.append(dict(arm=arm, ref=ref, metric=metric, n_pairs=n, n_cells=pf["cluster"].nunique(),
                             ref_value=ref_v, arm_value=arm_v, diff=point, diff_lo=lo, diff_hi=hi,
                             diff_name="mean cell difference" if kind == "binary" else "median cell difference",
                             p=p, effect=eff, effect_name=eff_name, direction=direction,
                             dz=tf.cohens_dz(pf["arm"].to_numpy(float) - pf["ref"].to_numpy(float)), **extra))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        for metric in set(out["metric"]):
            idx = out.index[out["metric"] == metric]
            out.loc[idx, "p_holm"] = st.holm(out.loc[idx, "p"].to_numpy())
    return out


def primary_table(df: pd.DataFrame, ref: str = "a0_directed", arm: str = "a9_deepjanus",
                  n_boot: int = st.N_BOOT) -> pd.DataFrame:
    """Crossing share of arm vs ref per (task, SUT) stratum (the secondary comparisons a7 and a9 vs a0),
    exact McNemar, Holm across strata."""
    rows = []
    for (task, sut), g in df.groupby(["task", "sut"], sort=True):
        if not {ref, arm} <= set(g["arm"]):
            continue
        pt = paired_table(g, ref, ("crossed",), [arm], n_boot)
        if len(pt):
            r = pt.iloc[0].to_dict()
            r.update(task=task, sut=sut)
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm_strata"] = st.holm(out["p"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# mixed model
# ---------------------------------------------------------------------------


def mixed_model(df: pd.DataFrame, ref: str, arm: str, seed: int = st.BOOT_SEED) -> dict:
    """crossed ~ arm + (1 | cell) + (1 | sut:task): Bayesian mixed logit by variational Bayes
    (statsmodels BinomialBayesMixedGLM). Returns the log odds ratio of *arm* against *ref* with posterior
    mean, sd and a 95% interval (mean +- 1.96 sd; approximate). Needs statsmodels; returns NaNs without it."""
    d = df[df["arm"].isin([ref, arm])].copy()
    nan = dict(arm=arm, ref=ref, log_or=float("nan"), sd=float("nan"), lo=float("nan"), hi=float("nan"),
               odds_ratio=float("nan"), n_runs=len(d), n_cells=int(d["cell_id"].nunique()) if len(d) else 0)
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        logger.warning("statsmodels not installed: mixed model skipped")
        return nan
    if d.empty or d["crossed"].nunique() < 2 or d["arm"].nunique() < 2:
        return nan
    d["y"] = d["crossed"].astype(int)
    d["is_arm"] = (d["arm"] == arm).astype(int)
    d["cellkey"] = d["task"].astype(str) + "/" + d["cell_id"].astype(str)
    d["sut_task"] = d["sut"].astype(str) + ":" + d["task"].astype(str)
    vc = {"cell": "0 + C(cellkey)"}
    if d["sut_task"].nunique() > 1:
        vc["sut_task"] = "0 + C(sut_task)"
    np.random.seed(seed)
    res = BinomialBayesMixedGLM.from_formula("y ~ is_arm", vc, d).fit_vb()
    i = list(res.model.exog_names).index("is_arm")
    m, s = float(res.fe_mean[i]), float(res.fe_sd[i])
    return dict(nan, log_or=m, sd=s, lo=m - 1.96 * s, hi=m + 1.96 * s, odds_ratio=float(np.exp(m)))

