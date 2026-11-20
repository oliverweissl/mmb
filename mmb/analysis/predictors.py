"""E2b: which cheap quantity predicts whether and how hard a run crosses?

Usage::

    python -m mmb.analysis.predictors results/e1.csv [more.csv ...] --out results/predictors \\
        [--arm a0_directed] [--by-arm] [--n-boot 10000]

Input: CSVs from :mod:`mmb.analysis.collect` (needs ``anchor_g, prior_g, probe_gen0, min_gen0``).

Predictors (all known before or at generation 0):
  anchor_g       seed confidence gap, the pair contrast of the unmodified seed
  prior_g        pair contrast of the content-free input (uniform gray image, same prompt)
  abs_prior_g    its magnitude
  probe_gen0     median |g| over the generation-0 evaluations (the paper's probe)
  min_gen0       min |g| over generation 0
  answer_words_diff  words(class_a) - words(class_b), a crude stand-in for answer length (tokens are not
                 derivable without the SUT tokenizer)
Outcomes: floor_nats (min |g| over the run), first_cross_eval (censored at the budget), crossed.

Statistics, per (sut, task) [and arm with --by-arm], at run level and, in a separate table, at cell level
(rows averaged over seeds within cell_id; the cell outcome "crossed" is a crossing share >= 0.5):
  * Spearman rho of every predictor with floor_nats and with first_cross_eval
  * AUC of every predictor for "crossed" (probability that a crossing run has the larger predictor value;
    below 0.5 means larger values go with fewer crossings)
  * partial Spearman rho of probe_gen0 given prior_g and min_gen0 (rank-residual method), and of
    abs_prior_g given probe_gen0 and min_gen0 (is the prior better than the probe?)
All intervals: percentile bootstrap, 10,000 resamples, seed 42, resampling runs (run level) or cells (cell level).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from mmb.analysis.stats import BOOT_SEED, N_BOOT, load_results, tex_escape, _booktabs

logger = logging.getLogger(__name__)

PREDICTORS = ["anchor_g", "prior_g", "abs_prior_g", "probe_gen0", "min_gen0", "answer_words_diff"]
OUTCOMES = ["floor_nats", "first_cross_eval"]
MIN_N = 5
#: partial correlations: (predictor, controls)
PARTIALS = [
    ("probe_gen0", ["prior_g", "min_gen0"]),
    ("probe_gen0", ["abs_prior_g", "min_gen0"]),
    ("abs_prior_g", ["probe_gen0", "min_gen0"]),
]


# ---------------------------------------------------------------------------
# Vectorised statistics (rows = bootstrap resamples)
# ---------------------------------------------------------------------------


def _rank_rows(a: np.ndarray) -> np.ndarray:
    return sps.rankdata(a, axis=-1)


def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman rho of each row pair of x and y (2-D arrays of equal shape); NaN for constant rows."""
    rx, ry = _rank_rows(x), _rank_rows(y)
    rx = rx - rx.mean(axis=-1, keepdims=True)
    ry = ry - ry.mean(axis=-1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(-1) / den


def auc_rows(score: np.ndarray, label: np.ndarray) -> np.ndarray:
    """AUC of each row: P(score of a positive > score of a negative), ties half; NaN if a class is empty."""
    r = _rank_rows(score)
    n1 = label.sum(-1)
    n0 = label.shape[-1] - n1
    u = (r * label).sum(-1) - n1 * (n1 + 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        out = u / (n1 * n0)
    out[(n1 == 0) | (n0 == 0)] = np.nan
    return out


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial Spearman rho of x and y given the columns of z (rank-residual method)."""
    rx, ry = sps.rankdata(x), sps.rankdata(y)
    rz = np.column_stack([sps.rankdata(z[:, j]) for j in range(z.shape[1])])
    design = np.column_stack([np.ones(len(x)), rz])

    def resid(v):
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ beta

    a, b = resid(rx), resid(ry)
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _ci(boot: np.ndarray) -> tuple[float, float]:
    boot = boot[np.isfinite(boot)]
    if len(boot) < 100:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _chunks(n: int, n_boot: int, rng: np.random.Generator):
    size = max(1, min(n_boot, 1_000_000 // max(n, 1)))
    done = 0
    while done < n_boot:
        m = min(size, n_boot - done)
        yield rng.integers(0, n, size=(m, n))
        done += m


def spearman_ci(x, y, n_boot=N_BOOT, seed=BOOT_SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearman_rows(x[None], y[None])[0])
    rng = np.random.default_rng(seed)
    boot = np.concatenate([spearman_rows(x[i], y[i]) for i in _chunks(len(x), n_boot, rng)])
    return (rho, *_ci(boot))


def auc_ci(score, label, n_boot=N_BOOT, seed=BOOT_SEED):
    score, label = np.asarray(score, float), np.asarray(label, bool).astype(float)
    a = float(auc_rows(score[None], label[None])[0])
    rng = np.random.default_rng(seed)
    boot = np.concatenate([auc_rows(score[i], label[i]) for i in _chunks(len(score), n_boot, rng)])
    return (a, *_ci(boot))


def partial_ci(x, y, z, n_boot=N_BOOT, seed=BOOT_SEED):
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    rho = partial_spearman(x, y, z)
    rng = np.random.default_rng(seed)
    boot = []
    for idx in _chunks(len(x), n_boot, rng):
        for i in idx:
            boot.append(partial_spearman(x[i], y[i], z[i]))
    return (rho, *_ci(np.asarray(boot)))


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["abs_prior_g"] = d["prior_g"].abs()
    if {"answer_words_a", "answer_words_b"} <= set(d.columns):
        d["answer_words_diff"] = d["answer_words_a"] - d["answer_words_b"]
    else:
        d["answer_words_diff"] = np.nan
    d["crossed"] = d["crossed"].astype(bool)
    return d


def cell_frame(d: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """One row per cell: predictors and outcomes averaged over seeds; crossed = crossing share >= 0.5."""
    num = [c for c in PREDICTORS + OUTCOMES if c in d]
    g = d.groupby(list(keys) + ["cell_id"], dropna=False)
    out = g[num].mean().reset_index()
    out["crossed_share"] = g["crossed"].mean().to_numpy()
    out["n_seeds"] = g.size().to_numpy()
    out["crossed"] = out["crossed_share"] >= 0.5
    return out


def _group_keys(by_arm: bool) -> list[str]:
    return ["sut", "task"] + (["arm"] if by_arm else [])


def correlations(d: pd.DataFrame, level: str, keys: Sequence[str], n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for gk, g in d.groupby(list(keys), dropna=False):
        gk = gk if isinstance(gk, tuple) else (gk,)
        lab = dict(zip(keys, gk))
        for pred in PREDICTORS:
            if pred not in g or g[pred].notna().sum() < MIN_N:
                continue
            for out in OUTCOMES:
                ok = g[pred].notna() & g[out].notna()
                if ok.sum() < MIN_N:
                    continue
                rho, lo, hi = spearman_ci(g.loc[ok, pred], g.loc[ok, out], n_boot)
                rows.append({**lab, "level": level, "predictor": pred, "outcome": out, "stat": "spearman",
                             "n": int(ok.sum()), "value": rho, "lo": lo, "hi": hi})
            ok = g[pred].notna()
            if ok.sum() >= MIN_N:
                a, lo, hi = auc_ci(g.loc[ok, pred], g.loc[ok, "crossed"], n_boot)
                rows.append({**lab, "level": level, "predictor": pred, "outcome": "crossed", "stat": "auc",
                             "n": int(ok.sum()), "value": a, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def partials(d: pd.DataFrame, level: str, keys: Sequence[str], n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for gk, g in d.groupby(list(keys), dropna=False):
        gk = gk if isinstance(gk, tuple) else (gk,)
        lab = dict(zip(keys, gk))
        for pred, controls in PARTIALS:
            for out in OUTCOMES:
                cols = [pred, out] + controls
                gg = g.dropna(subset=cols)
                if len(gg) < MIN_N + len(controls):
                    continue
                raw = float(spearman_rows(gg[[pred]].to_numpy().T, gg[[out]].to_numpy().T)[0])
                rho, lo, hi = partial_ci(gg[pred], gg[out], gg[controls], n_boot)
                rows.append({**lab, "level": level, "predictor": pred, "given": "+".join(controls),
                             "outcome": out, "n": len(gg), "rho_raw": raw, "rho_partial": rho, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def _fmt(v, lo, hi) -> str:
    return "--" if pd.isna(v) else f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]" if not pd.isna(lo) else f"{v:+.2f}"


def corr_tex(c: pd.DataFrame, keys: Sequence[str]) -> str:
    if c.empty:
        return ""
    rows = []
    for gk, g in c.groupby(list(keys) + ["predictor"], sort=False):
        gk = gk if isinstance(gk, tuple) else (gk,)
        def cell(out, stat):
            r = g[(g["outcome"] == out) & (g["stat"] == stat)]
            return _fmt(r["value"].iloc[0], r["lo"].iloc[0], r["hi"].iloc[0]) if len(r) else "--"
        n = int(g["n"].max())
        rows.append([tex_escape(x) for x in gk] + [str(n), cell("floor_nats", "spearman"),
                                                   cell("first_cross_eval", "spearman"), cell("crossed", "auc")])
    return _booktabs("l" * (len(keys) + 1) + "rrrr", [k.replace("_", " ") for k in keys] +
                     ["predictor", "$n$", r"$\rho$ floor [95\% CI]", r"$\rho$ evals to cross [95\% CI]",
                      r"AUC crossed [95\% CI]"], rows)


def partial_tex(p: pd.DataFrame, keys: Sequence[str]) -> str:
    if p.empty:
        return ""
    rows = []
    for _, r in p.iterrows():
        rows.append([tex_escape(r[k]) for k in keys] + [tex_escape(r["predictor"]), tex_escape(r["given"]),
                     tex_escape(r["outcome"]), str(int(r["n"])), f"{r['rho_raw']:+.2f}",
                     _fmt(r["rho_partial"], r["lo"], r["hi"])])
    return _booktabs("l" * (len(keys) + 3) + "rrr", [k.replace("_", " ") for k in keys] +
                     ["predictor", "given", "outcome", "$n$", r"$\rho$", r"partial $\rho$ [95\% CI]"], rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_all(df: pd.DataFrame, out: Path, arm: str | None = None, by_arm: bool = False,
            n_boot: int = N_BOOT) -> dict[str, pd.DataFrame]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = prepare(df)
    if arm:
        d = d[d["arm"] == arm]
    keys = _group_keys(by_arm)
    cells = cell_frame(d, keys)
    res = {
        "run_level": correlations(d, "run", keys, n_boot),
        "cell_level": correlations(cells, "cell", keys, n_boot),
        "partial_run": partials(d, "run", keys, n_boot),
        "partial_cell": partials(cells, "cell", keys, n_boot),
    }
    for name, frame in res.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "run_level.tex").write_text(corr_tex(res["run_level"], keys))
    (out / "cell_level.tex").write_text(corr_tex(res["cell_level"], keys))
    (out / "partial_run.tex").write_text(partial_tex(res["partial_run"], keys))
    (out / "partial_cell.tex").write_text(partial_tex(res["partial_cell"], keys))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/predictors"))
    ap.add_argument("--arm", default=None, help="restrict to one arm (e.g. a0_directed)")
    ap.add_argument("--by-arm", action="store_true", help="also split by arm")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = load_results(args.csv)
    missing = {"prior_g", "probe_gen0", "min_gen0"} - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns {sorted(missing)}: collect with a tester that stores prior_g")
    run_all(df, args.out, args.arm, args.by_arm, args.n_boot)
    logger.info("predictor tables written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
