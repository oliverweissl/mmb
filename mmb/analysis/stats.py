"""Statistics and LaTeX tables for the comparison of search arms.

Usage::

    python -m mmb.analysis.stats results/e1.csv [more.csv ...] --ref a0_directed \\
        --out results/tables [--budgets 30,100,206,500,1000,1500]

Input: CSVs written by :mod:`mmb.analysis.collect` (one row per run). Output (in ``--out``):
``summary.csv/.tex``, ``curve.csv``, ``curve_budgets.tex``, ``paired.csv/.tex``.

Method
------
* Shares: Wilson 95% CI. Curve bands and medians: percentile bootstrap, 10,000 resamples, seed 42,
  resampling runs (or whole clusters if a cluster column is given).
* Paired comparison of every arm against the reference arm on matched ``(cell_id, rng_seed)``:
  exact McNemar test for ``crossed`` (two-sided binomial test on the discordant pairs);
  Wilcoxon signed-rank for ``first_cross_eval`` (censored at the budget) and for
  ``log10(floor_nats)`` (log scale: floors span decades, and signed-rank tests depend on the
  scale through the ranks of the differences). Effect size: matched-pairs rank-biserial
  correlation r = (W+ - W-) / (W+ + W-) over the non-zero differences, sign = arm minus reference,
  so for ``first_cross_eval`` and ``floor`` a negative r means the arm is better.
  Holm correction across arms within each metric.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

logger = logging.getLogger(__name__)

N_BOOT = 10_000
BOOT_SEED = 42
DEFAULT_BUDGETS = (30, 100, 206, 500, 1000, 1500)
PAIR_KEYS = ["cell_id", "rng_seed"]
METRICS = ("crossed", "first_cross_eval", "floor")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(paths: Sequence[Path | str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for c in ("crossed", "crossed_at_gen0", "label_flip", "reached_band", "complete"):
        if c in df:
            df[c] = df[c].astype(str).str.lower().isin(("true", "1", "1.0"))
    return df


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial share."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def _boot_indices(n: int, n_boot: int, rng: np.random.Generator, clusters: np.ndarray | None):
    """Yield index matrices (chunk x n) of resampled runs (cluster bootstrap if given)."""
    chunk = max(1, min(n_boot, 2_000_000 // max(n, 1)))
    done = 0
    if clusters is None:
        while done < n_boot:
            m = min(chunk, n_boot - done)
            yield rng.integers(0, n, size=(m, n))
            done += m
    else:
        uniq, inv = np.unique(clusters, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        while done < n_boot:
            m = min(chunk, n_boot - done)
            out = []
            for _ in range(m):
                pick = rng.integers(0, len(uniq), size=len(uniq))
                out.append(np.concatenate([members[i] for i in pick]))
            width = max(len(o) for o in out)
            # clusters can differ in size; pad by repeating (keeps the resample size ~constant)
            out = [np.resize(o, width) for o in out]
            yield np.vstack(out)
            done += m


def bootstrap_stat(
    values: np.ndarray, stat=np.median, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
    clusters: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """(point, lo, hi): percentile 95% bootstrap CI of stat(values)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = np.concatenate([stat(v[idx], axis=1) for idx in _boot_indices(len(v), n_boot, rng, clusters)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(stat(v)), float(lo), float(hi)


def share_curve(
    first_cross: np.ndarray, crossed: np.ndarray, budgets: Sequence[int],
) -> np.ndarray:
    """Share of runs with a crossing within each budget (never-crossed runs count as not crossed)."""
    fc = np.where(crossed, first_cross, np.inf).astype(float)
    return np.array([(fc <= b).mean() for b in budgets])


def curve_with_bands(
    df: pd.DataFrame, budgets: Sequence[int], n_boot: int = N_BOOT, seed: int = BOOT_SEED,
    cluster_col: str | None = None,
) -> pd.DataFrame:
    """Per arm and budget: share, Wilson CI and percentile-bootstrap CI."""
    rows = []
    budgets = np.asarray(budgets)
    for arm, g in df.groupby("arm", sort=True):
        fc = np.where(g["crossed"].to_numpy(), g["first_cross_eval"].to_numpy(dtype=float), np.inf)
        ind = (fc[:, None] <= budgets[None, :]).astype(float)          # runs x budgets
        n = len(g)
        rng = np.random.default_rng(seed)
        clusters = g[cluster_col].to_numpy() if cluster_col else None
        boot = np.concatenate([ind[idx].mean(axis=1) for idx in _boot_indices(n, n_boot, rng, clusters)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for j, b in enumerate(budgets):
            k = int(ind[:, j].sum())
            wl, wh = wilson(k, n)
            rows.append(dict(arm=arm, budget=int(b), n=n, k=k, share=k / n,
                             wilson_lo=wl, wilson_hi=wh, boot_lo=float(lo[j]), boot_hi=float(hi[j])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-arm summary
# ---------------------------------------------------------------------------


def arm_summary(df: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for arm, g in df.groupby("arm", sort=True):
        n = len(g)
        k = int(g["crossed"].sum())
        lo, hi = wilson(k, n)
        k0 = int(g["crossed_at_gen0"].sum())
        l0, h0 = wilson(k0, n)
        med_fc, fc_lo, fc_hi = bootstrap_stat(g["first_cross_eval"].to_numpy(float), n_boot=n_boot)
        med_fl, fl_lo, fl_hi = bootstrap_stat(g["floor_nats"].to_numpy(float), n_boot=n_boot)
        q1, q3 = np.percentile(g["first_cross_eval"].to_numpy(float), [25, 75])
        rows.append(dict(
            arm=arm, n_runs=n, n_crossed=k, share_crossed=k / n, wilson_lo=lo, wilson_hi=hi,
            n_crossed_gen0=k0, share_gen0=k0 / n, gen0_lo=l0, gen0_hi=h0,
            share_band=float(g["reached_band"].mean()), share_label_flip=float(g["label_flip"].mean()),
            median_first_cross_eval=med_fc, mfc_lo=fc_lo, mfc_hi=fc_hi, fc_q1=float(q1), fc_q3=float(q3),
            median_floor_nats=med_fl, mfl_lo=fl_lo, mfl_hi=fl_hi,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar_exact(ref: np.ndarray, arm: np.ndarray) -> dict:
    """Exact McNemar on paired booleans. b: ref only crossed, c: arm only crossed."""
    ref, arm = np.asarray(ref, bool), np.asarray(arm, bool)
    b = int((ref & ~arm).sum())
    c = int((~ref & arm).sum())
    p = 1.0 if b + c == 0 else float(sps.binomtest(min(b, c), b + c, 0.5).pvalue)
    return dict(b_ref_only=b, c_arm_only=c, p=p, diff=float(arm.mean() - ref.mean()))


def rank_biserial_paired(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-) over non-zero differences."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(d))
    wp, wm = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def wilcoxon_paired(ref: np.ndarray, arm: np.ndarray) -> dict:
    d = np.asarray(arm, float) - np.asarray(ref, float)
    nz = d[d != 0]
    p = 1.0 if len(nz) == 0 else float(sps.wilcoxon(nz, zero_method="wilcox", method="auto").pvalue)
    return dict(p=p, r_rb=rank_biserial_paired(d), n_nonzero=int(len(nz)),
                median_diff=float(np.median(d)) if len(d) else float("nan"))


def matched(df: pd.DataFrame, ref_arm: str, arm: str) -> pd.DataFrame:
    """Inner join of the two arms on (cell_id, rng_seed); columns suffixed _ref / _arm."""
    cols = PAIR_KEYS + ["crossed", "first_cross_eval", "floor_nats"]
    out = []
    for name, a in (("ref", ref_arm), ("arm", arm)):
        sub = df.loc[df["arm"] == a, cols]
        if sub.duplicated(PAIR_KEYS).any():
            dup = sub[sub.duplicated(PAIR_KEYS, keep=False)][PAIR_KEYS].drop_duplicates().head(3)
            raise ValueError(f"arm {a!r} has several runs per (cell_id, rng_seed), e.g. {dup.values.tolist()}")
        out.append(sub.rename(columns={c: f"{c}_{name}" for c in cols if c not in PAIR_KEYS}))
    return out[0].merge(out[1], on=PAIR_KEYS, how="inner")


def paired_vs_reference(df: pd.DataFrame, ref_arm: str, arms: Sequence[str] | None = None) -> pd.DataFrame:
    """Every arm against *ref_arm*; Holm-adjusted across arms within each metric."""
    if ref_arm not in set(df["arm"]):
        raise ValueError(f"reference arm {ref_arm!r} not in data (arms: {sorted(set(df['arm']))})")
    arms = [a for a in (arms or sorted(set(df["arm"]))) if a != ref_arm]
    rows = []
    for arm in arms:
        m = matched(df, ref_arm, arm)
        n = len(m)
        if n == 0:
            logger.warning("no matched (cell_id, rng_seed) pairs between %s and %s", ref_arm, arm)
            continue
        mc = mcnemar_exact(m["crossed_ref"], m["crossed_arm"])
        rows.append(dict(arm=arm, ref=ref_arm, metric="crossed", n_pairs=n,
                         ref_value=float(m["crossed_ref"].mean()), arm_value=float(m["crossed_arm"].mean()),
                         p=mc["p"], effect=mc["diff"], effect_name="share diff",
                         b_ref_only=mc["b_ref_only"], c_arm_only=mc["c_arm_only"]))
        for metric, ref_v, arm_v in (
            ("first_cross_eval", m["first_cross_eval_ref"], m["first_cross_eval_arm"]),
            ("floor", np.log10(m["floor_nats_ref"].clip(lower=1e-300)),
             np.log10(m["floor_nats_arm"].clip(lower=1e-300))),
        ):
            w = wilcoxon_paired(ref_v.to_numpy(), arm_v.to_numpy())
            rows.append(dict(arm=arm, ref=ref_arm, metric=metric, n_pairs=n,
                             ref_value=float(np.median(ref_v)), arm_value=float(np.median(arm_v)),
                             p=w["p"], effect=w["r_rb"], effect_name="rank-biserial r",
                             n_nonzero=w["n_nonzero"], median_diff=w["median_diff"]))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        for metric in METRICS:
            idx = out.index[out["metric"] == metric]
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def _fmt_p(p: float) -> str:
    if p != p:
        return "--"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def _booktabs(colspec: str, header: list[str], rows: list[list[str]], caption: str | None = None,
              label: str | None = None) -> str:
    lines = []
    if caption:
        lines += [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}"]
        if label:
            lines.append(rf"\label{{{label}}}")
    lines += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def summary_tex(s: pd.DataFrame) -> str:
    rows = []
    for _, r in s.iterrows():
        rows.append([
            tex_escape(r.arm), str(int(r.n_runs)),
            f"{int(r.n_crossed)} ({100 * r.share_crossed:.1f}) [{100 * r.wilson_lo:.0f}--{100 * r.wilson_hi:.0f}]",
            f"{100 * r.share_gen0:.1f}",
            f"{r.median_first_cross_eval:.0f} [{r.mfc_lo:.0f}--{r.mfc_hi:.0f}]",
            f"{r.median_floor_nats:.2g}",
            f"{100 * r.share_band:.1f}",
        ])
    return _booktabs("lrrrrrr", ["Arm", "Runs", r"Crossed (\%) [95\% CI]", r"Gen.\,0 (\%)",
                                 "Med. evals to cross [95\\% CI]", "Med. floor (nats)", r"$\le10^{-2}$ (\%)"], rows)


def budgets_tex(curve: pd.DataFrame, budgets: Sequence[int]) -> str:
    arms = list(dict.fromkeys(curve["arm"]))
    rows = []
    for arm in arms:
        c = curve[curve["arm"] == arm].set_index("budget")
        rows.append([tex_escape(arm)] + [
            f"{100 * c.loc[b, 'share']:.1f} [{100 * c.loc[b, 'boot_lo']:.0f}--{100 * c.loc[b, 'boot_hi']:.0f}]"
            for b in budgets])
    return _booktabs("l" + "r" * len(budgets), ["Arm"] + [f"{b}" for b in budgets], rows)


def paired_tex(p: pd.DataFrame) -> str:
    names = {"crossed": "crossed", "first_cross_eval": "evals to cross", "floor": r"$\log_{10}$ floor"}
    rows = []
    for _, r in p.iterrows():
        eff = f"{r.effect:+.2f}"
        extra = f" ({int(r.b_ref_only)}/{int(r.c_arm_only)})" if r.metric == "crossed" else ""
        rows.append([tex_escape(r.arm), names[r.metric], str(int(r.n_pairs)), _fmt_p(r.p), _fmt_p(r.p_holm),
                     eff + extra])
    return _booktabs("llrrrr", ["Arm", "Metric", "Pairs", "$p$", r"$p_{\text{Holm}}$",
                                r"Effect (arm $-$ ref)"], rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_all(df: pd.DataFrame, ref_arm: str | None, out: Path, budgets: Sequence[int] = DEFAULT_BUDGETS,
            n_boot: int = N_BOOT, cluster_col: str | None = None) -> dict[str, pd.DataFrame]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict[str, pd.DataFrame] = {}
    res["summary"] = arm_summary(df, n_boot)
    res["curve"] = curve_with_bands(df, budgets, n_boot, cluster_col=cluster_col)
    res["summary"].to_csv(out / "summary.csv", index=False)
    res["curve"].to_csv(out / "curve.csv", index=False)
    (out / "summary.tex").write_text(summary_tex(res["summary"]))
    (out / "curve_budgets.tex").write_text(budgets_tex(res["curve"], budgets))
    if ref_arm:
        res["paired"] = paired_vs_reference(df, ref_arm)
        res["paired"].to_csv(out / "paired.csv", index=False)
        (out / "paired.tex").write_text(paired_tex(res["paired"]))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wilson CIs, paired tests vs a reference arm, LaTeX tables.")
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--ref", default=None, help="reference arm id (e.g. a0_directed); omit to skip paired tests")
    ap.add_argument("--out", type=Path, default=Path("results/tables"))
    ap.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--cluster", default=None, help="column to cluster the bootstrap on (e.g. cell_id)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = load_results(args.csv)
    if df["arm"].isna().any():
        raise SystemExit("some runs have no arm; re-run collect with --default arm=<id>")
    budgets = [int(b) for b in args.budgets.split(",")]
    run_all(df, args.ref, args.out, budgets, args.n_boot, args.cluster)
    logger.info("tables written to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
