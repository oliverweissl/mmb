"""Results -> paper: tables, figures and a numbers.tex of macros, written into <paper>/generated/.

Usage::

    python -m mmb.analysis.export results/main.csv --paper ../paper_repo [--validity results/validity_main.csv]
        [--runs-root runs/MAIN] [--ref a0_directed] [--grid-suts qwen,internvl,ministral] [--n-boot 10000]

(scripts/analysis/export_to_paper.sh runs collect + validity first and then calls this.)

Scopes. 'grid' = the balanced grid (SUTs in --grid-suts, every arm); the continuity SUT (llava) only has
a0/a1 on ImageNet and appears in the per-SUT table E3 and in the per-(task, SUT) tables. Pooling over tasks or
SUTs prefixes cell_id with task/SUT, so pairs (cell, seed) stay unique.

Written under <paper>/generated/:
  tables/e1.tex e1_budgets.tex e1_paired.tex   grid pooled, every arm (Tables e1, e1-budgets, e1-paired)
  tables/e3.tex                                a0 and a1 per SUT, ImageNet task, continuity SUT included
  tables/e4.tex                                a0 and a1 per task, grid SUTs pooled
  tables/e2b.tex                               cell-level predictor table per task and SUT (arm a0)
  tables/primary.tex                           pre-committed primary comparison a1 vs a0 per (task, SUT), Holm across strata
  tables/e1_cluster.tex cost.tex               paired differences with cell-cluster CIs; cost of crossing per arm
  figures/per-cell-diff.pdf                    per-cell a0 minus a1 crossing share
  tables/validity.tex                          proxy medians per arm (only with --validity)
  tables/transfer*.tex, ETransfer macros      cross-SUT transfer of crossings (only with --transfer)
  tables/agree.tex, EAgree macros              greedy agreement and re-verification (only with --agree)
  tables/per/<task>_<sut>_{summary,paired,budgets}.tex
  figures/crossing-curve.pdf predictor-main.pdf gallery.pdf
  numbers.tex numbers_map.csv                  one macro per headline number; names have no digits
Every table file holds one adjustbox-wrapped tabular; captions stay in the paper.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mmb.analysis import paired as pr
from mmb.analysis import predictors as pred
from mmb.analysis import transfer as tr
from mmb.analysis import stats as st

logger = logging.getLogger(__name__)

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
GRID_SUTS = ("qwen", "internvl", "ministral")
PRIMARY_ARMS = ("a0_directed", "a1_random_tiered")
SUT_NAMES = {"qwen": "Qwen3.5-4B (bf16)", "internvl": "InternVL3.5-4B (bf16)",
             "ministral": "Ministral-3-3B (bf16)", "llava": "LLaVA-NeXT-7B (NF4)"}
TASK_NAMES = {"imagenet": "ImageNet class pairs", "pope": "POPE yes/no"}
WRAP = "\\begin{adjustbox}{max width=\\linewidth}%s\\end{adjustbox}\n"


# ---------------------------------------------------------------------------
# macro names and values
# ---------------------------------------------------------------------------


def camel(s: str) -> str:
    """'probe_gen0' -> 'ProbeGenZero': letters only, digits spelled out."""
    out = []
    for tok in re.findall(r"[A-Za-z]+|[0-9]", str(s)):
        out.append(DIGITS[int(tok)] if tok.isdigit() else tok[:1].upper() + tok[1:].lower())
    return "".join(out)


def arm_short(arm: str) -> str:
    """'a0_directed' -> 'AZero' (the arm id only)."""
    return camel(str(arm).split("_")[0])


def macro_name(*parts: str) -> str:
    name = "".join(p if re.fullmatch(r"[A-Za-z]+", p) and p[:1].isupper() else camel(p) for p in parts)
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"bad macro name {name!r}")
    return name


def tex_value(v: str) -> str:
    """Make a plain-text number safe for text mode (minus sign, comparison signs)."""
    v = str(v).replace("<", r"\ensuremath{<}").replace(">", r"\ensuremath{>}")
    return re.sub(r"(?<![\w}\-])-(?=\d)", r"\\ensuremath{-}", v)


def f_pct(x: float, d: int = 1) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}"


def f_num(x: float, d: int = 2, sign: bool = False) -> str:
    return "n/a" if pd.isna(x) else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def f_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


class Numbers:
    """Collects (macro, value, description) triples; writes numbers.tex and the mapping CSV."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.seen: set[str] = set()

    def add(self, parts: Iterable[str], value: str, desc: str, **meta) -> None:
        name = macro_name(*parts)
        if name in self.seen:
            raise ValueError(f"duplicate macro {name}")
        self.seen.add(name)
        self.rows.append(dict(macro=name, value=str(value), description=desc, **meta))

    def write(self, out: Path) -> None:
        df = pd.DataFrame(self.rows)
        df.to_csv(out / "numbers_map.csv", index=False)
        lines = ["% Generated by mmb.analysis.export; do not edit. One macro per headline number.",
                 "% Use \\res{Name} in the text: it prints [TBD] while a macro is undefined."]
        lines += [f"\\newcommand{{\\{r['macro']}}}{{{tex_value(r['value'])}}}" for r in self.rows]
        (out / "numbers.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# scopes and tables
# ---------------------------------------------------------------------------


def with_scope_cells(df: pd.DataFrame) -> pd.DataFrame:
    """cell_id made unique across tasks and SUTs, so that pooled pairs (cell, seed) never collide."""
    d = df.copy()
    d["cluster_id"] = d["task"].astype(str) + "/" + d["cell_id"].astype(str)     # a cell is shared by SUTs
    d["cell_id"] = d["task"].astype(str) + "/" + d["sut"].astype(str) + "/" + d["cell_id"].astype(str)
    return d


def scopes(df: pd.DataFrame, grid: Iterable[str]) -> dict[tuple[str, str], pd.DataFrame]:
    """{(sut_label, task_label): frame}. 'All' pools; the grid scope excludes the continuity SUT."""
    grid = set(grid)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    g = df[df["sut"].isin(grid)]
    if len(g):
        out[("All", "All")] = with_scope_cells(g)
    for task in sorted(set(g["task"])):
        out[("All", task)] = with_scope_cells(g[g["task"] == task])
    for sut in sorted(set(df["sut"])):
        s = df[df["sut"] == sut]
        if sut in grid and s["task"].nunique() > 1:
            out[(sut, "All")] = with_scope_cells(s)
        for task in sorted(set(s["task"])):
            out[(sut, task)] = s[s["task"] == task]
    return out


def keyed_summary_tex(frames: list[tuple[list[str], pd.DataFrame]], key_headers: list[str]) -> str:
    """Summary table with leading key columns; a midrule between key groups."""
    header = key_headers + ["Arm", "Runs", r"Crossed (\%) [95\% CI]", r"Gen.\,0 (\%)",
                            r"Med.\ evals to cross [95\% CI]", r"Med.\ floor (nats)", r"$\le10^{-2}$ (\%)"]
    body: list[str] = []
    for gi, (keys, s) in enumerate(frames):
        if gi:
            body.append(r"\midrule")
        for ri, (_, r) in enumerate(s.iterrows()):
            ks = [st.tex_escape(k) if ri == 0 else "" for k in keys]
            body.append(" & ".join(ks + [
                st.tex_escape(st.arm_label(r.arm)), str(int(r.n_runs)),
                f"{int(r.n_crossed)} ({100 * r.share_crossed:.1f}) [{100 * r.wilson_lo:.0f}--{100 * r.wilson_hi:.0f}]",
                f"{100 * r.share_gen0:.1f}",
                f"{r.median_first_cross_eval:.0f} [{r.mfc_lo:.0f}--{r.mfc_hi:.0f}]",
                f"{r.median_floor_nats:.2g}", f"{100 * r.share_band:.1f}"]) + r" \\")
    colspec = "l" * (len(key_headers) + 1) + "rrrrrr"
    return ("\\begin{tabular}{@{}" + colspec + "@{}}\n\\toprule\n" + " & ".join(header) + " \\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}")


def wrapped(tabular: str) -> str:
    return WRAP % tabular.strip()


def write_tex(path: Path, tabular: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(wrapped(tabular))


# ---------------------------------------------------------------------------
# E2b table
# ---------------------------------------------------------------------------


def e2b_table(cells: pd.DataFrame, n_boot: int) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    keys = ["task", "sut"]
    corr = pred.correlations(cells, "cell", keys, n_boot)
    part = pred.partials(cells, "cell", keys, n_boot)
    rows = []
    for (task, sut), _ in cells.groupby(keys):
        def pick(p, out, stat):
            if corr.empty:
                return "--"
            r = corr[(corr.task == task) & (corr.sut == sut) & (corr.predictor == p) & (corr.outcome == out)
                     & (corr.stat == stat)]
            return pred._fmt(r["value"].iloc[0], r["lo"].iloc[0], r["hi"].iloc[0]) if len(r) else "--"
        pr = part if part.empty else part[(part.task == task) & (part.sut == sut) & (part.predictor == "probe_gen0")
                  & (part.outcome == "floor_nats")] if len(part) else part
        pv = "--"
        if len(pr):
            r0 = pr.iloc[0]
            pv = pred._fmt(r0["rho_partial"], r0["lo"], r0["hi"])
        n = int(cells[(cells.task == task) & (cells.sut == sut)].shape[0])
        rows.append([st.tex_escape(TASK_NAMES.get(task, task)), st.tex_escape(SUT_NAMES.get(sut, sut)), str(n),
                     pick("probe_gen0", "floor_nats", "spearman"), pick("abs_prior_g", "floor_nats", "spearman"),
                     pick("min_gen0", "floor_nats", "spearman"), pick("probe_gen0", "crossed", "auc"),
                     pick("abs_prior_g", "crossed", "auc"), pv])
    tab = st._booktabs("llrllllll", ["Task", "SUT", "Cells", r"$\rho$ floor: probe", r"$\rho$ floor: $|$prior$|$",
                                     r"$\rho$ floor: gen-0 min", "AUC crossed: probe", r"AUC crossed: $|$prior$|$",
                                     r"partial $\rho$ probe"], rows)
    return tab, corr, part


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------


def add_summary_numbers(num: Numbers, sut: str, task: str, summ: pd.DataFrame) -> None:
    for _, r in summ.iterrows():
        a = arm_short(r.arm)
        sc = dict(sut=sut, task=task, arm=r.arm)
        base = ["EOne"]
        tail = [a, camel(sut), camel(task)]
        ci = f"[{100 * r.wilson_lo:.0f}--{100 * r.wilson_hi:.0f}]"
        items = [
            ("Runs", str(int(r.n_runs)), "number of runs"),
            ("Crossed", str(int(r.n_crossed)), "number of crossed runs"),
            ("CrossShare", f_pct(r.share_crossed), "share of runs crossed (%)"),
            ("CrossLo", f_pct(r.wilson_lo, 0), "Wilson 95% CI lower (%)"),
            ("CrossHi", f_pct(r.wilson_hi, 0), "Wilson 95% CI upper (%)"),
            ("CrossCi", ci, "Wilson 95% CI (%)"),
            ("GenZeroShare", f_pct(r.share_gen0), "share crossed in generation 0 (%)"),
            ("MedEvals", f_num(r.median_first_cross_eval, 0), "median evaluations to first crossing (censored at budget)"),
            ("MedEvalsCi", f"[{r.mfc_lo:.0f}--{r.mfc_hi:.0f}]", "bootstrap 95% CI of the median evaluations"),
            ("MedFloor", f"{r.median_floor_nats:.2g}", "median floor min|g| (nats)"),
            ("BandShare", f_pct(r.share_band), "share reaching |g| <= 1e-2 nats (%)"),
            ("LabelFlipShare", f_pct(r.share_label_flip), "share with a label flip (%)"),
        ]
        for m, v, d in items:
            num.add(base + [m] + tail, v, f"{d}; {r.arm}, {sut}, {task}", metric=m, **sc)


def add_curve_numbers(num: Numbers, sut: str, task: str, curve: pd.DataFrame) -> None:
    for _, r in curve.iterrows():
        tail = [arm_short(r.arm), camel(sut), camel(task)]
        d = f"share of runs crossed within {int(r.budget)} evaluations (%); {r.arm}, {sut}, {task}"
        num.add(["EOne", "AtBudget", str(int(r.budget))] + tail, f_pct(r.share), d)
        num.add(["EOne", "AtBudgetCi", str(int(r.budget))] + tail,
                f"[{100 * r.boot_lo:.0f}--{100 * r.boot_hi:.0f}]", d + ", bootstrap 95% CI")


def add_paired_numbers(num: Numbers, sut: str, task: str, paired: pd.DataFrame) -> None:
    for _, r in paired.iterrows():
        a = arm_short(r.arm)
        tail = [a, camel(sut), camel(task)]
        met = {"crossed": "Crossed", "first_cross_eval": "Evals", "floor": "Floor"}[r.metric]
        desc = f"{r.arm} vs {r.ref} on {int(r.n_pairs)} matched pairs, {sut}, {task}"
        num.add(["EOne", "Pairs", met] + tail, str(int(r.n_pairs)), "matched pairs; " + desc)
        num.add(["EOne", "P", met] + tail, f_p(r.p), "raw p; " + desc)
        num.add(["EOne", "PHolm", met] + tail, f_p(r.p_holm), "Holm-adjusted p; " + desc)
        num.add(["EOne", "Effect", met] + tail, f_num(r.effect, 2, True), f"{r.effect_name}; " + desc)
        if r.metric == "crossed":
            num.add(["EOne", "DiscRef", met] + tail, str(int(r.b_ref_only)), "pairs crossed only by the reference; " + desc)
            num.add(["EOne", "DiscArm", met] + tail, str(int(r.c_arm_only)), "pairs crossed only by the arm; " + desc)


def add_predictor_numbers(num: Numbers, corr: pd.DataFrame, part: pd.DataFrame) -> None:
    for _, r in corr.iterrows():
        stat = "Rho" if r.stat == "spearman" else "Auc"
        tail = [stat, camel(r.outcome), camel(r.predictor), camel(r.sut), camel(r.task)]
        d = f"cell-level {r.stat} of {r.predictor} with {r.outcome}, {r.sut}, {r.task}, n={int(r.n)}"
        num.add(["ETwoB"] + tail, f_num(r.value, 2, True), d)
        num.add(["ETwoB", "Ci"] + tail, pred._fmt(r.value, r.lo, r.hi), d + " (with 95% CI)")
    for _, r in part.iterrows():
        tail = [camel(r.predictor), camel(r.given), camel(r.outcome), camel(r.sut), camel(r.task)]
        num.add(["ETwoB", "Partial"] + tail, f_num(r.rho_partial, 2, True),
                f"partial rho of {r.predictor} given {r.given} with {r.outcome}, {r.sut}, {r.task}")


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(("true", "1", "1.0"))


def load_agree(paths: Iterable[Path]) -> pd.DataFrame:
    """scripts/agreement_replay.py CSVs (one per SUT) -> one frame with boolean columns."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for c in ("anchor_agree", "cross_agree", "cross_verified"):
        if c in df:
            df[c] = df[c].where(df[c].isna(), _bool(df[c]))
    return df


AGREE_MEASURES = (("anchor_agree", "Anchor", "teacher-forced argmax equals the greedy answer on the anchor"),
                  ("cross_agree", "Cross", "teacher-forced argmax equals the greedy answer on the crossing input"),
                  ("cross_verified", "Verified", "crossing survives single-item re-scoring"))


def agree_rows(df: pd.DataFrame) -> list[dict]:
    """Shares with Wilson CIs per (sut, task) scope: per SUT, per task, pooled and per cell of both."""
    rows = []
    suts = sorted(set(df["sut"]))
    tasks = sorted(set(df["task"]))
    scopes_ = [("All", "All", df)]
    scopes_ += [("All", t, df[df["task"] == t]) for t in tasks]
    for s in suts:
        scopes_.append((s, "All", df[df["sut"] == s]))
        scopes_ += [(s, t, df[(df["sut"] == s) & (df["task"] == t)]) for t in tasks]
    for sut, task, d in scopes_:
        for col, tag, desc in AGREE_MEASURES:
            if col not in d:
                continue
            v = d[col].dropna()
            if not len(v):
                continue
            k, n = int(v.astype(bool).sum()), len(v)
            lo, hi = st.wilson(k, n)
            rows.append(dict(sut=sut, task=task, tag=tag, desc=desc, k=k, n=n, share=k / n, lo=lo, hi=hi))
    return rows


def add_agree_numbers(num: Numbers, df: pd.DataFrame) -> None:
    for r in agree_rows(df):
        tail = [camel(r["sut"]), camel(r["task"])]
        d = f"{r['desc']}; {r['sut']}, {r['task']}"
        num.add(["EAgree", r["tag"], "Count"] + tail, str(r["k"]), "count; " + d)
        num.add(["EAgree", r["tag"], "N"] + tail, str(r["n"]), "denominator; " + d)
        num.add(["EAgree", r["tag"], "Share"] + tail, f_pct(r["share"]), "share (%); " + d)
        num.add(["EAgree", r["tag"], "Ci"] + tail, f"[{100 * r['lo']:.0f}--{100 * r['hi']:.0f}]", "Wilson 95% CI (%); " + d)


def agree_table(df: pd.DataFrame) -> str:
    rows = []
    by = {(r["sut"], r["task"], r["tag"]): r for r in agree_rows(df)}
    keys = sorted({(r["sut"], r["task"]) for r in by.values() if r["sut"] != "All"} | {("All", "All")})
    def cell(k, tag):
        r = by.get((k[0], k[1], tag))
        return "--" if r is None else f"{100 * r['share']:.1f} [{100 * r['lo']:.0f}--{100 * r['hi']:.0f}]"
    for k in keys:
        n = by.get((k[0], k[1], "Cross"))
        rows.append([st.tex_escape(SUT_NAMES.get(k[0], k[0])), st.tex_escape(TASK_NAMES.get(k[1], k[1])),
                     str(n["n"]) if n else "--", cell(k, "Anchor"), cell(k, "Cross"), cell(k, "Verified")])
    return st._booktabs("llrrrr", ["SUT", "Task", "Crossings", r"Anchor agrees (\%)", r"Crossing agrees (\%)",
                                   r"Verified by re-scoring (\%)"], rows)


PAIRED_METRIC_NAMES = {"crossed": "Crossed", "first_cross_eval": "Evals", "floor": "Floor", "min_edits": "MinEdits",
                      "hv_edits": "HvEdits", "hv_mag": "HvMag", "img_mag": "ImgMag", "txt_mag": "TxtMag",
                      "chardist": "CharDist"}
METRIC_HEAD = {"crossed": "crossed (share)", "first_cross_eval": "evals to cross", "floor": r"$\log_{10}$ floor",
               "min_edits": "fewest edits", "hv_edits": "edit hypervolume"}


def _ci(v: float, lo: float, hi: float, d: int = 2) -> str:
    return "--" if pd.isna(v) else f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def add_cluster_numbers(num: Numbers, sut: str, task: str, t: pd.DataFrame) -> None:
    for _, r in t.iterrows():
        tail = [PAIRED_METRIC_NAMES[r.metric], arm_short(r.arm), camel(sut), camel(task)]
        d = f"{r.arm} vs {r.ref}, {r.metric}, {sut}, {task}, {int(r.n_pairs)} pairs in {int(r.n_cells)} cells"
        for f, v, what in (("Pairs", str(int(r.n_pairs)), "matched pairs"),
                           ("RefValue", f_num(r.ref_value, 3), "reference value (share or median)"),
                           ("ArmValue", f_num(r.arm_value, 3), "arm value (share or median)"),
                           ("Diff", f_num(r["diff"], 3, True), r.diff_name + " (arm - ref)"),
                           ("DiffLo", f_num(r.diff_lo, 3, True), "cell-cluster bootstrap 95% CI lower"),
                           ("DiffHi", f_num(r.diff_hi, 3, True), "cell-cluster bootstrap 95% CI upper"),
                           ("P", f_p(r.p), "raw p (exact McNemar / Wilcoxon)"),
                           ("PHolm", f_p(r.p_holm), "Holm-adjusted p across arms")):
            num.add(["ETwoA", f] + tail, v, f"{what}; {d}")


def cluster_table_tex(t: pd.DataFrame, metrics=pr.DEFAULT_METRICS) -> str:
    rows = []
    for arm in list(dict.fromkeys(t["arm"])):
        row = [st.tex_escape(st.arm_label(arm))]
        for m in metrics:
            r = t[(t.arm == arm) & (t.metric == m)]
            row.append("--" if r.empty else f"{_ci(r.iloc[0]['diff'], r.iloc[0].diff_lo, r.iloc[0].diff_hi, 2 if m in ('crossed', 'floor', 'hv_edits') else 0)}; {f_p(r.iloc[0].p_holm)}")
        rows.append(row)
    return st._booktabs("l" + "l" * len(metrics), ["Arm vs directed"] + [METRIC_HEAD.get(m, m) for m in metrics], rows)


def primary_tex(t: pd.DataFrame, short: str = "a1") -> str:
    rows = []
    for _, r in t.iterrows():
        rows.append([st.tex_escape(TASK_NAMES.get(r.task, r.task)), st.tex_escape(SUT_NAMES.get(r.sut, r.sut)),
                     str(int(r.n_pairs)), f_pct(r.ref_value), f_pct(r.arm_value),
                     _ci(100 * r["diff"], 100 * r.diff_lo, 100 * r.diff_hi, 1), f_p(r.p), f_p(r.p_holm_strata)])
    return st._booktabs("llrrrrrr", ["Task", "SUT", "Pairs", "Directed (\\%)", f"{st.arm_label(short)} (\\%)", f"{st.arm_label(short)} $-$ directed (pp) [95\\% CI]", "$p$",
                                     r"$p_{\text{Holm}}$"], rows)


def add_primary_numbers(num: Numbers, t: pd.DataFrame, family: str = "EPrimary", short: str = "a1") -> None:
    for _, r in t.iterrows():
        tail = [camel(r.sut), camel(r.task)]
        d = f"{'primary' if family == 'EPrimary' else 'secondary'} comparison {short} vs a0 crossing share, {r.sut}, {r.task}"
        for f, v, what in (("Pairs", str(int(r.n_pairs)), "matched pairs"),
                           ("RefShare", f_pct(r.ref_value), "a0 crossing share (%)"),
                           ("ArmShare", f_pct(r.arm_value), f"{short} crossing share (%)"),
                           ("Diff", f_num(100 * r["diff"], 1, True), f"{short} minus a0 (percentage points)"),
                           ("DiffLo", f_num(100 * r.diff_lo, 1, True), "cell-cluster bootstrap 95% CI lower (pp)"),
                           ("DiffHi", f_num(100 * r.diff_hi, 1, True), "cell-cluster bootstrap 95% CI upper (pp)"),
                           ("P", f_p(r.p), "exact McNemar p"),
                           ("PHolm", f_p(r.p_holm_strata), "Holm-adjusted across strata")):
            num.add([family, f] + tail, v, f"{what}; {d}")


def cost_summary(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for arm, g in df.groupby("arm", sort=True):
        me, lo, hi = st.bootstrap_stat(g["min_edits_cross"].to_numpy(float), n_boot=n_boot) if "min_edits_cross" in g else (np.nan,) * 3
        hv, hlo, hhi = st.bootstrap_stat(g["hv_edits_cross"].to_numpy(float), n_boot=n_boot) if "hv_edits_cross" in g else (np.nan,) * 3
        c = g[g["crossed"].astype(bool)]
        if "mag_source" in c and (c["mag_source"] == "diag").any():
            c = c[c["mag_source"] == "diag"]            # never mix diagnostic and optimised-fitness magnitudes
        rows.append(dict(arm=arm, n_runs=len(g), n_crossed=len(c), med_min_edits=me, me_lo=lo, me_hi=hi,
                         med_hv=hv, hv_lo=hlo, hv_hi=hhi,
                         med_img_mag=float(c["best_img_mag_cross"].median()) if "best_img_mag_cross" in c and c["best_img_mag_cross"].notna().any() else np.nan,
                         med_txt_mag=float(c["best_txt_mag_cross"].median()) if "best_txt_mag_cross" in c and c["best_txt_mag_cross"].notna().any() else np.nan))
    return pd.DataFrame(rows)


def cost_tex(s: pd.DataFrame) -> str:
    rows = []
    for _, r in s.iterrows():
        rows.append([st.tex_escape(st.arm_label(r.arm)), str(int(r.n_runs)), f"{r.med_min_edits:.0f} [{r.me_lo:.0f}--{r.me_hi:.0f}]",
                     f"{r.med_hv:.3f} [{r.hv_lo:.3f}--{r.hv_hi:.3f}]",
                     "--" if pd.isna(r.med_img_mag) else f"{r.med_img_mag:.3f}",
                     "--" if pd.isna(r.med_txt_mag) else f"{r.med_txt_mag:.2f}"])
    return st._booktabs("lrrrrr", ["Arm", "Runs", "Fewest edits at a crossing [95\\% CI]", "Edit hypervolume [95\\% CI]",
                                   "Best image RMS", "Best text cosine"], rows)


def add_cost_numbers(num: Numbers, sut: str, task: str, s: pd.DataFrame) -> None:
    for _, r in s.iterrows():
        tail = [arm_short(r.arm), camel(sut), camel(task)]
        d = f"{r.arm}, {sut}, {task}"
        num.add(["ECost", "MedMinEdits"] + tail, f_num(r.med_min_edits, 0), "median fewest active genes at a crossing (never-crossed runs censored at all genes); " + d)
        num.add(["ECost", "MedMinEditsCi"] + tail, f"[{r.me_lo:.0f}--{r.me_hi:.0f}]", "bootstrap 95% CI; " + d)
        num.add(["ECost", "MedHv"] + tail, f_num(r.med_hv, 3), "median edit hypervolume fraction (0 when never crossed); " + d)
        num.add(["ECost", "MedHvCi"] + tail, f"[{r.hv_lo:.3f}--{r.hv_hi:.3f}]", "bootstrap 95% CI; " + d)
        if not pd.isna(r.med_img_mag):
            num.add(["ECost", "MedImgMag"] + tail, f_num(r.med_img_mag, 3), "median best image RMS among crossing runs; " + d)
        if not pd.isna(r.med_txt_mag):
            num.add(["ECost", "MedTxtMag"] + tail, f_num(r.med_txt_mag, 2), "median best text cosine among crossing runs; " + d)


def add_mixed_numbers(num: Numbers, res: dict) -> None:
    if pd.isna(res["log_or"]):
        return
    tail = [arm_short(res["arm"])]
    d = f"mixed logit crossed ~ arm + (1|cell) + (1|sut:task), {res['arm']} vs {res['ref']}, {res['n_runs']} runs"
    num.add(["EMixed", "LogOr"] + tail, f_num(res["log_or"], 2, True), "log odds ratio (posterior mean); " + d)
    num.add(["EMixed", "LogOrCi"] + tail, f"[{res['lo']:+.2f}, {res['hi']:+.2f}]", "approximate 95% interval; " + d)
    num.add(["EMixed", "OddsRatio"] + tail, f_num(res["odds_ratio"], 2), "odds ratio; " + d)


def add_transfer_numbers(num: Numbers, rows: list[dict]) -> None:
    for r in rows:
        tail = [camel(r["src"]), camel(r["dst"]), camel(r["task"])]
        d = f"crossings found on {r['src']} replayed on {r['dst']}, {r['task']}"
        num.add(["ETransfer", "Count"] + tail, str(r["k"]), "flipped on the destination; " + d)
        num.add(["ETransfer", "N"] + tail, str(r["n"]), "crossings replayed; " + d)
        num.add(["ETransfer", "Share"] + tail, f_pct(r["share"]), "share (%); " + d)
        num.add(["ETransfer", "Ci"] + tail, f"[{100 * r['lo']:.0f}--{100 * r['hi']:.0f}]", "Wilson 95% CI (%); " + d)
    for task in sorted({r["task"] for r in rows}):
        m, n = tr.mean_off_diagonal(rows, task)
        if n:
            num.add(["ETransfer", "MeanOffDiagonal", camel(task)], f_pct(m), f"pooled share of crossings flipping another SUT (%), {task}")
            num.add(["ETransfer", "MeanOffDiagonalN", camel(task)], str(n), f"crossing-destination pairs behind the pooled share, {task}")


def human_tex(s: pd.DataFrame) -> str:
    rows = []
    for _, r in s.iterrows():
        rows.append([st.tex_escape(r["stratum"]), st.tex_escape(TASK_NAMES.get(r["task"], r["task"])),
                     st.tex_escape(st.arm_label(r["arm"]) if isinstance(r["arm"], str) and r["arm"] else "unedited anchor"),
                     str(int(r["items"])), f"{r['valid_share']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]",
                     f"{r['mean_comprehensibility']:.1f}", f"{100 * r['cant_share']:.0f}"])
    return st._booktabs("lllrrrr", ["Stratum", "Task", "Arm", "Items", "Valid share [95\\% CI]", "Comprehens.", "Can't tell (\\%)"], rows)


def add_human_numbers(num: Numbers, s: pd.DataFrame) -> None:
    for _, r in s.iterrows():
        arm = r["arm"] if isinstance(r["arm"], str) and r["arm"] else "control"
        tail = [camel(r["stratum"]), camel(r["task"]), arm_short(arm) if arm != "control" else "Control"]
        d = f"human validity, {r['stratum']}, {r['task']}, {arm}, {int(r['items'])} items, {int(r['raters'])} raters"
        num.add(["EHuman", "Share"] + tail, f_pct(r["valid_share"]), "valid share (%); " + d)
        num.add(["EHuman", "Ci"] + tail, f"[{100 * r['lo']:.0f}--{100 * r['hi']:.0f}]", "95% bootstrap CI over items (%); " + d)
        num.add(["EHuman", "Items"] + tail, str(int(r["items"])), "items; " + d)
        num.add(["EHuman", "Comprehensibility"] + tail, f_num(r["mean_comprehensibility"], 1), "mean 5-point comprehensibility; " + d)


def add_validity_numbers(num: Numbers, vdf: pd.DataFrame) -> None:
    cols = [c for c in ("img_clip_cos", "img_dino_cos", "img_lpips", "img_ssim", "txt_minilm_cos",
                        "txt_word_edit") if c in vdf]
    for arm, g in vdf.groupby("arm"):
        for c in cols:
            num.add(["ESeven", "Median", c, arm_short(arm)], f_num(float(g[c].median()), 3),
                    f"median {c} of crossing inputs, arm {arm}")
        n_all = len(g)
        if "question_preserved" in g and "question_keyword_sim" in g:
            g = g[g["question_keyword_sim"].notna()]    # defined with a protected keyword only (the POPE object); class-pair prompts have none
        if "question_preserved" in g and len(g):
            qp = g["question_preserved"].astype(str).str.lower().isin(("true", "1", "1.0"))
            num.add(["ESeven", "QuestionPreserved", arm_short(arm)], f_pct(float(qp.mean())),
                    f"share of crossing inputs whose question heuristic is preserved (%), arm {arm}")
        num.add(["ESeven", "Crossings", arm_short(arm)], str(n_all), f"crossing inputs scored, arm {arm}")


def validity_table(vdf: pd.DataFrame) -> str:
    cols = [("img_clip_cos", "CLIP cos"), ("img_dino_cos", "DINOv2 cos"), ("img_lpips", "LPIPS"),
            ("img_ssim", "SSIM"), ("txt_minilm_cos", "MiniLM cos"), ("txt_word_edit", "word edits")]
    cols = [(c, h) for c, h in cols if c in vdf]
    rows = []
    for arm, g in vdf.groupby("arm"):
        rows.append([st.tex_escape(st.arm_label(arm)), str(len(g))] + [f"{g[c].median():.2f}" for c, _ in cols])
    return st._booktabs("lr" + "r" * len(cols), ["Arm", "Crossings"] + [h for _, h in cols], rows)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def export(csv_paths: list[Path], paper: Path, *, ref: str = "a0_directed", grid_suts=GRID_SUTS,
           validity: Path | None = None, runs_root: Path | None = None, n_boot: int = st.N_BOOT,
           figures: bool = True, agree: list[Path] | None = None,
           transfer: list[Path] | None = None, human: Path | None = None) -> Numbers:
    from mmb.analysis import gallery, plots

    gen = Path(paper) / "generated"
    tabs, figs = gen / "tables", gen / "figures"
    tabs.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df = st.load_results(csv_paths)
    if df["arm"].isna().any():
        raise SystemExit("some runs have no arm; re-run collect with --default arm=<id>")
    num = Numbers()
    sc = scopes(df, grid_suts)
    budgets = list(st.DEFAULT_BUDGETS)
    num.add(["EOne", "Runs", "Total"], str(len(df)), "runs in the results CSV")
    if "complete" in df:
        num.add(["EOne", "Runs", "Incomplete"], str(int((~df["complete"]).sum())), "runs with missing generations")

    for (sut, task), d in sc.items():
        arms = set(d["arm"])
        res = st.run_all(d, ref if ref in arms else None, _tmpdir(gen, sut, task),
                         budgets, n_boot)
        add_summary_numbers(num, sut, task, res["summary"])
        add_curve_numbers(num, sut, task, res["curve"])
        if ref in arms:
            ct = pr.paired_table(d, ref, pr.DEFAULT_METRICS, n_boot=n_boot)
            add_cluster_numbers(num, sut, task, ct)
            cs_ = cost_summary(d, n_boot)
            add_cost_numbers(num, sut, task, cs_)
            if (sut, task) == ("All", "All"):
                if len(ct):
                    write_tex(tabs / "e1_cluster.tex", cluster_table_tex(ct))
                write_tex(tabs / "cost.tex", cost_tex(cs_))
                for a in sorted(arms - {ref}):
                    add_mixed_numbers(num, pr.mixed_model(d.assign(cell_id=d["cluster_id"]), ref, a))
        if "paired" in res:
            add_paired_numbers(num, sut, task, res["paired"])
        if sut != "All" and task != "All":
            stem = f"{task}_{sut}"
            write_tex(tabs / "per" / f"{stem}_summary.tex", st.summary_tex(res["summary"]))
            write_tex(tabs / "per" / f"{stem}_budgets.tex", st.budgets_tex(res["curve"], budgets))
            if "paired" in res:
                write_tex(tabs / "per" / f"{stem}_paired.tex", st.paired_tex(res["paired"]))
        if (sut, task) == ("All", "All"):
            write_tex(tabs / "e1.tex", st.summary_tex(res["summary"]))
            write_tex(tabs / "e1_budgets.tex", st.budgets_tex(res["curve"], budgets))
            if "paired" in res:
                write_tex(tabs / "e1_paired.tex", st.paired_tex(res["paired"]))
    _cleanup_tmp(gen)
    pt = pr.primary_table(df, ref, "a1_random_tiered", n_boot)
    if len(pt):
        write_tex(tabs / "primary.tex", primary_tex(pt))
        add_primary_numbers(num, pt)
    # secondary (declared after the primary plan, METHODOLOGY 2026-11-22 08:40 UTC): a7 (image only) vs a0 on matched pairs
    if "a7_image_only" in set(df["arm"]):
        st7 = pr.primary_table(df, ref, "a7_image_only", n_boot)
        if len(st7):
            write_tex(tabs / "secondary_a7.tex", primary_tex(st7, "a7"))
            add_primary_numbers(num, st7, "ESecondary", "a7")
    # secondary (METHODOLOGY 2026-11-22 09:10 UTC): a9 (DeepJanus-style) vs a0 on matched pairs
    if "a9_deepjanus" in set(df["arm"]):
        st9 = pr.primary_table(df, ref, "a9_deepjanus", n_boot)
        if len(st9):
            write_tex(tabs / "secondary_a9.tex", primary_tex(st9, "a9"))
            add_primary_numbers(num, st9, "ESecondaryNine", "a9")

    prim = df[df["arm"].isin(PRIMARY_ARMS)]
    # E3: a0 and a1 per SUT on ImageNet (continuity SUT included)
    frames = []
    for sut in [s for s in SUT_NAMES if s in set(prim["sut"])]:
        d = prim[(prim["sut"] == sut) & (prim["task"] == "imagenet")]
        if len(d):
            frames.append(([SUT_NAMES[sut]], st.arm_summary(d, n_boot)))
    if frames:
        write_tex(tabs / "e3.tex", keyed_summary_tex(frames, ["SUT"]))
    # E4: a0 and a1 per task, grid SUTs pooled
    frames = []
    for task in sorted(set(prim["task"])):
        d = prim[(prim["task"] == task) & prim["sut"].isin(set(grid_suts))]
        if len(d):
            frames.append(([TASK_NAMES.get(task, task)], st.arm_summary(with_scope_cells(d), n_boot)))
    if frames:
        write_tex(tabs / "e4.tex", keyed_summary_tex(frames, ["Task"]))

    # E2b
    need = {"prior_g", "probe_gen0", "min_gen0"}
    if need <= set(df.columns) and df["prior_g"].notna().any():
        a0 = df[df["arm"] == ref]
        if len(a0):
            cells = pred.cell_frame(pred.prepare(a0), ["sut", "task"])
            tab, corr, part = e2b_table(cells, n_boot)
            write_tex(tabs / "e2b.tex", tab)
            corr.to_csv(gen / "e2b_cell_correlations.csv", index=False)
            add_predictor_numbers(num, corr, part)
    else:
        logger.warning("no prior_g / probe columns: E2b skipped")

    if validity is not None and Path(validity).exists():
        vdf = pd.read_csv(validity)
        if len(vdf):
            v1 = vdf[vdf["view"] == "first"] if "view" in vdf else vdf
            write_tex(tabs / "validity.tex", validity_table(v1))
            add_validity_numbers(num, v1)
            from mmb.analysis import validity_cost as vc
            for view, stem, fam in (("first", "validity_cost", "EValid"), ("min_edit", "validity_cost_min", "EValidMin")):
                vct = vc.validity_cost(vdf, df, grid_suts, view=view)
                if len(vct):
                    write_tex(tabs / f"{stem}.tex", vc.table_tex(vct, TASK_NAMES, objective=(view == "first")))
                    vc.add_numbers(num, vct, fam)
                    write_tex(tabs / f"{stem.replace('cost', 'sensitivity')}.tex",
                              vc.sensitivity_tex(vc.sensitivity(vdf, df, grid_suts, view)))

    if agree:
        adf = load_agree(agree)
        if len(adf):
            write_tex(tabs / "agree.tex", agree_table(adf))
            add_agree_numbers(num, adf)

    if transfer:
        tdf = tr.load_transfer(transfer, results=csv_paths[0])
        trows = tr.matrix_rows(tdf)
        if trows:
            write_tex(tabs / "transfer.tex", tr.matrix_tex(trows, "All", SUT_NAMES))
            for task in sorted(set(tdf["task"])):
                write_tex(tabs / f"transfer_{task}.tex", tr.matrix_tex(trows, task, SUT_NAMES))
            add_transfer_numbers(num, trows)

    if human is not None and Path(human).exists():
        hdf = pd.read_csv(human, keep_default_na=False)
        if len(hdf):
            write_tex(tabs / "human_validity.tex", human_tex(hdf))
            add_human_numbers(num, hdf)

    if figures:
        grid_df = sc.get(("All", "All"))
        if grid_df is not None:
            plots.plot_crossing_curve(grid_df, figs / "crossing-curve.pdf", ref_arm=ref, vlines=[206], n_boot=n_boot)
        if len(pt):
            pr.plot_per_cell_diff(df, figs / "per-cell-diff.pdf", ref, "a1_random_tiered")
        if "probe_gen0" in df and df["probe_gen0"].notna().any():
            plots.plot_predictor_scatter(df[df["sut"].isin(set(grid_suts))], figs / "predictor-main.pdf", arm=ref)
        gdf = df[df["sut"].isin(set(grid_suts))]
        picks = gallery.select_runs(gdf, ref, 2)
        fails = gallery.select_failures(gdf, ref)
        if len(fails):
            picks = pd.concat([picks.assign(kind="cross"), fails], ignore_index=True)
        if len(picks):
            try:
                gallery.plot_gallery(picks, figs / "gallery.pdf", runs_root=runs_root)
            except FileNotFoundError as e:
                logger.warning("gallery skipped: %s", e)
    num.write(gen)
    return num


def _tmpdir(gen: Path, sut: str, task: str) -> Path:
    return gen / ".work" / f"{task}_{sut}"


def _cleanup_tmp(gen: Path) -> None:
    import shutil
    shutil.rmtree(gen / ".work", ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--paper", type=Path, required=True)
    ap.add_argument("--ref", default="a0_directed")
    ap.add_argument("--grid-suts", default=",".join(GRID_SUTS))
    ap.add_argument("--validity", type=Path, default=None)
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--agree", type=Path, nargs="*", default=[], help="agreement_replay CSVs (one per SUT)")
    ap.add_argument("--transfer", type=Path, nargs="*", default=[], help="cross-SUT replay CSVs")
    ap.add_argument("--human-validity", type=Path, default=None, help="analyze_ratings.py output CSV")
    ap.add_argument("--n-boot", type=int, default=st.N_BOOT)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    num = export(args.csv, args.paper, ref=args.ref, grid_suts=args.grid_suts.split(","),
                 validity=args.validity, runs_root=args.runs_root, n_boot=args.n_boot,
                 figures=not args.no_figures, agree=args.agree or None, transfer=args.transfer or None, human=args.human_validity)
    logger.info("wrote %d macros to %s", len(num.rows), args.paper / "generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
