"""Results -> paper: tables, figures and a numbers.tex of macros, written into <paper>/generated/.

Usage::

    python -m mmb.analysis.export results/main.csv --paper ../paper_repo [--validity results/validity_main.csv]
        [--runs-root runs/MAIN] [--ref a0_directed] [--grid-suts qwen,internvl,ministral] [--n-boot 10000]
        [--calib <dir with trajectory.csv>]

(scripts/analysis/export_to_paper.sh runs collect + validity first and then calls this.)

Design (continuous genotype, 2026-11-28): arms a0 (MMBA, the method), a5 (MIMICRY-style), a9 (DeepJanus-style) and
the ablations a6 (text only), a7 (image only), x1 (w/o guidance), on the balanced grid (--grid-suts) x tasks. Arms
outside the study (DROPPED_ARMS) and SUTs outside the grid are filtered before any output. Pooling over tasks or SUTs
prefixes cell_id with task/SUT, so pairs (cell, seed) stay unique.

Cost of a crossing is the smallest crossing found within the budget (magnitude = L2 of the weighted image and text
block magnitudes, mag_image, mag_text, min_text_d; smaller is better). Every budget-bound metric uses in-budget data
only; the boundary-pair calls (nb_extra_calls) are outside the budget and never enter a budget metric.

Written under <paper>/generated/:
  tables/e1.tex e1_paired.tex                  grid pooled, every arm
  tables/primary.tex                           primary comparisons a0 vs a5 and a0 vs a9 per (task, SUT), Holm across strata
  tables/e1_cluster.tex cost.tex               paired differences with cell-cluster CIs; smallest-crossing magnitude per arm
  tables/boundary_pair.tex                     boundary pair per arm (EPair* macros)
  tables/min_modality.tex                      modality of the smallest crossing per arm (EMinMod* macros)
  tables/arm_main.tex arm_cost.tex arm_ablation.tex   RQ tables (with --validity)
  tables/agree.tex, EAgree macros              greedy agreement and re-verification (only with --agree)
  tables/human_validity.tex                    human validity (only with --human-validity)
  figures/crossing-curve.pdf predictor-main.pdf gallery.pdf calib-trajectory.pdf (with --calib)
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
from mmb.analysis import tablefmt as tf

logger = logging.getLogger(__name__)

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
GRID_SUTS = ("qwen", "internvl", "ministral")
#: arm prefixes not in the study (random search, GA, NSGA-II, hill climber, patch edits): filtered before any output
DROPPED_ARMS = frozenset({"a1", "a2", "a3", "a4", "a8", "x2"})
#: arm prefixes kept in the data but omitted from the paper (none by default; --omit-arms)
PAPER_OMIT_ARMS = frozenset()
#: RQ4 ablation arms are reported on rng seed 0 only; their seed-1 runs are filtered before any output
SEED0_ONLY_ARMS = frozenset({"a6", "a7", "x1"})
#: primary comparisons: (compared arm, its id prefix); each against the reference a0, Holm across (task, SUT) strata
PRIMARY_ARMS = (("a5_mimicry_objective", "a5"), ("a9_deepjanus", "a9"))
SUT_NAMES = {"qwen": "Qwen3.5-4B (bf16)", "internvl": "InternVL3.5-4B (bf16)", "ministral": "Ministral-3-3B (bf16)"}
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


def f_smart(x: float, d: int = 3, sign: bool = False) -> str:
    """Like f_num, but integral values (medians of counts) without decimals."""
    if not pd.isna(x) and float(x).is_integer():
        return f_num(x, 0, sign)
    return f_num(x, d, sign)


def f_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def data_status(df: pd.DataFrame, planned_dir: Path, repo: Path, n_cells: int = 18) -> tuple[str, int, int]:
    """("complete"|"partial", configs complete, configs planned). A planned configuration (a YAML of
    configs/generated/MAIN) is complete when its save_dir holds complete runs for n_cells distinct cells."""
    import yaml

    planned = sorted(Path(planned_dir).glob("*.yaml"))
    if not planned or "run_dir" not in df:
        return "partial", 0, len(planned)
    done_df = df[df["complete"].astype(bool)] if "complete" in df else df
    dirs = done_df["run_dir"].map(lambda p: str(Path(p).resolve()))
    ok = 0
    for p in planned:
        save = (repo / str(yaml.safe_load(p.read_text()).get("save_dir", "")).strip()).resolve()
        hit = done_df[dirs.str.startswith(str(save) + "/")]
        ok += int(hit["cell_id"].nunique() >= n_cells)
    return ("complete" if ok == len(planned) else "partial"), ok, len(planned)


class Numbers:
    """Collects (macro, value, description) triples; writes numbers.tex and the mapping CSV."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.seen: set[str] = set()
        self.datastatus = "complete"   # "partial" while a planned configuration lacks cells (invisible marker)

    def add(self, parts: Iterable[str], value: str, desc: str, **meta) -> None:
        name = macro_name(*parts)
        if name in self.seen:
            raise ValueError(f"duplicate macro {name}")
        self.seen.add(name)
        self.rows.append(dict(macro=name, value=str(value), description=desc, **meta))

    def add_legacy(self, name: str, value: str, desc: str) -> None:
        """A macro kept verbatim from an earlier export (no recomputation), so text that still cites it compiles."""
        if name in self.seen:
            return
        self.seen.add(name)
        self.rows.append(dict(macro=name, value=str(value), description="LEGACY " + desc))

    def write(self, out: Path) -> None:
        df = pd.DataFrame(self.rows)
        df.to_csv(out / "numbers_map.csv", index=False)
        lines = ["% Generated by mmb.analysis.export; do not edit. One macro per headline number.",
                 "% Use \\res{Name} in the text: it prints [TBD] while a macro is undefined."]
        lines += [f"\\def\\datastatus{{{self.datastatus}}}"]
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
    """Summary table with leading key columns; a midrule between key groups; best values bold within each group."""
    header = key_headers + st.SUMMARY_HEADER
    body: list[str] = []
    for gi, (keys, s) in enumerate(frames):
        if gi:
            body.append(r"\midrule")
        for ri, row in enumerate(st.summary_rows(s)):
            ks = [st.tex_escape(k) if ri == 0 else "" for k in keys]
            body.append(" & ".join(ks + row) + r" \\")
    colspec = "l" * (len(key_headers) + 1) + "rrrrrr"
    return ("\\begin{tabular}{@{}" + colspec + "@{}}\n\\toprule\n" + " & ".join(header) + " \\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}")



def wrapped(tabular: str) -> str:
    return WRAP % tabular.strip()


# Tables the paper no longer uses (removed by the authors, paper 49bb955/d7d7e8c): never written again, so an export
# cannot bring them back. Their \res macros (EMinType*, ETransfer*, EValid*, ...) are still emitted.
REMOVED_TABLES = frozenset({"arm_failure", "e1_budgets", "validity", "validity_cost", "validity_cost_min", "validity_main",
                            "validity_sensitivity", "validity_sensitivity_min", "transfer", "transfer_imagenet",
                            "transfer_pope", "e2b"})


def is_removed_table(path: Path) -> bool:
    """generated/tables/<REMOVED_TABLES>.tex and everything under generated/tables/per/."""
    p = Path(path)
    if p.parent.name == "tables":
        return p.stem in REMOVED_TABLES
    return "per" in p.parts and p.parts[p.parts.index("per") - 1:p.parts.index("per")] == ("tables",)


def write_tex(path: Path, tabular: str) -> None:
    if is_removed_table(path):
        logger.debug("table %s removed from the paper: not written", path)
        return
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
            ("MinN", str(int(r.n_min)) if "n_min" in r else "0", "runs with a smallest crossing within the budget"),
            ("MedMag", f_smart(r.median_magnitude, 4) if "median_magnitude" in r else "n/a",
             "median magnitude of the smallest in-budget crossing (L2 of the weighted image and text magnitudes)"),
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
        met = PAIRED_METRIC_NAMES.get(r.metric, camel(r.metric))
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


PAIRED_METRIC_NAMES = {"crossed": "Crossed", "first_cross_eval": "Evals", "floor": "Floor", "magnitude": "Mag",
                      "mag_image": "MagImage", "mag_text": "MagText", "text_d": "TextD", "min_eval_idx": "MinEvals",
                      "wall": "Wall"}
METRIC_HEAD = {"crossed": "Crossing rate", "first_cross_eval": "Model calls to first crossing", "floor": "Closest approach",
               "magnitude": "Smallest-crossing magnitude", "mag_image": "Image magnitude", "mag_text": "Text magnitude",
               "text_d": "Question distance", "min_eval_idx": "Calls to smallest crossing", "wall": "Runtime"}


#: paired metrics whose values are counts (model calls): their medians print as integers or with one decimal
COUNT_METRICS = frozenset({"first_cross_eval", "min_eval_idx"})
#: paired metrics compared on a log10 scale (paired.METRICS transform)
LOG_METRICS = frozenset({"floor", "magnitude"})

def _ci(v: float, lo: float, hi: float, d: int = 2) -> str:
    return "--" if pd.isna(v) else f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def add_cluster_numbers(num: Numbers, sut: str, task: str, t: pd.DataFrame) -> None:
    for _, r in t.iterrows():
        tail = [PAIRED_METRIC_NAMES[r.metric], arm_short(r.arm), camel(sut), camel(task)]
        d = f"{r.arm} vs {r.ref}, {r.metric}, {sut}, {task}, {int(r.n_pairs)} pairs in {int(r.n_cells)} cells"
        if r.metric == "crossed":           # shares as percent, their difference in percentage points
            fv = lambda x, sign=False: f_num(100 * x, 1, sign)
            unit = " (%; difference in percentage points)"
        elif r.metric in COUNT_METRICS:     # medians of counts (edits, calls): integer, else one decimal (10.5)
            fv = lambda x, sign=False: f_smart(x, 1, sign)
            unit = ""
        else:
            fv = lambda x, sign=False: f_smart(x, 3, sign)
            unit = " (log10)" if r.metric in LOG_METRICS else ""
        for f, v, what in (("Pairs", str(int(r.n_pairs)), "matched pairs"),
                           ("RefValue", fv(r.ref_value), "reference value (share or median)" + unit),
                           ("ArmValue", fv(r.arm_value), "arm value (share or median)" + unit),
                           ("Diff", fv(r["diff"], True), r.diff_name + " (arm - ref)" + unit),
                           ("DiffLo", fv(r.diff_lo, True), "cell-cluster bootstrap 95% CI lower"),
                           ("DiffHi", fv(r.diff_hi, True), "cell-cluster bootstrap 95% CI upper"),
                           ("P", f_p(r.p), "raw p (exact McNemar / Wilcoxon)"),
                           ("PHolm", f_p(r.p_holm), "Holm-adjusted p across arms")):
            num.add(["ETwoA", f] + tail, v, f"{what}; {d}")


def _effect_text(row, metric: str) -> str:
    """Effect size of a paired comparison: percentage points for the crossing rate, d_z otherwise."""
    if metric == "crossed":
        return "--" if pd.isna(row["diff"]) else f"{100 * row['diff']:+.0f}"
    return "--" if pd.isna(row["dz"]) else f"{row['dz']:+.2f}"


def cluster_table_tex(t: pd.DataFrame, metrics=pr.DEFAULT_METRICS) -> str:
    """Paired differences against the directed arm: one effect size per comparison (bullet for the magnitude,
    dagger when significant after Holm)."""
    rows = []
    for arm in list(dict.fromkeys(t["arm"])):
        row = [st.tex_escape(st.arm_label(arm))]
        for m in metrics:
            r = t[(t.arm == arm) & (t.metric == m)]
            row.append("--" if r.empty else tf.effect_cell(_effect_text(r.iloc[0], m), r.iloc[0]["dz"], r.iloc[0].p_holm))
        rows.append(row)
    return st._booktabs("l" + "r" * len(metrics), ["Arm vs directed"] + [tf.head(METRIC_HEAD.get(m, m), m) for m in metrics], rows)



def primary_tex(t: pd.DataFrame) -> str:
    """Primary comparisons per stratum: MMBA's crossing rate, and for each compared arm (MIMICRY-style, DeepJanus-style)
    its crossing rate and the difference to MMBA in percentage points (arm minus MMBA) with its magnitude bullet and
    dagger (exact McNemar, Holm across strata within each comparison)."""
    shorts = [s for _a, s in PRIMARY_ARMS if s in set(t["short"])]
    rows = []
    for (task, sut), g in t.groupby(["task", "sut"], sort=False):
        ref = g.iloc[0]
        row = [st.tex_escape(TASK_NAMES.get(task, task)), st.tex_escape(SUT_NAMES.get(sut, sut)), f"{100 * ref.ref_value:.0f}"]
        for s in shorts:
            q = g[g["short"] == s]
            if q.empty:
                row += ["--", "--"]
                continue
            r = q.iloc[0]
            row += [f"{100 * r.arm_value:.0f}", tf.effect_cell(f"{100 * r['diff']:+.0f}", r["dz"], r.p_holm_strata)]
        rows.append(row)
    header = ["Task", "SUT", tf.head(r"\sysname{} (\%)", "crossed")]
    for s in shorts:
        header += [tf.head(st.tex_escape(st.arm_label(s)) + r" (\%)", "crossed"), "Difference (pp)"]
    return st._booktabs("llr" + "rr" * len(shorts), header, rows)


def primary_tables(df: pd.DataFrame, ref: str, n_boot: int) -> pd.DataFrame:
    """paired.primary_table for every primary comparison present (column 'short' = the compared arm's prefix)."""
    parts = []
    arms = set(df["arm"])
    for arm, short in PRIMARY_ARMS:
        if ref in arms and arm in arms:
            t = pr.primary_table(df, ref, arm, n_boot)
            if len(t):
                parts.append(t.assign(short=short))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def add_primary_numbers(num: Numbers, t: pd.DataFrame, family: str = "EPrimary") -> None:
    """EPrimary<Field><Arm><Sut><Task>: a0 vs the arm on the crossing share, per stratum."""
    for _, r in t.iterrows():
        tail = [arm_short(r["short"]), camel(r.sut), camel(r.task)]
        d = f"primary comparison {r['short']} vs a0 crossing share, {r.sut}, {r.task}"
        for f, v, what in (("Pairs", str(int(r.n_pairs)), "matched pairs"),
                           ("RefShare", f_pct(r.ref_value), "a0 crossing share (%)"),
                           ("ArmShare", f_pct(r.arm_value), f"{r['short']} crossing share (%)"),
                           ("Diff", f_num(100 * r["diff"], 1, True), f"{r['short']} minus a0 (percentage points)"),
                           ("DiffLo", f_num(100 * r.diff_lo, 1, True), "cell-cluster bootstrap 95% CI lower (pp)"),
                           ("DiffHi", f_num(100 * r.diff_hi, 1, True), "cell-cluster bootstrap 95% CI upper (pp)"),
                           ("P", f_p(r.p), "exact McNemar p"),
                           ("PHolm", f_p(r.p_holm_strata), "Holm-adjusted across strata")):
            num.add([family, f] + tail, v, f"{what}; {d}")


# ---------------------------------------------------------------------------
# cost of a crossing (smallest in-budget crossing), boundary pair, modality of the smallest crossing
# ---------------------------------------------------------------------------


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(("true", "1", "1.0"))


def in_budget_min(g: pd.DataFrame) -> pd.DataFrame:
    """Runs whose smallest crossing lies within the budget (min_within_budget) and has a magnitude."""
    if "magnitude" not in g:
        return g.iloc[0:0]
    ok = _truthy(g["min_within_budget"]) if "min_within_budget" in g else g["crossed"].astype(bool)
    return g[ok & pd.to_numeric(g["magnitude"], errors="coerce").notna()]


def _med(g: pd.DataFrame, col: str) -> float:
    v = pd.to_numeric(g[col], errors="coerce").dropna() if col in g else pd.Series(dtype=float)
    return float(v.median()) if len(v) else float("nan")


def cost_summary(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Per arm: the smallest crossing found within the budget (runs without one are left out, smaller is better)."""
    rows = []
    for arm, g in df.groupby("arm", sort=True):
        c = in_budget_min(g)
        me, lo, hi = st.bootstrap_stat(pd.to_numeric(c["magnitude"], errors="coerce").to_numpy(float), n_boot=n_boot) \
            if len(c) else (np.nan,) * 3
        rows.append(dict(arm=arm, n_runs=len(g), n_crossed=int(g["crossed"].astype(bool).sum()), n_min=len(c),
                         med_mag=me, mag_lo=lo, mag_hi=hi, med_mag_image=_med(c, "mag_image"),
                         med_mag_text=_med(c, "mag_text"), med_text_d=_med(c, "min_text_d"),
                         med_min_eval=_med(c, "min_eval_idx")))
    return pd.DataFrame(rows)


def _cells(vals, d: int, key: str | None) -> list[str]:
    cells = ["--" if pd.isna(v) else f"{v:.{d}f}" for v in vals]
    return tf.bold_best(cells, list(vals), key, cells) if key else cells


def cost_tex(s: pd.DataFrame) -> str:
    cols = [_cells(s.med_mag, 4, "magnitude"), _cells(s.med_mag_image, 4, "mag_image"), _cells(s.med_mag_text, 4, "mag_text"),
            _cells(s.med_text_d, 3, "text_d"), _cells(s.med_min_eval, 0, "min_eval_idx")]
    rows = [[st.tex_escape(st.arm_label(r.arm)), str(int(r.n_runs)), str(int(r.n_min))] + [c[i] for c in cols]
            for i, r in enumerate(s.itertuples())]
    return st._booktabs("lrrrrrrr", ["Arm", "Runs", "Smallest crossing in budget", tf.head("Magnitude", "magnitude"),
                                     tf.head("Image magnitude", "mag_image"), tf.head("Text magnitude", "mag_text"),
                                     tf.head("Question distance", "text_d"), tf.head("Calls to it", "min_eval_idx")], rows)


def add_cost_numbers(num: Numbers, sut: str, task: str, s: pd.DataFrame) -> None:
    for _, r in s.iterrows():
        tail = [arm_short(r.arm), camel(sut), camel(task)]
        d = f"smallest crossing within the budget, runs that have one; {r.arm}, {sut}, {task}"
        num.add(["ECost", "NMin"] + tail, str(int(r.n_min)), "runs with a smallest crossing within the budget; " + d)
        num.add(["ECost", "MedMag"] + tail, f_smart(r.med_mag, 4), "median magnitude (L2 of weighted image and text magnitudes); " + d)
        num.add(["ECost", "MedMagCi"] + tail, "--" if pd.isna(r.mag_lo) else f"[{r.mag_lo:.4f}--{r.mag_hi:.4f}]", "bootstrap 95% CI of the median magnitude; " + d)
        num.add(["ECost", "MedMagImage"] + tail, f_smart(r.med_mag_image, 4), "median weighted image magnitude (w_img x RMS of the image genes); " + d)
        num.add(["ECost", "MedMagText"] + tail, f_smart(r.med_mag_text, 4), "median weighted text magnitude (w_txt x t); " + d)
        num.add(["ECost", "MedTextD"] + tail, f_smart(r.med_text_d, 3), "median MiniLM distance of the edited question; " + d)
        num.add(["ECost", "MedMinEvals"] + tail, f_smart(r.med_min_eval, 1), "median model calls to the smallest crossing; " + d)


def pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm: boundary pairs of the crossing runs. A pair is confirmed when bp_source is own_phase2 or
    posthoc_bisection; gap, kind and extra calls over confirmed pairs. The extra calls of the boundary pair are spent
    outside the search budget and never enter a budget-bound metric."""
    rows = []
    for arm, g in df.groupby("arm", sort=True):
        c = g[g["crossed"].astype(bool)]
        src = c["bp_source"].astype(str) if "bp_source" in c else pd.Series("none", index=c.index)
        has = src.isin(st.PAIR_SOURCES)
        p = c[has]
        n, k = len(c), int(has.sum())
        kind = p["bp_kind"].astype(str) if "bp_kind" in p else pd.Series(dtype=str)
        rows.append(dict(arm=arm, n_runs=len(g), n_crossed=n, n_pair=k,
                         share_pair=k / n if n else np.nan,
                         share_own=float((src == "own_phase2").sum()) / n if n else np.nan,
                         share_posthoc=float((src == "posthoc_bisection").sum()) / n if n else np.nan,
                         med_gap=_med(p, "bp_gap"),
                         share_text_step=float((kind == "text_step").mean()) if k else np.nan,
                         share_continuous=float((kind == "continuous").mean()) if k else np.nan,
                         med_extra_calls=_med(p, "nb_extra_calls")))
    return pd.DataFrame(rows)


def pair_tex(s: pd.DataFrame) -> str:
    pct = lambda v: "--" if pd.isna(v) else f"{100 * v:.0f}"
    gap = _cells(s.med_gap, 4, "bp_gap")
    rows = [[st.tex_escape(st.arm_label(r.arm)), str(int(r.n_crossed)), pct(r.share_pair), pct(r.share_own),
             pct(r.share_posthoc), gap[i], pct(r.share_text_step), pct(r.share_continuous),
             "--" if pd.isna(r.med_extra_calls) else f_smart(r.med_extra_calls, 1)] for i, r in enumerate(s.itertuples())]
    return st._booktabs("lrrrrrrrr", ["Arm", "Crossed", r"Pair (\%)", r"own Phase 2 (\%)", r"post hoc (\%)",
                                      tf.head("Gap", "bp_gap"), r"text step (\%)", r"continuous (\%)",
                                      "Extra calls (outside budget)"], rows)


def add_pair_numbers(num: Numbers, sut: str, task: str, s: pd.DataFrame) -> None:
    for _, r in s.iterrows():
        tail = [arm_short(r.arm), camel(sut), camel(task)]
        d = f"boundary pair, crossing runs; {r.arm}, {sut}, {task}"
        for f, v, what in (("N", str(int(r.n_pair)), "crossing runs with a confirmed boundary pair"),
                           ("Crossed", str(int(r.n_crossed)), "crossing runs"),
                           ("Share", f_pct(r.share_pair), "share of crossing runs with a confirmed pair (%)"),
                           ("ShareOwn", f_pct(r.share_own), "share of crossing runs with a pair from the search's own Phase 2 (%)"),
                           ("SharePosthoc", f_pct(r.share_posthoc), "share of crossing runs with a post hoc bisection pair (%)"),
                           ("MedGap", f_smart(r.med_gap, 4), "median gap of the confirmed pairs"),
                           ("ShareTextStep", f_pct(r.share_text_step), "share of confirmed pairs that are a text step (%)"),
                           ("ShareContinuous", f_pct(r.share_continuous), "share of confirmed pairs that are continuous (%)"),
                           ("MedExtraCalls", f_smart(r.med_extra_calls, 1), "median extra model calls for the pair (outside the budget)")):
            num.add(["EPair", f] + tail, v, f"{what}; {d}")


MODALITIES = (("image", "Image"), ("text", "Text"), ("joint", "Joint"))


def modality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm: which blocks the smallest in-budget crossing changed (min_modality image | text | joint)."""
    rows = []
    for arm, g in df.groupby("arm", sort=True):
        c = in_budget_min(g)
        m = c["min_modality"].astype(str) if "min_modality" in c else pd.Series(dtype=str)
        m = m[m.isin([k for k, _ in MODALITIES])]
        n = len(m)
        rows.append(dict(arm=arm, n_min=n, **{f"share_{k}": float((m == k).mean()) if n else np.nan for k, _ in MODALITIES}))
    return pd.DataFrame(rows)


def modality_tex(s: pd.DataFrame) -> str:
    pct = lambda v: "--" if pd.isna(v) else f"{100 * v:.0f}"
    rows = [[st.tex_escape(st.arm_label(r["arm"])), str(int(r["n_min"]))] + [pct(r[f"share_{k}"]) for k, _ in MODALITIES]
            for _, r in s.iterrows()]
    return st._booktabs("lrrrr", ["Arm", "Smallest crossings", r"image only (\%)", r"text only (\%)", r"joint (\%)"], rows)


def add_modality_numbers(num: Numbers, sut: str, task: str, s: pd.DataFrame) -> None:
    for _, r in s.iterrows():
        tail = [arm_short(r["arm"]), camel(sut), camel(task)]
        d = f"smallest in-budget crossing; {r['arm']}, {sut}, {task}"
        num.add(["EMinMod", "N"] + tail, str(int(r["n_min"])), "smallest crossings with a modality; " + d)
        for k, nm in MODALITIES:
            num.add(["EMinMod", nm] + tail, f_pct(r[f"share_{k}"]), f"share whose change is {k} (%); " + d)


def add_wall_numbers(num: Numbers, sc: dict) -> None:
    """Median wall-clock minutes per run: per arm in every (SUT, task) scope, and pooled over arms."""
    for (sut, task), d in sc.items():
        if "wall_s" not in d or not d["wall_s"].notna().any():
            continue
        for arm, g in d.groupby("arm", sort=True):
            w = g["wall_s"].dropna()
            if len(w):
                num.add(["ECost", "WallMinMed", arm_short(arm), camel(sut), camel(task)], f_num(w.median() / 60, 1),
                        f"median wall-clock minutes per run; {arm}, {sut}, {task}")
        w = d["wall_s"].dropna()
        num.add(["ECost", "WallMinMedAllArms", camel(sut), camel(task)], f_num(w.median() / 60, 1),
                f"median wall-clock minutes per run over all arms; {sut}, {task}")


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


def add_transfer_paired(num: Numbers, tdf: pd.DataFrame, ref: str = "a0_directed") -> None:
    """Paired transfer test: per source run the share of other SUTs whose answer the crossing flips; runs of each
    arm are matched to the reference on (task, source SUT, rng seed, seed input); Wilcoxon signed-rank on the
    differences (zero differences dropped, Pratt would keep them), Holm across arms."""
    from scipy.stats import wilcoxon
    t = tdf.copy()
    key = t["run_dir"].astype(str).str.extract(r"_s(\d+)_seed_(\d+)_")
    t["cell_idx"] = key[1]
    if "rng_seed" not in t:          # older transfer CSVs: the rng seed is in the run name (_s<seed>_seed_<cell>_)
        t["rng_seed"] = pd.to_numeric(key[0], errors="coerce")
    t = t.dropna(subset=["cell_idx"])
    t["flip"] = t["flipped_on_dst"].astype(str).str.lower().isin(("true", "1", "1.0"))
    per = t.groupby(["arm", "task", "src_sut", "rng_seed", "cell_idx"])["flip"].mean().reset_index()
    keys = ["task", "src_sut", "rng_seed", "cell_idx"]
    a = per[per["arm"] == ref]
    rows = []
    for arm in sorted(set(per["arm"]) - {ref}):
        m = a.merge(per[per["arm"] == arm], on=keys, suffixes=("_ref", "_arm"))
        if len(m) < 5:
            continue
        diff = m["flip_arm"] - m["flip_ref"]
        p = float(wilcoxon(diff, zero_method="wilcox").pvalue) if (diff != 0).any() else 1.0
        rows.append(dict(arm=arm, n=len(m), ref=float(m["flip_ref"].mean()), val=float(m["flip_arm"].mean()),
                         diff=float(diff.mean()), p=p))
    if not rows:
        return
    ps = st.holm([r["p"] for r in rows]) if hasattr(st, "holm") else [min(1.0, r["p"] * len(rows)) for r in rows]
    for r, ph in zip(rows, ps):
        tail = [arm_short(r["arm"])]
        d = f"transfer, {r['arm']} vs {ref}, {r['n']} matched crossing runs (both crossed)"
        num.add(["ETransferPaired", "N"] + tail, str(r["n"]), "matched pairs; " + d)
        num.add(["ETransferPaired", "RefShare"] + tail, f_pct(r["ref"]), "mean share of other SUTs flipped, reference (%); " + d)
        num.add(["ETransferPaired", "ArmShare"] + tail, f_pct(r["val"]), "mean share of other SUTs flipped, arm (%); " + d)
        num.add(["ETransferPaired", "Diff"] + tail, f_num(100 * r["diff"], 1, True), "arm minus reference (pp); " + d)
        num.add(["ETransferPaired", "P"] + tail, f_p(r["p"]), "Wilcoxon signed-rank p; " + d)
        num.add(["ETransferPaired", "PHolm"] + tail, f_p(ph), "Holm across arms; " + d)


def human_tex(s: pd.DataFrame) -> str:
    rows = []
    for _, r in s.iterrows():
        rows.append([st.tex_escape(r["stratum"]), st.tex_escape(TASK_NAMES.get(r["task"], r["task"])),
                     st.tex_escape(st.arm_label(r["arm"]) if isinstance(r["arm"], str) and r["arm"] else "unedited control"),
                     str(int(r["items"])), f"{r['valid_share']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]",
                     f"{100 * r['cant_share']:.0f}"])
    return st._booktabs("lllrrr", ["Stratum", "Task", "Method", "Items", "Valid share [95\\% CI]", "Can't tell (\\%)"], rows)


def add_human_numbers(num: Numbers, s: pd.DataFrame) -> None:
    for _, r in s.iterrows():
        arm = r["arm"] if isinstance(r["arm"], str) and r["arm"] else "control"
        tail = [camel(r["stratum"]), camel(r["task"]), arm_short(arm) if arm != "control" else "Control"]
        d = f"human validity, {r['stratum']}, {r['task']}, {arm}, {int(r['items'])} items, {int(r['raters'])} raters"
        num.add(["EHuman", "Share"] + tail, f_pct(r["valid_share"]), "valid share (%); " + d)
        num.add(["EHuman", "Ci"] + tail, f"[{100 * r['lo']:.0f}--{100 * r['hi']:.0f}]", "95% bootstrap CI over items (%); " + d)
        num.add(["EHuman", "Items"] + tail, str(int(r["items"])), "items; " + d)


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
    cols = [("img_clip_cos", "CLIP cos", "clip"), ("img_dino_cos", "DINOv2 cos", "dino"), ("img_lpips", "LPIPS", "lpips"),
            ("img_ssim", "SSIM", "ssim"), ("txt_minilm_cos", "MiniLM cos", "minilm"), ("txt_word_edit", "word edits", "word_edit")]
    cols = [(c, h, k) for c, h, k in cols if c in vdf]
    arms = [(arm, g) for arm, g in vdf.groupby("arm")]
    columns = []
    for c, _, k in cols:
        vals = [float(g[c].median()) for _, g in arms]
        cells = [f"{v:.2f}" for v in vals]
        columns.append(tf.bold_best(cells, vals, k, cells))
    rows = [[st.tex_escape(st.arm_label(arm)), str(len(g))] + [columns[j][i] for j in range(len(cols))] for i, (arm, g) in enumerate(arms)]
    return st._booktabs("lr" + "r" * len(cols), ["Arm", "Crossings"] + [tf.head(h, k) for _, h, k in cols], rows)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _rq_tables(rq, tabs: Path, df: pd.DataFrame, vdf: pd.DataFrame, grid_suts, num, marks: dict | None = None) -> None:
    """Main-text tables: RQ1 (effectiveness and quality), RQ2 (efficiency), RQ4 (ablation, pooled, seed 0).
    marks: arm id -> superscript mark on its row label (see method_marks); a note row explains each mark used."""
    gd, gv = df[df["sut"].isin(grid_suts)], vdf[vdf["sut"].isin(grid_suts)]
    marks = {a: m for a, m in (marks or {}).items() if a in set(gd["arm"])}
    cells_m = rq.by_cell(gd, gv)
    write_tex(tabs / "arm_main.tex", rq.rq1_tex(cells_m, marks))
    write_tex(tabs / "arm_cost.tex", rq.rq2_tex(cells_m, marks))
    abl = rq.ablation(gd, gv)
    abl_e = rq.ablation_effects(abl)
    abl_marks = {rq.prefix(a): m for a, m in marks.items()}
    write_tex(tabs / "arm_ablation.tex", rq.ablation_tex(abl, abl_e, abl_marks))
    rq.add_ablation_numbers(num, abl)
    rq.add_ablation_effect_numbers(num, abl, abl_e)
    rq.add_pooled_numbers(num, gd, gv, arm_short)


def add_teaser_numbers(num: Numbers, facts: list[dict]) -> None:
    """Macros for the Fig. 1 rows (ETeaserImagenet*/ETeaserPope*), so the introduction quotes the figure it shows."""
    def odds(r):
        return f"{r:.0f}" if r >= 10 else f"{r:.1f}"
    for f in facts:
        row = ["ETeaser", camel(f["task"])]
        d = f"Fig. 1 row, {f['task']}"
        for k, v in (("Sut", f["sut"]), ("Before", f["before"]), ("After", f["after"]), ("Object", f["obj"]),
                     ("OddsBefore", odds(f["odds_before"])), ("OddsAfter", odds(f["odds_after"])),
                     ("Mag", f_smart(f["magnitude"], 4)), ("MagImage", f_smart(f["mag_image"], 4)),
                     ("MagText", f_smart(f["mag_text"], 4))):
            num.add(row + [k], st.tex_escape(str(v)), f"{k}; {d}")


def complete_blocks(df: pd.DataFrame, ref: str = "a0_directed", grid_suts=GRID_SUTS) -> pd.DataFrame:
    """Drop the runs of every (method, rng seed) block that does not yet cover all (task, SUT, seed input) keys the
    reference method has for that rng seed on the grid SUTs, so a method enters tables and figures only with a
    complete balanced block (a queue still running a block must not leak partial rows)."""
    if ref not in set(df["arm"]):
        return df
    keys = ["task", "sut", "cell_id"]
    grid = df["sut"].isin(list(grid_suts))
    keep = pd.Series(True, index=df.index)
    for (arm, seed), g in df[grid].groupby(["arm", "rng_seed"]):
        if arm == ref:
            continue
        need = set(map(tuple, df[grid & (df["arm"] == ref) & (df["rng_seed"] == seed)][keys].drop_duplicates().to_numpy()))
        have = set(map(tuple, g[keys].drop_duplicates().to_numpy()))
        if need and not need <= have:
            logger.info("held back: %s seed %s (%d of %d seed inputs x models x tasks)", arm, seed,
                        len(need & have), len(need))
            keep &= ~((df["arm"] == arm) & (df["rng_seed"] == seed))
    return df[keep]


class UnbalancedExport(SystemExit):
    """Raised when the model x task cells of the grid do not hold identical (method, rng seed) blocks."""



_NUM_WORDS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
              "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty")


def step_word(k: int) -> str:
    """Digit-free macro part for a step number (LaTeX macro names cannot hold digits)."""
    return _NUM_WORDS[k] if 0 <= k < len(_NUM_WORDS) else "Step" + "".join(_NUM_WORDS[int(c)] for c in str(k))


def add_calib_numbers(num: "Numbers", calib_dir: Path, figs: Path | None = None) -> None:
    """ECalibTraj*: the weight calibration (mmb.analysis.calibration) from <calib_dir>/trajectory.csv, pooled rows:
    ECalibTraj<Step|Final><Text|Image><Cr|Lo|Hi>, ...P, ...Pairs, ...Disc, ...WImg, ...WTxt (step words = the step
    number spelled out; the adopted step is 'Final'); figures/calib-trajectory.pdf via calibration.plot_trajectory."""
    traj = Path(calib_dir) / "trajectory.csv"
    if not traj.exists():
        logger.warning("calibration: no %s", traj)
        return
    tj = pd.read_csv(traj)
    pooled = tj[tj["scope"].astype(str) == "pooled"] if "scope" in tj else tj
    for _, r in pooled.iterrows():
        fin = str(r.get("final", "")).lower() in ("true", "1", "1.0")
        key = "Final" if fin else step_word(int(r["step"]))
        lab = f"calibration step {r['step']} ({r.get('label', '')})"
        for mod, nm in (("text", "Text"), ("image", "Image")):
            for col, suf in (("cr", "Cr"), ("lo", "Lo"), ("hi", "Hi")):
                v = r.get(f"{mod}_{col}")
                num.add(["ECalibTraj", key, nm, suf], "n/a" if pd.isna(v) else f"{float(v):.1f}",
                        f"{lab}: {mod}-only crossing share (%){'' if col == 'cr' else ', Wilson 95% CI ' + col}")
            m = r.get(f"{mod}_mag_med")
            if m is not None and not pd.isna(m):
                num.add(["ECalibTraj", key, nm, "Mag"], f_smart(float(m), 4), f"{lab}: median magnitude of {mod}-only smallest crossings")
        num.add(["ECalibTraj", key, "P"], f_p(float(r["p_mcnemar"])), f"{lab}: exact McNemar p, text-only vs image-only")
        num.add(["ECalibTraj", key, "Pairs"], str(int(r["pairs"])), f"{lab}: matched pairs")
        if "text_only" in r and not pd.isna(r["text_only"]):
            num.add(["ECalibTraj", key, "Disc"], f"{int(r['text_only'])}:{int(r['image_only'])}",
                    f"{lab}: discordant pairs (text only : image only)")
        for col, nm in (("w_img", "WImg"), ("w_txt", "WTxt")):
            if col in r and not pd.isna(r[col]):
                num.add(["ECalibTraj", key, nm], f"{float(r[col]):g}", f"{lab}: block weight {col}")
    if figs is not None and len(pooled):
        from mmb.analysis.calibration import plot_trajectory
        plot_trajectory(tj, Path(figs) / "calib-trajectory.pdf")


def check_balance(df: pd.DataFrame, grid_suts=GRID_SUTS) -> dict:
    """Balanced-design guard (user rule: every model x task gets the same methods, seeds and seed inputs).
    Every (task, SUT) cell of the grid must hold the same set of (arm, rng seed) blocks, each over the same seed
    inputs. Fails loudly (UnbalancedExport) otherwise, so an unbalanced row can never reach the paper.
    Returns {(arm, seed): n seed inputs} of the balanced design."""
    cells = {}
    for task in sorted(set(df["task"])):
        for sut in grid_suts:
            d = df[(df["task"] == task) & (df["sut"] == sut)]
            cells[(task, sut)] = {(a, int(s)): frozenset(g["cell_id"]) for (a, s), g in d.groupby(["arm", "rng_seed"])}
    if not cells:
        return {}
    ref_key, ref = next(iter(cells.items()))
    first_of_task = {}
    bad = []
    for k, blocks in cells.items():
        same_task = first_of_task.setdefault(k[0], blocks)        # seed inputs are task-specific: compare ids within a task
        miss = sorted(set(ref) - set(blocks)); extra = sorted(set(blocks) - set(ref))
        diff = sorted(b for b in set(ref) & set(blocks) if len(ref[b]) != len(blocks[b])
                      or (b in same_task and same_task[b] != blocks[b]))
        if miss or extra or diff:
            bad.append(f"{k[0]} x {k[1]}: missing {miss}, extra {extra}, other seed inputs {diff} (vs {ref_key[0]} x {ref_key[1]})")
    if bad:
        raise UnbalancedExport("unbalanced export, refusing to write the paper files:\n  " + "\n  ".join(bad))
    return {b: len(c) for b, c in ref.items()}


def pending_confirmation(df: pd.DataFrame) -> set[str]:
    """run_dirs whose smallest crossing exists but has not been confirmed by single-input scoring. Their smallest-crossing
    metrics are withheld; crossing share and calls to the first crossing come from the trace and stay."""
    if "min_confirmed" not in df:
        return set()
    ok = _truthy(df["min_confirmed"])
    has = _truthy(df["min_found"]) if "min_found" in df else df["crossed"].astype(bool)
    out = set(df.loc[has & ~ok, "run_dir"])
    if out:
        arms = sorted(set(df.loc[df["run_dir"].isin(out), "arm"]))
        logger.info("smallest crossings pending confirmation: %d runs (%s); smallest-crossing metrics withheld", len(out), ", ".join(arms))
    return out


def method_marks(df: pd.DataFrame, ref: str, pending: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Per arm id: (table marks, legend suffix). \\ddagger = minimal crossings pending single-input confirmation.
    No seed mark and no legend suffix (user instruction 2026-11-26: the paper treats every method as having 2 seeds;
    the (method, seed) gate and the balance guard still hold back partial blocks)."""
    marks, suffix = {}, {}
    for arm, d in df.groupby("arm"):
        m = ""
        if d["run_dir"].isin(pending).any():
            m += r"\ddagger"
        if m:
            marks[arm] = "$^{" + m + "}$"
    return marks, suffix


def export(csv_paths: list[Path], paper: Path, *, ref: str = "a0_directed", grid_suts=GRID_SUTS,
           validity: Path | None = None, runs_root: Path | None = None, n_boot: int = st.N_BOOT,
           figures: bool = True, agree: list[Path] | None = None,
           transfer: list[Path] | None = None, human: Path | None = None,
           planned: Path | None = None, omit_arms: frozenset = frozenset(), legacy: dict | None = None,
           calib: Path | None = None, modswap: list[Path] | None = None) -> Numbers:
    from mmb.analysis import gallery, plots

    gen = Path(paper) / "generated"
    tabs, figs = gen / "tables", gen / "figures"
    tabs.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df = st.load_results(csv_paths)
    all_runs = list(df["run_dir"])
    if df["arm"].isna().any():
        raise SystemExit("some runs have no arm; re-run collect with --default arm=<id>")
    if agree:                                   # per-run verification flag from the replay CSVs
        adf = load_agree(agree)
        if len(adf) and "run_dir" in adf:
            v = adf.drop_duplicates("run_dir").set_index("run_dir")["cross_verified"]
            df["verified"] = df["run_dir"].map(v)
    pre = df["arm"].astype(str).str.split("_").str[0]
    gone = pre.isin(DROPPED_ARMS | set(omit_arms))
    if gone.any():    # arms not in the study (or omitted): no table, macro, curve or Holm family
        logger.warning("dropped %d runs of arms outside the study: %s", int(gone.sum()), sorted(set(df.loc[gone, "arm"])))
        df = df[~gone]
    s0only = df["arm"].astype(str).str.split("_").str[0].isin(SEED0_ONLY_ARMS) & (df["rng_seed"] != 0)
    if s0only.any():   # ablation arms: seed 0 only in the paper (data of other seeds kept)
        logger.warning("ablation arms on seed 0 only: %d runs of other seeds left out", int(s0only.sum()))
        df = df[~s0only]
    off = ~df["sut"].isin(list(grid_suts))
    if off.any():     # only the balanced grid is exported
        logger.warning("dropped %d runs of SUTs outside the grid: %s", int(off.sum()), sorted(set(df.loc[off, "sut"])))
        df = df[~off]
    df = complete_blocks(df, ref, grid_suts)
    design = check_balance(df, grid_suts)
    bad_seed = [b for b in design if b[0].split("_")[0] in SEED0_ONLY_ARMS and b[1] != 0]
    if bad_seed:
        raise UnbalancedExport(f"ablation arms must be on seed 0 only, found {bad_seed}")
    excluded = set(all_runs) - set(df["run_dir"])
    logger.info("balanced design: %s", ", ".join(f"{a} s{s} ({n})" for (a, s), n in sorted(design.items())))
    df = df.copy()
    pending = pending_confirmation(df)
    df["min_pending"] = df["run_dir"].isin(pending)
    if pending:      # smallest crossings not yet confirmed by single-input scoring: their metrics are withheld
        for c in ("magnitude", "mag_image", "mag_text", "min_text_d", "min_eval_idx"):
            if c in df:
                df.loc[df["min_pending"], c] = np.nan
    num = Numbers()
    num.add(["EPending", "Confirm", "Runs"], str(len(pending)), "runs whose smallest crossing awaits single-input confirmation (its metrics withheld)")
    if planned is not None:
        num.datastatus, n_ok, n_plan = data_status(df, planned, Path(planned).resolve().parents[2])
        logger.info("data status: %s (%d of %d planned configurations complete)", num.datastatus, n_ok, n_plan)
    if df.empty:
        logger.warning("no runs left after the filters: only the run count is written")
        num.add(["EOne", "Runs", "Total"], "0", "runs in the results CSV")
        num.write(gen)
        return num
    sc = scopes(df, grid_suts)
    budgets = list(st.DEFAULT_BUDGETS)
    num.add(["EOne", "Runs", "Total"], str(len(df)), "runs in the results CSV")
    if "complete" in df:
        num.add(["EOne", "Runs", "Incomplete"], str(int((~df["complete"].astype(bool)).sum())), "runs with missing generations")

    for (sut, task), d in sc.items():
        arms = set(d["arm"])
        res = st.run_all(d, ref if ref in arms and len(arms) > 1 else None, _tmpdir(gen, sut, task), budgets, n_boot)
        add_summary_numbers(num, sut, task, res["summary"])
        add_curve_numbers(num, sut, task, res["curve"])
        cs_, ps_, ms_ = cost_summary(d, n_boot), pair_summary(d), modality_summary(d)
        add_cost_numbers(num, sut, task, cs_)
        add_pair_numbers(num, sut, task, ps_)
        add_modality_numbers(num, sut, task, ms_)
        ct = pd.DataFrame()
        if ref in arms and len(arms) > 1:
            ct = pr.paired_table(d, ref, pr.DEFAULT_METRICS, n_boot=n_boot)
            add_cluster_numbers(num, sut, task, ct)
            for a in sorted(arms - {ref}):
                npair = len(pr.pair_frame(d, ref, a, "magnitude"))
                num.add(["EOne", "PairsBothCrossed", arm_short(a), camel(sut), camel(task)], str(npair),
                        f"matched (cell, seed) pairs in which {a} and {ref} both have an in-budget smallest crossing; {sut}, {task}")
        if "paired" in res:
            add_paired_numbers(num, sut, task, res["paired"])
        if (sut, task) == ("All", "All"):
            write_tex(tabs / "e1.tex", st.summary_tex(res["summary"]))
            if "paired" in res:
                write_tex(tabs / "e1_paired.tex", st.paired_tex(res["paired"]))
            if len(ct):
                write_tex(tabs / "e1_cluster.tex", cluster_table_tex(ct))
            write_tex(tabs / "cost.tex", cost_tex(cs_))
            write_tex(tabs / "boundary_pair.tex", pair_tex(ps_))
            write_tex(tabs / "min_modality.tex", modality_tex(ms_))
            if ref in arms:
                for a in sorted(arms - {ref}):
                    add_mixed_numbers(num, pr.mixed_model(d.assign(cell_id=d["cluster_id"]), ref, a))
    _cleanup_tmp(gen)
    pt = primary_tables(df, ref, n_boot)
    if len(pt):
        write_tex(tabs / "primary.tex", primary_tex(pt))
        add_primary_numbers(num, pt)
    add_wall_numbers(num, sc)

    need = {"prior_g", "probe_gen0", "min_gen0"}
    if need <= set(df.columns) and df["prior_g"].notna().any():
        a0 = df[df["arm"] == ref]
        if len(a0):
            try:
                cells = pred.cell_frame(pred.prepare(a0), ["sut", "task"])
                tab, corr, part = e2b_table(cells, n_boot)
                corr.to_csv(gen / "e2b_cell_correlations.csv", index=False)
                add_predictor_numbers(num, corr, part)
            except (ValueError, KeyError) as e:
                logger.warning("E2b skipped: %s", e)
    else:
        logger.warning("no prior_g / probe columns: E2b skipped")
    from mmb.analysis import hardness as hd
    hdf = hd.frame(df)
    if len(hdf):
        hd.add_numbers(num, hd.correlations(hdf))

    if validity is not None and Path(validity).exists():
        vdf = pd.read_csv(validity)
        if len(vdf) and "run_dir" in vdf:
            vdf = vdf[vdf["run_dir"].isin(set(df["run_dir"]))]
        if len(vdf) and pending and "view" in vdf:
            vdf = vdf[~((vdf["view"] == "min") & vdf["run_dir"].isin(pending))]
        if len(vdf):
            v1 = vdf[vdf["view"] == "first"] if "view" in vdf else vdf
            add_validity_numbers(num, v1)
            from mmb.analysis import validity_cost as vc
            from mmb.analysis import rq_tables as rq
            if {"sut", "task", "arm", "view"} <= set(vdf.columns):
                _rq_tables(rq, tabs, df, vdf, grid_suts, num, method_marks(df, ref, pending)[0])
            for view, fam, lab in (("min", "EValid", "smallest crossing"), ("first", "EValidFirst", "first crossing")):
                vct = vc.validity_cost(vdf, df, grid_suts, view=view)
                if len(vct):
                    vc.add_numbers(num, vct, fam, lab)

    if agree:
        adf = load_agree(agree)
        if len(adf) and "run_dir" in adf:
            adf = adf[~adf["run_dir"].isin(excluded)]
        if len(adf):
            write_tex(tabs / "agree.tex", agree_table(adf))
            add_agree_numbers(num, adf)

    if transfer:
        tdf = tr.load_transfer(transfer, results=csv_paths[0])
        tdf = tdf[~tdf["run_dir"].isin(excluded)]
        trows = tr.matrix_rows(tdf) if len(tdf) else []
        if trows:
            add_transfer_numbers(num, trows)
            if "arm" in tdf:
                for arm_name, tg in tdf.groupby("arm"):
                    for task_name, tt in [("All", tg)] + [(t, tg[tg["task"] == t]) for t in sorted(set(tg["task"]))]:
                        m, n = tr.mean_off_diagonal(tr.matrix_rows(tt), "All")
                        if n:
                            num.add(["ETransferArm", "Share", arm_short(arm_name), camel(task_name)], f_pct(m), f"pooled share of crossings flipping another SUT (%), arm {arm_name}, {task_name}")
                            num.add(["ETransferArm", "N", arm_short(arm_name), camel(task_name)], str(n), f"crossing-destination pairs behind it, arm {arm_name}, {task_name}")
                add_transfer_paired(num, tdf, ref)

    if human is not None and Path(human).exists():
        hdf_ = pd.read_csv(human, keep_default_na=False)
        if len(hdf_):
            write_tex(tabs / "human_validity.tex", human_tex(hdf_))
            add_human_numbers(num, hdf_)

    if figures:
        grid_df = sc.get(("All", "All"))
        if grid_df is not None:
            plots.plot_crossing_curve(grid_df, figs / "crossing-curve.pdf", ref_arm=ref, n_boot=n_boot,
                                      label_suffix=method_marks(grid_df, ref, pending)[1])
        if "probe_gen0" in df and df["probe_gen0"].notna().any():
            plots.plot_predictor_scatter(df, figs / "predictor-main.pdf", arm=ref)
        examples = gallery.select_examples(df, ref, runs_root=runs_root)
        if not examples:
            logger.warning("gallery: no MMBA run with a stored in-budget smallest crossing yet")
        else:
            try:
                gallery.plot_teaser(examples, figs / "gallery.pdf", runs_root=runs_root)
                add_teaser_numbers(num, gallery.teaser_facts(examples, runs_root=runs_root))
            except FileNotFoundError as e:
                logger.warning("gallery skipped: %s", e)
    if calib is not None:
        add_calib_numbers(num, Path(calib), figs if figures else None)
    if modswap:
        from mmb.analysis import modality_swap as msw
        md = msw.load(modswap)
        if len(md):
            for view in ("min", "first"):
                if (md["view"] == view).any():
                    msw.add_numbers(num, md, camel, view)
            if figures and (md["view"] == "min").any():
                msw.plot(md, figs / "modality-swap.pdf", st.arm_label, view="min")
    if legacy:
        kept = []
        for name, (value, desc) in legacy.items():
            if name not in num.seen:
                num.add_legacy(name, value, desc)
                kept.append(name)
        (gen / "legacy_macros.txt").write_text("\n".join(sorted(kept)) + ("\n" if kept else ""))
    else:
        (gen / "legacy_macros.txt").unlink(missing_ok=True)
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
    ap.add_argument("--planned", type=Path, default=None, help="configs/generated/MAIN: planned configurations (data status)")
    ap.add_argument("--n-boot", type=int, default=st.N_BOOT)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--omit-arms", default=",".join(sorted(PAPER_OMIT_ARMS)),
                    help="arm prefixes kept in the data but omitted from the paper (default: %(default)s; '' = none)")
    ap.add_argument("--calib", type=Path, default=None, help="calibration output dir with trajectory.csv (ECalibTraj* macros)")
    ap.add_argument("--modswap", type=Path, nargs="*", default=None, help="scripts/modality_swap.py CSVs (EModSwap*, modality-swap.pdf)")
    ap.add_argument("--legacy", action="store_true",
                    help="also keep the macros of omitted arms from a full export (verbatim, listed in legacy_macros.txt)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    omit = frozenset(x for x in args.omit_arms.split(",") if x)
    kw = dict(ref=args.ref, grid_suts=args.grid_suts.split(","), validity=args.validity, runs_root=args.runs_root,
              n_boot=args.n_boot, agree=args.agree or None, transfer=args.transfer or None, human=args.human_validity,
              planned=args.planned)
    legacy = None
    if omit and args.legacy:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "main.tex").write_text("")
            full = export(args.csv, Path(tmp), figures=False, **kw)
            legacy = {r["macro"]: (r["value"], r["description"]) for r in full.rows}
    num = export(args.csv, args.paper, figures=not args.no_figures, omit_arms=omit, legacy=legacy, calib=args.calib, modswap=args.modswap, **kw)
    logger.info("wrote %d macros to %s", len(num.rows), args.paper / "generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())