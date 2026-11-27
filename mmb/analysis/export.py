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
from mmb.analysis import tablefmt as tf

logger = logging.getLogger(__name__)

DIGITS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
GRID_SUTS = ("qwen", "internvl", "ministral")
#: arm prefixes dropped from the study (user decision 2026-11-25: GA, NSGA-II, hill climber); filtered before any output
DROPPED_ARMS = frozenset({"a3", "a4", "a8"})
#: arm prefixes run on the full balanced grid but omitted from the paper (user decision 2026-11-27: random search, for
#: focus); their runs stay in runs/MAIN and results/*.csv. Filtered before any table, figure or Holm family.
PAPER_OMIT_ARMS = frozenset({"a1"})
#: RQ4 ablation arms are reported on rng seed 0 only (user decision 2026-11-27, limited resources); their seed-1 runs
#: stay on disk but are filtered before any output. RQ1/RQ2 methods (a0, a5, a9) use both seeds.
SEED0_ONLY_ARMS = frozenset({"a6", "a7", "x1", "x2"})
#: generated files that exist only to compare MMBA with an omitted arm; removed when that arm is omitted
OMIT_STALE = {"a1": ("tables/primary.tex", "tables/generalization_main.tex", "tables/e4.tex", "figures/per-cell-diff.pdf")}
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


# ---------------------------------------------------------------------------
# The three main-text tables (one per research question group): one value per cell, no intervals,
# significance as a marker explained once in the text.
# ---------------------------------------------------------------------------


def validity_main_tex(vct: pd.DataFrame, task_names: dict) -> str:
    """RQ3: per task and arm the image change (LPIPS against the unedited input) and the text change (visible
    character edit fraction) at the minimal crossing; best value bold within each task."""
    rows = []
    for task in [t for t in dict.fromkeys(vct["task"]) if t in task_names]:     # the pooled "All" block is not shown
        g = vct[vct["task"] == task]
        g = g[pd.to_numeric(g["n_cross"], errors="coerce").fillna(0) > 0]    # no scored minimal crossing (e.g. pending confirmation): no row
        lp = [float(v) for v in g["med_lpips"]]
        tx = [100 * float(v) for v in g["med_vis_char"]]
        lc = [f"{v:.3f}" for v in lp]
        tc = [f"{v:.0f}" for v in tx]
        lc, tc = tf.bold_best(lc, lp, "lpips", lc), tf.bold_best(tc, tx, "text_change", tc)
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append([st.tex_escape(task_names.get(task, task)) if i == 0 else "", st.tex_escape(st.arm_label(r["arm"])),
                         f"{int(r['n_cross'])}", lc[i], tc[i]])
    return st._booktabs("llrrr", ["Task", "Arm", "Crossings", tf.head("Image change (LPIPS)", "lpips"),
                                  tf.head(r"Text change (\% of characters)", "text_change")], rows)


def generalization_main_tex(primary: pd.DataFrame, transfer_rows: list[dict] | None = None) -> str:
    """RQ4: crossing rate of the directed and the random arm per model and task with the effect size of the
    difference (pp, bullet, dagger), and the transfer rate of the crossings found on each model."""
    suts = [s for s in SUT_NAMES if s in set(primary["sut"])]
    tasks = [t for t in ("imagenet", "pope") if t in set(primary["task"])]
    rate = {}
    if transfer_rows:
        for s in suts:
            r = [x for x in transfer_rows if x["task"] == "All" and x["src"] == s and x["dst"] != s]
            n = sum(x["n"] for x in r)
            if n:
                rate[s] = 100 * sum(x["k"] for x in r) / n
    rows = []
    for s in suts:
        row = [st.tex_escape(SUT_NAMES.get(s, s))]
        for t in tasks:
            q = primary[(primary["sut"] == s) & (primary["task"] == t)]
            if q.empty:
                row += ["--", "--", "--"]
                continue
            r = q.iloc[0]
            d, a = f"{100 * r.ref_value:.0f}", f"{100 * r.arm_value:.0f}"
            cells = tf.bold_best([d, a], [r.ref_value, r.arm_value], "crossed", [d, a])
            row += [cells[0], cells[1], tf.effect_cell(f"{100 * r['diff']:+.0f}", r["dz"], r.p_holm_strata)]
        if rate:
            row.append(f"{rate[s]:.0f}" if s in rate else "--")
        rows.append(row)
    header = ["Model"]
    for t in tasks:
        header += [tf.head(rf"{TASK_NAMES.get(t, t)}: directed (\%)", "crossed"), tf.head(r"random (\%)", "crossed"), r"effect (pp)"]
    if rate:
        header.append(r"Transfer rate (\%)")
    return st._booktabs("l" + "rrr" * len(tasks) + ("r" if rate else ""), header, rows)


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
                      "min_edits_x": "MinEditsX", "edits_img_x": "EditsImgX", "edits_txt_x": "EditsTxtX",
                      "hv_edits": "HvEdits", "hv_mag": "HvMag", "img_mag": "ImgMag", "txt_mag": "TxtMag",
                      "chardist": "CharDist"}
METRIC_HEAD = {"crossed": "Crossing rate", "first_cross_eval": "Model calls to first crossing", "floor": "Closest approach",
               "min_edits": "Edits, censored at all genes", "min_edits_x": "Minimal edits",
               "edits_img_x": "Image edits, minimal crossing", "edits_txt_x": "Text edits, minimal crossing", "hv_edits": "Edit hypervolume"}


#: paired metrics whose values are counts (model calls, edits): their medians print as integers or with one decimal
COUNT_METRICS = frozenset({"first_cross_eval", "min_edits_x", "edits_img_x", "edits_txt_x", "min_edits", "min_cross_eval"})


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
            unit = ""
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


def primary_tex(t: pd.DataFrame, short: str = "a1") -> str:
    """Crossing rate of the directed and the compared arm per stratum, and the effect size of the difference in
    percentage points with its magnitude bullet and dagger (Holm across strata)."""
    rows = []
    for _, r in t.iterrows():
        rows.append([st.tex_escape(TASK_NAMES.get(r.task, r.task)), st.tex_escape(SUT_NAMES.get(r.sut, r.sut)), str(int(r.n_pairs)),
                     f"{100 * r.ref_value:.0f}", f"{100 * r.arm_value:.0f}",
                     tf.effect_cell(f"{100 * r['diff']:+.0f}", r["dz"], r.p_holm_strata)])
    return st._booktabs("llrrrr", ["Task", "SUT", "Pairs", tf.head("Directed crossing rate (\\%)", "crossed"),
                                   tf.head(f"{st.arm_label(short)} crossing rate (\\%)", "crossed"),
                                   f"Difference (pp)"], rows)


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
        cx = g[g["crossed"].astype(bool)]            # edit counts of crossing runs only (no censoring)
        def _med(col):
            return float(cx[col].median()) if col in cx and cx[col].notna().any() else np.nan
        mx, xlo, xhi = (st.bootstrap_stat(cx["min_edits_cross"].to_numpy(float), n_boot=n_boot)
                        if len(cx) and "min_edits_cross" in cx else (np.nan,) * 3)
        rows.append(dict(arm=arm, n_runs=len(g), n_crossed=len(c), med_min_edits=me, me_lo=lo, me_hi=hi,
                         med_edits_x=mx, ex_lo=xlo, ex_hi=xhi, med_edits_img=_med("edits_img_at_min"), med_edits_txt=_med("edits_txt_at_min"),
                         med_hv=hv, hv_lo=hlo, hv_hi=hhi,
                         med_img_mag=float(c["best_img_mag_cross"].median()) if "best_img_mag_cross" in c and c["best_img_mag_cross"].notna().any() else np.nan,
                         med_txt_mag=float(c["best_txt_mag_cross"].median()) if "best_txt_mag_cross" in c and c["best_txt_mag_cross"].notna().any() else np.nan))
    return pd.DataFrame(rows)


def cost_tex(s: pd.DataFrame) -> str:
    def fmt(v, d):
        return "--" if pd.isna(v) else f"{v:.{d}f}"
    def col(vals, d, key):
        cells = [fmt(v, d) for v in vals]
        return tf.bold_best(cells, list(vals), key, cells)
    cols = [col(s.med_edits_x, 0, "min_edits_x"), col(s.med_edits_img, 0, "edits_img_x"), col(s.med_edits_txt, 0, "edits_txt_x"),
            col(s.med_min_edits, 0, "min_edits"), col(s.med_hv, 3, "hv_edits"), col(s.med_img_mag, 3, "img_mag"), col(s.med_txt_mag, 2, "txt_mag")]
    rows = [[st.tex_escape(st.arm_label(r.arm)), str(int(r.n_runs)), str(int(r.n_crossed))] + [cols[j][i] for j in range(7)] for i, r in enumerate(s.itertuples())]
    return st._booktabs("lrrrrrrrrr", ["Arm", "Runs", "Crossed", tf.head("Minimal edits", "min_edits_x"),
                                   tf.head("Image edits", "edits_img_x"), tf.head("Text edits", "edits_txt_x"),
                                   tf.head("Edits, censored at all genes", "min_edits"), tf.head("Edit hypervolume", "hv_edits"),
                                   tf.head("Smallest image change (RMS)", "img_mag"), tf.head("Smallest text change (cosine)", "txt_mag")], rows)


def add_cost_numbers(num: Numbers, sut: str, task: str, s: pd.DataFrame) -> None:
    for _, r in s.iterrows():
        tail = [arm_short(r.arm), camel(sut), camel(task)]
        d = f"{r.arm}, {sut}, {task}"
        num.add(["ECost", "MedEditsX"] + tail, f_num(r.med_edits_x, 0), "median fewest active genes at a crossing, crossing runs only; " + d)
        num.add(["ECost", "MedEditsXCi"] + tail, f"[{r.ex_lo:.0f}--{r.ex_hi:.0f}]" if not pd.isna(r.ex_lo) else "--", "bootstrap 95% CI; " + d)
        num.add(["ECost", "MedEditsImg"] + tail, f_num(r.med_edits_img, 0), "median image genes of the cheapest crossing input, crossing runs only; " + d)
        num.add(["ECost", "MedEditsTxt"] + tail, f_num(r.med_edits_txt, 0), "median text genes of the cheapest crossing input, crossing runs only; " + d)
        num.add(["ECost", "MedMinEdits"] + tail, f_num(r.med_min_edits, 0), "median fewest active genes, never-crossed runs censored at all genes; " + d)
        num.add(["ECost", "MedMinEditsCi"] + tail, f"[{r.me_lo:.0f}--{r.me_hi:.0f}]", "bootstrap 95% CI; " + d)
        num.add(["ECost", "MedHv"] + tail, f_num(r.med_hv, 3), "median edit hypervolume fraction (0 when never crossed); " + d)
        num.add(["ECost", "MedHvCi"] + tail, f"[{r.hv_lo:.3f}--{r.hv_hi:.3f}]", "bootstrap 95% CI; " + d)
        if not pd.isna(r.med_img_mag):
            num.add(["ECost", "MedImgMag"] + tail, f_num(r.med_img_mag, 3), "median best image RMS among crossing runs; " + d)
        if not pd.isna(r.med_txt_mag):
            num.add(["ECost", "MedTxtMag"] + tail, f_num(r.med_txt_mag, 2), "median best text cosine among crossing runs; " + d)


def add_failure_numbers(num: Numbers, minimal: pd.DataFrame) -> None:
    """Macros of the failure-type study: per (SUT, task) scope and arm group the number of reduced crossings, the share
    that re-verified, the median genes before and after the reduction and the share of each failure type."""
    from mmb.analysis import failure_types as ft

    for r in ft.summary(minimal).itertuples():
        tail = [{"directed": "Directed", "others": "Others"}[r.group], camel(r.sut), camel(r.task)]
        d = f"{r.group} arms, {r.sut}, {r.task}"
        num.add(["EFail", "N"] + tail, str(int(r.n)), "smallest crossings submitted to the minimiser; " + d)
        num.add(["EFail", "Verified"] + tail, f_pct(r.verified_share, 0), "share of the smallest (final minimal) crossings that verify by single-item re-scoring (%); " + d)
        num.add(["EFail", "MedGenesFirst"] + tail, f_num(r.med_genes_first, 0), "median changed genes of the smallest crossing in the trace; " + d)
        num.add(["EFail", "MedGenesMin"] + tail, f_num(r.med_genes_min, 0), "median changed genes after the reduction; " + d)
        num.add(["EFail", "MedImgMin"] + tail, f_num(r.med_img_min, 0), "median image genes of the minimal flip; " + d)
        num.add(["EFail", "MedTxtMin"] + tail, f_num(r.med_txt_min, 0), "median text genes of the minimal flip; " + d)
        num.add(["EFail", "OneGene"] + tail, f_pct(r.one_gene_share, 0), "share of minimal flips that need a single gene (%); " + d)
        for k in ft.TYPE_KEYS:
            num.add(["EFail", "Share", ft.TYPE_WORDS[k]] + tail, f_pct(getattr(r, "share_" + k), 0), f"share of minimal flips of type {k} (%); " + d)


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
    abl_marks = {rq.prefix(a): m.replace(r"\S", "") for a, m in marks.items()}     # RQ4 is seed 0 for every variant
    write_tex(tabs / "arm_ablation.tex", rq.ablation_tex(abl, abl_e, {k: v for k, v in abl_marks.items() if v != "$^{}$"}))
    rq.add_ablation_numbers(num, abl)
    rq.add_ablation_effect_numbers(num, abl, abl_e)
    extra = rq.ablation(gd, gv, variants=rq.VARIANTS[:1] + rq.MACRO_ONLY_VARIANTS)      # EAbl*Mimicry/Janus: macros only
    rq.add_ablation_numbers(num, extra[extra["pre"].isin([p for _l, p, _v in rq.MACRO_ONLY_VARIANTS])])
    rq.add_pooled_numbers(num, gd, vdf[vdf["sut"].isin(grid_suts)], arm_short)


def add_teaser_numbers(num: Numbers, facts: list[dict]) -> None:
    """Macros for the Fig. 1 rows (ETeaserImagenet*/ETeaserPope*), so the introduction quotes the figure it shows."""
    def odds(r):
        return f"{r:.0f}" if r >= 10 else f"{r:.1f}"
    for f in facts:
        row = ["ETeaser", camel(f["task"])]
        d = f"Fig. 1 row, {f['task']}"
        for k, v in (("Sut", f["sut"]), ("Before", f["before"]), ("After", f["after"]), ("Object", f["obj"]),
                     ("OddsBefore", odds(f["odds_before"])), ("OddsAfter", odds(f["odds_after"]))):
            num.add(row + [k], st.tex_escape(str(v)), f"{k}; {d}")
        for who, e in [("Dir", f["dir"])] + ([("Rnd", f["rnd"])] if "rnd" in f else []):
            for k in ("n", "img", "txt"):
                num.add(row + [who, {"n": "Edits", "img": "Img", "txt": "Txt"}[k]], str(int(e[k])), f"{who} {k}; {d}")


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
    """run_dirs whose minimal crossing has not been confirmed by single-input scoring yet (scripts/confirm_crossings.py
    has not run on them). Their min_cross.json is the search's batched output, so every minimal-crossing metric of
    these runs (calls to minimal, LPIPS, edit distances, stealth) is withheld; crossing share and calls to the first
    crossing come from the trace for every method alike and stay."""
    if "min_confirmed" not in df:
        return set()
    ok = df["min_confirmed"].astype(str).str.lower().isin(("true", "1", "1.0"))
    out = set(df.loc[~ok, "run_dir"])
    if out:
        arms = sorted(set(df.loc[df["run_dir"].isin(out), "arm"]))
        logger.info("minimal crossings pending confirmation: %d runs (%s); minimal-crossing metrics withheld", len(out), ", ".join(arms))
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
           planned: Path | None = None, minimal: list[Path] | None = None,
           omit_arms: frozenset = frozenset(), legacy: dict | None = None) -> Numbers:
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
    gone = df["arm"].astype(str).str.split("_").str[0].isin(DROPPED_ARMS)
    if gone.any():    # GA, NSGA-II, hill climber dropped 2026-11-25 (user decision): no table, macro or curve
        logger.warning("dropped %d runs of dropped methods: %s", int(gone.sum()), sorted(set(df.loc[gone, "arm"])))
        df = df[~gone]
    om = df["arm"].astype(str).str.split("_").str[0].isin(omit_arms)
    if om.any():      # run on the full grid, omitted from the paper (data kept): no table, figure, macro or Holm family
        logger.warning("omitted from the paper: %d runs of %s", int(om.sum()), sorted(set(df.loc[om, "arm"])))
        df = df[~om]
    s0only = df["arm"].astype(str).str.split("_").str[0].isin(SEED0_ONLY_ARMS) & (df["rng_seed"] != 0)
    if s0only.any():   # ablation arms: seed 0 only in the paper (data of other seeds kept)
        logger.warning("ablation arms on seed 0 only: %d runs of other seeds left out", int(s0only.sum()))
        df = df[~s0only]
    for pre in omit_arms:
        for rel in OMIT_STALE.get(pre, ()):
            (gen / rel).unlink(missing_ok=True)
    off = ~df["sut"].isin(list(grid_suts))
    if off.any():     # only the balanced grid is exported (LLaVA-NeXT dropped 2026-11-25: no row or macro for it)
        logger.warning("dropped %d runs of SUTs outside the grid: %s", int(off.sum()), sorted(set(df.loc[off, "sut"])))
        df = df[~off]
    df = complete_blocks(df, ref, grid_suts)
    design = check_balance(df, grid_suts)
    bad_seed = [b for b in design if b[0].split("_")[0] in SEED0_ONLY_ARMS and b[1] != 0]
    if bad_seed:
        raise UnbalancedExport(f"ablation arms must be on seed 0 only, found {bad_seed}")
    excluded = set(all_runs) - set(df["run_dir"])    # runs left out on purpose (dropped/omitted arm, partial block, non-grid)
    logger.info("balanced design: %s", ", ".join(f"{a} s{s} ({n})" for (a, s), n in sorted(design.items())))
    pending = pending_confirmation(df)
    df["min_pending"] = df["run_dir"].isin(pending)
    if pending:      # minimal crossings of these runs are the search's own, not yet confirmed by single-input scoring
        df.loc[df["min_pending"], "min_cross_eval"] = np.nan
    num = Numbers()
    num.add(["EPending", "Confirm", "Runs"], str(len(pending)), "runs whose minimal crossing awaits single-input confirmation (minimal-crossing metrics withheld)")
    if planned is not None:
        num.datastatus, n_ok, n_plan = data_status(df, planned, Path(planned).resolve().parents[2])
        logger.info("data status: %s (%d of %d planned configurations complete)", num.datastatus, n_ok, n_plan)
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
            for a in sorted(arms - {ref}):
                npair = len(pr.pair_frame(d, ref, a, "min_edits_x"))
                num.add(["EOne", "PairsBothCrossed", arm_short(a), camel(sut), camel(task)], str(npair),
                        f"matched (cell, seed) pairs in which {a} and {ref} both crossed; {sut}, {task}")
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

    add_wall_numbers(num, sc)

    prim = df[df["arm"].isin(PRIMARY_ARMS)]
    # (E3, the four-model ImageNet continuity table, is gone: LLaVA-NeXT dropped 2026-11-25, balanced grid only)
    # E4: a0 and a1 per task, grid SUTs pooled
    frames = []
    for task in sorted(set(prim["task"])):
        d = prim[(prim["task"] == task) & prim["sut"].isin(set(grid_suts))]
        if len(d):
            frames.append(([TASK_NAMES.get(task, task)], st.arm_summary(with_scope_cells(d), n_boot)))
    if frames and prim["arm"].nunique() > 1:      # a comparison table: not written with one arm left
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
        if len(vdf) and "run_dir" in vdf:        # only runs that passed the gates (dropped methods, partial blocks)
            vdf = vdf[vdf["run_dir"].isin(set(df["run_dir"]))]
        if len(vdf) and pending and "view" in vdf:      # unconfirmed minimal crossings never enter a table
            vdf = vdf[~((vdf["view"] == "min") & vdf["run_dir"].isin(pending))]
        if len(vdf):
            v1 = vdf[vdf["view"] == "first"] if "view" in vdf else vdf
            write_tex(tabs / "validity.tex", validity_table(v1))
            add_validity_numbers(num, v1)
            from mmb.analysis import validity_cost as vc
            from mmb.analysis import rq_tables as rq
            if {"sut", "task", "arm", "view"} <= set(vdf.columns):
                _rq_tables(rq, tabs, df, vdf, grid_suts, num, method_marks(df, ref, pending)[0])
            # primary view: the minimal crossing (EValid*, validity_main); the first-crossing view is supplement only (EValidFirst*)
            for view, stem, fam, lab in (("min", "validity_cost_min", "EValid", "minimal crossing"),
                                         ("first", "validity_cost", "EValidFirst", "first crossing")):
                vct = vc.validity_cost(vdf, df, grid_suts, view=view)
                if len(vct):
                    write_tex(tabs / f"{stem}.tex", vc.table_tex(vct, TASK_NAMES, objective=(view == "first")))
                    if view == "min":
                        write_tex(tabs / "validity_main.tex", validity_main_tex(vct, TASK_NAMES))
                    vc.add_numbers(num, vct, fam, lab)
                    write_tex(tabs / f"{stem.replace('cost', 'sensitivity')}.tex",
                              vc.sensitivity_tex(vc.sensitivity(vdf, df, grid_suts, view)))

    if agree:
        adf = load_agree(agree)
        if len(adf) and "run_dir" in adf:       # replayed runs of the exported arms only
            adf = adf[~adf["run_dir"].isin(excluded)]
        if len(adf):
            write_tex(tabs / "agree.tex", agree_table(adf))
            add_agree_numbers(num, adf)

    trows = None
    if transfer:
        tdf = tr.load_transfer(transfer, results=csv_paths[0])
        tdf = tdf[~tdf["run_dir"].isin(excluded)]                 # crossings of the exported arms only
        trows = tr.matrix_rows(tdf) if len(tdf) else []
        if trows:
            write_tex(tabs / "transfer.tex", tr.matrix_tex(trows, "All", SUT_NAMES))
            for task in sorted(set(tdf["task"])):
                write_tex(tabs / f"transfer_{task}.tex", tr.matrix_tex(trows, task, SUT_NAMES))
            add_transfer_numbers(num, trows)
            if "arm" in tdf:
                for arm_name, tg in tdf.groupby("arm"):
                    for task_name, tt in [("All", tg)] + [(t, tg[tg["task"] == t]) for t in sorted(set(tg["task"]))]:
                        m, n = tr.mean_off_diagonal(tr.matrix_rows(tt), "All")
                        if n:
                            num.add(["ETransferArm", "Share", arm_short(arm_name), camel(task_name)], f_pct(m), f"pooled share of crossings flipping another SUT (%), arm {arm_name}, {task_name}")
                            num.add(["ETransferArm", "N", arm_short(arm_name), camel(task_name)], str(n), f"crossing-destination pairs behind it, arm {arm_name}, {task_name}")
                add_transfer_paired(num, tdf, ref)

    if len(pt):
        write_tex(tabs / "generalization_main.tex", generalization_main_tex(pt, trows))

    if human is not None and Path(human).exists():
        hdf = pd.read_csv(human, keep_default_na=False)
        if len(hdf):
            write_tex(tabs / "human_validity.tex", human_tex(hdf))
            add_human_numbers(num, hdf)

    if figures:
        grid_df = sc.get(("All", "All"))
        if grid_df is not None:
            plots.plot_crossing_curve(grid_df, figs / "crossing-curve.pdf", ref_arm=ref, vlines=[206], n_boot=n_boot,
                                      label_suffix=method_marks(grid_df, ref, pending)[1])
        if len(pt):
            pr.plot_per_cell_diff(df, figs / "per-cell-diff.pdf", ref, "a1_random_tiered")
        if "probe_gen0" in df and df["probe_gen0"].notna().any():
            plots.plot_predictor_scatter(df[df["sut"].isin(set(grid_suts))], figs / "predictor-main.pdf", arm=ref)
        gdf = df[df["sut"].isin(set(grid_suts))]
        if "a1_random_tiered" in set(gdf["arm"]):     # teaser: directed vs random minimal crossing, with difference panes
            teaser = gallery.select_teaser(gdf, ref, "a1_random_tiered", runs_root=runs_root)
        else:                                          # random omitted from the paper: MMBA panels only (rule: gallery.select_teaser_single)
            teaser = gallery.select_teaser_single(gdf, ref, runs_root=runs_root)
        if not teaser:
            logger.warning("teaser: no cell with a stored minimal crossing for both arms yet")
        else:
            try:
                gallery.plot_teaser(teaser, figs / "gallery.pdf", runs_root=runs_root)
                add_teaser_numbers(num, gallery.teaser_facts(teaser, runs_root=runs_root))
            except FileNotFoundError as e:
                logger.warning("teaser skipped: %s", e)
    mdf = None
    if runs_root is not None and Path(runs_root).exists():
        from mmb.analysis import failure_types as ft, min_types as mtp
        mdf = mtp.load_runs(runs_root)
        if len(mdf):
            mdf = mdf[~mdf["run_dir"].isin(excluded)]                # runs of the exported arms only
        if len(mdf):
            write_tex(tabs / "arm_failure.tex", mtp.table_tex(mdf))
            mtp.add_numbers(num, mdf, camel, f_num, f_pct)
            from mmb.analysis import hardness as hd
            hdf = hd.frame(df, mdf)
            if len(hdf):
                hd.add_numbers(num, hd.correlations(hdf))
    elif minimal:
        from mmb.analysis import failure_types as ft
        mdf = ft.load_minimal(minimal)
    if mdf is not None:
        from mmb.analysis import failure_types as ft
        if len(mdf):
            add_failure_numbers(num, mdf)
            if figures:
                ft.figure(mdf, figs / "failure-types.pdf", sut_names={k: v.split(" (")[0] for k, v in SUT_NAMES.items()},
                          task_names=TASK_NAMES)
    if legacy:        # macros of omitted arms from the full export, kept until the text no longer cites them
        kept = []
        for name, (value, desc) in legacy.items():
            if name not in num.seen:
                num.add_legacy(name, value, desc)
                kept.append(name)
        (gen / "legacy_macros.txt").write_text("\n".join(sorted(kept)) + ("\n" if kept else ""))
        logger.info("legacy macros kept (omitted arms): %d, listed in generated/legacy_macros.txt", len(kept))
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
    ap.add_argument("--minimal", type=Path, nargs="*", default=[], help="minimize_crossings.py CSVs (failure types)")
    ap.add_argument("--planned", type=Path, default=None, help="configs/generated/MAIN: planned configurations (data status)")
    ap.add_argument("--n-boot", type=int, default=st.N_BOOT)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--omit-arms", default=",".join(sorted(PAPER_OMIT_ARMS)),
                    help="arm prefixes kept in the data but omitted from the paper (default: %(default)s; '' = none)")
    ap.add_argument("--legacy", action="store_true",
                    help="also keep the macros of omitted arms from a full export (verbatim, listed in legacy_macros.txt)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    omit = frozenset(x for x in args.omit_arms.split(",") if x)
    kw = dict(ref=args.ref, grid_suts=args.grid_suts.split(","), validity=args.validity, runs_root=args.runs_root,
              n_boot=args.n_boot, agree=args.agree or None, transfer=args.transfer or None, human=args.human_validity,
              planned=args.planned, minimal=args.minimal or None)
    legacy = None
    if omit and args.legacy:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "main.tex").write_text("")
            full = export(args.csv, Path(tmp), figures=False, **kw)
            legacy = {r["macro"]: (r["value"], r["description"]) for r in full.rows}
    num = export(args.csv, args.paper, figures=not args.no_figures, omit_arms=omit, legacy=legacy, **kw)
    logger.info("wrote %d macros to %s", len(num.rows), args.paper / "generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
