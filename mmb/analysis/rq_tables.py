"""Main-text tables of the paper: one table per research question (RQ1 effectiveness, RQ2 efficiency, RQ4 ablation).

Metrics (defined once in the paper's Metrics paragraph), all per run and summarised by the median over runs unless noted:

* CR      share of runs that found a crossing (%), higher is better
* LPIPS   image distance of the minimal crossing to the original input (the unedited seed at the model's input resolution)
* LPIPS_bp image distance across the boundary pair (a sentence in the paper, not a table column): LPIPS between the minimal crossing and its non-crossing partner
          (the same input minus one edit); shown when at least MIN_PARTNERS partners exist
* d_edit  visible character edit distance of the question at the minimal crossing against the unedited question, in
          percent of its characters (NFKC, zero-width characters removed, homoglyphs mapped)
* d_sem   semantic distance of the visible question at the minimal crossing: 1 - MiniLM cosine of the normalised strings (d_edit = 0 implies d_sem = 0)
* calls   model calls to the first crossing (runs without a crossing are censored at the budget) and to the minimal crossing
* runtime wall-clock seconds per run

RQ1 and RQ2 hold one block per SUT with the columns grouped by task, so every SUT x task cell is shown; RQ4 pools SUTs and
tasks (seed 0). Best value per block column in bold (neutral columns excepted).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mmb.analysis import stats as st
from mmb.analysis import tablefmt as tf
from mmb.analysis import validity_cost as vc

MIN_PARTNERS = 3
SUT_NAMES = {"qwen": "Qwen3.5-4B", "internvl": "InternVL3.5-4B", "ministral": "Ministral-3-3B", "llava": "LLaVA-NeXT-7B"}
TASK_NAMES = {"imagenet": "ImageNet", "pope": "POPE"}
#: search methods compared in RQ1 and RQ2 (arm-id prefixes, in table order)
METHODS = ("a0", "a1", "a3", "a4", "a5", "a8", "a9")
#: RQ4 variants: (label, arm prefix, crossing view)
VARIANTS = (("Full \\sysname", "a0", "min"), ("w/o margin guidance", "x1", "min"), ("w/o Phase 2", "a0", "first"),
            ("Image only", "a7", "min"), ("Text only", "a6", "min"), ("Patch edits", "x2", "min"),
            ("GA", "a3", "min"), ("NSGA-II", "a4", "min"), ("MIMICRY-style", "a5", "min"), ("Hill climber", "a8", "min"),
            ("DeepJanus-style", "a9", "min"))
VARIANT_KEYS = {"a0": "Full", "x1": "Unguided", "a7": "ImageOnly", "a6": "TextOnly", "x2": "Patch", "a3": "Ga", "a4": "Nsga",
                "a5": "Mimicry", "a8": "Hill", "a9": "Janus"}

CR = ("CR", "crossed", r"CR (\%)", 0)
SC = ("S", "stealth", r"$S$", 2)
SR = ("SCR", "scr", r"SCR (\%)", 0)
LP = ("lpips", "lpips", "LPIPS", 2)
DE = ("d_edit", "d_edit", r"$d_\mathrm{edit}$ (\%)", 0)
DS = ("d_sem", "d_sem", r"$d_\mathrm{sem}$", 2)
FC = ("first_cross_eval", "first_cross_eval", "Calls to first", 0)
MC = ("min_cross_eval", "min_cross_eval", "Calls to minimal", 0)
WL = ("wall", "wall", "Runtime (s)", 0)


def prefix(arm: str) -> str:
    return str(arm).split("_")[0]


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(("true", "1", "1.0"))


def _stealth(r: pd.Series) -> float:
    """Stealth of one minimal crossing, adapted from the stealth-weighted score of MoST: S = 1 - mean over the channels the crossing
    changed of their aggressiveness; image: LPIPS against the unedited input clipped to [0, 1]; text: the normalised visible edit
    distance. NaN when no channel changed."""
    a = []
    if (r.get("edits_img_at_min") or 0) > 0:
        a.append(min(max(float(r["img_lpips"]), 0.0), 1.0))
    if (r.get("edits_txt_at_min") or 0) > 0:
        a.append(min(max(float(r["vis_char_dist"]), 0.0), 1.0))
    return 1.0 - float(np.mean(a)) if a else float("nan")


def metrics(d: pd.DataFrame, v: pd.DataFrame, v_first: pd.DataFrame | None = None) -> dict:
    """One row of metrics for the runs d (results CSV rows) and the validity rows v (minimal-crossing view) of the same runs."""
    n = len(d)
    crossed = _truthy(d["crossed"]) if n else pd.Series(dtype=bool)
    out = dict(n_runs=n, CR=100.0 * crossed.mean() if n else np.nan,
               first_cross_eval=float(d["first_cross_eval"].median()) if n else np.nan,
               min_cross_eval=float(pd.to_numeric(d.loc[crossed, "min_cross_eval"], errors="coerce").median()) if n and crossed.any() and "min_cross_eval" in d else np.nan,
               wall=float(pd.to_numeric(d["wall_s"], errors="coerce").median()) if n else np.nan)
    vv = vc.prepare(v) if len(v) else v
    if len(vv) and "edits_img_at_min" in d:
        vv = vv.merge(d[["run_dir", "edits_img_at_min", "edits_txt_at_min"]].drop_duplicates("run_dir"), on="run_dir", how="left")
        s_run = vv.apply(_stealth, axis=1)
    else:
        s_run = pd.Series(dtype=float)
    out["S"] = float(s_run.mean()) if s_run.notna().any() else np.nan
    out["SCR"] = out["CR"] * out["S"] if not pd.isna(out["S"]) else np.nan
    out["n_cross_scored"] = len(vv)
    out["d_edit"] = 100.0 * float(pd.to_numeric(vv["vis_char_dist"], errors="coerce").median()) if len(vv) and "vis_char_dist" in vv else np.nan
    out["lpips"] = float(pd.to_numeric(vv["img_lpips"], errors="coerce").median()) if len(vv) and "img_lpips" in vv else np.nan
    out["d_sem"] = float((1.0 - pd.to_numeric(vv["vis_minilm_cos"], errors="coerce")).median()) if len(vv) and "vis_minilm_cos" in vv else np.nan   # visible text, like d_edit
    hp = _truthy(vv["has_partner"]) if len(vv) and "has_partner" in vv else pd.Series(dtype=bool)
    out["n_partner"] = int(hp.sum())
    out["lpips_bp"] = float(pd.to_numeric(vv.loc[hp, "pair_lpips"], errors="coerce").median()) if hp.sum() >= MIN_PARTNERS else np.nan
    return out


def by_cell(df: pd.DataFrame, vdf: pd.DataFrame) -> pd.DataFrame:
    """Metrics per (task, sut, arm) with the minimal-crossing validity rows."""
    vmin = vdf[vdf["view"] == "min"] if "view" in vdf else vdf.iloc[0:0]
    rows = []
    for (task, sut, arm), d in df.groupby(["task", "sut", "arm"]):
        v = vmin[(vmin["task"] == task) & (vmin["sut"] == sut) & (vmin["arm"] == arm)]
        rows.append(dict(task=task, sut=sut, arm=arm, **metrics(d, v)))
    return pd.DataFrame(rows)


def _fmt(x, d):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def _head(label: str, key: str) -> str:
    return tf.head(label, key)


def _method_label(arm: str) -> str:
    return r"\sysname" if prefix(arm) == "a0" else st.tex_escape(st.arm_label(arm))


def blocks_tex(heads: list[str], blocks: list[tuple[str, list[tuple[str, list[list[str]]]]]]) -> str:
    """Model | Task | rows: the model name rotated by 90 degrees in a multirow spanning its rows, the task in a multirow
    spanning its methods. blocks: [(model label, [(task label, [[method, cell, ...], ...]), ...]), ...]."""
    k = len(heads)
    lines = [r"\begin{tabular}{ll" + "l" + "r" * (k - 1) + "}", r"\toprule",
             " & ".join(["Model", "Task"] + heads) + r" \\"]
    for model, tasks in blocks:
        n_model = sum(len(rows) for _t, rows in tasks)
        if not n_model:
            continue
        lines.append(r"\midrule")
        first_model = True
        for ti, (task, rows) in enumerate(tasks):
            if ti:
                lines.append(rf"\cmidrule{{2-{2 + k - 1}}}")
            for ri, row in enumerate(rows):
                mcell = (rf"\multirow{{{n_model}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\footnotesize {model}}}}}"
                         if first_model else "")
                tcell = rf"\multirow{{{len(rows)}}}{{*}}{{{task}}}" if ri == 0 else ""
                first_model = False
                lines.append(" & ".join([mcell, tcell] + row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def grouped_tex(m: pd.DataFrame, cols: list[tuple], methods=METHODS) -> str:
    """One block per SUT (rotated model name), one band per task (rows: methods present); best value per band and
    column in bold."""
    heads = ["Method"] + [_head(lab, key) for (_, key, lab, _d) in cols]
    blocks = []
    for s in [x for x in SUT_NAMES if x in set(m["sut"])]:
        tasks = []
        for t in [x for x in TASK_NAMES if x in set(m["task"])]:
            ms = m[(m["sut"] == s) & (m["task"] == t)]
            arms = sorted({a for a in ms["arm"] if prefix(a) in methods}, key=lambda a: methods.index(prefix(a)))
            if not arms:
                continue
            cells = {}
            for (col, key, _lab, dec) in cols:
                vals = [float(ms[ms["arm"] == a].iloc[0][col]) for a in arms]
                txt = [_fmt(x, dec) for x in vals]
                cells[col] = tf.bold_best(txt, vals, key, txt)
            rows = [[_method_label(a)] + [cells[col][i] for (col, *_r) in cols] for i, a in enumerate(arms)]
            tasks.append((TASK_NAMES[t], rows))
        blocks.append((SUT_NAMES[s], tasks))
    return blocks_tex(heads, blocks)


def rq1_tex(m: pd.DataFrame) -> str:
    return grouped_tex(m, [CR, SC, SR, LP, DE, DS])


def rq2_tex(m: pd.DataFrame) -> str:
    return grouped_tex(m, [WL, FC, MC])


def ablation(df: pd.DataFrame, vdf: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per RQ4 variant (all SUTs and tasks, seed 0). 'w/o Phase 2' is the first crossing of the full method
    (no partner and no run time of its own)."""
    df = df[df["rng_seed"] == 0] if "rng_seed" in df else df
    rows = []
    for label, pre, view in VARIANTS:
        d = df[df["arm"].map(prefix) == pre]
        if d.empty:
            rows.append(dict(label=label, pre=pre, n_runs=0))
            continue
        v = vdf[(vdf["view"] == view) & vdf["run_dir"].isin(set(d["run_dir"]))] if "view" in vdf else vdf.iloc[0:0]
        r = metrics(d, v)
        if view == "first":
            r.update(lpips_bp=np.nan, wall=np.nan, min_cross_eval=np.nan)
        rows.append(dict(label=label, pre=pre, **r))
    return pd.DataFrame(rows)


def ablation_tex(a: pd.DataFrame) -> str:
    cols = [CR, SR, LP, DE, DS, FC, WL]
    a = a[a["n_runs"] > 0].reset_index(drop=True)
    cells = {}
    for (col, key, _lab, dec) in cols:
        vals = [float(x) for x in a[col]]
        txt = [_fmt(x, dec) for x in vals]
        cells[col] = tf.bold_best(txt, vals, key, txt)
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", " & ".join(["Variant"] + [_head(lab, key) for (_, key, lab, _d) in cols]) + r" \\", r"\midrule"]
    for i, r in a.iterrows():
        lab = r["label"] if r["label"].startswith("Full") else st.tex_escape(r["label"])
        lines.append(" & ".join([lab] + [cells[col][i] for (col, *_r) in cols]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def add_ablation_numbers(num, a: pd.DataFrame) -> None:
    names = {"CR": "Cr", "S": "S", "SCR": "Scr", "lpips": "Lpips", "lpips_bp": "LpipsBp", "d_edit": "DEdit", "d_sem": "DSem", "first_cross_eval": "Calls", "wall": "Wall", "n_runs": "Runs"}
    for _, r in a.iterrows():
        key = VARIANT_KEYS.get(r["pre"], r["pre"])
        if r["label"] == "w/o Phase 2":
            key = "NoPhaseTwo"
        if r["n_runs"] == 0:
            continue
        for col, nm in names.items():
            if col in r and not pd.isna(r[col]):
                num.add(["EAbl", nm, key], f"{r[col]:.{0 if col in ('CR', 'SCR', 'first_cross_eval', 'wall', 'n_runs', 'd_edit') else 2}f}", f"ablation variant {r['label']}, pooled over SUTs and tasks, seed 0: {col}")


POOL_NAMES = {"lpips": "Lpips", "CR": "Cr", "S": "S", "SCR": "Scr", "lpips_bp": "LpipsBp", "d_edit": "DEdit", "d_sem": "DSem",
              "first_cross_eval": "Calls", "min_cross_eval": "CallsMin", "wall": "Wall", "n_partner": "NPartner"}


def add_pooled_numbers(num, df: pd.DataFrame, vdf: pd.DataFrame, arm_short) -> None:
    """Macros ERqPool<Metric><Arm>: the table metrics pooled over all SUTs and tasks and all seeds, per arm (prose of RQ1 and RQ2)."""
    vmin = vdf[vdf["view"] == "min"]
    for arm, d in df.groupby("arm"):
        r = metrics(d, vmin[vmin["arm"] == arm])
        for col, nm in POOL_NAMES.items():
            if col in r and not pd.isna(r[col]):
                dec = 0 if col in ("CR", "SCR", "first_cross_eval", "min_cross_eval", "wall", "d_edit", "n_partner") else 2
                num.add(["ERqPool", nm, arm_short(arm)], f"{r[col]:.{dec}f}", f"table metric {col} pooled over SUTs, tasks and seeds, arm {arm}")
