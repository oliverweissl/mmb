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
METHODS = ("a0", "a1", "a5", "a9")      # GA (a3), NSGA-II (a4), hill climber (a8) dropped 2026-11-25
#: RQ4 variants: (label, arm prefix, crossing view)
VARIANTS = (("Full \\sysname", "a0", "min"), ("w/o margin guidance", "x1", "min"), ("w/o Phase 2", "a0", "first"),
            ("Image only", "a7", "min"), ("Text only", "a6", "min"), ("Patch edits", "x2", "min"))
# (MIMICRY-style and DeepJanus-style are compared with MMBA in RQ1/RQ2, not in the ablation table: user edit 2026-11-26)
#: variants computed for EAbl* macros only (no table row, not in the paired-effect/Holm family); kept so text citing them compiles
MACRO_ONLY_VARIANTS = (("MIMICRY-style", "a5", "min"), ("DeepJanus-style", "a9", "min"))
VARIANT_KEYS = {"a0": "Full", "x1": "Unguided", "a7": "ImageOnly", "a6": "TextOnly", "x2": "Patch",
                "a5": "Mimicry", "a9": "Janus"}

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
    if n and "min_pending" in d and d["min_pending"].astype(bool).all():
        out["n_partner"] = np.nan          # no confirmed minimal crossing yet: partner count unknown
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


def _method_label(arm: str, marks: dict | None = None) -> str:
    lab = r"\sysname" if prefix(arm) == "a0" else st.tex_escape(st.arm_label(arm))
    return lab + (marks or {}).get(arm, "")


#: explanation of each row mark (see mmb.analysis.export.method_marks), printed as a note row under the table
MARK_NOTES = ((r"\ddagger", r"$^{\ddagger}$\,minimal crossing not yet confirmed by single-input re-scoring: its columns are withheld (--).",),)


def note_rows(marks: dict | None, n_cols: int) -> list[str]:
    used = "".join((marks or {}).values())
    return [rf"\multicolumn{{{n_cols}}}{{l}}{{\footnotesize {txt}}} \\" for key, txt in MARK_NOTES if key in used]


def blocks_tex(heads: list[str], blocks: list[tuple[str, list[tuple[str, list[list[str]]]]]],
               notes: list[str] | None = None) -> str:
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
                lines.append(rf"\cmidrule{{2-{2 + k}}}")
            for ri, row in enumerate(rows):
                mcell = (rf"\multirow{{{n_model}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\footnotesize {model}}}}}"
                         if first_model else "")
                tcell = rf"\multirow{{{len(rows)}}}{{*}}{{{task}}}" if ri == 0 else ""
                first_model = False
                lines.append(" & ".join([mcell, tcell] + row) + r" \\")
    lines += [r"\bottomrule"] + list(notes or []) + [r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def grouped_tex(m: pd.DataFrame, cols: list[tuple], methods=METHODS, marks: dict | None = None) -> str:
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
            rows = [[_method_label(a, marks)] + [cells[col][i] for (col, *_r) in cols] for i, a in enumerate(arms)]
            tasks.append((TASK_NAMES[t], rows))
        blocks.append((SUT_NAMES[s], tasks))
    used = {a: mk for a, mk in (marks or {}).items() if a in set(m["arm"]) and prefix(a) in methods}
    return blocks_tex(heads, blocks, note_rows(used, 2 + len(heads)))


def rq1_tex(m: pd.DataFrame, marks: dict | None = None) -> str:
    return grouped_tex(m, [CR, SC, SR, LP, DE, DS], marks=marks)


def rq2_tex(m: pd.DataFrame, marks: dict | None = None) -> str:
    return grouped_tex(m, [WL, FC, MC], marks=marks)


ABL_KEYS = ("task", "sut", "seed_idx")
#: RQ4 table columns: (column, direction key, header, decimals); per-run values come from ablation_runs()
ABL_COLS = (("CR", "crossed", r"CR (\%)", 0), ("SCR", "scr", r"SCR (\%)", 0), ("lpips", "lpips", "LPIPS", 2),
            ("edits", "min_edits_x", "Edits", 0), ("first_cross_eval", "first_cross_eval", "Calls to first", 0),
            ("wall", "wall", "Runtime (s)", 0))
#: per-run column, paired test ("binary": McNemar; "num": Wilcoxon), pairs restricted to runs where both crossed
ABL_TESTS = {"CR": ("crossed", "binary", False), "SCR": ("scr", "num", False), "lpips": ("lpips", "num", True),
             "edits": ("edits", "num", True), "first_cross_eval": ("first_cross_eval", "num", False),
             "wall": ("wall", "num", False)}


def _abl_keys(df: pd.DataFrame) -> list[str]:
    return [k for k in ABL_KEYS if k in df]


def _first_edits(run_dir) -> float:
    import json
    from pathlib import Path
    p = Path(str(run_dir)) / "first_cross.json"
    if not p.exists():
        return float("nan")
    g = json.loads(p.read_text()).get("genotype")
    return float(np.count_nonzero(np.asarray(g, dtype=float))) if g is not None else float("nan")


def ablation_runs(df: pd.DataFrame, vdf: pd.DataFrame, view: str) -> pd.DataFrame:
    """One row per run: the per-run values behind the RQ4 columns, for the minimal ('min') or first ('first') crossing."""
    v = vdf[vdf["view"] == view] if "view" in vdf else vdf.iloc[0:0]
    v = vc.prepare(v) if len(v) else v
    cols = ["run_dir", "img_lpips", "vis_char_dist", "n_edits"]
    r = df.merge(v[[c for c in cols if c in v]].drop_duplicates("run_dir"), on="run_dir", how="left")
    crossed = _truthy(r["crossed"])
    if view == "first":     # channels changed at the first crossing are not stored per run; S uses the edited channels' change
        r = r.assign(edits_img_at_min=np.where(pd.to_numeric(r["img_lpips"], errors="coerce") > 0, 1, 0),
                     edits_txt_at_min=np.where(pd.to_numeric(r["vis_char_dist"], errors="coerce") > 0, 1, 0))
    s = r.apply(_stealth, axis=1) if len(r) else pd.Series(dtype=float)
    out = pd.DataFrame({k: r[k] for k in _abl_keys(r)})
    out["run_dir"] = r["run_dir"]
    out["crossed"] = crossed.astype(float)
    out["scr"] = np.where(crossed, 100.0 * s.fillna(0.0), 0.0)
    if "min_pending" in r:       # minimal crossing awaiting confirmation: its stealth is unknown, not zero
        out["scr"] = out["scr"].where(~r["min_pending"].astype(bool).to_numpy(), np.nan)
    out["lpips"] = pd.to_numeric(r["img_lpips"], errors="coerce").where(crossed)
    if view == "first":     # edits of the first crossing: non-zero genes of its stored genotype
        out["edits"] = pd.Series([_first_edits(p) for p in r["run_dir"]], index=r.index, dtype=float).where(crossed)
    else:
        out["edits"] = pd.to_numeric(r["n_edits"], errors="coerce").where(crossed) if "n_edits" in r else np.nan
    out["first_cross_eval"] = pd.to_numeric(r["first_cross_eval"], errors="coerce")
    out["wall"] = pd.to_numeric(r["wall_s"], errors="coerce") if view == "min" else np.nan
    return out


def ablation(df: pd.DataFrame, vdf: pd.DataFrame, variants=None) -> pd.DataFrame:
    """Pooled metrics per RQ4 variant (all SUTs and tasks, seed 0). 'w/o Phase 2' is the first crossing of the full method
    (no run time of its own). A variant is listed only once it covers every seed input, model and task of the full method."""
    df = df[df["rng_seed"] == 0] if "rng_seed" in df else df
    keys = _abl_keys(df)

    def cover(d: pd.DataFrame) -> set:
        return set(map(tuple, d[keys].astype(str).values)) if keys else set()

    full_cover = None
    rows = []
    for label, pre, view in (VARIANTS if variants is None else variants):
        d = df[df["arm"].map(prefix) == pre]
        if d.empty:
            rows.append(dict(label=label, pre=pre, n_runs=0))
            continue
        if full_cover is None:
            full_cover = cover(d)
        elif not full_cover <= cover(d):
            rows.append(dict(label=label, pre=pre, n_runs=0, n_partial=len(d)))
            continue
        v = vdf[(vdf["view"] == view) & vdf["run_dir"].isin(set(d["run_dir"]))] if "view" in vdf else vdf.iloc[0:0]
        r = metrics(d, v)
        runs = ablation_runs(d, vdf, view)
        r["edits"] = float(runs["edits"].median()) if runs["edits"].notna().any() else np.nan
        r["SCR"] = float(runs["scr"].mean())
        if view == "first":
            r.update(lpips_bp=np.nan, wall=np.nan, min_cross_eval=np.nan)
        rows.append(dict(label=label, pre=pre, view=view, _runs=runs, **r))
    return pd.DataFrame(rows)


def ablation_effects(a: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison of every listed variant with the full method per RQ4 column: pairs matched on
    (task, model, seed input) at seed 0; McNemar for CR, Wilcoxon signed-rank otherwise (crossing-only metrics on
    pairs where both runs crossed); Cohen's d_z of the pair differences; Holm across variants within a column.
    A comparison whose pair differences are all zero (identical by construction) gets no test."""
    a = a[a["n_runs"] > 0].reset_index(drop=True)
    if a.empty:
        return pd.DataFrame()
    full = a.iloc[0]["_runs"]
    keys = _abl_keys(full)
    if not keys:
        return pd.DataFrame()
    rows = []
    for i in range(1, len(a)):
        var = a.iloc[i]["_runs"]
        m = full.merge(var, on=keys, suffixes=("_ref", "_arm"))
        for col, (c, kind, cross_only) in ABL_TESTS.items():
            pf = m[[c + "_ref", c + "_arm"]].astype(float).dropna()
            if cross_only:
                pf = pf[(m.loc[pf.index, "crossed_ref"] > 0) & (m.loc[pf.index, "crossed_arm"] > 0)]
            n = len(pf)
            diff = (pf[c + "_arm"] - pf[c + "_ref"]).to_numpy()
            if n < 3 or np.all(diff == 0):
                rows.append(dict(i=i, col=col, n=n, dz=np.nan, p=np.nan, same=bool(n and np.all(diff == 0))))
                continue
            if kind == "binary":
                p = st.mcnemar_exact(pf[c + "_ref"].to_numpy() > 0.5, pf[c + "_arm"].to_numpy() > 0.5)["p"]
            else:
                p = st.wilcoxon_paired(pf[c + "_ref"].to_numpy(), pf[c + "_arm"].to_numpy())["p"]
            rows.append(dict(i=i, col=col, n=n, dz=tf.cohens_dz(diff), p=p, same=False,
                             diff=float(np.mean(diff)) if kind == "binary" else float(np.median(diff))))
    e = pd.DataFrame(rows)
    if e.empty:
        return e
    e["p_holm"] = np.nan
    for col in set(e["col"]):
        idx = e.index[(e["col"] == col) & e["p"].notna()]
        if len(idx):
            e.loc[idx, "p_holm"] = st.holm(e.loc[idx, "p"].to_numpy())
    return e


def _no_image_edits(r: pd.Series) -> bool:
    """True when none of the variant's crossing runs changed the image (LPIPS of every scored crossing <= 1e-3; an
    unedited image scores ~3e-6 from PNG round-trips)."""
    runs = r.get("_runs")
    if not isinstance(runs, pd.DataFrame) or "lpips" not in runs:
        return False
    v = pd.to_numeric(runs["lpips"], errors="coerce").dropna()
    return len(v) > 0 and bool((v <= 1e-3).all())


def ablation_tex(a: pd.DataFrame, e: pd.DataFrame | None = None, marks: dict | None = None) -> str:
    """RQ4 table: absolute values per variant (bold: best of the column); each variant cell carries the magnitude bullet of
    its paired effect against the full method and a dagger when the Holm-corrected paired test is significant."""
    a = a[a["n_runs"] > 0].reset_index(drop=True)
    e = ablation_effects(a) if e is None else e
    no_img = [_no_image_edits(r) for _, r in a.iterrows()]
    cells = {}
    for (col, key, _lab, dec) in ABL_COLS:
        vals = [float(x) for x in a[col]]
        if col == "lpips":      # a variant that never edits the image has no LPIPS: "--", not a (bold) 0.00
            vals = [np.nan if ni else v for v, ni in zip(vals, no_img)]
        txt = [_fmt(x, dec) for x in vals]
        out = tf.bold_best(txt, vals, key, txt)
        for i in range(1, len(a)):
            hit = e[(e["i"] == i) & (e["col"] == col)] if len(e) else e
            if txt[i] == "--" or hit.empty or bool(hit.iloc[0]["same"]):
                continue
            h = hit.iloc[0]
            m = tf.magnitude(h["dz"])
            out[i] = out[i] + (" " + tf.BULLET[m] if m else "") + (tf.DAGGER if tf._ok(h["p_holm"]) and h["p_holm"] < tf.ALPHA else "")
        cells[col] = out
    lines = [r"\begin{tabular}{l" + "r" * len(ABL_COLS) + "}", r"\toprule",
             " & ".join(["Variant"] + [_head(lab, key) for (_, key, lab, _d) in ABL_COLS]) + r" \\", r"\midrule"]
    for i, r in a.iterrows():
        lab = r["label"] if r["label"].startswith("Full") else st.tex_escape(r["label"])
        if r["label"] != "w/o Phase 2":          # marks are keyed by arm prefix (marks: prefix -> superscript)
            lab += (marks or {}).get(r["pre"], "")
        lines.append(" & ".join([lab] + [cells[col][i] for (col, *_r) in ABL_COLS]) + r" \\")
    shown = {p: mk for p, mk in (marks or {}).items() if p in set(a["pre"])}
    lines += [r"\bottomrule"] + note_rows(shown, 1 + len(ABL_COLS)) + [r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def add_ablation_numbers(num, a: pd.DataFrame) -> None:
    names = {"CR": "Cr", "S": "S", "SCR": "Scr", "lpips": "Lpips", "lpips_bp": "LpipsBp", "d_edit": "DEdit", "d_sem": "DSem", "first_cross_eval": "Calls", "wall": "Wall", "n_runs": "Runs", "edits": "Edits"}
    for _, r in a.iterrows():
        key = VARIANT_KEYS.get(r["pre"], r["pre"])
        if r["label"] == "w/o Phase 2":
            key = "NoPhaseTwo"
        if r["n_runs"] == 0:
            continue
        for col, nm in names.items():
            if col in r and not pd.isna(r[col]):
                num.add(["EAbl", nm, key], f"{r[col]:.{0 if col in ('CR', 'SCR', 'first_cross_eval', 'wall', 'n_runs', 'd_edit', 'edits') else 2}f}", f"ablation variant {r['label']}, pooled over SUTs and tasks, seed 0: {col}")


def add_ablation_effect_numbers(num, a: pd.DataFrame, e: pd.DataFrame) -> None:
    """EAbl{Dz,PHolm,Diff}<Column><Variant>: the paired comparison of each RQ4 variant with the full method."""
    names = {"CR": "Cr", "SCR": "Scr", "lpips": "Lpips", "edits": "Edits", "first_cross_eval": "Calls", "wall": "Wall"}
    a = a[a["n_runs"] > 0].reset_index(drop=True)
    for _, h in e.iterrows() if len(e) else []:
        r = a.iloc[int(h["i"])]
        key = "NoPhaseTwo" if r["label"] == "w/o Phase 2" else VARIANT_KEYS.get(r["pre"], r["pre"])
        what = f"RQ4 {r['label']} vs full method, {h['col']}"
        if tf._ok(h["dz"]):
            num.add(["EAbl", "Dz", names[h["col"]], key], f"{h['dz']:+.2f}", f"paired Cohen's d_z, {what}")
        if tf._ok(h["p_holm"]):
            p = float(h["p_holm"])
            num.add(["EAbl", "PHolm", names[h["col"]], key], "<0.001" if p < 0.001 else f"{p:.3f}", f"Holm-corrected p, {what}")
        num.add(["EAbl", "Pairs", names[h["col"]], key], str(int(h["n"])), f"matched pairs, {what}")


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
