"""Crossing cost against descriptive validity proxies, per arm and task.

Thresholds are fixed a priori (METHODOLOGY.md, change 2026-11-22 07:12 UTC): LPIPS < 0.2 and text cosine
distance (1 - MiniLM cosine) < 0.1 at the first crossing input. They are descriptive and uncalibrated: they say
how far an input moved, not whether a human reads it as the same item. Sensitivity rows (other cut-offs) are
labelled as such.

    python -m mmb.analysis.validity_cost results/MAIN.csv results/validity_MAIN.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mmb.analysis import stats as st
from mmb.analysis import tablefmt as tf

LPIPS_MAX = 0.2
TXT_MAX = 0.1
SENS_LPIPS = (0.1, 0.2, 0.3, 0.4)
SENS_TXT = (0.05, 0.1, 0.2, 0.3)
MIN_RHO_N = 6


def prepare(vdf: pd.DataFrame) -> pd.DataFrame:
    """Add txt_dist = 1 - MiniLM cosine (0 when the prompt was not edited)."""
    v = vdf.copy()
    v["txt_dist"] = 1.0 - pd.to_numeric(v.get("txt_minilm_cos"), errors="coerce")
    if "txt_edited" in v:
        unedited = ~v["txt_edited"].astype(str).str.lower().isin(("true", "1", "1.0"))
        v.loc[unedited & v["txt_dist"].isna(), "txt_dist"] = 0.0
    v["img_lpips"] = pd.to_numeric(v["img_lpips"], errors="coerce")
    return v


def _share(k: int, n: int) -> dict:
    lo, hi = st.wilson(k, n) if n else (np.nan, np.nan)
    return dict(k=k, n=n, share=k / n if n else np.nan, lo=lo, hi=hi)


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < MIN_RHO_N or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return float("nan")
    return float(spearmanr(x[m], y[m])[0])


def summarize(g: pd.DataFrame, n_runs: int, lp: float = LPIPS_MAX, tx: float = TXT_MAX) -> dict:
    """One row: crossings of an arm (g) against the n_runs runs the arm made in the scope."""
    n = len(g)
    ok_i = g["img_lpips"] < lp
    ok_t = g["txt_dist"] < tx
    both = ok_i & ok_t
    out = dict(n_runs=n_runs, n_cross=n,
               med_lpips=float(g["img_lpips"].median()) if n else np.nan,
               med_txt=float(g["txt_dist"].median()) if n else np.nan,
               rho_eval_lpips=_rho(g["first_cross_eval"].to_numpy(float), g["img_lpips"].to_numpy(float)),
               rho_eval_txt=_rho(g["first_cross_eval"].to_numpy(float), g["txt_dist"].to_numpy(float)))
    if "vis_identical" in g:
        ident = g["vis_identical"].astype(str).str.lower().isin(("true", "1", "1.0"))
        within = g["vis_within_one_per_word"].astype(str).str.lower().isin(("true", "1", "1.0")) | ident
        for name, ok in (("visid", ident), ("visone", within)):
            s_ = _share(int(ok.sum()), n)
            out.update({f"{name}_{k}": v for k, v in s_.items() if k != "n"})
        out["med_vis_char"] = float(pd.to_numeric(g["vis_char_dist"], errors="coerce").median()) if n else np.nan
        out["med_vis_cos"] = float(pd.to_numeric(g["vis_minilm_cos"], errors="coerce").median()) if n else np.nan
    for name, ok in (("img", ok_i), ("txt", ok_t), ("both", both)):
        s = _share(int(ok.sum()), n)
        out.update({f"{name}_{k}": v for k, v in s.items() if k != "n"})
    c = _share(int(both.sum()), n_runs)       # conservative crossing share: every run of the arm in the denominator
    out.update(cons_k=c["k"], cons_share=c["share"], cons_lo=c["lo"], cons_hi=c["hi"])
    return out


def objective_text(d: pd.DataFrame, arm: str, tx: float = TXT_MAX) -> dict:
    """The search's own text magnitude (cosine distance of the objective's embedding, diagnostic column) at the
    cheapest crossing of each run that crossed. Added after the first-crossing reading; see METHODOLOGY."""
    r = d[d["arm"] == arm]
    if "mag_source" in r:
        r = r[r["mag_source"] == "diag"]            # never mix diagnostic and optimised-fitness magnitudes
    c = r[r["crossed"].astype(bool)] if "crossed" in r else r.iloc[0:0]
    x = pd.to_numeric(c.get("best_txt_mag_cross"), errors="coerce").dropna()
    k = int((x < tx).sum())
    s = _share(k, len(x))
    c_all = _share(k, len(r))
    return dict(obj_n=len(x), obj_txt_k=k, obj_txt_share=s["share"], obj_txt_lo=s["lo"], obj_txt_hi=s["hi"],
                obj_med=float(x.median()) if len(x) else np.nan, obj_cons_share=c_all["share"],
                obj_cons_lo=c_all["lo"], obj_cons_hi=c_all["hi"])


def _view(vdf: pd.DataFrame, view: str) -> pd.DataFrame:
    """Rows of one crossing view: 'first' (the first crossing input) or 'min' (the run's smallest crossing, the
    search's output). Frames without a 'view' column hold first crossings only."""
    if "view" not in vdf:
        return vdf if view == "first" else vdf.iloc[0:0]
    return vdf[vdf["view"] == view]


def validity_cost(vdf: pd.DataFrame, df: pd.DataFrame, grid_suts=None, lp: float = LPIPS_MAX,
                  tx: float = TXT_MAX, view: str = "first") -> pd.DataFrame:
    """Rows per (task, arm) with task 'All' pooling the tasks; df is the results CSV (denominators)."""
    if "img_lpips" not in vdf or "txt_minilm_cos" not in vdf:
        return pd.DataFrame()                       # proxies not computed for this tree
    v = prepare(_view(vdf, view))
    d = df
    if grid_suts:
        v, d = v[v["sut"].isin(grid_suts)], d[d["sut"].isin(grid_suts)]
    rows = []
    for task in [*sorted(set(d["task"])), "All"]:
        dt = d if task == "All" else d[d["task"] == task]
        vt = v if task == "All" else v[v["task"] == task]
        for arm in sorted(set(dt["arm"])):
            g = vt[vt["arm"] == arm]
            rows.append(dict(task=task, arm=arm, **summarize(g, int((dt["arm"] == arm).sum()), lp, tx),
                             **objective_text(dt, arm, tx)))
    return pd.DataFrame(rows)


def sensitivity(vdf: pd.DataFrame, df: pd.DataFrame, grid_suts=None, view: str = "first") -> pd.DataFrame:
    """Share of crossings under both cut-offs for other values (labelled sensitivity, not a result)."""
    rows = []
    for lp in SENS_LPIPS:
        for tx in SENS_TXT:
            t = validity_cost(vdf, df, grid_suts, lp, tx, view)
            t = t[t["task"] == "All"] if len(t) else t
            for _, r in t.iterrows():
                rows.append(dict(arm=r["arm"], lpips=lp, txt=tx, both_share=r["both_share"], cons_share=r["cons_share"]))
    return pd.DataFrame(rows)


def _f(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def _pct(r, k):
    return "--" if pd.isna(r[f"{k}_share"]) else f"{100 * r[f'{k}_share']:.0f}"


def _colvals(g: pd.DataFrame, col: str, scale: float = 1.0) -> list:
    return [np.nan if col not in g or pd.isna(v) else scale * float(v) for v in (g[col] if col in g else [np.nan] * len(g))]


def table_tex(t: pd.DataFrame, task_names: dict, objective: bool = True) -> str:
    # (header label, direction key, cell strings, numeric values, tie labels); best bold within each task block
    def block(g: pd.DataFrame) -> list[list[str]]:
        n = len(g)
        spec = [
            ([_f(v) for v in _colvals(g, "med_lpips")], "lpips", _colvals(g, "med_lpips"), 2),
            ([_f(v) for v in _colvals(g, "med_txt")], "txt_dist", _colvals(g, "med_txt"), 2),
            ([_pct(r, "img") for _, r in g.iterrows()], "lpips_share", _colvals(g, "img_share", 100), 0),
            ([_pct(r, "txt") for _, r in g.iterrows()], "txt_share", _colvals(g, "txt_share", 100), 0),
            ([_pct(r, "both") for _, r in g.iterrows()], "both_share", _colvals(g, "both_share", 100), 0),
            (["--" if pd.isna(r["cons_share"]) else f"{100 * r['cons_share']:.0f}"
              for _, r in g.iterrows()], "cons_share", _colvals(g, "cons_share", 100), 0),
            ([_f(r["rho_eval_lpips"]) for _, r in g.iterrows()], None, [np.nan] * n, 2),
            ([_f(r["rho_eval_txt"]) for _, r in g.iterrows()], None, [np.nan] * n, 2),
            ([_f(r.get("med_vis_cos", np.nan)) for _, r in g.iterrows()], "vis_cos", _colvals(g, "med_vis_cos"), 2),
            ([_pct(r, "visid") if "visid_share" in r else "--" for _, r in g.iterrows()], "vis_identical", _colvals(g, "visid_share", 100), 0),
            ([_pct(r, "visone") if "visone_share" in r else "--" for _, r in g.iterrows()], "vis_within_one", _colvals(g, "visone_share", 100), 0),
        ]
        if objective:
            spec += [([_f(r["obj_med"], 3) for _, r in g.iterrows()], "obj_txt_dist", _colvals(g, "obj_med"), 3),
                     ([_pct(r, "obj_txt") for _, r in g.iterrows()], "obj_share", _colvals(g, "obj_txt_share", 100), 0)]
        cols = []
        for cells, key, vals, dig in spec:
            ties = [None if (isinstance(v, float) and np.isnan(v)) else f"{round(v, dig)}" for v in vals]
            cols.append(tf.bold_best(cells, vals, key, ties) if key else cells)
        return [[st.tex_escape(st.arm_label(r["arm"])), f"{int(r['n_cross'])}/{int(r['n_runs'])}"] + [cols[j][i] for j in range(len(cols))]
                for i, (_, r) in enumerate(g.iterrows())]

    rows = []
    for task in dict.fromkeys(t["task"]):
        g = t[t["task"] == task]
        for i, body in enumerate(block(g)):
            rows.append([st.tex_escape(task_names.get(task, task)) if i == 0 else ""] + body)
    header = ["Task", "Arm", "Crossed", tf.head("Med. LPIPS", "lpips"), tf.head("Med. text dist.", "txt_dist"),
              tf.head(r"LPIPS $<0.2$ (\%)", "lpips_share"), tf.head(r"text $<0.1$ (\%)", "txt_share"),
              tf.head("both (\\%)", "both_share"), tf.head("Both, of all runs (\\%)", "cons_share"),
              r"$\rho$(evals, LPIPS)", r"$\rho$(evals, text)",
              tf.head("Visible text cos. (med.)", "vis_cos"), tf.head("visibly identical (\\%)", "vis_identical"),
              tf.head(r"$\le$1 edit/word (\%)", "vis_within_one"),
              tf.head("Minimal-edit crossing: text dist.", "obj_txt_dist"), tf.head(r"$<0.1$ (\%)", "obj_share")]
    if not objective:
        header = header[:-2]
    return st._booktabs("llrrrrrrrrrrrr" + ("rr" if objective else ""), header, rows)


def sensitivity_tex(s: pd.DataFrame) -> str:
    arms = list(dict.fromkeys(s["arm"]))
    rows = []
    for lp in SENS_LPIPS:
        for tx in SENS_TXT:
            sub = s[(s["lpips"] == lp) & (s["txt"] == tx)].set_index("arm")
            rows.append([f"{lp:g}", f"{tx:g}"] + [("--" if pd.isna(sub.loc[a, "cons_share"]) else f"{100 * sub.loc[a, 'cons_share']:.0f}") if a in sub.index else "--" for a in arms])
    return st._booktabs("rr" + "r" * len(arms), ["LPIPS $<$", "text $<$"] + [st.tex_escape(st.arm_label(a)) for a in arms], rows)


def add_numbers(num, t: pd.DataFrame, family: str = "EValid", label: str = "first crossing") -> None:
    """Macros EValid<Metric><Arm><Task> (task All pools the tasks)."""
    from mmb.analysis.export import arm_short, camel, f_num, f_pct
    for _, r in t.iterrows():
        tail = [arm_short(r["arm"]), camel(r["task"])]
        d = f"{r['arm']}, {r['task']}, {label} (descriptive thresholds fixed a priori)"
        num.add([family, "Runs"] + tail, str(int(r["n_runs"])), "runs of the arm; " + d)
        num.add([family, "Crossed"] + tail, str(int(r["n_cross"])), "runs with a validity-scored first crossing; " + d)
        num.add([family, "MedLpips"] + tail, f_num(r["med_lpips"], 2), f"median LPIPS at the {label}; " + d)
        num.add([family, "MedTxt"] + tail, f_num(r["med_txt"], 2), f"median 1 - MiniLM cosine at the {label}; " + d)
        for key, what in (("img", "LPIPS < 0.2"), ("txt", "text distance < 0.1"), ("both", "both thresholds")):
            num.add([family, "Share", key] + tail, f_pct(r[f"{key}_share"], 0), f"% of crossings under {what}; " + d)
        num.add([family, "ConsShare"] + tail, f_pct(r["cons_share"], 0), "% of ALL runs that crossed under both thresholds; " + d)
        num.add([family, "ConsCi"] + tail, f"[{100 * r['cons_lo']:.0f}--{100 * r['cons_hi']:.0f}]", "Wilson 95% interval; " + d)
        if "visid_share" in r:
            num.add([family, "VisIdentical"] + tail, f_pct(r["visid_share"], 0), "% of crossings whose question is visibly identical (NFKC, invisible characters removed, confusables mapped); " + d)
            num.add([family, "VisWithinOne"] + tail, f_pct(r["visone_share"], 0), "% of crossings visibly identical or within one character edit per word; " + d)
            num.add([family, "VisCharDist"] + tail, f_num(r["med_vis_char"], 3), "median visible character edit distance / length; " + d)
            num.add([family, "VisMinilm"] + tail, f_num(r["med_vis_cos"], 2), "median MiniLM cosine of the normalised questions; " + d)
        num.add([family, "RhoEvalLpips"] + tail, f_num(r["rho_eval_lpips"], 2, True), "Spearman, evaluations to first crossing vs LPIPS; " + d)
        num.add([family, "RhoEvalTxt"] + tail, f_num(r["rho_eval_txt"], 2, True), "Spearman, evaluations to first crossing vs text distance; " + d)
        num.add([family, "ObjTxtMed"] + tail, f_num(r["obj_med"], 3), "median text cosine of the search's own minimal-edit crossing (post hoc addition); " + d)
        num.add([family, "ObjTxtShare"] + tail, f_pct(r["obj_txt_share"], 0), "% of crossed runs whose minimal-edit crossing has text cosine < 0.1 (post hoc addition); " + d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results"); ap.add_argument("validity")
    ap.add_argument("--grid-suts", default="qwen,internvl,ministral")
    a = ap.parse_args()
    t = validity_cost(pd.read_csv(a.validity), pd.read_csv(a.results), a.grid_suts.split(","))
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
