#!/usr/bin/env python
"""Summary of one calibration round (T-calib): crossing share per modality x task x SUT and pooled, calls to the first
crossing, and the plausibility proxies of the minimal crossings: image LPIPS (image-only) and question_preserved /
visible character change / MiniLM similarity of the question (text-only). Search-time (batched) crossings, as recorded
by the runs; no single-input confirmation (same treatment for both modalities).

    python scripts/analysis/calib_summary.py r1 [--no-validity]
reads runs/CALIB/<round>; writes results/CALIB_<round>.csv, results/validity_CALIB_<round>.csv, results/calib_<round>_summary.csv
"""
import argparse, subprocess, sys
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("round"); ap.add_argument("--no-validity", action="store_true")
ap.add_argument("--text-from", default=None, help="take the text-only (a6) runs from another round (shared text setting)")
a = ap.parse_args()
root = Path(f"runs/CALIB/{a.round}")
res, val = Path(f"results/CALIB_{a.round}.csv"), Path(f"results/validity_CALIB_{a.round}.csv")
py = sys.executable
subprocess.run([py, "-m", "mmb.analysis.collect", str(root), "--out", str(res), "--exp-id", f"CALIB_{a.round}", "--require-stats"], check=True)
if not a.no_validity:
    subprocess.run([py, "-m", "mmb.validity.score", str(root), "--out", str(val), "--default", f"exp_id=CALIB_{a.round}", "--offline"], check=True)
d = pd.read_csv(res)
if a.text_from:
    other = pd.read_csv(f"results/CALIB_{a.text_from}.csv")
    d = pd.concat([d[d["arm"] != "a6_text_only"], other[other["arm"] == "a6_text_only"]], ignore_index=True)
    ov = Path(f"results/validity_CALIB_{a.text_from}.csv")
    if ov.exists() and val.exists():
        val2 = pd.concat([pd.read_csv(val), pd.read_csv(ov)], ignore_index=True)
        val = Path(f"results/validity_CALIB_{a.round}_with_{a.text_from}.csv"); val2.to_csv(val, index=False)
d["crossed"] = d["crossed"].astype(str).str.lower().isin(("true", "1"))
d["mod"] = d["arm"].map({"a7_image_only": "image", "a6_text_only": "text", "a0_directed": "both"})
v = pd.read_csv(val) if val.exists() else pd.DataFrame()
vm = v[v["view"] == "min"].drop_duplicates("run_dir").set_index("run_dir") if len(v) else pd.DataFrame()
if len(vm):
    for c in ("img_lpips", "vis_char_dist", "txt_minilm_cos", "question_preserved", "g_at_cross"):
        d[c] = d["run_dir"].map(vm[c]) if c in vm else None
    d["question_preserved"] = d["question_preserved"].astype(str).str.lower().isin(("true", "1", "1.0")).where(d["question_preserved"].notna())


def agg(g):
    out = dict(n=len(g), CR=round(100 * g["crossed"].mean(), 1), calls_med=float(g["first_cross_eval"].median()))
    c = g[g["crossed"]]
    out["edits_med"] = float(pd.to_numeric(c["min_edits_cross"], errors="coerce").median()) if len(c) else None
    if "img_lpips" in g:
        out["lpips_med"] = round(float(pd.to_numeric(c["img_lpips"], errors="coerce").median()), 3) if len(c) else None
        out["vis_char_med"] = round(float(pd.to_numeric(c["vis_char_dist"], errors="coerce").median()), 3) if len(c) else None
        out["minilm_med"] = round(float(pd.to_numeric(c["txt_minilm_cos"], errors="coerce").median()), 3) if len(c) else None
        out["out_margin_med"] = round(float(pd.to_numeric(c["g_at_cross"], errors="coerce").abs().median()), 2) if len(c) else None
        q = c["question_preserved"].dropna()
        out["q_preserved"] = round(100 * float(q.astype(float).mean()), 1) if len(q) else None
    return pd.Series(out)


rows = []
for keys, name in ((["mod"], "pooled"), (["mod", "task"], "task"), (["mod", "task", "sut"], "task_sut")):
    t = d.groupby(keys).apply(agg, include_groups=False).reset_index()
    t.insert(0, "level", name)
    rows.append(t)
S = pd.concat(rows, ignore_index=True)
S.to_csv(f"results/calib_{a.round}_summary.csv", index=False)
pd.set_option("display.width", 200)
print(S.to_string(index=False))


# Equilibrium test (user, 2026-11-27): text-only and image-only are "in equilibrium" when their crossing shares cannot
# be distinguished by an exact paired McNemar test at alpha = 0.05 on matched (task, SUT, seed input) runs. Also: full
# MMBA vs each single modality (one-sided reading: is "both" higher?). Reported with the discordant counts, because with
# few pairs a non-significant result is weak evidence of equality (low power).
from scipy.stats import binomtest

key = ["task", "sut", "seed_idx"]
tests = []
for scope, sub in [("pooled", d)] + [(t, d[d["task"] == t]) for t in sorted(set(d["task"]))]:
    w = sub.pivot_table(index=key, columns="mod", values="crossed", aggfunc="first")
    for a_, b_ in (("text", "image"), ("both", "text"), ("both", "image")):
        if a_ not in w or b_ not in w:
            continue
        m = w[[a_, b_]].dropna().astype(bool)
        n10, n01 = int((m[a_] & ~m[b_]).sum()), int((~m[a_] & m[b_]).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        tests.append(dict(scope=scope, a=a_, b=b_, pairs=len(m), share_a=round(100 * m[a_].mean(), 1),
                          share_b=round(100 * m[b_].mean(), 1), a_only=n10, b_only=n01, p_mcnemar=round(p, 4),
                          distinguishable=p < 0.05))
T = pd.DataFrame(tests)
T.to_csv(f"results/calib_{a.round}_tests.csv", index=False)
print()
print(T.to_string(index=False))
