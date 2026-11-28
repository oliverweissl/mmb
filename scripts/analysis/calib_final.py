#!/usr/bin/env python
"""Final check of the candidate operator setting (T-calib): combine rounds that ran the SAME setting on held-out set 1
and set 2, then report per modality (text-only a6, image-only a7, both = full MMBA a0): crossing share, calls to first
crossing, minimal edits, image LPIPS, visible text change, question_preserved, output margin; and exact paired McNemar
tests text vs image (equilibrium: not distinguishable at p = 0.05, the user's criterion) and both vs each single
modality, and both vs the baselines MIMICRY-style (a5) and DeepJanus-style (a9) at the same operator setting (rule:
MMBA's crossing share must be >= each baseline). Validity proxies are (re)scored per round if missing.

    python scripts/analysis/calib_final.py --part r2b:a7_image_only --part r3t:a6_text_only --part fc_old:a0_directed \
        --part fc_new:all --part fcb1:all --part fcb2:all --out results/calib_final
"""
import argparse, subprocess, sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ap = argparse.ArgumentParser(); ap.add_argument("--part", action="append", required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
py = sys.executable
frames, vals = [], []
for part in a.part:
    rnd, arm = part.split(":")
    res, val = Path(f"results/CALIB_{rnd}.csv"), Path(f"results/validity_CALIB_{rnd}.csv")
    root = f"runs/CALIB/{rnd}"
    subprocess.run([py, "-m", "mmb.analysis.collect", root, "--out", str(res), "--exp-id", f"CALIB_{rnd}", "--require-stats"], check=True)
    if not val.exists() or val.stat().st_mtime < res.stat().st_mtime:
        subprocess.run([py, "-m", "mmb.validity.score", root, "--out", str(val), "--default", f"exp_id=CALIB_{rnd}", "--offline"], check=True)
    d = pd.read_csv(res)
    if arm != "all":
        d = d[d["arm"] == arm]
    d["set"] = "set2" if rnd in ("fc_new", "fcb2") else "set1"
    frames.append(d); vals.append(pd.read_csv(val))
d = pd.concat(frames, ignore_index=True)
v = pd.concat(vals, ignore_index=True)
vm = v[v["view"] == "min"].drop_duplicates("run_dir").set_index("run_dir")
for c in ("img_lpips", "vis_char_dist", "question_preserved", "g_at_cross"):
    d[c] = d["run_dir"].map(vm[c])
d["crossed"] = d["crossed"].astype(str).str.lower().isin(("true", "1"))
d["mod"] = d["arm"].map({"a7_image_only": "image", "a6_text_only": "text", "a0_directed": "both",
                         "a5_mimicry_objective": "mimicry", "a9_deepjanus": "deepjanus"})
qp = d["question_preserved"].astype(str).str.lower()
d["qp"] = qp.isin(("true", "1", "1.0")).where(d["question_preserved"].notna())


def agg(g):
    c = g[g["crossed"]]
    num = lambda col: pd.to_numeric(c[col], errors="coerce")
    return pd.Series(dict(n=len(g), CR=round(100 * g["crossed"].mean(), 1), calls_med=float(g["first_cross_eval"].median()),
                          edits_med=float(num("min_edits_cross").median()) if len(c) else None,
                          lpips_med=round(float(num("img_lpips").median()), 3) if len(c) else None,
                          vis_char_med=round(float(num("vis_char_dist").median()), 3) if len(c) else None,
                          q_preserved=round(100 * float(c["qp"].dropna().astype(float).mean()), 1) if c["qp"].notna().any() else None,
                          out_margin_med=round(float(num("g_at_cross").abs().median()), 2) if len(c) else None))


rows = []
for keys, lvl in ((["mod"], "pooled"), (["mod", "task"], "task"), (["mod", "set"], "set"), (["mod", "task", "sut"], "task_sut")):
    t = d.groupby(keys).apply(agg, include_groups=False).reset_index(); t.insert(0, "level", lvl); rows.append(t)
S = pd.concat(rows, ignore_index=True)
key = ["set", "task", "sut", "seed_idx"]
tests = []
for scope, sub in [("pooled", d)] + [(t, d[d["task"] == t]) for t in sorted(set(d["task"]))]:
    w = sub.pivot_table(index=key, columns="mod", values="crossed", aggfunc="first")
    for x, y in (("text", "image"), ("both", "text"), ("both", "image"), ("both", "mimicry"), ("both", "deepjanus")):
        if x not in w or y not in w:
            continue
        m = w[[x, y]].dropna().astype(bool)
        n10, n01 = int((m[x] & ~m[y]).sum()), int((~m[x] & m[y]).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        tests.append(dict(scope=scope, a=x, b=y, pairs=len(m), share_a=round(100 * m[x].mean(), 1), share_b=round(100 * m[y].mean(), 1),
                          a_only=n10, b_only=n01, p_mcnemar=round(p, 4), distinguishable=p < 0.05))
T = pd.DataFrame(tests)
a.out.mkdir(parents=True, exist_ok=True)
S.to_csv(a.out / "summary.csv", index=False); T.to_csv(a.out / "tests.csv", index=False); d.to_csv(a.out / "runs.csv", index=False)
pd.set_option("display.width", 220)
print(S[S["level"] != "task_sut"].to_string(index=False)); print(); print(T.to_string(index=False))
