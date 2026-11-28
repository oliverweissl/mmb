#!/usr/bin/env python
"""Calibration trajectory (T-calib, user request 2026-11-28): how text-only vs image-only crossing shares move from the
UNCALIBRATED operators of the MAIN study (round r0) through the calibration rounds to the calibrated setting C*.
Every step on the SAME held-out set 1 (4 seed inputs per task x 3 SUTs = 24 matched pairs, rng seed 0, 1,500 calls);
the C* final check on sets 1+2 (48 pairs) is a separate point.

Per step: crossing share with Wilson 95% CI per modality, exact paired McNemar p (text vs image), median image LPIPS
of image-only minimal crossings, median visible text change of text-only minimal crossings.

    python scripts/analysis/calib_trajectory.py --out results/calib_final/trajectory.csv
"""
import argparse, math, subprocess, sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

# (step key, label, [(round, arm) for text-only], [(round, arm) for image-only], final, status, reason)
# status: start = the uncalibrated MAIN operators; rejected = a setting the calibration moved away from (reason why);
# adopted = the calibrated setting; None = from the test (p < 0.05: rejected "unbalanced", else "balanced").
# T-A trajectory (2026-11-28): each step changes ONE channel. R0 -> T-A text (x moves) -> image ladder (y moves).
T6, I7 = "a6_text_only", "a7_image_only"
STEPS_TA = [
    ("Zero", "R0\nuncalibrated", [("r0", T6)], [("r0", I7)], False, "start", "image edits hit dead codewords"),
    ("Text", "T-A\ntext", [("ta1", T6)], [("r0", I7)], False, None, ""),
    ("ImgOneTwoEight", "image\nrank≤128", [("ta1", T6)], [("r1", I7)], False, None, ""),
    ("ImgThirtyTwo", "image\nrank≤32", [("ta1", T6)], [("r2a", I7)], False, None, ""),
    ("ImgEight", "image\nrank≤8", [("ta1", T6)], [("r2b", I7)], False, None, ""),
    ("ImgFour", "image\nrank≤4", [("ta1", T6)], [("im4", I7)], False, None, ""),
]
# superseded (character-level text rounds, before T-A); kept to regenerate the earlier figure: --steps char
STEPS_CHAR = [
    ("Zero", "uncalibrated (MAIN study)", [("r0", "a6_text_only")], [("r0", "a7_image_only")], False, "start",
     "image edits hit dead codewords"),
    ("One", "R1", [("r1", "a6_text_only")], [("r1", "a7_image_only")], False, "rejected", "text invalid"),
    ("TwoA", "R2a", [("r2a", "a6_text_only")], [("r2a", "a7_image_only")], False, "rejected", "text too weak"),
    ("TwoB", "R2b", [("r2a", "a6_text_only")], [("r2b", "a7_image_only")], False, "rejected", "text too weak"),  # T2 ran once (r2a)
    ("Three", "C*", [("r3t", "a6_text_only")], [("r2b", "a7_image_only")], False, "adopted", ""),
    ("Final", "C* final check", [("r3t", "a6_text_only"), ("fc_new", "a6_text_only")],
     [("r2b", "a7_image_only"), ("fc_new", "a7_image_only")], True, "adopted", ""),
]
SET2 = ("fc_new", "fcb2")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


_cache: dict[str, pd.DataFrame] = {}


def min_margin(run_dir: str) -> float:
    """Closest approach of a run, relative to its anchor: min over all evaluations of g * sign(anchor g) / |anchor g|,
    g = log p(a) - log p(b). 1 = the anchor's margin, 0 = the border, < 0 = the run crossed (user, 2026-11-28: normalised
    so that models with different margin scales are comparable). Continuous, so the per-step distributions overlap."""
    import json
    import numpy as np
    p = Path(run_dir)
    try:
        st = json.loads((p / "stats.json").read_text())
        lp = pd.read_parquet(p / "trace.parquet", columns=["logprobs"])["logprobs"]
    except Exception:  # noqa: BLE001
        return float("nan")
    g = np.array([float(x[0]) - float(x[1]) for x in lp])
    return float((g * np.sign(st["anchor_g"])).min()) / abs(float(st["anchor_g"])) if len(g) and st["anchor_g"] else float("nan")


def load(rnd: str) -> pd.DataFrame | None:
    if rnd in _cache:
        return _cache[rnd]
    root = Path(f"runs/CALIB/{rnd}")
    if not root.exists():
        return None
    res, val = Path(f"results/CALIB_{rnd}.csv"), Path(f"results/validity_CALIB_{rnd}.csv")
    py = sys.executable
    if not res.exists() or res.stat().st_mtime < max(p.stat().st_mtime for p in root.rglob("stats.json")):
        subprocess.run([py, "-m", "mmb.analysis.collect", str(root), "--out", str(res), "--exp-id", f"CALIB_{rnd}", "--require-stats"], check=True)
    if not val.exists() or val.stat().st_mtime < res.stat().st_mtime:
        subprocess.run([py, "-m", "mmb.validity.score", str(root), "--out", str(val), "--default", f"exp_id=CALIB_{rnd}", "--offline"], check=True)
    d = pd.read_csv(res)
    v = pd.read_csv(val)
    vm = v[v["view"] == "min"].drop_duplicates("run_dir").set_index("run_dir")
    for c in ("img_lpips", "vis_char_dist", "txt_minilm_cos"):
        d[c] = d["run_dir"].map(vm[c]) if c in vm else float("nan")
    d["crossed"] = d["crossed"].astype(str).str.lower().isin(("true", "1"))
    d["set"] = "set2" if (rnd in SET2 or rnd.endswith("s2")) else "set1"      # set-2 rounds: fc_new, fcb2, *s2
    d["min_margin"] = d["run_dir"].map(min_margin)
    _cache[rnd] = d
    return d


def part(parts) -> pd.DataFrame | None:
    fr = []
    for rnd, arm in parts:
        d = load(rnd)
        if d is None or not (d["arm"] == arm).any():
            return None
        fr.append(d[d["arm"] == arm])
    return pd.concat(fr, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", choices=("ta", "char"), default="ta")
    ap.add_argument("--adopt", default=None, help="step key of the adopted (calibrated) setting")
    ap.add_argument("--final", default=None, help="final check: text rounds/image rounds, e.g. 'ta1+tf2:im4+tf2' (arms a6/a7)")
    ap.add_argument("--reason", action="append", default=[], help="KEY=reason override for a rejected step")
    a = ap.parse_args()
    steps = list(STEPS_TA if a.steps == "ta" else STEPS_CHAR)
    if a.final:
        tr, ir = a.final.split(":")
        steps.append(("Final", "final\ncheck", [(r, T6) for r in tr.split("+")], [(r, I7) for r in ir.split("+")], True, "adopted", ""))
    reasons = dict(x.split("=", 1) for x in a.reason)
    key = ["set", "task", "sut", "seed_idx"]
    rows, runs = [], []
    prev_t = prev_i = None
    for k, label, tp, ip, final, status, reason in steps:
        t, i = part(tp), part(ip)
        if t is None or i is None:
            print(f"step {k}: runs missing, skipped"); continue
        if len(t) != len(i):                 # a round still running: never plot a partial step
            print(f"step {k}: incomplete ({len(t)} text vs {len(i)} image runs), skipped"); continue
        for mod, dd, src, prev in (("text", t, tp, prev_t), ("image", i, ip, prev_i)):
            for _, rr in dd.iterrows():
                runs.append(dict(step=k, label=label, final=final, modality=mod, carried=(not final and prev == src),
                                 run_dir=rr["run_dir"], task=rr["task"], sut=rr["sut"], set=rr["set"], seed_idx=rr["seed_idx"],
                                 crossed=bool(rr["crossed"]), min_margin=rr["min_margin"]))
        if not final:
            prev_t, prev_i = tp, ip
        m = t.set_index(key)["crossed"].rename("text").to_frame().join(i.set_index(key)["crossed"].rename("image"), how="inner")
        n = len(m)
        kt, ki = int(m["text"].sum()), int(m["image"].sum())
        n10, n01 = int((m["text"] & ~m["image"]).sum()), int((~m["text"] & m["image"]).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        tl, th = wilson(kt, n); il, ih = wilson(ki, n)
        med = lambda d, c: float(pd.to_numeric(d.loc[d["crossed"], c], errors="coerce").median()) if d["crossed"].any() else float("nan")
        if status is None:
            status = "adopted" if k == a.adopt else ("rejected" if p < 0.05 else "balanced")
            reason = reasons.get(k) or ("unbalanced" if p < 0.05 else "")
        if final and a.adopt:
            label = f"final check\n({n} pairs)"
        rows.append(dict(step=k, label=label, final=final, status=status, reason=reason, pairs=n, text_cr=100 * kt / n, text_lo=100 * tl, text_hi=100 * th,
                         image_cr=100 * ki / n, image_lo=100 * il, image_hi=100 * ih, text_only=n10, image_only=n01, p_mcnemar=p,
                         image_lpips_med=med(i, "img_lpips"), text_vis_char_med=med(t, "vis_char_dist"),
                         text_dist_med=1 - med(t, "txt_minilm_cos") if t["crossed"].any() else float("nan"),
                         rounds_text="+".join(r for r, _ in tp), rounds_image="+".join(r for r, _ in ip)))
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.DataFrame(runs).to_csv(a.out.with_name(a.out.stem + "_runs.csv"), index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
