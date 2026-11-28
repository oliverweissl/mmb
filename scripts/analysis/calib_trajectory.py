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

# (step key, label, [(round, arm) for text-only], [(round, arm) for image-only], final)
STEPS = [
    ("Zero", "uncalibrated", [("r0", "a6_text_only")], [("r0", "a7_image_only")], False),
    ("One", "R1", [("r1", "a6_text_only")], [("r1", "a7_image_only")], False),
    ("TwoA", "R2a", [("r2a", "a6_text_only")], [("r2a", "a7_image_only")], False),
    ("TwoB", "R2b", [("r2a", "a6_text_only")], [("r2b", "a7_image_only")], False),   # T2 text ran once (r2a)
    ("Three", "C*", [("r3t", "a6_text_only")], [("r2b", "a7_image_only")], False),
    ("Final", "C* final check", [("r3t", "a6_text_only"), ("fc_new", "a6_text_only")],
     [("r2b", "a7_image_only"), ("fc_new", "a7_image_only")], True),
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
    for c in ("img_lpips", "vis_char_dist"):
        d[c] = d["run_dir"].map(vm[c]) if c in vm else float("nan")
    d["crossed"] = d["crossed"].astype(str).str.lower().isin(("true", "1"))
    d["set"] = "set2" if rnd in SET2 else "set1"
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
    a = ap.parse_args()
    key = ["set", "task", "sut", "seed_idx"]
    rows = []
    for k, label, tp, ip, final in STEPS:
        t, i = part(tp), part(ip)
        if t is None or i is None:
            print(f"step {k}: runs missing, skipped"); continue
        m = t.set_index(key)["crossed"].rename("text").to_frame().join(i.set_index(key)["crossed"].rename("image"), how="inner")
        n = len(m)
        kt, ki = int(m["text"].sum()), int(m["image"].sum())
        n10, n01 = int((m["text"] & ~m["image"]).sum()), int((~m["text"] & m["image"]).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        tl, th = wilson(kt, n); il, ih = wilson(ki, n)
        med = lambda d, c: float(pd.to_numeric(d.loc[d["crossed"], c], errors="coerce").median()) if d["crossed"].any() else float("nan")
        rows.append(dict(step=k, label=label, final=final, pairs=n, text_cr=100 * kt / n, text_lo=100 * tl, text_hi=100 * th,
                         image_cr=100 * ki / n, image_lo=100 * il, image_hi=100 * ih, text_only=n10, image_only=n01, p_mcnemar=p,
                         image_lpips_med=med(i, "img_lpips"), text_vis_char_med=med(t, "vis_char_dist"),
                         rounds_text="+".join(r for r, _ in tp), rounds_image="+".join(r for r, _ in ip)))
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
