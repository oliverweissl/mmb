"""Modality attribution by swapping (user request 2026-11-28).

For every crossing a run produced (view "min": the minimal crossing; view "first": the first crossing), score alone,
by teacher forcing with the same SUT and answer format:
  anchor:  original image (the codec reconstruction anchor_recon.png) + original question
  cross:   perturbed image + perturbed question (the crossing itself, re-verified alone)
  (a)      original image + perturbed question   -> does the TEXT edit suffice?
  (b)      perturbed image + original question   -> does the IMAGE edit suffice?
g = log p(class_a) - log p(class_b); an input "crosses" when sign(g) differs from the anchor's.
Class per verified crossing: image-sufficient (b crosses, a not), text-sufficient (a crosses, b not),
either-sufficient (both), joint-necessary (neither: needs both channels). Crossings that do not re-verify alone are
"unverified" and kept out of the shares. 1-minimality does not imply the class (the search space is not monotone).

    python scripts/modality_swap.py runs/CALIB/fc_new --sut qwen --out results/modswap/fc_new_qwen.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

SUT_IDS = {"qwen": ("Qwen/Qwen3.5-4B", False), "ministral": ("mistralai/Ministral-3-3B-Instruct-2512", False),
           "internvl": ("OpenGVLab/InternVL3_5-4B-HF", False)}
VIEWS = {"min": "min_cross", "first": "first_cross"}


def classify(a_cross: bool, b_cross: bool) -> str:
    return {(False, True): "image", (True, False): "text", (True, True): "either", (False, False): "joint"}[(a_cross, b_cross)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--sut", required=True, choices=list(SUT_IDS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", default="min,first")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="", help="free-text label stored per row (e.g. 'MAIN dry run, old operators')")
    a = ap.parse_args()
    from mmb.runners.run_boundary_test import _disable_hub_kernels, _warm_up_numba
    _warm_up_numba()
    if a.sut != "ministral":
        _disable_hub_kernels()
    from mmb.sut.scorer import create_scorer

    model_id, q4 = SUT_IDS[a.sut]
    sc = create_scorer(model_id, a.device, enable_thinking=False, load_in_4bit=q4)
    views = [v for v in a.views.split(",") if v]
    dirs = sorted({p.parent for r in a.runs for p in r.rglob("min_cross.json")} |
                  {p.parent for r in a.runs for p in r.rglob("first_cross.json")})
    rows = []
    for d in dirs:
        st = json.loads((d / "stats.json").read_text())
        meta = st.get("run_meta", {})
        if meta.get("sut") not in (None, a.sut):
            continue
        cands = list(st["categories"])
        suffix = st["answer_format"].format(categories=", ".join(cands))
        q0 = (st.get("seed_metadata") or {}).get("prompt_template") or st["prompt_template"]
        p0 = q0 + suffix
        img0 = Image.open(d / "anchor_recon.png").convert("RGB")

        def g(img, prompt):
            s = sc.score_batch_tensor([img], [prompt], cands)[0].tolist()
            return float(s[0] - s[1])

        g0 = g(img0, p0)
        a0 = np.asarray(img0, dtype=np.int16)
        for v in views:
            j = d / f"{VIEWS[v]}.json"
            if not j.exists():
                continue
            fc = json.loads(j.read_text())
            img1 = Image.open(d / f"{VIEWS[v]}.png").convert("RGB")
            p1 = fc["full_prompt"]
            gx, ga, gb = g(img1, p1), g(img0, p1), g(img1, p0)
            side = np.sign(g0)
            crosses = lambda x: bool(np.sign(x) != side)
            verified = crosses(gx)
            row = dict(run_dir=str(d), view=v, label=a.label, sut=a.sut,
                       **{k: meta.get(k) for k in ("exp_id", "task", "arm", "rng_seed")},
                       seed_idx=st.get("seed_idx"), g_anchor=g0, g_anchor_search=st.get("anchor_g"), g_cross=gx,
                       g_cross_search=fc.get("g"), g_text_only=ga, g_image_only=gb,
                       text_edited=p1 != p0, image_edited=bool(np.abs(np.asarray(img1, dtype=np.int16) - a0).max() > 0),
                       verified=verified, a_crosses=crosses(ga), b_crosses=crosses(gb),
                       cls=classify(crosses(ga), crosses(gb)) if verified else "unverified",
                       # margin left of the anchor's (1 = unchanged, 0 = at the border, < 0 = crossed)
                       rel_text_only=ga / g0 if g0 else np.nan, rel_image_only=gb / g0 if g0 else np.nan,
                       rel_cross=gx / g0 if g0 else np.nan)
            rows.append(row)
        if a.limit and len(rows) >= a.limit:
            break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["run_dir"])
        w.writeheader()
        w.writerows(rows)
    import collections
    for v in views:
        c = collections.Counter((r["arm"], r["cls"]) for r in rows if r["view"] == v)
        print(v, dict(sorted(c.items())))
    dev = [abs(r["g_anchor"] - r["g_anchor_search"]) for r in rows if r.get("g_anchor_search") is not None]
    if dev:
        print(f"anchor g vs search: max |diff| = {max(dev):.4f} nats over {len(dev)} rows")


if __name__ == "__main__":
    main()
