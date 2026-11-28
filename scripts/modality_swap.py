#!/usr/bin/env python
"""Modality attribution by swapping (paper): for every stored crossing of --sut (view min: the smallest in-budget
crossing; view first: the first crossing), score alone on the canonical path the anchor, the crossing, (a) the original
image with the edited question and (b) the edited image with the original question. (a) and (b) are the renderings of
the crossing's genotype with the image block, resp. the text block, set to zero. Classes per verified crossing:
image (b crosses, a not), text (a crosses, b not), either (both), joint (neither); crossings that do not re-verify
alone are "unverified". Analysis: mmb.analysis.modality_swap.

    ../.venv/bin/python scripts/modality_swap.py runs/MAIN --sut qwen --out results/modswap/MAIN_qwen.csv
"""
import argparse, collections
from pathlib import Path

from mmb.analysis import replay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--sut", required=True, choices=list(replay.SUT_IDS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", default="min,first")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="", help="free-text label stored per row")
    a = ap.parse_args()
    views = [v for v in a.views.split(",") if v]
    its = [it for v in views for it in replay.items(a.runs, v, a.arms, a.sut)]
    if a.limit:
        its = its[:a.limit]
    sc = replay.load_scorer(a.sut, a.device)
    rows = [replay.swap_row(it, sc, a.label) for it in its]
    replay.write_csv(rows, a.out)
    for v in views:
        print(v, dict(sorted(collections.Counter((r["arm"], r["cls"]) for r in rows if r["view"] == v).items())))


if __name__ == "__main__":
    main()
