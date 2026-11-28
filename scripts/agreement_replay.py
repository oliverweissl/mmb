#!/usr/bin/env python
"""Greedy agreement (Discussion): does the teacher-forced argmax agree with the SUT's greedy answer? For every run of
--sut with a stored smallest crossing, the anchor (anchor_recon.png + original question) and the crossing
(min_cross.png + its question) are answered greedily and scored alone on the canonical path; the free-text answer is
mapped to the candidates. Also re-verifies the crossing alone (cross_verified).

    ../.venv/bin/python scripts/agreement_replay.py runs/MAIN --sut qwen --out results/agree_MAIN_qwen.csv
"""
import argparse
from pathlib import Path

from mmb.analysis import replay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--sut", required=True, choices=list(replay.SUT_IDS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    its = replay.items(a.runs, "min", sut=a.sut)
    if a.limit:
        its = its[:a.limit]
    sc = replay.load_scorer(a.sut, a.device)
    rows = [replay.agreement_row(it, sc) for it in its]
    replay.write_csv(rows, a.out)
    n = len(rows)
    if n:
        for tag in ("anchor", "cross"):
            k = sum(r[f"{tag}_agree"] for r in rows)
            print(f"{tag}: greedy == teacher-forced argmax in {k}/{n} ({k / n:.1%})")
        v = sum(r["cross_verified"] for r in rows)
        print(f"crossings verified alone: {v}/{n} ({v / n:.1%})")


if __name__ == "__main__":
    main()
