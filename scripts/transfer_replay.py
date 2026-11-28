#!/usr/bin/env python
"""Transfer of crossings across SUTs (Discussion): every stored smallest crossing (min_cross.*) and its anchor
(anchor_recon.png) replayed on every OTHER SUT, each input scored alone on the canonical path. A crossing transfers if,
on the destination SUT, the crossing has the opposite sign of g to the destination's anchor. One SUT on the GPU at a time.

    ../.venv/bin/python scripts/transfer_replay.py runs/MAIN --arms a0_directed a5_mimicry_objective a9_deepjanus \
        --out results/transfer_MAIN.csv
"""
import argparse, gc
from pathlib import Path

from mmb.analysis import replay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--dst", nargs="+", default=list(replay.SUT_IDS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    its = replay.items(a.runs, "min", a.arms)
    rows = []
    for dst in a.dst:
        sc = replay.load_scorer(dst, a.device)
        rows += replay.transfer_rows(its, sc, dst)
        del sc
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    replay.write_csv(rows, a.out)
    print(f"{len(rows)} replays -> {a.out}")


if __name__ == "__main__":
    main()
