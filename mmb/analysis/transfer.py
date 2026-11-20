"""Cross-SUT transfer of crossing inputs: does a crossing found on SUT A also flip SUT B?

Input: the replay CSV (one row per crossing input per destination SUT) with columns
run_dir, src_sut, dst_sut, task, g_dst, flipped_on_dst (bool: the input, scored on dst against dst's own
anchor, has the opposite sign). Optional columns arm / rng_seed / cell_id are used when present; otherwise
arm is joined from a collect.py CSV via run_dir (--results).

Output (via mmb.analysis.export or the CLI): transfer matrix (rows = source SUT, columns =
destination SUT) of the share of crossings that flip the destination, with Wilson 95% intervals, pooled over
tasks and per task; the diagonal (src == dst), when present, is the single-item re-verification rate.
Macros ETransfer{Count,N,Share,Ci}{Src}{Dst}{Task} and the mean off-diagonal share.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mmb.analysis import stats as st

REQUIRED = ("run_dir", "src_sut", "dst_sut", "task", "flipped_on_dst")


def load_transfer(paths, results: Path | None = None, arm: str | None = None) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    miss = [c for c in REQUIRED if c not in df]
    if miss:
        raise ValueError(f"transfer CSV lacks columns {miss}")
    df["flipped_on_dst"] = df["flipped_on_dst"].astype(str).str.lower().isin(("true", "1", "1.0"))
    if "arm" not in df and results is not None:
        res = pd.read_csv(results)[["run_dir", "arm"]].drop_duplicates("run_dir")
        df = df.merge(res, on="run_dir", how="left")
    if arm is not None:
        if "arm" not in df:
            raise ValueError("no arm column and no --results to join it from")
        df = df[df["arm"] == arm]
    return df.drop_duplicates(["run_dir", "dst_sut"]).reset_index(drop=True)


def matrix_rows(df: pd.DataFrame) -> list[dict]:
    """Wilson shares per (src, dst, task) for task in {'All', each task}."""
    rows = []
    tasks = ["All"] + sorted(set(df["task"]))
    for task in tasks:
        d = df if task == "All" else df[df["task"] == task]
        for (src, dst), g in d.groupby(["src_sut", "dst_sut"], sort=True):
            k, n = int(g["flipped_on_dst"].sum()), len(g)
            lo, hi = st.wilson(k, n)
            rows.append(dict(src=src, dst=dst, task=task, k=k, n=n, share=k / n, lo=lo, hi=hi))
    return rows


def mean_off_diagonal(rows: list[dict], task: str = "All") -> tuple[float, int]:
    """Pooled share over src != dst cells (sum of flips over sum of crossings) and its denominator."""
    r = [x for x in rows if x["task"] == task and x["src"] != x["dst"]]
    n = sum(x["n"] for x in r)
    return (sum(x["k"] for x in r) / n if n else float("nan")), n


def matrix_tex(rows: list[dict], task: str = "All", names: dict | None = None) -> str:
    names = names or {}
    r = [x for x in rows if x["task"] == task]
    srcs = sorted({x["src"] for x in r})
    dsts = sorted({x["dst"] for x in r})
    by = {(x["src"], x["dst"]): x for x in r}
    body = []
    for s in srcs:
        cells = []
        for d in dsts:
            x = by.get((s, d))
            cells.append("--" if x is None else f"{100 * x['share']:.1f} [{100 * x['lo']:.0f}--{100 * x['hi']:.0f}] ($n={x['n']}$)")
        body.append(" & ".join([st.tex_escape(names.get(s, s))] + cells) + r" \\")
    head = " & ".join([r"Found on $\downarrow$ / flips on $\rightarrow$"] + [st.tex_escape(names.get(d, d)) for d in dsts])
    return ("\\begin{tabular}{@{}l" + "r" * len(dsts) + "@{}}\n\\toprule\n" + head + " \\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--results", type=Path, default=None, help="collect.py CSV to join arm via run_dir")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/transfer_matrix.csv"))
    args = ap.parse_args(argv)
    rows = matrix_rows(load_transfer(args.csv, args.results, args.arm))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
