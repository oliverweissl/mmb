"""Join the ratings CSVs returned by raters with key.csv and summarise validity.

Usage::

    python -m mmb.validity.analyze_ratings study/main_validity/key.csv ratings_*.csv --out results/validity_human.csv

Per rating, valid =
  imagenet: the answer equals the anchor class word (key anchor_label);
  pope:     the answer to "does the image contain the object" equals the ground truth (key gt) AND the
            answer to "does the question still ask that" is yes.
Per item the validity is the mean over its raters; a group's share is the mean over its items, with a 95%
percentile bootstrap over items (10,000 resamples, seed 42). Groups: (stratum, task, arm); controls
(unedited anchors) have an empty arm. Also: mean comprehensibility, share of "can't tell" style answers and
the number of raters. Descriptive: with few items per group, do not read non-significance as equivalence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mmb.analysis import stats as st

CANT = ("__cant", "__nothing", "__else")


def rating_valid(row) -> bool:
    if row["task"] == "pope":
        return (str(row["answer"]).lower() == str(row["gt"]).lower()
                and str(row["answer_question_still_asked"]).lower() == "yes")
    return str(row["answer"]) == str(row["anchor_label"])


def load(key: Path, ratings: list[Path]) -> pd.DataFrame:
    k = pd.read_csv(key, dtype=str, keep_default_na=False)
    r = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in ratings], ignore_index=True)
    r = r.drop_duplicates(["rater_id", "item_id"], keep="last")
    d = r.merge(k, on="item_id", how="inner", validate="many_to_one")
    d["valid"] = d.apply(rating_valid, axis=1)
    d["comprehensibility"] = pd.to_numeric(d["comprehensibility"], errors="coerce")
    d["cant"] = d["answer"].isin(CANT)
    return d


def summarize(d: pd.DataFrame, n_boot: int = st.N_BOOT, by=("stratum", "task", "arm")) -> pd.DataFrame:
    rows = []
    for gk, g in d.groupby(list(by), dropna=False):
        per_item = g.groupby("item_id").agg(valid=("valid", "mean"), comp=("comprehensibility", "mean"),
                                            cant=("cant", "mean"), n=("rater_id", "nunique"))
        v = per_item["valid"].to_numpy(float)
        pt, lo, hi = st.bootstrap_stat(v, stat=np.mean, n_boot=n_boot)
        rows.append({**dict(zip(by, gk if isinstance(gk, tuple) else (gk,))), "items": len(per_item),
                     "ratings": len(g), "raters": g["rater_id"].nunique(), "valid_share": pt, "lo": lo, "hi": hi,
                     "mean_comprehensibility": float(per_item["comp"].mean()), "cant_share": float(per_item["cant"].mean())})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("key", type=Path)
    ap.add_argument("ratings", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("results/validity_human.csv"))
    ap.add_argument("--n-boot", type=int, default=st.N_BOOT)
    a = ap.parse_args(argv)
    s = summarize(load(a.key, a.ratings), a.n_boot)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(a.out, index=False)
    print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
