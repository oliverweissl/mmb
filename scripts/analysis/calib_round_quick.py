#!/usr/bin/env python
"""Quick overview of one calibration round (continuous design, 2026-11-28). C1 owns the full analysis / figures.

    ../.venv/bin/python scripts/analysis/calib_round_quick.py runs/CALIB/w1.0_1.0 --out docs/figs/calib/w1.0_1.0

Reads every finished seed run (stats.json, first_cross.json, min_cross.json, boundary_pair.json) and writes
  summary.csv            one row per (task, SUT, arm, cell): crossed, first-cross eval, min magnitude, confirmed pair
  summary.md             crossing share per arm x SUT x task; exact McNemar text-only (a6) vs image-only (a7) on
                         matched cells, per SUT and pooled; median magnitude of the min crossing
  montage_<arm>.png      6 min crossings per arm: unedited image + question (top) vs crossing image + question (bottom)
"""
import argparse, json, math, textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = {"a6_text_only": "text-only (a6)", "a7_image_only": "image-only (a7)", "a0_directed": "MMBA (a0)"}


def load(root: Path) -> pd.DataFrame:
    rows = []
    for st in sorted(root.glob("*/*/*/s*/*/stats.json")):
        d = st.parent
        s = json.loads(st.read_text())
        meta = s.get("run_meta", {})
        fc = json.loads((d / "first_cross.json").read_text()) if (d / "first_cross.json").exists() else None
        mc = json.loads((d / "min_cross.json").read_text()) if (d / "min_cross.json").exists() else None
        bp = json.loads((d / "boundary_pair.json").read_text()) if (d / "boundary_pair.json").exists() else None
        # margin closed: 1 - min over the trace of sign(g0) g / |g0| (>= 1 means crossed)
        closed = None
        tp = d / "trace.parquet"
        if tp.exists() and s.get("anchor_g"):
            import pyarrow.parquet as pq
            ia, ib = s.get("target_classes", [0, 1])
            lp = pq.read_table(tp, columns=["logprobs"]).column("logprobs").to_pylist()
            g = np.array([v[ia] - v[ib] for v in lp])
            g0 = float(s["anchor_g"])
            closed = float(1.0 - (np.sign(g0) * g).min() / abs(g0)) if g.size else None
        rows.append({"margin_closed": closed, "task": meta.get("task"), "sut": meta.get("sut"), "arm": meta.get("arm"),
                     "cell": s.get("seed_idx"), "run_dir": str(d), "anchor_g": s.get("anchor_g"),
                     "question": ((s.get("seed_metadata") or {}).get("prompt_template")
                                  or (s.get("seed_metadata") or {}).get("prompt") or s.get("prompt_template")),
                     "crossed": fc is not None, "first_cross_eval": fc["eval_idx"] if fc else None,
                     "first_text": fc["text"] if fc else None,
                     "min_magnitude": mc.get("magnitude") if mc else None,
                     "min_mag_image": mc.get("mag_image") if mc else None,
                     "min_mag_text": mc.get("mag_text") if mc else None,
                     "min_text": mc.get("text") if mc else None, "min_g": mc.get("g") if mc else None,
                     "min_confirmed": mc.get("confirmed") if mc else None,
                     "pair_source": bp.get("pair_source") if bp else None,
                     "pair_kind": bp.get("kind") if bp else None,
                     "canonical_consistent": bp.get("canonical_consistent") if bp else None})
    return pd.DataFrame(rows)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summary_md(df: pd.DataFrame, rnd: str) -> str:
    L = [f"# Calibration round {rnd}: quick summary", "",
         "Crossing = a crossing within the budget (canonical scoring). Cells: held-out calibration cells.", ""]
    tab = df.groupby(["sut", "task", "arm"]).agg(n=("crossed", "size"), crossed=("crossed", "sum"),
                                                 med_mag=("min_magnitude", "median"),
                                                 closed=("margin_closed", "median")).reset_index()
    L += ["margin closed = 1 - min(sign(g0) g) / |g0| over the run (>= 100 % = crossed); median over cells.", "",
          "| SUT | task | arm | crossed / n | share | median margin closed | median min magnitude |",
          "|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        L.append(f"| {r.sut} | {r.task} | {ARMS.get(r.arm, r.arm)} | {int(r.crossed)} / {r.n} | "
                 f"{r.crossed / r.n:.0%} | {r.closed:.0%} | {r.med_mag:.3f} |")
    L += ["", "## Text-only vs image-only (exact McNemar on matched cells)", "",
          "| SUT | n matched | both | text only | image only | neither | p |", "|---|---|---|---|---|---|---|"]
    piv = df[df.arm.isin(["a6_text_only", "a7_image_only"])].pivot_table(
        index=["sut", "task", "cell"], columns="arm", values="crossed", aggfunc="first")
    if not {"a6_text_only", "a7_image_only"} <= set(piv.columns):
        piv = piv.iloc[0:0].reindex(columns=["a6_text_only", "a7_image_only"])
    piv = piv.dropna()
    groups = [(s, piv.xs(s, level="sut")) for s in sorted(piv.index.get_level_values("sut").unique())] if len(piv) else []
    groups.append(("all", piv))
    for name, g in groups:
        t, i = g["a6_text_only"].astype(bool), g["a7_image_only"].astype(bool)
        b, c = int((t & ~i).sum()), int((~t & i).sum())
        L.append(f"| {name} | {len(g)} | {int((t & i).sum())} | {b} | {c} | {int((~t & ~i).sum())} | "
                 f"{mcnemar_exact(b, c):.3g} |")
    n_unconf = int((df["canonical_consistent"] == False).sum())  # noqa: E712
    L += ["", f"Canonical consistency: {n_unconf} run(s) with canonical_consistent = false.",
          f"Pairs: {df.pair_source.value_counts().to_dict()}; kinds: {df.pair_kind.value_counts().to_dict()}."]
    return "\n".join(L) + "\n"


def montage(df: pd.DataFrame, arm: str, out: Path, n: int = 6) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    sel = df[(df.arm == arm) & df.crossed].sort_values("min_magnitude")
    if sel.empty:
        return
    # spread over SUT x task: take round-robin from each group, smallest magnitude first
    picks, groups = [], [g for _, g in sel.groupby(["sut", "task"])]
    while len(picks) < n and any(len(g) for g in groups):
        for k, g in enumerate(groups):
            if len(g) and len(picks) < n:
                picks.append(g.iloc[0]); groups[k] = g.iloc[1:]
    fig, ax = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 6.2), squeeze=False)
    for j, r in enumerate(picks):
        d = Path(r.run_dir)
        q0 = r.question or ""
        for i, (img, txt, head) in enumerate([
                ("anchor_recon.png", q0, f"{r.sut} | {r.task} | cell {r.cell}\nunedited  g = {r.anchor_g:+.2f}"),
                ("min_cross.png", r.min_text or "", f"min crossing  g = {r.min_g:+.2f}\n"
                 f"|img| {r.min_mag_image:.3f}  |txt| {r.min_mag_text:.3f}")]):
            a = ax[i, j]
            a.imshow(Image.open(d / img).convert("RGB")); a.set_xticks([]); a.set_yticks([])
            a.set_title(head, fontsize=7)
            changed = i == 1 and txt != q0
            a.set_xlabel("\n".join(textwrap.wrap(txt, 34)), fontsize=6.5,
                         color="#B22222" if changed else "#333333")
    fig.suptitle(f"{ARMS.get(arm, arm)}: smallest crossings (bottom)\nvs unedited input (top); red = edited question",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(out / f"montage_{arm}.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("root", type=Path); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = load(a.root)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "summary.csv", index=False)
    md = summary_md(df, a.root.name)
    (a.out / "summary.md").write_text(md)
    for arm in ARMS:
        montage(df, arm, a.out)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
