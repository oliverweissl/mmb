"""Gallery figure: anchor beside first-crossing image, per task and SUT (the paper's teaser).

Usage::

    python -m mmb.analysis.gallery results/main.csv --out results/figures/gallery.pdf \
        [--arm a0_directed] [--per-task 2] [--runs-root runs/MAIN] [--runs picks.csv]

Layout: one row per (task, example), one column per SUT; every tile shows the anchor (origin.png) and the
first-crossing image (first_cross.png) side by side, with the original question, the edited question and the
pair contrast g before (anchor reference) and after (at the crossing). Both images are shown at the codec
resolution of first_cross.png (origin.png is resized to it, no cropping).

Selection (default): among crossing runs of --arm, per task the per_task cells that most SUTs cross
(so a row compares SUTs on the same cell), ties broken by the smallest mean image magnitude at the crossing;
within a (cell, SUT) the run with the fewest evaluations to cross. --runs takes a CSV with columns
task,sut,run_dir to override the selection. Missing images leave a labelled empty tile.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from mmb.analysis.stats import load_results  # noqa: E402

SUT_LABELS = {"qwen": "Qwen3.5-4B", "internvl": "InternVL3.5-4B", "ministral": "Ministral-3-3B",
              "llava": "LLaVA-NeXT-7B"}
TASK_LABELS = {"imagenet": "ImageNet pairs", "pope": "POPE yes/no"}


def select_runs(df: pd.DataFrame, arm: str = "a0_directed", per_task: int = 2,
                suts: list[str] | None = None) -> pd.DataFrame:
    """Rows of df (one per chosen run) with an added 'example' index; see the module docstring."""
    d = df[(df["arm"] == arm) & df["crossed"].astype(bool)].copy()
    if suts:
        d = d[d["sut"].isin(suts)]
    picks = []
    for task, t in d.groupby("task", sort=True):
        best = t.sort_values("first_cross_eval").groupby(["cell_id", "sut"], as_index=False).first()
        stat = best.groupby("cell_id").agg(n_sut=("sut", "nunique"), mag=("img_mag_at_cross", "mean"))
        order = stat.reset_index().sort_values(["n_sut", "mag", "cell_id"], ascending=[False, True, True])
        for ex, cell in enumerate(order["cell_id"].head(per_task)):
            rows = best[best["cell_id"] == cell].copy()
            rows["example"] = ex
            picks.append(rows)
    return pd.concat(picks, ignore_index=True) if picks else d.iloc[0:0].assign(example=[])


def _load(run_dir: Path, runs_root: Path | None):
    p = Path(run_dir)
    if not p.is_absolute() and runs_root is not None and not p.exists():
        p = runs_root / p
    stats = json.loads((p / "stats.json").read_text())
    fc = json.loads((p / "first_cross.json").read_text()) if (p / "first_cross.json").exists() else {}
    o = Image.open(p / "origin.png").convert("RGB") if (p / "origin.png").exists() else None
    c = Image.open(p / "first_cross.png").convert("RGB") if (p / "first_cross.png").exists() else None
    if o is not None and c is not None and o.size != c.size:
        o = o.resize(c.size, Image.BICUBIC)
    return stats, fc, o, c


def _fmt_g(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):+.2f}"


def _wrap(s: str, width: int, lines: int = 2) -> str:
    w = textwrap.wrap(s, width)
    if len(w) > lines:
        w = w[:lines]
        w[-1] = w[-1][: max(0, width - 3)] + "..."
    return "\n".join(w)


def plot_gallery(picks: pd.DataFrame, out: Path | str, *, runs_root: Path | None = None,
                 width_in: float = 7.0, img_in: float = 0.85) -> Path:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6})
    tasks = [t for t in ("imagenet", "pope") if t in set(picks["task"])] or sorted(set(picks["task"]))
    suts = [s for s in SUT_LABELS if s in set(picks["sut"])] or sorted(set(picks["sut"]))
    n_ex = int(picks["example"].max()) + 1 if len(picks) else 1
    rows = [(t, e) for t in tasks for e in range(n_ex)]
    lab_w, head_h, txt_h, gap, title_h = 0.35, 0.22, 0.62, 0.08, 0.14
    tile_w = (width_in - lab_w) / max(1, len(suts))
    tile_h = title_h + img_in + txt_h + gap
    height = head_h + tile_h * len(rows)
    fig = plt.figure(figsize=(width_in, height))

    def axes(x, y, w, h):
        a = fig.add_axes([x / width_in, 1 - (y + h) / height, w / width_in, h / height])
        a.set_axis_off()
        return a

    for j, s in enumerate(suts):
        a = axes(lab_w + j * tile_w, 0, tile_w, head_h)
        a.text(0.5, 0.5, SUT_LABELS.get(s, s), ha="center", va="center", fontsize=7, fontweight="bold",
               transform=a.transAxes)
    for i, (t, e) in enumerate(rows):
        y0 = head_h + i * tile_h
        if e == 0:
            a = axes(0, y0, lab_w, tile_h * n_ex)
            a.text(0.5, 0.5, TASK_LABELS.get(t, t), rotation=90, ha="center", va="center", fontsize=7,
                   fontweight="bold", transform=a.transAxes)
        for j, s in enumerate(suts):
            r = picks[(picks["task"] == t) & (picks["sut"] == s) & (picks["example"] == e)]
            x0 = lab_w + j * tile_w + (tile_w - 2 * img_in - 0.06) / 2
            if r.empty:
                a = axes(x0, y0 + title_h, 2 * img_in + 0.06, img_in)
                a.text(0.5, 0.5, "no crossing run", ha="center", va="center", fontsize=6, color="0.5",
                       transform=a.transAxes)
                continue
            r = r.iloc[0]
            stats, fc, o, c = _load(Path(r["run_dir"]), runs_root)
            for k, (img, cap) in enumerate(((o, "anchor"), (c, "first crossing"))):
                a = axes(x0 + k * (img_in + 0.06), y0 + title_h, img_in, img_in)
                if img is not None:
                    a.imshow(np.asarray(img))
                else:
                    a.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6, color="0.5",
                           transform=a.transAxes)
                a.set_title(cap, fontsize=5.5, pad=1)
            meta = stats.get("seed_metadata") or {}
            q0 = meta.get("prompt_template") or stats.get("prompt_template") or ""
            q1 = fc.get("text") or ""
            g0 = stats.get("anchor_g")
            cand = " / ".join(map(str, (stats.get("class_a"), stats.get("class_b"))))
            txt = [f"options: {cand}", "Q: " + _wrap(q0, 46, 1)]
            if q1 and q1 != q0:
                txt.append("Q': " + _wrap(q1, 46, 2))
            else:
                txt.append("Q': unchanged")
            txt.append(f"g: {_fmt_g(g0)} → {_fmt_g(fc.get('g'))}  (eval {fc.get('eval_idx', '?')})")
            a = axes(x0, y0 + title_h + img_in + 0.04, 2 * img_in + 0.06, txt_h)
            a.text(0, 1, "\n".join(txt), va="top", ha="left", fontsize=5, linespacing=1.15,
                   transform=a.transAxes)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/figures/gallery.pdf"))
    ap.add_argument("--arm", default="a0_directed")
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--runs-root", type=Path, default=None, help="base for relative run_dir values")
    ap.add_argument("--runs", type=Path, default=None, help="CSV task,sut,run_dir overriding the selection")
    ap.add_argument("--sut", action="append", default=[], help="restrict to these SUTs")
    args = ap.parse_args(argv)
    df = load_results(args.csv)
    if args.runs:
        picks = pd.read_csv(args.runs)
        picks["example"] = picks.groupby(["task", "sut"]).cumcount()
    else:
        picks = select_runs(df, args.arm, args.per_task, args.sut or None)
    if picks.empty:
        raise SystemExit("no crossing runs to show")
    print(f"wrote {plot_gallery(picks, args.out, runs_root=args.runs_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
