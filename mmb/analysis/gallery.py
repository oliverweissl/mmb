"""Gallery figure: anchor beside first-crossing image, per task and SUT (the paper's teaser).

Usage::

    python -m mmb.analysis.gallery results/main.csv --out results/figures/gallery.pdf \
        [--arm a0_directed] [--per-task 2] [--runs-root runs/MAIN] [--runs picks.csv]

Layout: one row per (task, example), one column per SUT; every tile shows the anchor (origin.png) and the
first-crossing image (first_cross.png) side by side, with the original question, the edited question and the
pair contrast g before (anchor reference) and after (at the crossing). Both images are shown at the codec
resolution of first_cross.png (origin.png is resized to it, no cropping).

With --failures an extra row per task shows a typical unsuccessful run: the run without a crossing whose floor
is the median over such runs, the anchor beside the Pareto member closest to the boundary.

Selection (default): among crossing runs of --arm, per task the per_task cells that most SUTs cross (so a row
compares SUTs on the same cell), then the smallest mean text cosine distance at the crossing (diag_txt_cosdist),
ties broken by the smallest image RMS; within a (cell, SUT) the crossing with the smallest text distance, then
image RMS. One further image-dominated row (a cell with a crossing of text cosine distance < 0.05) is added
when one exists. A garbled question is therefore never picked in preference to a lightly edited one. --runs takes a CSV with columns
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
                suts: list[str] | None = None, img_dominated_max: float = 0.05) -> pd.DataFrame:
    """Rows of df (one per chosen run) with an added 'example' index and a 'kind'; see the module docstring.

    Text distance is the cosine distance of the text embedding at the first crossing (txt_mag_at_cross), image
    distance the RMS (img_mag_at_cross); only diagnostic magnitudes are used when the data has them (never mixed
    with optimised-fitness ones). Without magnitudes the fallback is the fewest evaluations to cross."""
    d = df[(df["arm"] == arm) & df["crossed"].astype(bool)].copy()
    if suts:
        d = d[d["sut"].isin(suts)]
    if "mag_source" in d and (d["mag_source"] == "diag").any():
        d = d[d["mag_source"] == "diag"]
    has_mag = "txt_mag_at_cross" in d and d["txt_mag_at_cross"].notna().any()
    if has_mag:
        d = d[d["txt_mag_at_cross"].notna()]
    picks = []
    for task, t in d.groupby("task", sort=True):
        if has_mag:
            key = ["txt_mag_at_cross", "img_mag_at_cross", "first_cross_eval", "run_dir"]
        else:
            key = ["first_cross_eval", "run_dir"]
        best = t.sort_values(key).groupby(["cell_id", "sut"], as_index=False).first()
        if has_mag:
            stat = best.groupby("cell_id").agg(n_sut=("sut", "nunique"), txt=("txt_mag_at_cross", "mean"),
                                               img=("img_mag_at_cross", "mean"), tmin=("txt_mag_at_cross", "min"))
            order = stat.reset_index().sort_values(["n_sut", "txt", "img", "cell_id"], ascending=[False, True, True, True])
        else:
            stat = best.groupby("cell_id").agg(n_sut=("sut", "nunique"), mag=("img_mag_at_cross", "mean"))
            order = stat.reset_index().sort_values(["n_sut", "mag", "cell_id"], ascending=[False, True, True])
        chosen = list(order["cell_id"].head(per_task))
        for ex, cell in enumerate(chosen):
            rows = best[best["cell_id"] == cell].copy()
            rows["example"], rows["kind"] = ex, "cross"
            picks.append(rows)
        if has_mag:   # one image-dominated example (text cosine < img_dominated_max), if any cell has one
            cand = order[(order["tmin"] < img_dominated_max) & ~order["cell_id"].isin(chosen)]
            if len(cand):
                cand = cand.sort_values(["n_sut", "img", "txt", "cell_id"], ascending=[False, True, True, True])
                rows = best[best["cell_id"] == cand.iloc[0]["cell_id"]].copy()
                rows["example"], rows["kind"] = per_task, "img"
                picks.append(rows)
    return pd.concat(picks, ignore_index=True) if picks else d.iloc[0:0].assign(example=[], kind=[])


def select_failures(df: pd.DataFrame, arm: str = "a0_directed", suts: list[str] | None = None) -> pd.DataFrame:
    """One typical failure per (task, SUT): among runs of arm without a crossing, the run with the median floor
    (not the best or the worst), so the row shows what a typical unsuccessful search looks like."""
    d = df[(df["arm"] == arm) & ~df["crossed"].astype(bool)].copy()
    if suts:
        d = d[d["sut"].isin(suts)]
    rows = []
    for (task, sut), g in d.groupby(["task", "sut"], sort=True):
        g = g.sort_values(["floor_nats", "run_dir"]).reset_index(drop=True)
        rows.append(g.iloc[[len(g) // 2]].assign(kind="fail"))
    return pd.concat(rows, ignore_index=True) if rows else d.iloc[0:0]


def _closest_pareto(run_dir: Path):
    """(png, json dict) of the Pareto member with the smallest |g| (first fitness value), or (None, {})."""
    best = None
    for jf in sorted(run_dir.glob("pareto_*.json")):
        try:
            j = json.loads(jf.read_text())
        except ValueError:
            continue
        f = j.get("fitness") or [float("inf")]
        if best is None or f[0] < best[0]:
            best = (f[0], jf, j)
    if best is None:
        return None, {}
    return best[1].with_suffix(".png"), best[2]


def _load(run_dir: Path, runs_root: Path | None, fail: bool = False):
    p = Path(run_dir)
    if not p.is_absolute() and runs_root is not None and not p.exists():
        p = runs_root / p
    stats = json.loads((p / "stats.json").read_text())
    fc = json.loads((p / "first_cross.json").read_text()) if (p / "first_cross.json").exists() else {}
    o = None
    if (p / "anchor_recon.png").exists():          # the unedited input as the SUT sees it (codec reconstruction)
        o = Image.open(p / "anchor_recon.png").convert("RGB")
    elif (p / "origin.png").exists():              # older runs: raw photograph, resized + centre-cropped like the codec input
        raw = Image.open(p / "origin.png").convert("RGB")
        side = min(raw.size)
        o = raw.crop(((raw.width - side) // 2, (raw.height - side) // 2, (raw.width + side) // 2, (raw.height + side) // 2))
    c = Image.open(p / "first_cross.png").convert("RGB") if (p / "first_cross.png").exists() else None
    if fail:
        png, j = _closest_pareto(p)
        fc = {"text": j.get("text"), "min_abs_g": (j.get("fitness") or [None])[0]}
        c = Image.open(png).convert("RGB") if png is not None and png.exists() else None
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
    is_fail = picks["kind"].eq("fail") if "kind" in picks else pd.Series(False, index=picks.index)
    n_ex = int(picks.loc[~is_fail, "example"].max()) + 1 if (~is_fail).any() else 1
    has_fail = bool(is_fail.any())
    rows = [(t, e) for t in tasks for e in range(n_ex + (1 if has_fail else 0))]
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
            a = axes(0, y0, lab_w, tile_h * (n_ex + (1 if has_fail else 0)))
            a.text(0.5, 0.5, TASK_LABELS.get(t, t), rotation=90, ha="center", va="center", fontsize=7,
                   fontweight="bold", transform=a.transAxes)
        for j, s in enumerate(suts):
            fail = has_fail and e == n_ex
            if fail:
                r = picks[(picks["task"] == t) & (picks["sut"] == s) & is_fail]
            else:
                r = picks[(picks["task"] == t) & (picks["sut"] == s) & (picks["example"] == e) & ~is_fail]
            x0 = lab_w + j * tile_w + (tile_w - 2 * img_in - 0.06) / 2
            if r.empty:
                a = axes(x0, y0 + title_h, 2 * img_in + 0.06, img_in)
                a.text(0.5, 0.5, "no failure run" if fail else "no crossing run", ha="center", va="center", fontsize=6, color="0.5",
                       transform=a.transAxes)
                continue
            r = r.iloc[0]
            stats, fc, o, c = _load(Path(r["run_dir"]), runs_root, fail)
            for k, (img, cap) in enumerate(((o, "anchor"), (c, "closest input, no crossing" if fail else "first crossing"))):
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
            kind = r.get("kind") if hasattr(r, "get") else None
            if not fail and kind == "img":
                txt[0] += "  [image-dominated]"
            if q1 and q1 != q0:
                txt.append("Q': " + _wrap(q1, 46, 2))
            else:
                txt.append("Q': unchanged")
            if fail:
                m = fc.get("min_abs_g")
                txt.append(f"g: {_fmt_g(g0)}; closest |g| = " + ("n/a" if m is None else f"{m:.2f}") + " (budget exhausted)")
            else:
                txt.append(f"g: {_fmt_g(g0)} \u2192 {_fmt_g(fc.get('g'))}  (eval {fc.get('eval_idx', '?')})")
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
    ap.add_argument("--failures", action="store_true", help="add a typical no-crossing row (median floor) per task")
    ap.add_argument("--sut", action="append", default=[], help="restrict to these SUTs")
    args = ap.parse_args(argv)
    df = load_results(args.csv)
    if args.runs:
        picks = pd.read_csv(args.runs)
        picks["example"] = picks.groupby(["task", "sut"]).cumcount()
    else:
        picks = select_runs(df, args.arm, args.per_task, args.sut or None)
    if args.failures and not args.runs:
        picks = pd.concat([picks.assign(kind="cross"), select_failures(df, args.arm, args.sut or None)], ignore_index=True)
    if picks.empty:
        raise SystemExit("no crossing runs to show")
    print(f"wrote {plot_gallery(picks, args.out, runs_root=args.runs_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
