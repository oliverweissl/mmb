"""Calibration of the per-modality weights (w_img, w_txt) of the continuous genotype (2026-11-28).

One knob per modality: the manipulator multiplies the image block by w_img and the text magnitude by w_txt (the
optimizer never sees them; (1, 1) = uncalibrated). A calibration STEP is one (w_img, w_txt) setting, run with the
text-only (a6) and the image-only (a7) arm on the held-out calibration cells (cells/<task>_calib.yaml), same SUTs,
seeds and budget as the study. Target: text-only and image-only cross about equally often (exact McNemar on matched
(task, SUT, cell, seed) pairs, p >= 0.05 = "balanced"), both mostly crossing; raise w_txt before lowering w_img.

Usage::

    # configs of one step from base configs of the a6 / a7 arms (one YAML per task x SUT x arm)
    python -m mmb.analysis.calibration configs --base 'configs/generated/MAIN/*a6*' 'configs/generated/MAIN/*a7*' \
        --step s1 --w-img 1.0 --w-txt 2.0 [--cells calib] [--out configs/generated/CALIB_s1]

    # trajectory over the steps: CSVs of mmb.analysis.collect or run roots (collected here), in step order
    python -m mmb.analysis.calibration analyze runs/CALIB/s0 runs/CALIB/s1 ... --out results/calib [--final CALIB_s3]

Outputs of analyze: <out>/trajectory.csv (one row per step and scope: pooled and per task; columns TRAJ_COLUMNS)
and <out>/calib-trajectory.pdf (plot_trajectory: the steps in the (w_img, w_txt) plane and the two crossing shares).
"""
from __future__ import annotations

import argparse
import copy
import glob
import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from mmb.analysis.stats import load_results, mcnemar_exact, wilson  # noqa: E402

logger = logging.getLogger(__name__)

TEXT_ARM, IMAGE_ARM = "a6_text_only", "a7_image_only"
PAIR_KEYS = ["task", "sut", "cell_id", "rng_seed"]
ALPHA = 0.05
TRAJ_COLUMNS = ["step", "label", "exp_id", "w_img", "w_txt", "scope", "pairs",
                "text_k", "text_n", "text_cr", "text_lo", "text_hi",
                "image_k", "image_n", "image_cr", "image_lo", "image_hi",
                "text_only", "image_only", "p_mcnemar", "balanced", "text_mag_med", "image_mag_med", "final"]


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------


def step_config(base: dict, *, step: str, w_img: float, w_txt: float, cells: list | None = None,
                runs_root: str = "runs/CALIB") -> dict:
    """A copy of a base run config with the step's weights, the calibration cells and its own name / save_dir / exp_id."""
    c = copy.deepcopy(base)
    c.setdefault("image", {}).setdefault("latent", {})["w_img"] = float(w_img)
    comp = c.setdefault("text", {}).setdefault("composite", {})
    comp.setdefault("overrides", {}).setdefault("question_pool", {})["w_txt"] = float(w_txt)
    if cells is not None:
        c.setdefault("seeds", {}).setdefault("slot_items", {})["items"] = copy.deepcopy(cells)
    tags = c.setdefault("tags", {})
    task, sut, arm = tags.get("task", "task"), tags.get("sut", "sut"), tags.get("arm", "arm")
    seed = c.get("seed", 0)
    tags["exp_id"] = f"CALIB_{step}"
    c["name"] = f"CALIB_{step}_{task}_{sut}_{arm}_s{seed}"
    c["save_dir"] = f"{runs_root}/{step}/{task}/{sut}/{arm}/s{seed}"
    return c


def make_configs(base_paths: Sequence[Path], *, step: str, w_img: float, w_txt: float, cells: str | None = "calib",
                 out: Path | None = None, runs_root: str = "runs/CALIB", cells_dir: Path = Path("cells")) -> list[Path]:
    out = Path(out or f"configs/generated/CALIB_{step}")
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for p in base_paths:
        base = yaml.safe_load(Path(p).read_text())
        items = None
        if cells:
            task = (base.get("tags") or {}).get("task")
            items = yaml.safe_load((Path(cells_dir) / f"{task}_{cells}.yaml").read_text())["seeds"]["slot_items"]["items"]
        c = step_config(base, step=step, w_img=w_img, w_txt=w_txt, cells=items, runs_root=runs_root)
        q = out / f"{step}__{Path(p).name}"
        q.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
        written.append(q)
    return written


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _mode(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mode().iloc[0]) if len(s) else float("nan")


def step_rows(df: pd.DataFrame, step: int, label: str, final: bool = False) -> list[dict]:
    """Rows of one step (pooled and per task) from collect rows of the a6 and a7 arms."""
    rows = []
    scopes = [("pooled", df)] + [(t, g) for t, g in df.groupby("task", sort=True)]
    for scope, d in scopes:
        t = d[d["arm"] == TEXT_ARM].set_index(PAIR_KEYS)
        i = d[d["arm"] == IMAGE_ARM].set_index(PAIR_KEYS)
        both = t.index.intersection(i.index)
        tc, ic = t.loc[both, "crossed"].astype(bool), i.loc[both, "crossed"].astype(bool)
        mc = mcnemar_exact(tc.to_numpy(), ic.to_numpy()) if len(both) else dict(b_ref_only=0, c_arm_only=0, p=float("nan"))
        tk, ik, n = int(tc.sum()), int(ic.sum()), int(len(both))
        tl, th = wilson(tk, n)
        il, ih = wilson(ik, n)
        mag = lambda x: float(x["magnitude"].dropna().median()) if "magnitude" in x and x["magnitude"].notna().any() else float("nan")
        rows.append(dict(step=step, label=label, exp_id=";".join(sorted(set(d["exp_id"].dropna().astype(str)))),
                         w_img=_mode(d["w_img"]) if "w_img" in d else float("nan"),
                         w_txt=_mode(d["w_txt"]) if "w_txt" in d else float("nan"), scope=scope, pairs=n,
                         text_k=tk, text_n=n, text_cr=100 * tk / n if n else float("nan"), text_lo=100 * tl, text_hi=100 * th,
                         image_k=ik, image_n=n, image_cr=100 * ik / n if n else float("nan"), image_lo=100 * il,
                         image_hi=100 * ih, text_only=mc["b_ref_only"], image_only=mc["c_arm_only"], p_mcnemar=mc["p"],
                         balanced=bool(mc["p"] >= ALPHA) if mc["p"] == mc["p"] else False,
                         text_mag_med=mag(t.loc[both]), image_mag_med=mag(i.loc[both]), final=final))
    return rows


def trajectory(frames: Sequence[tuple[str, pd.DataFrame]], final: str | None = None) -> pd.DataFrame:
    """One block of rows per step, in the given order; *final* = the exp_id (or label) of the adopted step."""
    rows = []
    for k, (label, d) in enumerate(frames, 1):
        is_final = final is not None and (label == final or final in set(d["exp_id"].dropna().astype(str)))
        rows += step_rows(d, k, label, is_final)
    return pd.DataFrame(rows, columns=TRAJ_COLUMNS)


def _load(path: Path) -> tuple[str, pd.DataFrame]:
    path = Path(path)
    if path.suffix == ".csv":
        df = load_results([path])
    else:
        from mmb.analysis.collect import collect
        df = collect(path, require_stats=True)
        for c in ("crossed",):
            df[c] = df[c].astype(bool)
    ids = sorted(set(df["exp_id"].dropna().astype(str))) if len(df) else []
    return (ids[0] if len(ids) == 1 else path.stem), df


def _fmt_w(v: float) -> str:
    return f"{v:g}" if v == v else "?"


def plot_trajectory(traj: pd.DataFrame, out: Path | str) -> Path:
    """Greyscale two-panel figure of the pooled rows: (a) the steps in the (w_img, w_txt) plane, arrows in step order,
    filled = balanced (McNemar p >= 0.05), star = final; (b) text-only and image-only crossing share per step with
    Wilson 95% intervals and the McNemar p above each step."""
    t = traj[traj["scope"] == "pooled"].sort_values("step")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6})
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1, 1.4]})
    x, y = t["w_img"].to_numpy(float), t["w_txt"].to_numpy(float)
    for k in range(1, len(t)):
        if np.isfinite([x[k - 1], y[k - 1], x[k], y[k]]).all() and (x[k - 1], y[k - 1]) != (x[k], y[k]):
            a.annotate("", xy=(x[k], y[k]), xytext=(x[k - 1], y[k - 1]),
                       arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8, shrinkA=4, shrinkB=4))
    for r, xi, yi in zip(t.itertuples(), x, y):
        a.scatter([xi], [yi], s=46 if r.final else 22, marker="*" if r.final else "o",
                  facecolors="0.1" if r.balanced else "white", edgecolors="0.1", linewidths=0.8, zorder=3)
        a.annotate(str(r.step), (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=6.5)
    if np.all(x[np.isfinite(x)] > 0) and np.all(y[np.isfinite(y)] > 0) and np.isfinite(x).any():
        a.set_xscale("log", base=2)
        a.set_yscale("log", base=2)
        from matplotlib.ticker import FuncFormatter, LogLocator
        for ax_ in (a.xaxis, a.yaxis):
            ax_.set_major_locator(LogLocator(base=2, subs=(1.0, 1.5)))
            ax_.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax_.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        a.margins(0.25)
    a.set_xlabel("image weight $w_{img}$")
    a.set_ylabel("text weight $w_{txt}$")
    a.set_title("(a) weights per step (filled: balanced)", fontsize=7, loc="left")
    a.grid(True, color="0.9", linewidth=0.4)
    s = t["step"].to_numpy()
    for col, lab, ls, mk, off in (("text", "text-only", "-", "o", -0.06), ("image", "image-only", "--", "s", 0.06)):
        v = t[f"{col}_cr"].to_numpy(float)
        err = np.clip(np.vstack([v - t[f"{col}_lo"].to_numpy(float), t[f"{col}_hi"].to_numpy(float) - v]), 0, None)
        b.errorbar(s + off, v, yerr=err, color="0.1" if col == "text" else "0.5", linestyle=ls, marker=mk,
                   markersize=3, linewidth=0.9, elinewidth=0.6, capsize=1.5, label=lab)
    for r in t.itertuples():
        if r.p_mcnemar == r.p_mcnemar:
            b.annotate(f"p={r.p_mcnemar:.2g}", (r.step, 103), fontsize=5.5, ha="center", va="bottom")
    b.set_xticks(s)
    b.set_xticklabels([f"{r.step}\n{_fmt_w(r.w_img)}/{_fmt_w(r.w_txt)}" for r in t.itertuples()], fontsize=6)
    b.set_xlabel("step ($w_{img}$/$w_{txt}$)")
    b.set_ylabel("runs crossed (%)")
    b.set_ylim(0, 112)
    b.set_title("(b) crossing share, Wilson 95%; exact McNemar p", fontsize=7, loc="left")
    b.grid(True, axis="y", color="0.9", linewidth=0.4)
    b.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.4)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs", help="write the configs of one (w_img, w_txt) step")
    c.add_argument("--base", nargs="+", required=True, help="base YAML configs (globs allowed) of the a6 / a7 runs")
    c.add_argument("--step", required=True)
    c.add_argument("--w-img", type=float, required=True)
    c.add_argument("--w-txt", type=float, required=True)
    c.add_argument("--cells", default="calib", help="cells/<task>_<cells>.yaml; 'none' keeps the base items")
    c.add_argument("--out", type=Path, default=None)
    c.add_argument("--runs-root", default="runs/CALIB")
    a = sub.add_parser("analyze", help="trajectory table and figure over the steps")
    a.add_argument("inputs", nargs="+", type=Path, help="per step: a collect CSV or a runs root, in step order")
    a.add_argument("--out", type=Path, default=Path("results/calib"))
    a.add_argument("--final", default=None, help="exp_id (or label) of the adopted step")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.cmd == "configs":
        paths = sorted({Path(q) for p in args.base for q in (glob.glob(p) or [p])})
        w = make_configs(paths, step=args.step, w_img=args.w_img, w_txt=args.w_txt,
                         cells=None if args.cells == "none" else args.cells, out=args.out, runs_root=args.runs_root)
        logger.info("wrote %d configs to %s", len(w), w[0].parent if w else args.out)
        return 0
    traj = trajectory([_load(p) for p in args.inputs], final=args.final)
    args.out.mkdir(parents=True, exist_ok=True)
    traj.to_csv(args.out / "trajectory.csv", index=False)
    plot_trajectory(traj, args.out / "calib-trajectory.pdf")
    logger.info("trajectory of %d steps -> %s", traj["step"].nunique(), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
