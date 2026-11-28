"""Crossing-share-vs-evaluation-budget figure.

Usage::

    python -m mmb.analysis.plots results/e1.csv [more.csv ...] --out results/figures/crossing_curve.pdf \\
        [--ref a0_directed] [--max-budget 1500] [--vline 206]

One line per arm: share of runs with a crossing within b evaluations (log x), 95% percentile-bootstrap
band (10,000 resamples, seed 42). Single-column width, greyscale shades with
distinct line styles and markers.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mmb.analysis.stats import BOOT_SEED, N_BOOT, arm_label, curve_with_bands, load_results  # noqa: E402

#: Okabe-Ito palette (colour-blind safe), ordered so neighbouring arms differ strongly.
PALETTE = ["0.0", "0.55", "0.3", "0.7", "0.15", "0.45", "0.6", "0.8"]   # greyscale: arms differ by shade, line style and marker
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), "-", "--"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
COLUMN_WIDTH_IN = 3.3   # acmart sigconf column width


def budget_grid(max_budget: int, n: int = 60) -> np.ndarray:
    return np.unique(np.round(np.geomspace(1, max_budget, n)).astype(int))


def plot_crossing_curve(
    df: pd.DataFrame,
    out: Path | str,
    *,
    ref_arm: str | None = None,
    max_budget: int | None = None,
    vlines: Sequence[int] = (),
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    cluster_col: str | None = None,
    height_in: float = 2.3,
    label_suffix: dict[str, str] | None = None,
) -> Path:
    """Write the figure to *out* (PDF by extension) and return its path. label_suffix: arm -> text appended to its legend label
    (e.g. ' (seed 0)' for a method that so far has fewer rng seeds than the reference)."""
    label_suffix = label_suffix or {}
    max_budget = int(max_budget or df["budget"].max())
    grid = budget_grid(max_budget)
    curve = curve_with_bands(df, grid, n_boot=n_boot, seed=seed, cluster_col=cluster_col)

    arms = sorted(set(df["arm"]))
    if ref_arm in arms:
        arms = [ref_arm] + [a for a in arms if a != ref_arm]

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, height_in))
    for i, arm in enumerate(arms):
        c = curve[curve["arm"] == arm].sort_values("budget")
        col = PALETTE[i % len(PALETTE)]
        n = int(c["n"].iloc[0])
        ax.fill_between(c["budget"], c["boot_lo"], c["boot_hi"], color=col, alpha=0.18, linewidth=0)
        ax.plot(c["budget"], c["share"], color=col, linestyle=LINESTYLES[i % len(LINESTYLES)],
                linewidth=1.3 if arm == ref_arm else 1.0, label=("MMBA" if arm == ref_arm else arm_label(arm)) + label_suffix.get(arm, ""),
                marker=MARKERS[i % len(MARKERS)], markevery=(i % 5, 12), markersize=3)
    for v in vlines:
        ax.axvline(v, color="0.6", linewidth=0.5, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlim(1, max_budget)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("model calls")
    ax.set_ylabel("share of runs crossed")
    ax.grid(True, which="major", color="0.9", linewidth=0.4)
    ax.legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
              handlelength=2.2, columnspacing=1.0)   # below the axes: never over a curve
    fig.tight_layout(pad=0.3)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def _wilson(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


LABEL_OFFSETS = {"Zero": ((5, 3), "left"), "One": ((5, 3), "left"), "TwoA": ((5, 3), "left"),
                 "TwoB": ((-4, -10), "right"), "Three": ((4, 5), "left"), "Final": ((5, -11), "left")}
CALIB_MODS = (("text", "text-only"), ("image", "image-only"), ("both", "MMBA"), ("mimicry", "MIMICRY-style"),
              ("deepjanus", "DeepJanus-style"))


def plot_calib_trajectory(traj: pd.DataFrame, out: Path | str, final: pd.DataFrame | None = None, *,
                          width_in: float = 5.4, height_in: float = 2.6) -> Path:
    """Operator calibration, main-paper figure. Left: text-only vs image-only crossing share (%) per calibration step
    (scripts/analysis/calib_trajectory.py), Wilson 95% CIs, the diagonal = balance; steps on held-out set 1 joined by
    arrows from the uncalibrated MAIN operators to C*; the C* final check (sets 1+2) as a hollow diamond. Labels carry
    the median LPIPS of image-only crossings. Right (if *final*, the pooled rows of calib_final.py summary.csv):
    crossing share of every method at C* on the final-check inputs, with a reference line at full MMBA."""
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6})
    two = final is not None and len(final)
    fig, axs = plt.subplots(1, 2 if two else 1, figsize=(width_in if two else width_in * 0.55, height_in),
                            gridspec_kw={"width_ratios": [1.1, 1]} if two else None)
    ax = axs[0] if two else axs
    ax.plot([0, 100], [0, 100], color="0.55", linestyle="--", linewidth=0.7, zorder=0)
    ax.text(80, 77, "balance: text = image", rotation=45, rotation_mode="anchor", ha="center", va="top", fontsize=5.5, color="0.4")
    fin_mask = traj["final"].astype(bool).to_numpy()
    chain = traj[~fin_mask].reset_index(drop=True)
    fin = traj[fin_mask]
    for _, r in traj.iterrows():
        ax.errorbar(r["text_cr"], r["image_cr"], xerr=[[r["text_cr"] - r["text_lo"]], [r["text_hi"] - r["text_cr"]]],
                    yerr=[[r["image_cr"] - r["image_lo"]], [r["image_hi"] - r["image_cr"]]], fmt="none",
                    ecolor="0.72", elinewidth=0.5, capsize=1.2, zorder=1)
    for a_, b_ in zip(chain.index[:-1], chain.index[1:]):
        ax.annotate("", xy=(chain.at[b_, "text_cr"], chain.at[b_, "image_cr"]), xytext=(chain.at[a_, "text_cr"], chain.at[a_, "image_cr"]),
                    arrowprops=dict(arrowstyle="-|>", color="0.15", linewidth=0.9, shrinkA=4, shrinkB=4, mutation_scale=7), zorder=2)
    ax.scatter(chain["text_cr"], chain["image_cr"], s=18, color="0.0", zorder=3)
    if len(chain) and chain.at[0, "step"] == "Zero":
        ax.scatter(chain.at[0, "text_cr"], chain.at[0, "image_cr"], s=48, facecolors="none", edgecolors="0.0", linewidths=0.8, zorder=3)
    if len(fin):
        ax.scatter(fin["text_cr"], fin["image_cr"], s=30, facecolors="white", edgecolors="0.0", linewidths=1.0, marker="D", zorder=4)
    for _, r in traj.iterrows():
        lp = "" if pd.isna(r.get("image_lpips_med")) else f" ({r['image_lpips_med']:.3f})"
        off, ha = LABEL_OFFSETS.get(str(r["step"]), ((5, 3), "left"))
        ax.annotate(f"{r['label']}{lp}", (r["text_cr"], r["image_cr"]), xytext=off, textcoords="offset points", fontsize=5.8, ha=ha)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("text-only crossing share (%)"); ax.set_ylabel("image-only crossing share (%)")
    ax.set_title("calibration steps (median image LPIPS)", fontsize=6.5)
    ax.grid(True, color="0.9", linewidth=0.4)
    if two:
        bx = axs[1]
        f = final.set_index("mod")
        rows = [(lab, f.loc[m] if m in f.index else None) for m, lab in CALIB_MODS]
        y = np.arange(len(rows))[::-1]
        for yi, (lab, r) in zip(y, rows):
            if r is None:
                bx.text(3, yi, "(running)", va="center", fontsize=6, color="0.4")
                continue
            n = float(r["n"]); cr = float(r["CR"]); lo, hi = _wilson(round(cr * n / 100), n)
            mm = lab == "MMBA"
            bx.barh(yi, cr, color="0.15" if mm else ("0.6" if lab in ("MIMICRY-style", "DeepJanus-style") else "0.85"),
                    edgecolor="0.0", linewidth=0.5, height=0.62)
            bx.errorbar(cr, yi, xerr=[[cr - 100 * lo], [100 * hi - cr]], fmt="none", ecolor="0.0", elinewidth=0.5, capsize=1.5)
            bx.text(100 * hi + 2, yi, f"{cr:.0f}%", va="center", fontsize=6)
        if "both" in f.index:
            bx.axvline(float(f.loc["both", "CR"]), color="0.15", linestyle=":", linewidth=0.8)
        bx.set_yticks(y); bx.set_yticklabels([lab for lab, _ in rows]); bx.set_ylim(-0.6, len(rows) - 0.4)
        bx.set_axisbelow(True)
        bx.set_xlim(0, 100); bx.set_xlabel("crossing share at C* (%)")
        bx.set_title(f"C* final check, {int(float(f['n'].iloc[0]))} matched runs per method", fontsize=6.5)
        bx.grid(True, axis="x", color="0.9", linewidth=0.4)
    fig.tight_layout(pad=0.3)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_predictor_scatter(df: pd.DataFrame, out: Path | str, *, arm: str | None = "a0_directed",
                           height_in: float = 2.3) -> Path:
    """Generation-0 probe (median |g| of the initial population) against the run floor, log-log.

    Colour = SUT, marker = task; one point per run of *arm* (all arms if None). Zero floors are drawn at
    the smallest positive floor / 2 so they stay visible.
    """
    d = df if arm is None else df[df["arm"] == arm]
    d = d[d["probe_gen0"].notna() & d["floor_nats"].notna() & (d["probe_gen0"] > 0)]
    fl = d["floor_nats"].to_numpy(float)
    pos = fl[fl > 0]
    fl = np.where(fl > 0, fl, (pos.min() / 2) if len(pos) else 1e-6)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, height_in))
    suts = sorted(set(d["sut"]))
    tasks = sorted(set(d["task"]))
    for i, s in enumerate(suts):
        for k, t in enumerate(tasks):
            m = ((d["sut"] == s) & (d["task"] == t)).to_numpy()
            if m.any():
                ax.scatter(d.loc[m, "probe_gen0"], fl[m], s=9, color=PALETTE[i % len(PALETTE)],
                           marker=MARKERS[k % len(MARKERS)], linewidths=0.3, edgecolors="white", alpha=0.85,
                           label=f"{s}, {t}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("generation-0 probe: median $|g|$ (nats)")
    ax.set_ylabel("run floor: min $|g|$ (nats)")
    ax.grid(True, which="major", color="0.9", linewidth=0.4)
    ax.legend(fontsize=5, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout(pad=0.3)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/figures/crossing_curve.pdf"))
    ap.add_argument("--ref", default=None)
    ap.add_argument("--max-budget", type=int, default=None)
    ap.add_argument("--vline", type=int, action="append", default=[])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--cluster", default=None)
    args = ap.parse_args(argv)
    df = load_results(args.csv)
    p = plot_crossing_curve(df, args.out, ref_arm=args.ref, max_budget=args.max_budget,
                            vlines=args.vline, n_boot=args.n_boot, cluster_col=args.cluster)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
