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


CALIB_MODS = (("text", "text-only"), ("image", "image-only"), ("both", "MMBA"), ("mimicry", "MIMICRY-style"),
              ("deepjanus", "DeepJanus-style"))
# label placement (offset in points, ha, va) per step: panel (a) full view, panel (b) zoom on the lower cluster
CALIB_LABEL = {"Zero": ((3.0, 84.0), "left", "top"), "One": ((57.0, 81.0), "left", "bottom")}     # data coordinates
CALIB_ZOOM_LABEL = {"TwoA": ((6, 0), "left", "center"), "TwoB": ((0, -8), "center", "top"),
                    "Three": ((7, 5), "left", "bottom"), "Final": ((-3, 7), "left", "bottom")}
CALIB_ZOOM = (0.0, 45.0, 20.0, 55.0)        # x0, x1, y0, y1 (%) of panel (b)


def _calib_marker(ax, r, s=22):
    st = str(r.get("status", "adopted"))
    x, y = r["text_cr"], r["image_cr"]
    if st == "start":
        ax.scatter(x, y, s=s * 1.6, marker="s", color="0.5", edgecolors="0.0", linewidths=0.7, zorder=4)
    elif st == "rejected":
        ax.scatter(x, y, s=s, marker="o", facecolors="white", edgecolors="0.0", linewidths=0.8, zorder=4)
    elif bool(r["final"]):
        ax.scatter(x, y, s=s * 1.3, marker="D", facecolors="0.0", edgecolors="0.0", linewidths=0.8, zorder=5)
    else:
        ax.scatter(x, y, s=s * 1.2, marker="o", color="0.0", zorder=5)


def _calib_text(r, stacked: bool = False) -> str:
    t = str(r["label"])
    if stacked:
        t = t.replace(" (", "\n(")
    reason = r.get("reason")
    if isinstance(reason, str) and reason and str(r.get("status")) == "rejected":
        t += ("\n" if stacked else ": ") + reason
    if not pd.isna(r.get("image_lpips_med")):
        t += f"\nLPIPS {float(r['image_lpips_med']):.3f}"
    return t


def _calib_final_link(a, traj, box=None):
    """thin dashed link (no arrowhead) from the adopted step to its final check"""
    fin = traj[traj["final"].astype(bool)]
    ad = traj[(~traj["final"].astype(bool)) & (traj.get("status", pd.Series("", index=traj.index)).astype(str) == "adopted")]
    if len(fin) and len(ad):
        p, q = ad.iloc[-1], fin.iloc[0]
        a.plot([p["text_cr"], q["text_cr"]], [p["image_cr"], q["image_cr"]], color="0.35", linestyle=(0, (2, 1.5)),
               linewidth=0.6, zorder=2)


def _calib_arrows(a, sub, box=None):
    for i, j in zip(sub.index[:-1], sub.index[1:]):
        p, q = (sub.at[i, "text_cr"], sub.at[i, "image_cr"]), (sub.at[j, "text_cr"], sub.at[j, "image_cr"])
        if box is not None and not all(box[0] <= x <= box[1] and box[2] <= y <= box[3] for x, y in (p, q)):
            continue
        a.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="0.2", linewidth=0.8, shrinkA=4, shrinkB=4,
                                                       mutation_scale=7), zorder=2)


def plot_calib_trajectory(traj: pd.DataFrame, out: Path | str, final: pd.DataFrame | None = None, *,
                          missing_note: str = "(on hold)", width_in: float = 5.4, height_in: float = 3.75) -> Path:
    """Operator calibration, main-paper figure (scripts/analysis/calib_trajectory.py output), three panels.
    (a) text-only vs image-only crossing share (%) per calibration step on the same held-out inputs; dashed diagonal =
        balance. Grey square = the uncalibrated MAIN operators (start); hollow circles = settings the calibration moved
        away from, labelled with the reason; filled circle = C*; filled diamond = C* final check (sets 1+2). Labels carry
        the median LPIPS of the image-only crossings. The box marks panel (b).
    (b) zoom on the lower cluster; Wilson 95% CIs on C* and the final check.
    (c) if *final* (pooled rows of calib_final.py summary.csv): crossing share of every method at C* on the final-check
        inputs with Wilson 95% CIs; dashed line = full MMBA; methods without runs show *missing_note*."""
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6})
    three = final is not None
    fig = plt.figure(figsize=(width_in, height_in if three else height_in * 0.66))
    gs = fig.add_gridspec(2 if three else 1, 2, height_ratios=[2.0, 1.0] if three else None)
    ax, zx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    traj = traj.reset_index(drop=True)
    chain = traj[~traj["final"].astype(bool)].reset_index(drop=True)
    x0, x1, y0, y1 = CALIB_ZOOM
    in_box = lambda r: x0 <= r["text_cr"] <= x1 and y0 <= r["image_cr"] <= y1
    # (a) full view
    ax.plot([0, 100], [0, 100], color="0.55", linestyle="--", linewidth=0.7, zorder=0)
    ax.text(70, 71.5, "balance", rotation=45, rotation_mode="anchor", ha="center", va="bottom", fontsize=5.5, color="0.4")
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="0.45", linewidth=0.5, linestyle=":", zorder=1))
    ax.text(x1 - 1, y0 + 1, "(b)", ha="right", va="bottom", fontsize=5.5, color="0.35")
    _calib_arrows(ax, chain)
    _calib_final_link(ax, traj)
    for _, r in traj.iterrows():
        _calib_marker(ax, r)
        if not in_box(r):
            pos = CALIB_LABEL.get(str(r["step"]))
            if pos is None:
                ax.annotate(_calib_text(r, True), (r["text_cr"], r["image_cr"]), xytext=(6, 2), textcoords="offset points",
                            fontsize=5.8, ha="left", va="bottom")
            else:
                ax.annotate(_calib_text(r, True), (r["text_cr"], r["image_cr"]), xytext=pos[0], textcoords="data",
                            fontsize=5.8, ha=pos[1], va=pos[2], arrowprops=dict(arrowstyle="-", color="0.5", linewidth=0.4,
                                                                                shrinkA=1, shrinkB=4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("text-only crossing share (%)"); ax.set_ylabel("image-only crossing share (%)")
    ax.set_title("(a) calibration steps", fontsize=6.5)
    ax.grid(True, color="0.92", linewidth=0.4); ax.set_axisbelow(True)
    # (b) zoom
    zx.plot([0, 100], [0, 100], color="0.55", linestyle="--", linewidth=0.6, zorder=0)
    _calib_arrows(zx, chain, CALIB_ZOOM)
    _calib_final_link(zx, traj)
    for _, r in traj.iterrows():
        if not in_box(r):
            continue
        if str(r.get("status")) == "adopted":
            zx.errorbar(r["text_cr"], r["image_cr"], xerr=[[r["text_cr"] - r["text_lo"]], [r["text_hi"] - r["text_cr"]]],
                        yerr=[[r["image_cr"] - r["image_lo"]], [r["image_hi"] - r["image_cr"]]], fmt="none",
                        ecolor="0.7", elinewidth=0.45, capsize=1.0, zorder=1)
        _calib_marker(zx, r)
        off, ha, va = CALIB_ZOOM_LABEL.get(str(r["step"]), ((6, 0), "left", "center"))
        zx.annotate(_calib_text(r), (r["text_cr"], r["image_cr"]), xytext=off, textcoords="offset points", fontsize=5.3,
                    ha=ha, va=va, zorder=6, bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8))
    zx.set_xlim(x0, x1); zx.set_ylim(y0, y1); zx.set_aspect("equal", adjustable="box")
    zx.set_xlabel("text-only crossing share (%)")
    zx.set_title("(b) zoom", fontsize=6.5)
    zx.grid(True, color="0.92", linewidth=0.4); zx.set_axisbelow(True)
    if three:
        bx = fig.add_subplot(gs[1, :])
        f = final.set_index("mod") if len(final) else pd.DataFrame()
        rows = [(lab, f.loc[m] if m in f.index else None) for m, lab in CALIB_MODS]
        y = np.arange(len(rows))[::-1]
        style = {"text-only": ("0.88", ""), "image-only": ("0.88", ""), "MMBA": ("0.1", ""),
                 "MIMICRY-style": ("white", "////"), "DeepJanus-style": ("white", "....")}
        for yi, (lab, r) in zip(y, rows):
            if r is None:
                bx.text(3, yi, missing_note, va="center", fontsize=5.8, color="0.35")
                continue
            n = float(r["n"]); cr = float(r["CR"]); lo, hi = _wilson(round(cr * n / 100), n)
            col, hatch = style[lab]
            bx.barh(yi, cr, color=col, hatch=hatch, edgecolor="0.0", linewidth=0.6, height=0.6)
            bx.errorbar(cr, yi, xerr=[[cr - 100 * lo], [100 * hi - cr]], fmt="none", ecolor="0.55" if lab == "MMBA" else "0.0",
                        elinewidth=0.6, capsize=1.5, zorder=3)
            bx.text(100 * hi + 1, yi, f"{cr:.0f}%", va="center", fontsize=5.8, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
        if "both" in f.index:
            bx.axvline(float(f.loc["both", "CR"]), color="0.0", linestyle=(0, (2, 1.5)), linewidth=0.8, zorder=4)
        bx.set_yticks(y); bx.set_yticklabels([lab for lab, _ in rows], fontsize=6); bx.set_ylim(-0.6, len(rows) - 0.4)
        bx.set_xlim(0, 100); bx.set_xlabel("crossing share at C* (%)")
        nn = int(float(f["n"].iloc[0])) if len(f) else 0
        bx.set_title(f"(c) crossing share of each method with the calibrated operators C* ({nn} matched runs each)", fontsize=6.5)
        bx.grid(True, axis="x", color="0.9", linewidth=0.4); bx.set_axisbelow(True)
    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=0.8)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


OKABE_BLUE, OKABE_ORANGE = "#0072B2", "#E69F00"      # colour-blind safe: text = blue, image = orange


def _asinh_ticks(ax, vmin: float, vmax: float) -> None:
    cand = [-30, -10, -3, -1, 0, 1, 3, 10, 30]
    t = [v for v in cand if vmin - 1e-9 <= v <= vmax + 1e-9]
    ax.set_yticks(np.arcsinh(t)); ax.set_yticklabels([str(v) for v in t])


def plot_calib_steps(traj: pd.DataFrame, runs: pd.DataFrame, out: Path | str, final: pd.DataFrame | None = None, *,
                     missing_note: str = "on hold", width_in: float = 5.4, height_in: float = 3.3) -> Path:
    """Calibration, main-paper figure (scripts/analysis/calib_trajectory.py: trajectory.csv + trajectory_runs.csv).
    x = calibration step (R0 uncalibrated -> T-A text -> image ladder -> final check on 48 pairs).
    (a) crossing share per step, text-only (blue) and image-only (orange), Wilson 95% CIs.
    (b) per step, the distribution over runs of the closest approach to the decision border (minimum signed margin,
        nats, asinh scale; < 0 = crossed): text left, image right; a channel carried over unchanged from an earlier
        step is faded. The share below 0 is the crossing share of (a).
    (c) if *final*: crossing share of each method with the calibrated operators (baselines grey)."""
    import matplotlib.gridspec as gridspec
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(width_in, height_in))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3.5, 1.35] if final is not None else [1, 0.001],
                           height_ratios=[1.0, 2.1], wspace=0.35, hspace=0.32)
    ax_cr = fig.add_subplot(gs[0, 0]); ax = fig.add_subplot(gs[1, 0], sharex=ax_cr)
    traj = traj.reset_index(drop=True)
    xs = np.arange(len(traj))
    col = {"text": OKABE_BLUE, "image": OKABE_ORANGE}
    off = {"text": -0.17, "image": 0.17}
    carried = runs.groupby(["step", "modality"])["carried"].any().to_dict() if len(runs) else {}
    # (a) crossing share
    for mod in ("text", "image"):
        y = traj[f"{mod}_cr"].to_numpy(float)
        lo, hi = traj[f"{mod}_lo"].to_numpy(float), traj[f"{mod}_hi"].to_numpy(float)
        ax_cr.plot(xs + off[mod] * 0.5, y, color=col[mod], linewidth=1.0, marker="o", markersize=3, zorder=3,
                   label=f"{mod}-only")
        ax_cr.errorbar(xs + off[mod] * 0.5, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt="none", ecolor=col[mod], elinewidth=0.6,
                       capsize=1.2, alpha=0.7)
        for k, (x, st) in enumerate(zip(xs, traj["step"])):
            if carried.get((st, mod), False):
                ax_cr.scatter(x + off[mod] * 0.5, y[k], s=10, facecolors="white", edgecolors=col[mod], linewidths=0.8, zorder=4)
    ax_cr.set_ylim(-3, 103); ax_cr.set_ylabel("crossing\nshare (%)")
    ax_cr.legend(ncol=2, fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1)
    ax_cr.grid(True, axis="y", color="0.92", linewidth=0.4); ax_cr.set_axisbelow(True)
    ax_cr.tick_params(labelbottom=False)
    ax_cr.set_title("(a) crossing share per step", fontsize=6.5, loc="right")
    # (b) closest-approach distributions (asinh scale)
    allv = []
    for k, st in enumerate(traj["step"]):
        for mod in ("text", "image"):
            v = runs[(runs["step"] == st) & (runs["modality"] == mod)]["min_margin"].dropna().to_numpy(float)
            if not len(v):
                continue
            allv.append(v)
            tv = np.arcsinh(v)
            faded = carried.get((st, mod), False)
            parts = ax.violinplot([tv], positions=[k + off[mod]], widths=0.32, showextrema=False, showmedians=False)
            for b in parts["bodies"]:
                b.set_facecolor(col[mod]); b.set_edgecolor(col[mod]); b.set_alpha(0.18 if faded else 0.55); b.set_linewidth(0.4)
            rng = np.random.default_rng(k * 2 + (mod == "image"))
            ax.scatter(k + off[mod] + rng.uniform(-0.05, 0.05, len(tv)), tv, s=2.5, color=col[mod],
                       alpha=0.25 if faded else 0.8, linewidths=0, zorder=3)
            ax.plot([k + off[mod] - 0.08, k + off[mod] + 0.08], [np.median(tv)] * 2, color="0.0", linewidth=0.8, zorder=4)
    ax.axhline(0, color="0.0", linestyle="--", linewidth=0.7, zorder=2)
    if allv:
        v = np.concatenate(allv); _asinh_ticks(ax, float(v.min()), float(v.max()))
    ax.text(-0.55, -0.12, "crossed ↓", ha="left", va="top", fontsize=5.5, color="0.3")
    ax.set_ylabel("min. signed margin (nats)")
    ax.set_xticks(xs); ax.set_xticklabels([str(l).replace("\\n", "\n") for l in traj["label"]], fontsize=5.8)
    ax.set_xlim(-0.6, len(traj) - 0.4)
    fin = np.where(traj["final"].astype(bool).to_numpy())[0]
    for a_ in (ax, ax_cr):
        for f in fin:
            a_.axvline(f - 0.5, color="0.6", linestyle=":", linewidth=0.6)
    ax.grid(True, axis="y", color="0.92", linewidth=0.4); ax.set_axisbelow(True)
    ax.set_title("(b) closest approach per run (faded: channel unchanged, reused)", fontsize=6.5, loc="left")
    # (c) methods
    if final is not None:
        bx = fig.add_subplot(gs[:, 1])
        f = final.set_index("mod") if len(final) else pd.DataFrame()
        sty = {"text": (OKABE_BLUE, ""), "image": (OKABE_ORANGE, ""), "both": ("0.1", ""), "mimicry": ("0.75", "////"),
               "deepjanus": ("0.75", "....")}
        lab = {"text": "text-\nonly", "image": "image-\nonly", "both": "MMBA", "mimicry": "MIMICRY-\nstyle", "deepjanus": "DeepJanus-\nstyle"}
        mods = ["text", "image", "both", "mimicry", "deepjanus"]
        for i, m in enumerate(mods):
            if m not in f.index:
                bx.text(i, 3, missing_note, rotation=90, ha="center", va="bottom", fontsize=5.5, color="0.35")
                continue
            n, cr = float(f.at[m, "n"]), float(f.at[m, "CR"]); lo, hi = _wilson(round(cr * n / 100), n)
            c, h = sty[m]
            bx.bar(i, cr, color=c, hatch=h, edgecolor="0.0", linewidth=0.5, width=0.7)
            bx.errorbar(i, cr, yerr=[[cr - 100 * lo], [100 * hi - cr]], fmt="none", ecolor="0.0" if m != "both" else "0.6",
                        elinewidth=0.6, capsize=1.5)
            bx.text(i, 100 * hi + 2, f"{cr:.0f}", ha="center", va="bottom", fontsize=5.5)
        if "both" in f.index:
            bx.axhline(float(f.at["both", "CR"]), color="0.0", linestyle=(0, (2, 1.5)), linewidth=0.7)
        bx.set_xticks(range(len(mods))); bx.set_xticklabels([lab[m] for m in mods], fontsize=5.5, rotation=90)
        bx.set_ylim(0, 100); bx.set_ylabel("crossing share (%)")
        nn = int(float(f["n"].iloc[0])) if len(f) else 0
        bx.set_title(f"(c) final setting, n={nn}", fontsize=6.5, loc="left")
        bx.grid(True, axis="y", color="0.92", linewidth=0.4); bx.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.16)
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
