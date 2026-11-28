"""Figure: MMBA on the synthetic SUT (real-coded genotype), CPU only.

Panel A: Phase-1 signed margin per generation (min and median of the generation).
Panel B: Phase-2 lines: the bracket [lo, hi] of the scale parameter per
          generation (thin: interval sampled this generation; thick: new bracket).
Panel C: the same brackets as width hi - lo (log), against eps.
Panel D: incumbent weighted block magnitudes (w_img*RMS(c), w_txt*t) per
          generation; markers: guided restarts that found a smaller crossing.

Usage: ../.venv/bin/python scripts/analysis/search_real_fig.py [--out docs/figs/search_real]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from mmb.optimizer.genotype import GenotypeSpace
from mmb.optimizer.mmba import MMBAOptimizer
from mmb.optimizer.synthetic_sut import SyntheticSUT, run_search

BLUE, ORANGE, INK, MUTED = "#0072B2", "#E69F00", "#222222", "#8a8a8a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figs/search_real")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-img", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1e-3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sp = GenotypeSpace.contract(a.d_img)
    sut = SyntheticSUT(sp, base=2.4, k_img=3.0, k_txt=1.2, n_bands=4)
    opt = MMBAOptimizer(sp, 1, pop_size=50, seed=a.seed, eps=a.eps)
    n_evals = run_search(opt, sut, 30)
    tr = opt.trace
    pair = opt.boundary_pair
    if pair is not None:   # the tester does this from rendered texts; here the SUT's text bands
        opt.annotate_pair(bool(sut.text_score(pair["off"])[0] != sut.text_score(pair["on"])[0]))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK})
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.4), constrained_layout=True)
    axs = axs.ravel()

    # A: phase-1 margin
    ax = axs[0]
    p1 = [r for r in tr if r["phase"] == 1]
    g = [r["gen"] for r in p1]
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    ax.plot(g, [r["min_margin"] for r in p1], color=INK, lw=2, marker="o", ms=4, label="min of generation")
    ax.plot(g, [r["median_margin"] for r in p1], color=MUTED, lw=2, ls="--", label="median")
    ax.text(g[-1], 0.05, "boundary m = 0", color=MUTED, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("generation"); ax.set_ylabel("signed margin m")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("A  Phase 1: margin until first crossing", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    p2 = [r for r in tr if r.get("stage") in ("global", "image", "text")]
    sty = {"global": (INK, "o", "global lam"), "image": (ORANGE, "s", "image block mu"),
           "text": (BLUE, "^", "text block mu")}

    # B: phase-2 bracket values
    ax = axs[1]
    seen = set()
    for r in p2:
        col, mk, lab = sty[r["stage"]]
        x = r["gen"]
        ax.plot([x - 0.18, x - 0.18], [r["lo_before"], r["hi_before"]], color=col, lw=1, alpha=0.6)
        ax.plot([x + 0.18, x + 0.18], [r["lo"], r["hi"]], color=col, lw=4, solid_capstyle="round",
                label=None if lab in seen else lab)
        ax.plot(x + 0.18, r["hi"], marker=mk, color=col, ms=6, ls="none")
        seen.add(lab)
    ax.set_ylim(0.88, 1.005)  # first generation of each line samples (0, 1): its thin bar is clipped
    ax.set_xlabel("generation"); ax.set_ylabel("scale parameter (lam or mu)")
    ax.set_title("B  Phase 2: bracket [lo, hi] (thin: sampled, thick: kept)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    # C: bracket width
    ax = axs[2]
    seen = set()
    for r in p2:
        col, mk, lab = sty[r["stage"]]
        w = max(r["hi"] - r["lo"], 1e-6) if not r.get("removed") else None
        if w is None:
            ax.annotate("block dropped", (r["gen"], 0.5), color=col, fontsize=7, ha="center")
            continue
        ax.plot(r["gen"], w, marker=mk, color=col, ms=7, ls="none",
                label=None if lab in seen else lab)
        seen.add(lab)
    ax.axhline(a.eps, color=MUTED, lw=1, ls=":")
    ax.text(p2[0]["gen"] if p2 else 0, a.eps * 1.3, f"eps = {a.eps:g}", color=MUTED, fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(a.eps / 10, 1.5)
    ax.set_xlabel("generation"); ax.set_ylabel("bracket width hi - lo")
    ax.set_title("C  Phase 2: batched bisection, bracket width", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # D: incumbent magnitudes
    ax = axs[3]
    q = [r for r in tr if r["phase"] == 2]
    gq = [r["gen"] for r in q]
    ax.plot(gq, [r["mag_image"] for r in q], color=ORANGE, lw=2, label="image RMS(c)")
    ax.plot(gq, [r["mag_text"] for r in q], color=BLUE, lw=2, ls="--", label="text t")
    ax.plot(gq, [r["mag"] for r in q], color=INK, lw=1.5, ls=":", label="genotype (L2)")
    rc = [r for r in q if r.get("restart_cross")]
    if rc:
        ax.plot([r["gen"] for r in rc], [r["mag"] for r in rc], ls="none", marker="v", ms=7,
                color=INK, label="restart: smaller crossing")
    ax.set_xlabel("generation"); ax.set_ylabel("incumbent block magnitude")
    ax.set_ylim(0, None)
    ax.set_title("D  Incumbent crossing: weighted magnitudes (w = 1)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"MMBA on a synthetic SUT (d_img={a.d_img}, pop 50 x 30 generations = {n_evals} calls, seed {a.seed})",
                 fontsize=9, x=0.01, ha="left")
    fig.savefig(out / "mmba_synthetic_trace.png", dpi=200)
    fig.savefig(out / "mmba_synthetic_trace.pdf")
    rec = {"trace": tr, "n_evals": n_evals,
           "pair": None if pair is None else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pair.items()},
           "sut": {"base": sut.base, "k_img": sut.k_img, "k_txt": sut.k_txt, "n_bands": sut.n_bands}}
    (out / "mmba_synthetic_trace.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({"n_evals": n_evals, "pair_gap": None if pair is None else pair["gap"],
                      "pair_line": None if pair is None else pair["line"],
                      "pair_kind": None if pair is None else pair["kind"],
                      "restart_crossings": sum(1 for r in tr if r.get("restart_cross")),
                      "final_mag": tr[-1].get("mag")}))


if __name__ == "__main__":
    main()
