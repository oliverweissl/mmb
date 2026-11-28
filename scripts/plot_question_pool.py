#!/usr/bin/env python
"""Figures of the question pool (docs/figs/text_pool/), greyscale, for the text-block decision.

    ../.venv/bin/python scripts/plot_question_pool.py [--pool configs/text/question_pool.json] [--out docs/figs/text_pool]

pool_size.png      accepted pool entries per question, by source
distances.png      every pool distance per question (one dot each), d_max and the band edges; pooled histogram
resolution.png     largest and median step between neighbouring distances in [0, d_max] (= text resolution)
staircase.png      selected distance d(t) for u = 0 and w_txt = 1, all questions (the stepwise text output)
examples.png       the question rendered along t = 0 .. 1 (u = 0, w_txt = 1) for 4 questions
"""
import argparse, json, textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mmb.manipulator.text.operators.question_pool import select_index

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555",
                     "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444", "savefig.dpi": 200})
SRC = (("substitution", "#222222", "word substitutions (1-3 sites)"), ("bank", "#666666", "paraphrase bank"),
       ("t5", "#aaaaaa", "T5 paraphrases"), ("paraphrase+substitution", "#dddddd", "substitutions on paraphrases (growth)"))
EXAMPLES = ("What is the main subject in this image?", "Is there a dog in the image?", "Is there a broccoli in the image?",
            "Is there a car in the image?")


def short(q):
    return q.replace("Is there a ", "").replace(" in the image?", "").replace("What is the main subject in this image?", "ImageNet: main subject")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="configs/text/question_pool.json")
    ap.add_argument("--out", default="docs/figs/text_pool")
    a = ap.parse_args()
    P = json.loads(Path(a.pool).read_text()); meta, Q = P["meta"], P["questions"]
    d_max, nb = float(meta["d_max"]), int(meta["n_bands"]); h = d_max / nb
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    qs = sorted(Q, key=lambda q: Q[q]["n_pool"])
    lab = [short(q) for q in qs]
    dists = {q: np.array([0.0] + [x["dist"] for x in Q[q]["pool"]]) for q in qs}

    # 1 pool size by (first) source
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    left = np.zeros(len(qs))
    for s, c, name in SRC:
        v = np.array([sum(x["sources"][0] == s for x in Q[q]["pool"]) for q in qs])
        ax.barh(lab, v, left=np.maximum(left, 1), color=c, edgecolor="#888", linewidth=0.5, label=name, height=0.7)
        left += v
    for i, n in enumerate(left):
        ax.text(n * 1.08, i, str(int(n)), va="center", fontsize=8, color="#222")
    ax.set_xlabel("accepted whole questions in the pool (log scale)"); ax.set_xscale("log"); ax.set_xlim(1, None)
    ax.set_title("Pool size per question (whole questions, validated)", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "pool_size.png", bbox_inches="tight"); plt.close(fig)

    # 2 distances: strip per question + pooled histogram
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.8, 6.2), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    for k in range(nb + 1):
        ax.axvline(k * h, color="#e6e6e6", lw=0.6, zorder=0)
    for i, q in enumerate(qs):
        d = dists[q][1:]
        ax.plot(d, np.full(d.size, i), "|", color="#222", ms=9, mew=1)
    ax.axvline(d_max, color="#000", lw=1.2, ls="--")
    ax.text(d_max, len(qs) - 0.4, f" d_max = {d_max:.3f}\n ({meta['d_max_quantile']:.0%} quantile)", fontsize=8, va="top")
    ax.set_yticks(range(len(qs))); ax.set_yticklabels(lab)
    ax.set_title(f"Pool distances ({nb} bands of width {h:.4f}, grey)", loc="left", fontsize=10)
    alld = np.concatenate([dists[q][1:] for q in qs])
    ax2.hist(alld, bins=np.arange(0, alld.max() + h, h), color="#777", edgecolor="white")
    ax2.axvline(d_max, color="#000", lw=1.2, ls="--")
    ax2.set_xlabel("MiniLM sentence distance d = 1 - cos to the original question"); ax2.set_ylabel("entries (all questions)")
    fig.tight_layout(); fig.savefig(out / "distances.png"); plt.close(fig)

    # 3 resolution: steps between neighbouring distances in [0, d_max]
    big, med = [], []
    for q in qs:
        d = np.unique(np.concatenate([dists[q][dists[q] <= d_max], [d_max]]))
        g = np.diff(d)
        big.append(g.max()); med.append(np.median(g))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(qs))
    ax.barh(y + 0.18, big, height=0.36, color="#222", label="largest step")
    ax.barh(y - 0.18, med, height=0.36, color="#aaa", label="median step")
    for i, b in enumerate(big):
        ax.text(b + 0.002, i + 0.18, f"{b:.3f} (t {b / d_max:.2f})", va="center", fontsize=7.5)
    ax.axvline(h, color="#000", ls=":", lw=1); ax.text(h, len(qs) - 0.3, f" band width {h:.4f}", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(lab)
    ax.set_xlabel("step between neighbouring pool distances in [0, d_max] (incl. 0 and d_max)")
    ax.set_title("Text resolution: largest jump in distance", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "resolution.png"); plt.close(fig)

    # 4 staircase
    ts = np.linspace(0, 1, 1001)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    greys = np.linspace(0.0, 0.7, len(qs))
    labels = []
    for g, q in zip(greys, qs):
        sel = [dists[q][select_index(dists[q], t, 0.0, w_txt=1.0, d_max=d_max, n_bands=nb)] for t in ts]
        ax.step(ts, sel, where="post", color=str(g), lw=1.1)
        if sel[-1] < 0.9 * d_max:        # label only the questions whose pool ends below d_max (ceiling)
            labels.append([sel[-1], short(q) + " (ceiling)", str(g)])
    labels.sort()
    for i in range(1, len(labels)):                 # keep ceiling labels apart
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 0.009)
    for y, s, g in labels:
        ax.text(1.01, y, s, fontsize=7, color=g, va="center")
    ax.plot([0, 1], [0, d_max], color="#000", ls="--", lw=0.8); ax.text(0.55, 0.55 * d_max + 0.02, "target w t d_max", fontsize=8, rotation=20)
    ax.set_xlabel("t (u = 0, w_txt = 1)"); ax.set_ylabel("distance of the rendered question")
    ax.set_title("Selected distance along t (all 11 questions)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "staircase.png", bbox_inches="tight"); plt.close(fig)

    # 5 example sheet
    tt = np.round(np.arange(0, 1.0001, 0.1), 2)
    fig, axs = plt.subplots(len(EXAMPLES), 1, figsize=(8.6, 2.15 * len(EXAMPLES)))
    for ax, q in zip(axs, EXAMPLES):
        d = dists[q]; texts = [q] + [x["text"] for x in Q[q]["pool"]]
        ax.axis("off"); ax.set_title(f"{q}   (pool {len(texts) - 1})", loc="left", fontsize=9, fontweight="bold")
        prev = None
        for j, t in enumerate(tt):
            i = select_index(d, t, 0.0, w_txt=1.0, d_max=d_max, n_bands=nb)
            txt = texts[i] if i != prev else "  (same)"
            prev = i
            ax.text(0.0, 1 - (j + 0.6) / len(tt), f"t={t:.1f}", fontsize=7.5, family="monospace", transform=ax.transAxes)
            ax.text(0.08, 1 - (j + 0.6) / len(tt), f"d={d[i]:.3f}", fontsize=7.5, family="monospace", transform=ax.transAxes)
            ax.text(0.19, 1 - (j + 0.6) / len(tt), textwrap.shorten(txt, 95), fontsize=7.5, transform=ax.transAxes,
                    color="#000" if txt != "  (same)" else "#888")
    fig.suptitle("Rendered question along t (u = 0, w_txt = 1); u picks other members of the same band", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "examples.png"); plt.close(fig)
    print("\n".join(str(p) for p in sorted(out.glob("*.png"))))


if __name__ == "__main__":
    main()
