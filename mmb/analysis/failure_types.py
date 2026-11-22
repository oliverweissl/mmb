"""Failure types of minimal flipping perturbations (contraction study; RQ3).

Input: the CSVs of scripts/minimize_crossings.py (one row per run with a robust first crossing; columns run_dir, task, sut,
arm, rng_seed, verified, n_genes_first, n_genes_min, img_genes_min, txt_genes_min, channel, txt_zero_width, txt_homoglyph,
txt_typo_words, txt_synonym_words, txt_visible_identical, ...). A row is classified only if it verified, i.e. the stored
crossing was re-scored as a robust crossing and then reduced to a 1-minimal flipping perturbation.

Types: image-only by patch-count bin, text-only by the dominant text operator, and joint (image and text genes).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

#: (key, label) in plotting order
TYPES = [
    ("image_few", "Image, 1-9 patches"), ("image_mid", "Image, 10-49 patches"), ("image_many", "Image, 50+ patches"),
    ("text_typo", "Text, typo"), ("text_homoglyph", "Text, homoglyph"), ("text_zero_width", "Text, zero-width"),
    ("text_synonym", "Text, near-synonym"), ("text_other", "Text, other"), ("joint", "Image and text"),
]
TYPE_KEYS = [k for k, _ in TYPES]
TYPE_LABELS = dict(TYPES)
#: macro-name words (no digits)
TYPE_WORDS = dict(image_few="ImageFew", image_mid="ImageMid", image_many="ImageMany", text_typo="TextTypo",
                  text_homoglyph="TextHomoglyph", text_zero_width="TextZeroWidth", text_synonym="TextSynonym",
                  text_other="TextOther", joint="Joint")
GROUPS = [("directed", "Directed arm"), ("others", "Other arms")]
IMAGE_BINS = ((1, 9, "image_few"), (10, 49, "image_mid"), (50, 10 ** 9, "image_many"))
#: tie order between text operators
TEXT_OPS = (("txt_typo_words", "text_typo"), ("txt_synonym_words", "text_synonym"),
            ("txt_homoglyph", "text_homoglyph"), ("txt_zero_width", "text_zero_width"))


def load_minimal(paths: Sequence[Path | str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["verified"] = df["verified"].astype(str).str.lower().isin(("true", "1", "1.0"))
    for c in ("n_genes_first", "n_genes_min", "img_genes_min", "txt_genes_min", "txt_zero_width", "txt_homoglyph",
              "txt_typo_words", "txt_synonym_words"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["group"] = np.where(df["arm"] == "a0_directed", "directed", "others")
    df["ftype"] = [classify(r) if r["verified"] else None for _, r in df.iterrows()]
    return df


def classify(row) -> str | None:
    """Failure type of one minimised crossing."""
    n_img, n_txt = row.get("img_genes_min"), row.get("txt_genes_min")
    if pd.isna(n_img) or pd.isna(n_txt):
        return None
    n_img, n_txt = int(n_img), int(n_txt)
    if n_img and n_txt:
        return "joint"
    if n_img:
        return next((k for lo, hi, k in IMAGE_BINS if lo <= n_img <= hi), "image_many")
    counts = [(int(row.get(c) or 0), k) for c, k in TEXT_OPS]
    best = max(counts, key=lambda t: t[0])
    return best[1] if best[0] > 0 else "text_other"


def scopes(df: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """(sut, task) -> rows, with 'All' for the pooled scopes."""
    out = {("All", "All"): df}
    for sut in sorted(df["sut"].dropna().unique()):
        out[(sut, "All")] = df[df["sut"] == sut]
    for task in sorted(df["task"].dropna().unique()):
        out[("All", task)] = df[df["task"] == task]
    for (sut, task), g in df.groupby(["sut", "task"]):
        out[(sut, task)] = g
    return out


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (sut, task, group): runs reduced, verified share, median genes before/after, share per type."""
    rows = []
    for (sut, task), d in scopes(df).items():
        for gk, _ in GROUPS:
            g = d[d["group"] == gk]
            if g.empty:
                continue
            v = g[g["verified"]]
            row = dict(sut=sut, task=task, group=gk, n=len(g), n_verified=len(v),
                       verified_share=len(v) / len(g),
                       med_genes_first=float(v["n_genes_first"].median()) if len(v) else np.nan,
                       med_genes_min=float(v["n_genes_min"].median()) if len(v) else np.nan,
                       med_img_min=float(v["img_genes_min"].median()) if len(v) else np.nan,
                       med_txt_min=float(v["txt_genes_min"].median()) if len(v) else np.nan,
                       one_gene_share=float((v["n_genes_min"] == 1).mean()) if len(v) else np.nan)
            for k in TYPE_KEYS:
                row["share_" + k] = float((v["ftype"] == k).mean()) if len(v) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


PALETTE = {"image_few": "#0072B2", "image_mid": "#56B4E9", "image_many": "#9CC9E8", "text_typo": "#D55E00",
           "text_homoglyph": "#E69F00", "text_zero_width": "#F0E442", "text_synonym": "#CC79A7",
           "text_other": "#999999", "joint": "#009E73"}


def figure(df: pd.DataFrame, path: Path, sut_names: dict | None = None, task_names: dict | None = None) -> bool:
    """Stacked bars (share of verified minimal flips per type), one panel per arm group, one bar per model x task."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sut_names, task_names = sut_names or {}, task_names or {}
    v = df[df["verified"] & df["ftype"].notna()]
    if v.empty:
        return False
    combos = sorted({(s, t) for s, t in zip(v["sut"], v["task"])})
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(7.0, 2.6), sharey=True)
    for ax, (gk, gname) in zip(axes, GROUPS):
        sub = v[v["group"] == gk]
        bottoms = np.zeros(len(combos))
        for k in TYPE_KEYS:
            share = np.array([
                (sub[(sub["sut"] == s) & (sub["task"] == t)]["ftype"] == k).mean()
                if len(sub[(sub["sut"] == s) & (sub["task"] == t)]) else 0.0 for s, t in combos]) * 100
            ax.bar(range(len(combos)), share, bottom=bottoms, color=PALETTE[k], label=TYPE_LABELS[k], width=0.7,
                   edgecolor="white", linewidth=0.4)
            bottoms += share
        for i, (s, t) in enumerate(combos):
            n = len(sub[(sub["sut"] == s) & (sub["task"] == t)])
            ax.text(i, 101, f"n={n}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels([f"{sut_names.get(s, s)}\n{task_names.get(t, t)}" for s, t in combos], fontsize=6)
        ax.set_title(gname, fontsize=8)
        ax.set_ylim(0, 112)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Minimal flips (%)", fontsize=8)
    axes[-1].legend(fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True
