"""One place for the table conventions of the paper (user rules, STYLE.md).

1. Every metric column header shows its direction: an up arrow for higher-is-better and a down arrow for
   lower-is-better. Neutral columns (counts, n, p-values, correlations, differences) get no arrow.
2. The best value in each metric column is bold within each comparison group (ties: all tied values; when every
   value of a group ties there is no best, so nothing is bold). Counts and p-values are never bold. In paired
   tables the arm is bold only when it is significantly better after Holm correction.

DIRECTION is the single map from a metric key to its direction; every table writer goes through head(),
bold_best() and bold_if().
"""
from __future__ import annotations

import math
from typing import Sequence

UP, DOWN = "up", "down"
ARROW = {UP: r" $\uparrow$", DOWN: r" $\downarrow$"}
ALPHA = 0.05

DIRECTION: dict[str, str] = {
    # effectiveness (the generation-0 share and the budget-30 share are descriptive: every arm draws generation 0 from
    # the same sampler, so they are not in this map and stay neutral)
    "crossed": UP, "band": UP, "label_flip": UP, "budget_share": UP,
    # efficiency
    "first_cross_eval": DOWN, "evals": DOWN, "floor": DOWN, "min_edits": DOWN, "hv_edits": UP,
    "img_mag": DOWN, "txt_mag": DOWN, "wall": DOWN, "hv_mag": UP, "chardist": DOWN,
    # validity proxies
    "lpips": DOWN, "txt_dist": DOWN, "text_change": DOWN, "lpips_share": UP, "txt_share": UP, "both_share": UP, "cons_share": UP,
    "vis_cos": UP, "vis_identical": UP, "vis_within_one": UP, "obj_txt_dist": DOWN, "obj_share": UP,
    "clip": UP, "dino": UP, "ssim": UP, "minilm": UP, "word_edit": DOWN,
}


def head(label: str, key: str | None) -> str:
    """Header text with the direction arrow of metric *key* (no arrow for neutral or unknown keys)."""
    return label + ARROW.get(DIRECTION.get(key or "", ""), "")


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def best_indices(values: Sequence, key: str, ties: Sequence[str] | None = None) -> set[int]:
    """Indices of the best values of a comparison group. *ties* are the displayed strings: two values tie when they
    display the same. No best (empty set) when fewer than two values exist or all of them tie."""
    d = DIRECTION.get(key)
    idx = [i for i, v in enumerate(values) if _ok(v)]
    if d is None or len(idx) < 2:
        return set()
    best = (max if d == UP else min)(values[i] for i in idx)
    label = (lambda i: ties[i]) if ties is not None else (lambda i: values[i])
    win = {i for i in idx if (label(i) == label(next(j for j in idx if values[j] == best)))}
    return set() if len(win) == len(idx) else win


def stars(p: float) -> str:
    """Significance marker of a Holm-corrected p-value: *** p<0.001, ** p<0.01, * p<0.05 (explained once in the text)."""
    if not _ok(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < ALPHA else ""


def mark(cell: str, p: float) -> str:
    """The cell followed by its significance marker as a superscript (nothing when not significant)."""
    s = stars(p)
    return cell + (rf"$^{{{s}}}$" if s else "")


def bold(cell: str, on: bool) -> str:
    return rf"\textbf{{{cell}}}" if on else cell


def bold_best(cells: Sequence[str], values: Sequence, key: str, ties: Sequence[str] | None = None) -> list[str]:
    best = best_indices(values, key, ties)
    return [bold(c, i in best) for i, c in enumerate(cells)]


def favourable(diff: float, key: str) -> bool:
    """True when a positive (diff > 0) or negative difference of metric *key* is an improvement."""
    d = DIRECTION.get(key)
    return _ok(diff) and ((d == UP and diff > 0) or (d == DOWN and diff < 0))


def bold_if_better(cell: str, diff: float, p_holm: float, key: str) -> str:
    """Paired tables: bold the arm only when it is significantly better after Holm correction."""
    return bold(cell, _ok(p_holm) and p_holm < ALPHA and favourable(diff, key))
