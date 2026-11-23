"""One place for the table conventions of the paper (user rules, STYLE.md).

1. Every metric column header shows its direction: an up arrow for higher-is-better and a down arrow for
   lower-is-better. Neutral columns (counts, n, p-values, correlations, differences) get no arrow.
2. The best value in each metric column is bold within each comparison group (ties: all tied values; when every
   value of a group ties there is no best, so nothing is bold). Counts are never bold.
3. Comparison cells show one effect size against the reference arm, a magnitude bullet and a dagger when the
   Holm-corrected paired test is significant (effect_cell); they are not bold.

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
    "first_cross_eval": DOWN, "evals": DOWN, "floor": DOWN, "min_edits": DOWN, "min_edits_x": DOWN, "edits_img_x": DOWN, "edits_txt_x": DOWN, "hv_edits": UP,
    "img_mag": DOWN, "txt_mag": DOWN, "wall": DOWN, "min_cross_eval": DOWN, "sut_calls": DOWN, "hv_mag": UP, "chardist": DOWN,
    # validity proxies
    "lpips": DOWN, "pair_lpips": DOWN, "pair_text": DOWN, "txt_dist": DOWN, "text_change": DOWN, "lpips_share": UP, "txt_share": UP, "both_share": UP, "cons_share": UP,
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


D_LARGE, D_MEDIUM = 1.0, 0.5     # Cohen's d magnitude thresholds of the MIMICRY paper: large d>1, medium 0.5<d<=1, small d<=0.5
BULLET = {"L": r"\effL{}", "M": r"\effM{}", "S": r"\effS{}"}     # filled, half-filled, empty (macros in preamble.tex)
DAGGER = r"$^{\dagger}$"


def magnitude(dz: float) -> str | None:
    """'L' (large), 'M' (medium) or 'S' (small) for an effect size d; None when it is undefined."""
    if not _ok(dz):
        return None
    a = abs(dz)
    return "L" if a > D_LARGE else "M" if a > D_MEDIUM else "S"


def effect_cell(text: str, dz: float, p_holm: float) -> str:
    """One comparison cell: the effect size as text (percentage points for the crossing rate, Cohen's d_z for continuous
    metrics), a magnitude bullet, and a dagger when the Holm-corrected paired test is significant (alpha 0.05).
    Bold is not used here: it is reserved for the best value of a metric column."""
    m = magnitude(dz)
    cell = text + (" " + BULLET[m] if m else "")
    return cell + (DAGGER if _ok(p_holm) and p_holm < ALPHA else "")


def cohens_dz(diffs) -> float:
    """Paired Cohen's d_z: mean of the pair differences over their standard deviation (+-9.99 when they are all equal
    and non-zero, 0 when they are all zero, nan below two pairs)."""
    import numpy as np
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return 0.0 if float(d.mean()) == 0.0 else math.copysign(9.99, float(d.mean()))
    return float(d.mean()) / sd


def bold(cell: str, on: bool) -> str:
    return rf"\textbf{{{cell}}}" if on else cell


def bold_best(cells: Sequence[str], values: Sequence, key: str, ties: Sequence[str] | None = None) -> list[str]:
    best = best_indices(values, key, ties)
    return [bold(c, i in best) for i, c in enumerate(cells)]
