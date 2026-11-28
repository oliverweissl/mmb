"""Modality attribution by swapping: shares, macros (EModSwap*) and figure from scripts/modality_swap.py output.

Classes per verified crossing: image (image edit suffices), text (text edit suffices), either (each alone suffices),
joint (neither alone suffices). Shares are over verified crossings; unverified crossings are counted separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("image", "text", "either", "joint")
CLASS_LABELS = {"image": "image-sufficient", "text": "text-sufficient", "either": "either-sufficient",
                "joint": "joint-necessary"}
SHADES = {"image": "0.25", "text": "0.85", "either": "0.55", "joint": "white"}
HATCH = {"image": "", "text": "", "either": "", "joint": "////"}


def load(paths) -> pd.DataFrame:
    fr = [pd.read_csv(p) for p in paths if Path(p).exists() and Path(p).stat().st_size > 0]
    d = pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()
    for c in ("verified", "a_crosses", "b_crosses", "text_edited", "image_edited"):
        if c in d:
            d[c] = d[c].astype(str).str.lower().isin(("true", "1"))
    return d


def shares(d: pd.DataFrame, keys=("arm",)) -> pd.DataFrame:
    """Per group: n crossings, n verified, share (%) of each class among verified, median relative margins."""
    rows = []
    for k, g in d.groupby(list(keys)):
        v = g[g["verified"]]
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)), n=len(g), n_verified=len(v))
        for c in CLASSES:
            r[c] = 100 * float((v["cls"] == c).mean()) if len(v) else np.nan
        for col in ("rel_text_only", "rel_image_only", "g_text_only", "g_image_only"):
            r[f"{col}_med"] = float(v[col].median()) if len(v) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def add_numbers(num, d: pd.DataFrame, camel, view: str = "min") -> None:
    """EModSwap{Arm}{Class}{Scope}: share (%) of verified crossings; ...N, ...Verified; ...RelText/RelImage medians.
    Scope = All (pooled) or the task. First-crossing view: prefix EModSwapFirst."""
    from mmb.analysis.export import arm_short
    pre = "EModSwap" if view == "min" else "EModSwapFirst"
    d = d[d["view"] == view]
    for scope_keys, scope_of in ((("arm",), lambda r: "All"), (("arm", "task"), lambda r: camel(str(r["task"])))):
        for _, r in shares(d, scope_keys).iterrows():
            arm, sc = arm_short(str(r["arm"])), scope_of(r)
            for c in CLASSES:
                if not pd.isna(r[c]):
                    num.add([pre, arm, camel(c), sc], f"{r[c]:.1f}", f"modality swap ({view}): {CLASS_LABELS[c]} share (%), {r['arm']}, {sc}")
            num.add([pre, arm, "N", sc], str(int(r["n"])), f"modality swap ({view}): crossings, {r['arm']}, {sc}")
            num.add([pre, arm, "Verified", sc], str(int(r["n_verified"])), f"modality swap ({view}): crossings re-verified alone, {r['arm']}, {sc}")
            for col, nm in (("rel_text_only_med", "RelText"), ("rel_image_only_med", "RelImage")):
                if not pd.isna(r[col]):
                    num.add([pre, arm, nm, sc], f"{r[col]:.2f}", f"modality swap ({view}): median margin of the {nm[3:].lower()}-only swap relative to the anchor's, {r['arm']}, {sc}")


def plot(d: pd.DataFrame, out, arm_label, *, view: str = "min", order=None, title: str | None = None,
         width_in: float = 5.4, height_in: float = 1.7) -> Path:
    """One stacked horizontal bar per method (share of verified crossings per class); right: n verified."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6})
    s = shares(d[d["view"] == view], ("arm",)).set_index("arm")
    arms = [a for a in (order or sorted(s.index)) if a in s.index]
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    y = np.arange(len(arms))[::-1]
    for yi, arm in zip(y, arms):
        left = 0.0
        for c in CLASSES:
            v = float(s.at[arm, c]) if not pd.isna(s.at[arm, c]) else 0.0
            ax.barh(yi, v, left=left, color=SHADES[c], hatch=HATCH[c], edgecolor="0.0", linewidth=0.5, height=0.6,
                    label=CLASS_LABELS[c] if yi == y[0] else None)
            if v >= 8:
                ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if c == "image" else "0.0")
            left += v
        ax.text(101, yi, f"n={int(s.at[arm, 'n_verified'])}", va="center", fontsize=5.5)
    ax.set_yticks(y); ax.set_yticklabels([arm_label(a) for a in arms])
    ax.set_xlim(0, 100); ax.set_xlabel("share of crossings (%)")
    ax.legend(ncol=4, fontsize=5.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    if title:
        ax.set_title(title, fontsize=6.5, pad=14)
    fig.tight_layout(pad=0.3)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
