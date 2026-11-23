"""Failure types of the minimal crossings, read straight from the stored runs (RQ1, Further Analysis).

For every run of runs/MAIN/<task>/<sut>/<arm>/s*/<run>/ with a minimal crossing (min_cross.json, written by the tester and
confirmed by single-input scoring) the genotype is split with context.json: the first len(image_patch_positions) genes are
image region genes, the rest are text genes in the contiguous per-operator blocks text_op_order / text_op_gene_dims
(synonym, fragmentation, character_noise, saliency = typo). A character-noise gene is split into homoglyph and
zero-width by the characters present in the edited question. The result has the columns of failure_types
(verified, n_genes_first, n_genes_min, img_genes_min, txt_genes_min, txt_*), so failure_types.summary/figure and the
EFail macros apply unchanged.

Usage::

    python -m mmb.analysis.min_types runs/MAIN --out results/minimal_types_MAIN.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmb.analysis import failure_types as ft
from mmb.manipulator.text.operators.character_noise import HOMOGLYPHS, ZERO_WIDTH

ARMS = {"a0_directed": "directed", "a1_random_tiered": "others"}
_HOMO_CHARS = frozenset(HOMOGLYPHS.values())


def _nonzero(g) -> int:
    return int(np.count_nonzero(np.asarray(g)))


def classify_genotype(genotype, ctx: dict, text: str) -> dict:
    """Gene counts of one crossing: image regions, text genes per operator family."""
    g = np.asarray(genotype, dtype=int)
    n_img_slots = len(ctx.get("image_patch_positions", []))
    img, txt = g[:n_img_slots], g[n_img_slots:]
    out = dict(img_genes=_nonzero(img), txt_genes=_nonzero(txt), txt_synonym_words=0, txt_typo_words=0,
               txt_zero_width=0, txt_homoglyph=0, txt_other_genes=0)
    pos = 0
    for name, dim in zip(ctx.get("text_op_order", []), ctx.get("text_op_gene_dims", [])):
        block = txt[pos:pos + dim]
        pos += dim
        k = _nonzero(block)
        if not k:
            continue
        if name == "synonym":
            out["txt_synonym_words"] += k
        elif name == "saliency":
            out["txt_typo_words"] += k
        elif name == "character_noise":
            out["txt_zero_width"] += sum(text.count(c) for c in ZERO_WIDTH)
            out["txt_homoglyph"] += sum(1 for c in text if c in _HOMO_CHARS)
        else:
            out["txt_other_genes"] += k
    return out


def load_runs(runs_root: Path | str, arms=ARMS) -> pd.DataFrame:
    rows = []
    for mc in sorted(Path(runs_root).glob("*/*/*/s*/*/min_cross.json")):
        rd = mc.parent
        task, sut, arm = rd.parts[-5], rd.parts[-4], rd.parts[-3]
        if arm not in arms or not (rd / "context.json").exists():
            continue
        m, ctx = json.loads(mc.read_text()), json.loads((rd / "context.json").read_text())
        c = classify_genotype(m["genotype"], ctx, m.get("text", ""))
        first = np.nan
        fj = rd / "first_cross.json"
        if fj.exists():
            f = json.loads(fj.read_text())
            if "genotype" in f:
                first = _nonzero(f["genotype"])
        rows.append(dict(run_dir=str(rd), task=task, sut=sut, arm=arm, rng_seed=int(rd.parts[-2][1:]),
                         verified=bool(m.get("confirmed", False)), n_genes_first=first,
                         n_genes_min=c["img_genes"] + c["txt_genes"], img_genes_min=c["img_genes"],
                         txt_genes_min=c["txt_genes"], txt_synonym_words=c["txt_synonym_words"],
                         txt_typo_words=c["txt_typo_words"], txt_zero_width=c["txt_zero_width"],
                         txt_homoglyph=c["txt_homoglyph"]))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["group"] = df["arm"].map(arms)
    df["ftype"] = [ft.classify(r) if r["verified"] else None for _, r in df.iterrows()]
    return df


# ---- composition table and macros -----------------------------------------------------------------------------
SUT_NAMES = {"qwen": "Qwen3.5-4B", "internvl": "InternVL3.5-4B", "ministral": "Ministral-3-3B"}
TASK_NAMES = {"imagenet": "ImageNet pairs", "pope": "POPE yes/no"}
GROUP_LABELS = {"directed": "Directed", "others": "Random"}


def composition(g: pd.DataFrame) -> dict:
    """Channel mix and edit families of the verified minimal crossings of one scope."""
    v = g[g["verified"]]
    n = len(v)
    if not n:
        return dict(n=0)
    img, txt = v["img_genes_min"] > 0, v["txt_genes_min"] > 0
    return dict(n=n, img_only=100 * float((img & ~txt).mean()), txt_only=100 * float((~img & txt).mean()),
                joint=100 * float((img & txt).mean()), regions=float(v["img_genes_min"].median()),
                synonym=float(v["txt_synonym_words"].mean()), typo=float(v["txt_typo_words"].mean()),
                charnoise=float((v["txt_zero_width"] + v["txt_homoglyph"]).mean()),
                text_genes=float(v["txt_genes_min"].mean()))


COLS = [("img_only", r"Image only (\%)", 0), ("txt_only", r"Text only (\%)", 0), ("joint", r"Joint (\%)", 0),
        ("regions", "Regions", 0), ("synonym", "Synonym", 1), ("typo", "Typo", 1), ("charnoise", "Char.\\ noise", 1)]


def table_tex(df: pd.DataFrame) -> str:
    """One block per model, one band per task, rows directed and random: channel mix, median image regions and the mean
    number of text edits per crossing by family (synonym, typo, character noise = homoglyph and zero-width)."""
    k = len(COLS)
    lines = [r"\begin{tabular}{l" + "r" * k + "}", r"\toprule", " & ".join(["Method"] + [c[1] for c in COLS]) + r" \\"]
    for s in [x for x in SUT_NAMES if x in set(df["sut"])]:
        lines += [r"\midrule", rf"\multicolumn{{{1 + k}}}{{l}}{{\emph{{{SUT_NAMES[s]}}}}} \\"]
        for t in [x for x in TASK_NAMES if x in set(df["task"])]:
            lines.append(rf"\multicolumn{{{1 + k}}}{{l}}{{\hspace{{1em}}{TASK_NAMES[t]}}} \\")
            for gk, lab in GROUP_LABELS.items():
                c = composition(df[(df["sut"] == s) & (df["task"] == t) & (df["group"] == gk)])
                if not c["n"]:
                    continue
                cells = [f"{c[key]:.{dec}f}" for key, _l, dec in COLS]
                lines.append(" & ".join([r"\hspace{2em}" + lab] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def add_numbers(num, df: pd.DataFrame, camel, f_num, f_pct) -> None:
    """Macros EMinType<Kind><Group><Sut><Task> (Kind: N, ImgOnly, TxtOnly, Joint, Regions, Synonym, Typo, CharNoise)."""
    for (sut, task), d in ft.scopes(df).items():
        for gk, gname in (("directed", "Directed"), ("others", "Random")):
            c = composition(d[d["group"] == gk])
            if not c["n"]:
                continue
            tail = [gname, camel(sut), camel(task)]
            where = f"{gk} arm, {sut}, {task}, verified minimal crossings"
            num.add(["EMinType", "N"] + tail, str(c["n"]), "crossings; " + where)
            for key, nm in (("img_only", "ImgOnly"), ("txt_only", "TxtOnly"), ("joint", "Joint")):
                num.add(["EMinType", nm] + tail, f_pct(c[key] / 100, 0), f"share (%) of image-only / text-only / joint minimal crossings; {where}")
            num.add(["EMinType", "Regions"] + tail, f_num(c["regions"], 0), "median image regions; " + where)
            for key, nm in (("synonym", "Synonym"), ("typo", "Typo"), ("charnoise", "CharNoise")):
                num.add(["EMinType", nm] + tail, f_num(c[key], 1), f"mean text edits per crossing of this family; {where}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    df = load_runs(a.runs_root)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} minimal crossings -> {a.out}")
    print(ft.summary(df).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
