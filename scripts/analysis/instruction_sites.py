#!/usr/bin/env python
"""T-A sites if the answer-format instruction were editable too (question for the user, 2026-11-28).
For each distinct (question, options) of the cell files: T-A (semantic_substitution, ladder [1,2,4,8]) on
question + instruction, with the protected words AND the option words fixed. Reports sites and variants of the question
part vs the instruction part.

    python scripts/analysis/instruction_sites.py --out docs/T-A_INSTRUCTION_SITES.md
"""
import argparse, glob
from pathlib import Path

import numpy as np
import spacy
import yaml

from mmb.manipulator.text.operators.semantic import SemanticSubstitutionOperator
from mmb.manipulator.text.types import tokenize

FMT = " Answer with exactly one of these options: {categories}."


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--max-per-prompt", type=int, default=2)
    a = ap.parse_args()
    nlp = spacy.load("en_core_web_sm")
    op = SemanticSubstitutionOperator(ladder=(1, 2, 4, 8), device="cpu")
    seen, rows = {}, []
    for f in sorted(set(glob.glob("cells/*_calib.yaml") + glob.glob("cells/*_calib2.yaml") + glob.glob("cells/*_cells.yaml"))):
        for it in yaml.safe_load(open(f))["seeds"]["slot_items"]["items"]:
            q = it["prompt"]; meta = it.get("meta") or {}
            opts = [meta.get("anchor_label_in_prompt"), meta.get("target_label_in_prompt")] if "imagenet" in f else ["Yes", "No"]
            if any(o is None for o in opts):
                continue
            if seen.get(q, 0) >= a.max_per_prompt or (q, tuple(opts)) in {(r[0], r[1]) for r in rows}:
                continue
            seen[q] = seen.get(q, 0) + 1
            prot = {str(w).lower() for w in meta.get("protect_words") or []} | {w.lower() for o in opts for w in str(o).split()}
            full = q + FMT.format(categories=", ".join(opts))
            toks = tokenize(nlp, full); nq = len(tokenize(nlp, q).tokens)
            ctx = op.prepare(toks, frozenset(prot))
            qs = [(w, c) for p, w, c in zip(ctx.positions, ctx.original_words, ctx.candidates) if p < nq]
            ins = [(w, c) for p, w, c in zip(ctx.positions, ctx.original_words, ctx.candidates) if p >= nq]
            vq = int(np.prod([len(c) + 1 for _, c in qs])) if qs else 1
            vi = int(np.prod([len(c) + 1 for _, c in ins])) if ins else 1
            rows.append((q, tuple(opts), qs, ins, vq, vi))
            print(q, opts, f"question sites {len(qs)} variants {vq}; instruction sites {len(ins)} variants {vi}")
    md = ["# T-A: question vs answer-format instruction sites", "",
          "T-A (embedding-ranked word substitution, ladder 1/2/4/8, NLI >= 0.8, CoLA >= 0.9 x original, dictionary words, "
          "no auxiliaries) on question + instruction; protected words and the option words fixed. Variants = product over "
          "sites of (levels + 1), i.e. every combination including 'unchanged'.", "",
          "| question | options | question sites | question variants | instruction sites | instruction variants | combined |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {q} | {', '.join(o)} | {len(qs)} | {vq} | {len(ins)} | {vi} | {vq * vi} |" for q, o, qs, ins, vq, vi in rows]
    md += ["", "## Substitutes per site", ""]
    for q, o, qs, ins, vq, vi in rows:
        md += [f"**{q}** (options {', '.join(o)})", "",
               "- question: " + ("; ".join(f"{w} -> {', '.join(c)}" for w, c in qs) or "none"),
               "- instruction: " + ("; ".join(f"{w} -> {', '.join(c)}" for w, c in ins) or "none"), ""]
    a.out.write_text("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
