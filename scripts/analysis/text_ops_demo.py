#!/usr/bin/env python
"""Text-operator dry run (CPU): genes per operator, levels per gene, and sample renderings at increasing edit counts
for the calibration questions, for a calibration settings file.
    python scripts/analysis/text_ops_demo.py configs/calib/r1.yaml [--n 6]"""
import argparse, sys
from pathlib import Path

import numpy as np
import yaml

ap = argparse.ArgumentParser(); ap.add_argument("settings", type=Path); ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()
from mmb.manipulator.text.composite import CompositeTextManipulator
from mmb.manipulator.text.config import TextCompositeConfig, TextConfig
s = yaml.safe_load(a.settings.read_text())
cm = CompositeTextManipulator.from_config(TextConfig(composite=TextCompositeConfig(operators=tuple(s["text"]["operators"]))), device="cpu")
rng = np.random.default_rng(0)
seen = set()
for task in ("imagenet", "pope"):
    for it in yaml.safe_load(Path(f"cells/{task}_calib.yaml").read_text())["seeds"]["slot_items"]["items"]:
        q, prot = it["prompt"], frozenset(w.lower() for w in it["meta"]["protect_words"])
        if q in seen:
            continue
        seen.add(q)
        ctx = cm.prepare(q, exclude_words=prot)
        print(f"\n[{task}] {q}  protected={sorted(prot)}")
        print("  genes per op:", dict(zip(ctx.op_order, ctx.op_gene_dims)), "bounds:", ctx.gene_bounds.tolist())
        b = ctx.gene_bounds
        for k in (1, 2, 4, len(b)):
            for _ in range(2):
                g = np.zeros(len(b), dtype=np.int64)
                idx = rng.choice(len(b), size=min(k, len(b)), replace=False)
                g[idx] = [rng.integers(1, bb) for bb in b[idx]]
                print(f"  {k} edits: {cm.apply(ctx, g)!r}")
