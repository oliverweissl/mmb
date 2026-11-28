#!/usr/bin/env python
"""Re-run one generation of a finished run exactly as the tester scored it (2026-11-28).

Loads the components the way the runner does (init_shared_components: SUT, image and text loaded in parallel
threads), re-renders the generation's genotypes in one manipulate() call and scores them with process_batch
(sut.batched_scoring) or process_input, then compares g with the trace. Also scores the generation's
individuals alone. Isolates rendering vs scoring nondeterminism.

  ../.venv/bin/python scripts/repro_generation.py RUN_DIR CONFIG.yaml --gen 6
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir", type=Path); ap.add_argument("config", type=Path)
    ap.add_argument("--gen", type=int, required=True); ap.add_argument("--serial", action="store_true",
                                                                       help="load the image backend after the SUT")
    a = ap.parse_args()
    from mmb.config import resolve_categories
    from mmb.data import ImageNetCache
    from mmb.common.pipeline_bootstrap import init_shared_components
    from mmb.manipulator.vlm_manipulator import VLMManipulator
    from mmb.runners.run_boundary_test import load_config
    from PIL import Image
    raw = yaml.safe_load(a.config.read_text()); raw["device"] = "cuda"
    exp = load_config(raw)
    ds = ImageNetCache(dirs=exp.cache_dirs); exp = resolve_categories(exp, ds.labels())
    comp = init_shared_components(exp, ds)
    vq = comp.image_manip._model
    dtypes = sorted({str(p.dtype) for p in vq.parameters()})
    run = a.run_dir
    stats = json.loads((run / "stats.json").read_text()); fc = json.loads((run / "min_cross.json").read_text())
    suffix = fc["full_prompt"][len(fc["text"]):]
    cats = tuple(stats["categories"]); ia, ib = stats["target_classes"]
    item = exp.seeds.slot_items.items[0]
    man = VLMManipulator(comp.image_manip, comp.text_manip)
    man.prepare(Image.open(Path(item.image).expanduser()).convert("RGB"), item.prompt,  # the per-seed question (stats.prompt_template is the global one)
                exclude_words=frozenset(w.lower() for w in (item.meta or {}).get("protect_words", [])) or None)
    tr = pd.read_parquet(run / "trace.parquet"); tr = tr[tr.generation == a.gen].sort_values("individual")
    G = np.stack(tr.genotype.to_numpy())
    g_trace = np.array([lp[ia] - lp[ib] for lp in tr.logprobs])
    ims, txts = man.manipulate(weights=G)
    T = [t + suffix for t in txts]
    with torch.no_grad():
        lp = comp.sut.process_batch(ims, T, cats).float().cpu().numpy()
        g_batch = lp[:, ia] - lp[:, ib]
        g_single = np.array([float((lambda v: v[ia] - v[ib])(comp.sut.process_input(i, text=t, categories=cats)))
                             for i, t in zip(ims, T)])
        ims1 = [man.manipulate(weights=G[i:i + 1])[0][0] for i in range(len(G))]   # rendered alone
        pix = [float(np.abs(np.asarray(x, float) - np.asarray(y, float)).max()) for x, y in zip(ims, ims1)]
        lp1 = comp.sut.process_batch(ims1, T, cats).float().cpu().numpy()
        g_batch_alone_render = lp1[:, ia] - lp1[:, ib]
    with torch.no_grad():
        zi, zt = man.manipulate(weights=np.zeros((1, G.shape[1])))
        a_lp = comp.sut.process_input(zi[0], text=zt[0] + suffix, categories=cats).flatten()
    out = {"anchor_g_run": stats.get("anchor_g"), "anchor_g_repro": float(a_lp[ia] - a_lp[ib]),
           "vqgan_param_dtypes": dtypes, "text_equal_trace": [x == y for x, y in zip(txts, tr.decoded_text)],
           "g_trace": g_trace.round(4).tolist(), "g_rebatch": g_batch.round(4).tolist(),
           "g_single": g_single.round(4).tolist(), "g_rendered_alone_batch": g_batch_alone_render.round(4).tolist(),
           "max_pixel_diff_batch_vs_alone_render": pix,
           "max_abs_trace_vs_rebatch": float(np.abs(g_trace - g_batch).max()),
           "max_abs_batch_vs_single": float(np.abs(g_batch - g_single).max())}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
