#!/usr/bin/env python
"""Determinism of a boundary pair's scores (scoring invariant, AGENTS.md), 2026-11-28.

Re-renders the ON/OFF genotypes of a run's boundary_pair.json with the run's manipulators and scores each side
(a) alone, (b) repeatedly, (c) inside batches of different composition / size / position (fillers = other trace
genotypes), (d) batched path (score_batch_tensor) vs single path (score_categories_tensor, as process_input).
Prints the spread of g = l(a) - l(b) per side and compares it with |g_ON - g_OFF|.

  ../.venv/bin/python scripts/check_pair_determinism.py RUN_DIR CONFIG.yaml [--sut MODEL_ID] [--out results/x.json]
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path); ap.add_argument("config", type=Path)
    ap.add_argument("--sut", default=None); ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from mmb.runners.run_boundary_test import load_config
    from mmb.manipulator.image_factory import build_image_backend
    from mmb.manipulator.text.composite import CompositeTextManipulator
    from mmb.manipulator.vlm_manipulator import VLMManipulator
    from mmb.sut.scorer import create_scorer
    from PIL import Image

    cfg = load_config(yaml.safe_load(a.config.read_text()))
    run = a.run_dir
    bp = json.loads((run / "boundary_pair.json").read_text())
    fc = json.loads((run / "min_cross.json").read_text())
    stats = json.loads((run / "stats.json").read_text())
    item = cfg.seeds.slot_items.items[0]
    img = Image.open(Path(item.image).expanduser()).convert("RGB")
    suffix = fc["full_prompt"][len(fc["text"]):]
    cats = list(stats["categories"])
    ia, ib = stats["target_classes"]

    man = VLMManipulator(build_image_backend(image_config=cfg.image, device="cuda"),
                         CompositeTextManipulator.from_config(cfg.text, device="cuda"))
    # per-seed question (stats.prompt_template is the global template)
    man.prepare(img, item.prompt, exclude_words=frozenset(
        w.lower() for w in (item.meta or {}).get("protect_words", [])) or None)
    if "on" in bp:
        on, off = np.asarray(bp["on"]["genotype"]), np.asarray(bp["off"]["genotype"])
        bp_g = {"on": bp["on"]["g"], "off": bp["off"]["g"]}
    else:   # no confirmed pair: the in-budget min crossing C and its eps-neighbour on the scale line
        on = np.asarray(fc["genotype"], float)
        off = man.scale(on, 1.0 - 1e-3)
        bp_g = {"on": fc["g"], "off": None}
    trace = pd.read_parquet(run / "trace.parquet")
    fill_g = np.stack(trace["genotype"].to_numpy())[:: max(1, len(trace) // 20)][:20]
    ims, txts = man.manipulate(weights=np.vstack([on, off, on, off, fill_g]))
    render_same = (np.array_equal(np.asarray(ims[0]), np.asarray(ims[2])),
                   np.array_equal(np.asarray(ims[1]), np.asarray(ims[3])))
    I = {"on": ims[0], "off": ims[1]}; T = {"on": txts[0] + suffix, "off": txts[1] + suffix}
    zero_im, zero_t = man.manipulate(weights=np.zeros((1, len(on))))
    saved = {s: Image.open(run / f).convert("RGB") for s, f in (("on", "min_cross.png"), ("off", "min_cross_neighbour.png"))
             if (run / f).exists() and "on" in bp}
    pix = {s: float(np.abs(np.asarray(saved[s], float) - np.asarray(I[s], float)).max()) for s in saved}
    FI, FT = ims[4:], [t + suffix for t in txts[4:]]

    del man                                      # free the VQGAN before the SUT (shared GPU)
    torch.cuda.empty_cache()
    model_id = a.sut or cfg.sut.model_id
    sc = create_scorer(model_id, "cuda", enable_thinking=False)
    if hasattr(sc, "batch_chunk"):
        sc.batch_chunk = a.chunk
    def g_of(lp, i):
        return float(lp[i, ia] - lp[i, ib])
    res = {s: {} for s in ("on", "off")}
    with torch.no_grad():
        for s in ("on", "off"):
            res[s]["single_path"] = [float(sc.score_categories_tensor(I[s], T[s], cats)[ia]
                                           - sc.score_categories_tensor(I[s], T[s], cats)[ib]) for _ in range(1)]
            res[s]["single_path"] += [float((lambda v: v[ia] - v[ib])(sc.score_categories_tensor(I[s], T[s], cats)))
                                      for _ in range(a.reps - 1)]
            res[s]["batch_alone"] = [g_of(sc.score_batch_tensor([I[s]], [T[s]], cats), 0) for _ in range(a.reps)]
        lp = sc.score_batch_tensor([I["on"], I["off"]], [T["on"], T["off"]], cats)
        res["on"]["pair_batch"], res["off"]["pair_batch"] = [g_of(lp, 0)], [g_of(lp, 1)]
        lp = sc.score_batch_tensor([I["off"], I["on"]], [T["off"], T["on"]], cats)
        res["on"]["pair_batch"].append(g_of(lp, 1)); res["off"]["pair_batch"].append(g_of(lp, 0))
        rng = random.Random(0)
        for s in ("on", "off"):
            res[s]["mixed_batch"] = []
            for n in (2, 4, 8, 12):
                for pos in (0, n // 2, n - 1):
                    idx = rng.sample(range(len(FI)), n - 1)
                    bi = [FI[i] for i in idx]; bt = [FT[i] for i in idx]
                    bi.insert(pos, I[s]); bt.insert(pos, T[s])
                    res[s]["mixed_batch"].append(g_of(sc.score_batch_tensor(bi, bt, cats), pos))
    with torch.no_grad():
        anchor_g = float((lambda v: v[ia] - v[ib])(sc.score_categories_tensor(zero_im[0], zero_t[0] + suffix, cats)))
        saved_g = {s: float((lambda v: v[ia] - v[ib])(sc.score_categories_tensor(saved[s], T[s], cats))) for s in saved}
    out = {"sut": model_id, "render_deterministic": render_same, "texts_equal": T["on"] == T["off"],
           "anchor_g_rescored": anchor_g, "anchor_g_run": stats.get("anchor_g"),
           "saved_png_g": saved_g, "max_pixel_diff_render_vs_saved_png": pix,
           "g_trace": bp_g}
    for s in ("on", "off"):
        allg = [g for v in res[s].values() for g in v]
        out[s] = {"conditions": {k: [round(x, 4) for x in v] for k, v in res[s].items()},
                  "min": min(allg), "max": max(allg), "spread": max(allg) - min(allg), "mean": float(np.mean(allg))}
    out["pair_diff_mean"] = out["on"]["mean"] - out["off"]["mean"]
    out["sign_flips_on"] = sum(1 for v in res["on"].values() for g in v if g >= 0)
    out["sign_flips_off"] = sum(1 for v in res["off"].values() for g in v if g < 0)
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
