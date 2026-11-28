#!/usr/bin/env python
"""Batch-composition spread of g and the cost of canonical scoring, per SUT (2026-11-28).

For one SUT (as configured for MAIN in scripts/make_matrix.py, bf16 unless quantised there):
 1. anchor g (canonical: codec reconstruction, original question, single SUT call) for every fixed cell
    (cells/*_cells.yaml, cells/*_calib*.yaml);
 2. spread of g on the calibration cells: per cell the unedited input + N edited inputs drawn by the shared
    sampler (realistic magnitudes); each input scored (a) alone on the single path (canonical, rendered alone),
    (b) in R random permutations of the cell's batch on the batched path (rendered in a batch, as in the search),
    (c) in one batch with every input rendered alone (isolates the scorer's share);
 3. wall time of ONE generation of 50 (render + score): batched (decode_batch_size from the config, batched
    scoring with the MAIN batch_chunk) vs canonical (decode one at a time, one single-path call each).

  ../.venv/bin/python scripts/measure_batch_spread.py --sut qwen --out results/determinism/spread_qwen.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_matrix import SUTS  # noqa: E402

CELL_FILES = ("cells/imagenet_cells.yaml", "cells/pope_cells.yaml", "cells/imagenet_calib.yaml",
              "cells/pope_calib.yaml", "cells/imagenet_calib2.yaml", "cells/pope_calib2.yaml")
SPREAD_FILES = ("cells/imagenet_calib.yaml", "cells/pope_calib.yaml")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sut", required=True, choices=sorted(SUTS)); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-edits", type=int, default=12); ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--pop", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from mmb.common.slot_items_seed_generator import slot_items_seeds
    from mmb.config import apply_modality
    from mmb.evolutionary.vlm_boundary_tester import effective_candidates, effective_prompt_template
    from mmb.manipulator.image_factory import build_image_backend
    from mmb.manipulator.text.composite import CompositeTextManipulator
    from mmb.manipulator.vlm_manipulator import VLMManipulator
    from mmb.optimizer.genotype import GenotypeSpace
    from mmb.optimizer.sparse_sampling import MagnitudeStratifiedSampling
    from mmb.runners.run_boundary_test import load_config
    from mmb.sut import VLMSUT
    from pymoo.core.problem import Problem

    raw = yaml.safe_load(open("configs/paper/e1/base.yaml"))
    raw["sut"] = {**SUTS[a.sut], "redis_url": "redis://localhost:1"}       # no inference cache
    raw["device"] = "cuda"
    exp = apply_modality(load_config(raw))
    sut = VLMSUT(exp)
    torch.cuda.empty_cache()
    man = VLMManipulator(build_image_backend(image_config=exp.image, device="cuda"),
                         CompositeTextManipulator.from_config(exp.text, device="cuda"))
    rng = np.random.default_rng(a.seed)
    out = {"sut": a.sut, "model_id": exp.sut.model_id, "batch_chunk": exp.sut.batch_chunk,
           "decode_batch_size": exp.image.decode_batch_size, "anchors": [], "spread": [], "timing": {}}

    def setup(seed):
        cands = effective_candidates(seed)
        ia, ib = cands.index(seed.class_a), cands.index(seed.class_b)
        suffix = exp.answer_format.format(categories=", ".join(cands))
        protect = (seed.metadata or {}).get("protect_words") or None
        man.prepare(seed.image, effective_prompt_template(seed, exp),
                    exclude_words=frozenset(w.lower() for w in protect) if protect else None)
        return tuple(cands), ia, ib, suffix

    def g_single(img, txt, cats, ia, ib):
        v = sut.process_input(img, text=txt, categories=cats).float().cpu().numpy().ravel()
        return float(v[ia] - v[ib])

    def g_batch(imgs, txts, cats, ia, ib):
        v = sut.process_batch(imgs, txts, cats).float().cpu().numpy()
        return v[:, ia] - v[:, ib]

    def seeds_of(path):
        r = dict(raw); r["seeds"] = yaml.safe_load(open(path))["seeds"]
        return slot_items_seeds(None, load_config(r), None)

    with torch.no_grad():
        # 1. anchors
        for f in CELL_FILES:
            for i, s in enumerate(seeds_of(f)):
                cats, ia, ib, suffix = setup(s)
                ims, txs = man.manipulate(weights=man.zero_genotype()[None])
                stored = (s.metadata or {}).get("anchor_g_by_sut", {})
                out["anchors"].append({"file": f, "idx": i, "cell": (s.metadata or {}).get("cell_key"),
                                       "anchor_g": g_single(ims[0], txs[0] + suffix, cats, ia, ib),
                                       "anchor_g_stored": {k: v for k, v in stored.items()}})
        # 2. spread
        for f in SPREAD_FILES:
            for i, s in enumerate(seeds_of(f)):
                cats, ia, ib, suffix = setup(s)
                sp = GenotypeSpace.from_manipulator(man)
                smp = MagnitudeStratifiedSampling(sp, seed=a.seed + i)
                G = np.vstack([np.zeros(sp.n_var),
                               smp.do(Problem(n_var=sp.n_var, n_obj=1, xl=sp.xl, xu=sp.xu), a.n_edits).get("X")])
                alone = [man.manipulate(weights=G[j:j + 1]) for j in range(len(G))]
                I1 = [x[0][0] for x in alone]; T = [x[1][0] + suffix for x in alone]
                single = np.array([g_single(im, t, cats, ia, ib) for im, t in zip(I1, T)])
                runs = []
                for _ in range(a.perms):
                    p = rng.permutation(len(G))
                    ims, txs = man.manipulate(weights=G[p])          # rendered in a batch, as in the search
                    gb = np.empty(len(G)); gb[p] = g_batch(ims, [t + suffix for t in txs], cats, ia, ib)
                    runs.append(gb)
                p = rng.permutation(len(G))
                gb = np.empty(len(G)); gb[p] = g_batch([I1[j] for j in p], [T[j] for j in p], cats, ia, ib)
                allg = np.vstack([single[None], np.stack(runs), gb[None]])
                out["spread"].append({"file": f, "idx": i, "magnitude": sp.magnitude(G).tolist(),
                                      "g_single": single.tolist(),
                                      "spread": (allg.max(0) - allg.min(0)).tolist(),
                                      "max_dev_batch_vs_single": np.abs(np.stack(runs) - single).max(0).tolist(),
                                      "dev_scorer_only": np.abs(gb - single).tolist()})
        # 3. timing: one generation of pop on the first calibration cell
        s = seeds_of(SPREAD_FILES[0])[0]
        cats, ia, ib, suffix = setup(s)
        sp = GenotypeSpace.from_manipulator(man)
        G = MagnitudeStratifiedSampling(sp, seed=a.seed).do(
            Problem(n_var=sp.n_var, n_obj=1, xl=sp.xl, xu=sp.xu), a.pop).get("X")

        def batched():
            ims, txs = man.manipulate(weights=G)
            g_batch(ims, [t + suffix for t in txs], cats, ia, ib)

        def canonical():
            for j in range(len(G)):
                ims, txs = man.manipulate(weights=G[j:j + 1])
                g_single(ims[0], txs[0] + suffix, cats, ia, ib)

        for name, fn in (("batched", batched), ("canonical", canonical)):
            fn(); sync()                                            # warm-up
            ts = []
            for _ in range(2):
                t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
            out["timing"][name] = {"seconds_per_generation": float(np.median(ts)), "runs": ts}
        out["timing"]["ratio"] = out["timing"]["canonical"]["seconds_per_generation"] / out["timing"]["batched"]["seconds_per_generation"]
        out["timing"]["per_run_hours_1500_calls"] = {k: out["timing"][k]["seconds_per_generation"] * 30 / 3600
                                                     for k in ("batched", "canonical")}
    sp_all = np.concatenate([np.asarray(r["spread"]) for r in out["spread"]])
    dev_all = np.concatenate([np.asarray(r["max_dev_batch_vs_single"]) for r in out["spread"]])
    out["summary"] = {"n_inputs": int(sp_all.size), "spread_median": float(np.median(sp_all)),
                      "spread_p90": float(np.quantile(sp_all, 0.9)), "spread_max": float(sp_all.max()),
                      "dev_batch_vs_single_median": float(np.median(dev_all)),
                      "dev_batch_vs_single_p90": float(np.quantile(dev_all, 0.9)),
                      "dev_batch_vs_single_max": float(dev_all.max())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"summary": out["summary"], "timing": out["timing"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
