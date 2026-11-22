"""Minimise crossings and classify what flips the decision (contraction study).

For every run with a first crossing, reduce the crossing genotype to a 1-minimal
flipping perturbation: repeatedly try resetting a chunk of the changed genes to the
unedited value (gene 0), keep the reset if the input still crosses (sign of
g = l(a) - l(b) opposite to the anchor), halve the chunk size when no chunk can be
reset, and stop when no single gene can be reset (ddmin-style; capped by --max-calls).
Then classify the minimal perturbation by channel and operator.

  python scripts/minimize_crossings.py runs/MAIN --sut qwen --out results/minimal_MAIN_qwen.csv
One SUT per invocation (16 GB GPU). Scoring is batched full-sequence teacher forcing.
"""
from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path

import numpy as np

ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿"}


def classify_text(orig: str, new: str) -> dict:
    """Operator-level description of a text edit from the two strings."""
    from mmb.validity.proxies import normalise_visible
    zw = sum(ch in ZERO_WIDTH for ch in new) - sum(ch in ZERO_WIDTH for ch in orig)
    nz = "".join(ch for ch in new if ch not in ZERO_WIDTH)
    homo = sum(1 for ch in nz if ord(ch) > 127 and not unicodedata.category(ch).startswith("P"))
    ow, nw = normalise_visible(orig).split(), normalise_visible(nz).split()
    changed = [(a, b) for a, b in zip(ow, nw) if a != b]
    typo = sum(1 for a, b in changed if len(a) == len(b) or abs(len(a) - len(b)) <= 1)
    syn = len(changed) - typo + abs(len(ow) - len(nw))
    return {"txt_zero_width": max(zw, 0), "txt_homoglyph": homo, "txt_typo_words": typo,
            "txt_synonym_words": syn, "txt_visible_identical": normalise_visible(nz) == normalise_visible(orig)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--sut", required=True)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-calls", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0, help="crossing: sign reversed and |g| > tau (default 0: any behaviour change)")
    a = ap.parse_args()

    from mmb.common.pipeline_bootstrap import init_shared_components, prepare_pipeline_seeds
    from mmb.common.seed_context import seed_target_class
    from mmb.data import ImageNetCache
    from mmb.evolutionary.vlm_boundary_tester import effective_candidates, effective_prompt_template
    from mmb.manipulator.vlm_manipulator import VLMManipulator
    from mmb.runners.run_boundary_test import _disable_hub_kernels, load_config

    runs = []
    for fc in sorted(a.runs.rglob("first_cross.json")):
        st = json.loads((fc.parent / "stats.json").read_text())
        meta = st.get("run_meta", {})
        if meta.get("sut") != a.sut or (a.arms and meta.get("arm") not in a.arms):
            continue
        runs.append((fc.parent, st))
        if a.limit and len(runs) >= a.limit:
            break
    if not runs:
        print("no crossing runs"); return
    if a.sut != "ministral":
        _disable_hub_kernels()
    # one component set per SUT; configs of one SUT share model + operators
    by_task: dict[str, list] = {}
    for d, st in runs:
        by_task.setdefault(st["run_meta"]["task"], []).append((d, st))
    rows = []
    comps = None
    for task, items in by_task.items():
        m0 = items[0][1]["run_meta"]
        cfg_path = next(Path("configs/generated/MAIN").glob(
            f"*__s{m0['rng_seed']}__{m0['task']}__{m0['sut']}__{m0['arm']}.yaml"))
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()); cfg["device"] = a.device
        exp = load_config(cfg)
        if comps is None:
            comps = init_shared_components(exp, ImageNetCache(dirs=exp.cache_dirs))
        seeds = prepare_pipeline_seeds(comps, exp)
        man = VLMManipulator(comps.image_manip, comps.text_manip)
        for d, st in items:
            seed = seeds[int(st["seed_idx"])]
            cands = effective_candidates(seed) or tuple(st["categories"])
            suffix = exp.answer_format.format(categories=", ".join(cands))
            prompt = effective_prompt_template(seed, exp)
            protect = (seed.metadata or {}).get("protect_words")
            man.prepare(seed.image, prompt, exclude_words=frozenset(w.lower() for w in protect) if protect else None,
                        target_class=seed_target_class(seed), origin_class=seed.class_a)
            ctx0 = json.loads((d / "context.json").read_text())
            if man.image_dim != len(ctx0.get("image_patch_positions") or []) or \
                    man.text_dim != int(sum(ctx0.get("text_op_gene_dims") or [])):
                rows.append({"run_dir": str(d), "task": task, "sut": a.sut, "arm": st["run_meta"]["arm"],
                             "verified": False, "layout_mismatch": True}); continue
            ia, ib = st["target_classes"]
            sign0 = np.sign(st["anchor_g"])
            calls = 0

            def g_of(G):
                nonlocal calls
                G = np.atleast_2d(G)
                if len(G) == 1:
                    G = np.vstack([G, G])
                    imgs, txts = man.manipulate(candidates=None, weights=G); imgs, txts = imgs[:1], txts[:1]
                else:
                    imgs, txts = man.manipulate(candidates=None, weights=G)
                lp = comps.sut.process_batch(imgs, [t + suffix for t in txts], tuple(cands))
                calls += len(imgs)
                return (lp[:, ia] - lp[:, ib]).float().numpy()

            import pyarrow.parquet as pq
            tr = pq.read_table(d / "trace.parquet", columns=["genotype", "logprobs"]).to_pylist()
            robust = [r for r in tr if np.sign(r["logprobs"][ia] - r["logprobs"][ib]) == -sign0
                      and abs(r["logprobs"][ia] - r["logprobs"][ib]) > a.tau]
            if not robust:
                rows.append({"run_dir": str(d), "task": task, "sut": a.sut, "arm": st["run_meta"]["arm"],
                             "verified": False, "robust_in_trace": False}); continue
            # start from the run's smallest crossing (MMBA: its final minimal crossing)
            best = min(robust, key=lambda r: sum(1 for v in r["genotype"] if v))
            G = np.asarray(best["genotype"], dtype=np.int64)
            n0 = int((G != 0).sum())

            def save_input(Gv, stem, extra=None):
                """Render a genotype and store it as <stem>.png/.json (same fields as first_cross.json)."""
                imgs, txts = man.manipulate(candidates=None, weights=np.vstack([Gv, Gv]))
                imgs[0].save(d / f"{stem}.png")
                gv = float(g_of(Gv)[0])
                (d / f"{stem}.json").write_text(json.dumps(dict(
                    text=txts[0], full_prompt=txts[0] + suffix, g=gv, genotype=[int(x) for x in Gv], **(extra or {}))))

            def is_cross(gv):
                return np.sign(gv) == -sign0 and abs(gv) > a.tau

            if not is_cross(g_of(G)[0]):   # does not re-verify as a robust crossing
                rows.append({"run_dir": str(d), "task": task, "sut": a.sut, "arm": st["run_meta"]["arm"],
                             "verified": False, "n_genes_first": n0}); continue
            # the first ROBUST crossing input: the validity proxies and the gallery read it when the first sign flip was weaker
            fcj = d / "first_cross.json"
            if not (fcj.exists() and abs(json.loads(fcj.read_text()).get("g", 0.0)) >= a.tau):
                idx_r = next(i for i, r in enumerate(tr) if r["genotype"] == robust[0]["genotype"])
                save_input(G, "first_cross_robust", dict(eval_idx=idx_r + 1))
            active = list(np.nonzero(G)[0]); chunk = max(1, len(active) // 2)
            while chunk >= 1 and calls < a.max_calls and active:
                groups = [active[i:i + chunk] for i in range(0, len(active), chunk)]
                cand = []
                for grp in groups:
                    H = G.copy(); H[grp] = 0; cand.append(H)
                gs = []
                for i in range(0, len(cand), 15):
                    gs.extend(g_of(np.stack(cand[i:i + 15])))
                ok = [j for j, gv in enumerate(gs) if is_cross(gv)]
                if ok:
                    j = ok[0]; G = cand[j]; active = list(np.nonzero(G)[0])
                    chunk = max(1, min(chunk, len(active) // 2)) if len(active) > 1 else 1
                    if len(active) <= 1:
                        break
                elif chunk == 1:
                    break
                else:
                    chunk //= 2
            gmin = float(g_of(G)[0])
            save_input(G, "minimal_cross")
            n_img_dim = man.image_dim
            img_idx = [int(i) for i in np.nonzero(G[:n_img_dim])[0]]
            txt = man.manipulate(candidates=None, weights=np.vstack([G, G]))[1][0]
            ctx = json.loads((d / "context.json").read_text())
            pos = [ctx["image_patch_positions"][i] for i in img_idx] if ctx.get("image_patch_positions") else []
            row = {"run_dir": str(d), "task": task, "sut": a.sut, "arm": st["run_meta"]["arm"],
                   "rng_seed": st["run_meta"].get("rng_seed"), "verified": True, "n_genes_first": n0,
                   "n_genes_min": int((G != 0).sum()), "img_genes_min": len(img_idx),
                   "txt_genes_min": int((G[n_img_dim:] != 0).sum()), "g_min": gmin, "calls": calls,
                   "img_patch_positions": json.dumps(pos), "text_min": txt, "prompt": prompt,
                   "genotype_min": json.dumps(G.tolist())}
            row.update(classify_text(prompt, txt))
            row["channel"] = ("joint" if row["img_genes_min"] and row["txt_genes_min"] else
                              "image" if row["img_genes_min"] else "text")
            rows.append(row)
            print(f"{d.name}: {n0} -> {row['n_genes_min']} genes ({row['channel']}), {calls} calls")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
