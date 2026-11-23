"""Confirm each run's minimal crossing under single-input scoring (all arms alike).

Search scoring is batched; for crossings whose margin is at the numeric resolution of
bf16 logits, the sign can depend on the batch composition. The reported minimal
crossing is therefore the smallest trace crossing (fewest edits; ties: larger margin)
whose decision change reproduces when the input is scored alone. Its boundary partner
is the 'minus one edit' input that does not cross when scored alone and lies closest
to the border. Extra calls are recorded (confirm_calls, nb_extra_calls).
The search output is kept as min_cross_search.json.
  python scripts/confirm_crossings.py runs/MAIN --sut qwen
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np, yaml, pyarrow.parquet as pq

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", type=Path); ap.add_argument("--sut", required=True)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--max-tries", type=int, default=25)
    a = ap.parse_args()
    from mmb.common.pipeline_bootstrap import init_shared_components
    from mmb.common.slot_items_seed_generator import build_seed_triples, load_item_image, resolve_item_image
    from mmb.common.seed_context import seed_target_class
    from mmb.data import ImageNetCache
    from mmb.evolutionary.vlm_boundary_tester import effective_prompt_template
    from mmb.manipulator.image_factory import build_image_backend
    from mmb.manipulator.text.composite import CompositeTextManipulator
    from mmb.manipulator.vlm_manipulator import VLMManipulator
    from mmb.runners.run_boundary_test import _disable_hub_kernels, load_config
    if a.sut != "ministral":
        _disable_hub_kernels()
    todo = []
    for st_p in sorted(a.runs.rglob("stats.json")):
        d = st_p.parent
        if not (d / "trace.parquet").exists() or (not a.force and (d / "min_cross_search.json").exists()):
            continue
        st = json.loads(st_p.read_text())
        if st.get("run_meta", {}).get("sut") == a.sut and "anchor_g" in st:
            todo.append((d, st))
    print(len(todo), "runs to confirm")
    comps, cache, n_ok, n_none = None, {}, 0, 0
    for d, st in todo:
        m = st["run_meta"]
        cfgp = next(Path("configs/generated/MAIN").glob(f"*__s{m['rng_seed']}__{m['task']}__{m['sut']}__{m['arm']}.yaml"))
        cfg = yaml.safe_load(cfgp.read_text()); cfg["device"] = "cuda"
        exp = load_config(cfg)
        if comps is None:
            comps = init_shared_components(exp, ImageNetCache(dirs=exp.cache_dirs))
        key = (m["task"], m["arm"])
        if key not in cache:
            img = build_image_backend(image_config=exp.image, device="cuda")
            txt = CompositeTextManipulator.from_config(exp.text, device="cpu")
            items = list(exp.seeds.slot_items.items)
            paths = [resolve_item_image(it.image) for it in items]
            cache[key] = (VLMManipulator(img, txt), build_seed_triples(items, [load_item_image(p) for p in paths], paths), exp)
        man, seeds, exp = cache[key]
        seed = seeds[int(st["seed_idx"])]
        protect = (seed.metadata or {}).get("protect_words")
        man.prepare(seed.image, effective_prompt_template(seed, exp),
                    exclude_words=frozenset(w.lower() for w in protect) if protect else None,
                    target_class=seed_target_class(seed), origin_class=seed.class_a)
        cats = tuple(st["categories"]); suffix = exp.answer_format.format(categories=", ".join(cats))
        ia, ib = st["target_classes"]; s0 = np.sign(st["anchor_g"]) or 1.0
        rows = pq.read_table(d / "trace.parquet", columns=["genotype", "logprobs", "decoded_text"]).to_pylist()
        text_of = {tuple(r["genotype"]): r["decoded_text"] for r in rows}
        nd = man.image_dim

        def render(Gs):
            """Images from the genotype; texts from the trace (exact scored text)."""
            Gs = [np.asarray(G, dtype=np.int64) for G in Gs]
            ims, txs = man.manipulate(candidates=None, weights=np.stack(Gs if len(Gs) > 1 else Gs * 2))
            return ims[:len(Gs)], txs[:len(Gs)]

        def single(im, t):
            lp = comps.sut.process_batch([im], [t + suffix], cats)
            return float(s0 * (lp[0, ia] - lp[0, ib]))

        cross = []
        for k, r in enumerate(rows):
            g = r["logprobs"][ia] - r["logprobs"][ib]
            if s0 * g < 0:
                cross.append((sum(1 for v in r["genotype"] if v), -abs(g), k))
        cross.sort()
        mcp = d / "min_cross.json"
        if mcp.exists() and not (d / "min_cross_search.json").exists():
            shutil.copy(mcp, d / "min_cross_search.json")
        for p in ("min_cross.json", "min_cross.png", "min_cross_neighbour.json", "min_cross_neighbour.png"):
            (d / p).unlink(missing_ok=True)
        if not cross:
            (d / "min_cross_search.json").write_text("null") if not (d / "min_cross_search.json").exists() else None
            continue
        calls, chosen = 0, None
        for n, _, k in cross[: a.max_tries]:
            G = np.asarray(rows[k]["genotype"], dtype=np.int64)
            ims, _ = render([G]); t = rows[k]["decoded_text"]
            ms = single(ims[0], t); calls += 1
            if ms < 0:
                chosen = (k, G, ims[0], t, ms); break
        if chosen is None:
            n_none += 1
            (d / "confirm.json").write_text(json.dumps({"confirmed": False, "confirm_calls": calls, "n_trace_crossings": len(cross)}))
            continue
        k, G, im, t, ms = chosen
        r = rows[k]
        im.save(d / "min_cross.png")
        mcp.write_text(json.dumps({"eval_idx": k + 1, "generation": r["generation"] if "generation" in r else None,
            "g": float(s0 * ms), "g_search": float(r["logprobs"][ia] - r["logprobs"][ib]), "n_edits": int((G != 0).sum()),
            "text": t, "full_prompt": t + suffix, "genotype": G.tolist(), "confirmed": True, "confirm_calls": calls}, indent=2))
        # partner: C minus one edit, scored alone, non-crossing and closest to the border
        same_render = render([G])[1][0] == t
        cands, texts = [], []
        for i in np.nonzero(G)[0]:
            H = G.copy(); H[i] = 0
            if i < nd:
                tt = t
            elif tuple(H.tolist()) in text_of:
                tt = text_of[tuple(H.tolist())]
            elif same_render:
                tt = None
            else:
                continue
            cands.append(H); texts.append(tt)
        best = None
        if cands:
            ims, txs = render(cands)
            for H, im2, tt, tr in zip(cands, ims, texts, txs):
                tt = tr if tt is None else tt
                g2 = single(im2, tt)
                if g2 >= 0 and (best is None or g2 < best[0]):
                    best = (g2, H, im2, tt)
        if best is not None:
            g2, H, im2, tt = best
            im2.save(d / "min_cross_neighbour.png")
            (d / "min_cross_neighbour.json").write_text(json.dumps({"g": float(s0 * g2), "n_edits": int((H != 0).sum()),
                "text": tt, "full_prompt": tt + suffix, "genotype": H.tolist(), "from_trace": False,
                "nb_extra_calls": len(cands), "confirmed": True}, indent=2))
        n_ok += 1
    print(f"confirmed {n_ok}, none confirmed {n_none}")

if __name__ == "__main__":
    main()
