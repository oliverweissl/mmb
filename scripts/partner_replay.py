"""Score boundary partners for runs without one (any arm).

For each run with min_cross.json but no min_cross_neighbour.json: score every
'minimal crossing minus one edit' candidate with the run's SUT (batched, single GPU)
and keep the non-crossing one closest to the border. These calls are outside the
search budget and recorded in nb_extra_calls.
  python scripts/partner_replay.py runs/MAIN --sut qwen
"""
import argparse, json
from pathlib import Path
import numpy as np, yaml

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", type=Path); ap.add_argument("--sut", required=True)
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
    for mc in sorted(a.runs.rglob("min_cross.json")):
        d = mc.parent
        if (d / "min_cross_neighbour.json").exists():
            continue
        st = json.loads((d / "stats.json").read_text()); m = st.get("run_meta", {})
        if m.get("sut") == a.sut:
            todo.append((d, st))
    print(len(todo), "runs without partner")
    if not todo:
        return
    comps, cache, done = None, {}, 0
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
        prompt = effective_prompt_template(seed, exp)
        protect = (seed.metadata or {}).get("protect_words")
        man.prepare(seed.image, prompt, exclude_words=frozenset(w.lower() for w in protect) if protect else None,
                    target_class=seed_target_class(seed), origin_class=seed.class_a)
        mc = json.loads((d / "min_cross.json").read_text())
        G = np.asarray(mc["genotype"], dtype=np.int64)
        act = np.nonzero(G)[0]
        if len(act) == 0 or len(G) != man.image_dim + man.text_dim:
            continue
        cands = []
        for i in act:
            H = G.copy(); H[i] = 0; cands.append(H)
        Hs = np.stack(cands if len(cands) > 1 else cands * 2)
        ims, txs = man.manipulate(candidates=None, weights=Hs)
        cats = tuple(st["categories"]); suffix = exp.answer_format.format(categories=", ".join(cats))
        ia, ib = st["target_classes"]; s0 = np.sign(st["anchor_g"]) or 1.0
        gs = []
        for j in range(0, len(cands), 15):
            lp = comps.sut.process_batch(ims[j:j + 15], [t + suffix for t in txs[j:j + 15]], cats)
            gs += list((lp[:, ia] - lp[:, ib]).float().numpy())
        ok = [j for j in range(len(cands)) if s0 * gs[j] >= 0]
        if not ok:
            continue
        j = min(ok, key=lambda j: abs(gs[j]))
        ims[j].save(d / "min_cross_neighbour.png")
        (d / "min_cross_neighbour.json").write_text(json.dumps({"g": float(gs[j]), "n_edits": int((cands[j] != 0).sum()),
            "text": txs[j], "full_prompt": txs[j] + suffix, "genotype": cands[j].tolist(), "from_trace": False,
            "nb_extra_calls": len(cands), "replayed": True}, indent=2))
        done += 1
    print("partners", done)

if __name__ == "__main__":
    main()
