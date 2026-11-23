"""Backfill min_cross.png/json for runs made before the tester saved them.
Renders each run's smallest crossing (fewest edits; ties: smallest |g|) from
trace.parquet with the run's own image/text manipulators on the GPU (no VLM needed).
  python scripts/backfill_min_cross.py runs/MAIN
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from mmb.common.slot_items_seed_generator import build_seed_triples, load_item_image, resolve_item_image
from mmb.common.seed_context import seed_target_class
from mmb.evolutionary.vlm_boundary_tester import effective_prompt_template
from mmb.manipulator.image_factory import build_image_backend
from mmb.manipulator.text.composite import CompositeTextManipulator
from mmb.manipulator.vlm_manipulator import VLMManipulator
from mmb.runners.run_boundary_test import load_config

root = Path(sys.argv[1])
done = 0
cache = {}
for st_path in sorted(root.rglob("stats.json")):
    d = st_path.parent
    need_min = not (d / "min_cross.json").exists()
    need_nb = not (d / "min_cross_neighbour.json").exists()
    if not (need_min or need_nb) or not (d / "trace.parquet").exists():
        continue
    st = json.loads(st_path.read_text())
    meta = st.get("run_meta", {})
    cfgp = next(Path("configs/generated/MAIN").glob(
        f"*__s{meta['rng_seed']}__{meta['task']}__{meta['sut']}__{meta['arm']}.yaml"))
    cfg = yaml.safe_load(cfgp.read_text()); cfg["device"] = "cuda"
    exp = load_config(cfg)
    key = (meta["task"], meta["arm"])  # operators depend on task cells and modality only
    if key not in cache:
        img = build_image_backend(image_config=exp.image, device="cuda")
        txt = CompositeTextManipulator.from_config(exp.text, device="cpu")
        items = list(exp.seeds.slot_items.items)
        paths = [resolve_item_image(it.image) for it in items]
        seeds = build_seed_triples(items, [load_item_image(p) for p in paths], paths)
        cache[key] = (VLMManipulator(img, txt), seeds, exp)
    man, seeds, exp = cache[key]
    seed = seeds[int(st["seed_idx"])]
    prompt = effective_prompt_template(seed, exp)
    protect = (seed.metadata or {}).get("protect_words")
    man.prepare(seed.image, prompt, exclude_words=frozenset(w.lower() for w in protect) if protect else None,
                target_class=seed_target_class(seed), origin_class=seed.class_a)
    ia, ib = st["target_classes"]; sign0 = np.sign(st["anchor_g"]) or 1.0
    rows = pq.read_table(d / "trace.parquet", columns=["generation", "individual", "genotype", "logprobs"]).to_pylist()
    best = None
    for k, r in enumerate(rows):
        g = r["logprobs"][ia] - r["logprobs"][ib]
        if sign0 * g < 0:
            key2 = (sum(1 for v in r["genotype"] if v), abs(g), k)
            if best is None or key2 < best[0]:
                best = (key2, r, g, k)
    if best is None:
        continue
    _, r, g, k = best
    G = np.asarray(r["genotype"], dtype=np.int64)
    if len(G) != man.image_dim + man.text_dim:
        print("layout mismatch", d); continue
    # boundary partner from the trace only (no VLM here): C minus one edit, not crossing
    seen = {}
    for rr in rows:
        H = np.asarray(rr["genotype"], dtype=np.int64)
        dd = np.nonzero(H != G)[0]
        if len(dd) == 1 and H[dd[0]] == 0 and G[dd[0]] != 0:
            gg = rr["logprobs"][ia] - rr["logprobs"][ib]
            if sign0 * gg >= 0:
                seen[int(dd[0])] = gg
    nb = None
    if seen:
        i = min(seen, key=lambda j: abs(seen[j])); nb = G.copy(); nb[i] = 0
    imgs, txts = man.manipulate(candidates=None, weights=np.stack([G, G if nb is None else nb]))
    if nb is not None and need_nb:
        imgs[1].save(d / "min_cross_neighbour.png")
        (d / "min_cross_neighbour.json").write_text(json.dumps({"g": float(seen[i]), "n_edits": int((nb != 0).sum()),
            "text": txts[1], "genotype": nb.tolist(), "from_trace": True, "nb_extra_calls": 0, "backfilled": True}, indent=2))
    if not need_min:
        done += 1; continue
    imgs[0].save(d / "min_cross.png")
    suffix = exp.answer_format.format(categories=", ".join(st["categories"]))
    (d / "min_cross.json").write_text(json.dumps({
        "eval_idx": k + 1, "generation": r["generation"], "individual": r["individual"], "g": float(g),
        "n_edits": int((G != 0).sum()), "text": txts[0], "full_prompt": txts[0] + suffix,
        "genotype": G.tolist(), "backfilled": True}, indent=2))
    done += 1
print("backfilled", done)
