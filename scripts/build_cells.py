"""Build fixed, SUT-independent cell lists for the balanced design (EXP_SETUP.md).

Every task x SUT x arm runs the SAME K cells (balanced-design rule, METHODOLOGY.md s4).
A cell is one (image, question, candidate pair) item in the slot_items seed format, so
the same YAML block feeds every SUT. Steps:

  pool     build a candidate pool for a task           (no GPU)
  score    score the VQGAN reconstruction of every pool item with one SUT (GPU)
  select   keep items every listed SUT answers as the anchor with margin >= tau,
           then draw K items stratified, with a fixed seed   (no GPU)

  python scripts/build_cells.py pool imagenet --out cells/imagenet_pool.yaml
  python scripts/build_cells.py pool pope --out cells/pope_pool.yaml
  python scripts/build_cells.py score cells/imagenet_pool.yaml --sut llava-hf/llava-v1.6-mistral-7b-hf --int8
  python scripts/build_cells.py select cells/imagenet_pool.yaml --k 24 --tau 1.0 --out cells/imagenet_K24.yaml
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

DATA = Path(os.environ.get("MMB_DATA", Path.home() / "data"))
ROSTER = ["junco", "ostrich", "green iguana", "boa constrictor", "cello", "marimba"]
PROMPT_IMAGENET = "What is the main subject in this image?"
ANSWER_FORMAT = " Answer with exactly one of these options: {categories}."


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _tilde(p: Path) -> str:
    """Home-relative path (no user name in committed cell lists)."""
    return str(p).replace(str(Path.home()), "~", 1)


def _imagenet_labels() -> list[str]:
    from mmb.data.imagenet import ImageNetCache
    return list(ImageNetCache(dirs=[Path.home() / ".cache" / "imagenet"]).labels())


# --------------------------------------------------------------------------- pool
def pool_imagenet(per_class: int) -> list[dict]:
    """Roster x combinatorial cells (thesis Exp-100 design), photographs from the
    local ImageNet-val parquet snapshot, NOT filtered by any SUT."""
    import pyarrow.parquet as pq
    from PIL import Image

    from mmb.common.combinatorial_pair_generator import combinatorial_pairs
    from mmb.common.roster_seed_generator import SeedImage
    from mmb.config import AbstractionConfig

    labels = _imagenet_labels()
    want = {labels.index(c): c for c in ROSTER}
    got: dict[str, list[Path]] = defaultdict(list)
    out_dir = DATA / "cells" / "imagenet"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(DATA / "imagenet-val-hf" / "data" / "validation-*.parquet")))
    for f in files:
        t = pq.read_table(f, columns=["label"])
        idx = [i for i, lab in enumerate(t.column("label").to_pylist()) if lab in want]
        if not idx:
            continue
        rows = pq.read_table(f).take(idx).to_pylist()
        for r in rows:
            c = want[r["label"]]
            if len(got[c]) >= per_class:
                continue
            p = out_dir / f"{_safe(c)}_{len(got[c]):02d}.png"
            if not p.exists():
                Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(p)
            got[c].append(p)
        if all(len(got[c]) >= per_class for c in ROSTER):
            break
    seeds = [SeedImage(image=None, class_concrete=c, seed_idx_in_class=i)
             for c in ROSTER for i in range(len(got[c]))]
    triples = combinatorial_pairs(
        seeds, ROSTER,
        AbstractionConfig(levels_anchor=(0, 1, 2), levels_target=(0, 1, 2),
                          apply_disjointness=True, directions="both"),
    )
    items = []
    for t in triples:
        m = dict(t.metadata)
        img = got[m["anchor_class_concrete"]][m["seed_idx_in_class"]]
        key = (f"{m['anchor_class_concrete']}|{m['target_class_concrete']}|"
               f"La{m['level_anchor']}-Lt{m['level_target']}|p{m['seed_idx_in_class']}")
        items.append({
            "image": _tilde(img), "prompt": PROMPT_IMAGENET, "template": "{slot}",
            "fillers": [t.class_a, t.class_b], "pair": [t.class_a, t.class_b],
            "meta": {**m, "task": "imagenet", "cell_key": key},
        })
    return items


def pool_pope(n: int, seed: int) -> list[dict]:
    """POPE (adversarial split, COCO val2014): balanced yes/no, one item per image."""
    import pyarrow.parquet as pq
    from PIL import Image

    rows = []
    for f in sorted(glob.glob(str(DATA / "pope" / "data" / "*.parquet"))):
        rows += [r for r in pq.read_table(f).to_pylist() if r["category"] == "adversarial"]
    rng = random.Random(seed)
    rng.shuffle(rows)
    seen, items, cnt = set(), [], {"yes": 0, "no": 0}
    out_dir = DATA / "cells" / "pope"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        a = r["answer"].strip().lower()
        if a not in cnt or cnt[a] >= n // 2 or r["image_source"] in seen:
            continue
        seen.add(r["image_source"])
        cnt[a] += 1
        p = out_dir / f"{r['image_source']}.jpg"
        if not p.exists():
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(p, quality=95)
        gt, other = ("Yes", "No") if a == "yes" else ("No", "Yes")
        obj = re.sub(r"^Is there an? (.*) in the image\?$", r"\1", r["question"].strip())
        items.append({
            "image": _tilde(p), "prompt": r["question"].strip(), "template": "{slot}",
            "fillers": [gt, other], "pair": [gt, other],
            "meta": {"task": "pope", "pope_question_id": r["question_id"], "gt": a,
                     "object": obj, "image_source": r["image_source"],
                     "cell_key": f"pope|{r['image_source']}|{obj}|{a}"},
        })
        if len(items) >= n:
            break
    return items


# -------------------------------------------------------------------------- score
def score(pool_path: Path, sut: str, int8: bool, device: str) -> Path:
    """g = l(a) - l(b) of the VQGAN-f8 reconstruction (what the search sees at
    genotype 0) for every pool item, with the exact prompt + answer suffix."""
    from PIL import Image

    from mmb.manipulator.image.codec import VQGANCodec
    from mmb.manipulator.image.loading import load_vqgan
    from mmb.sut.scorer import create_scorer

    items = yaml.safe_load(pool_path.read_text())["items"]
    codec = VQGANCodec(load_vqgan("f8-16384"), device=device)
    sc = create_scorer(sut, device, enable_thinking=False, load_in_8bit=int8)
    out = pool_path.with_name(pool_path.stem + f".scores.{_safe(sut)}.csv")
    cache: dict[str, Image.Image] = {}
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_key", "sut", "lp_a", "lp_b", "g", "argmax"])
        for it in items:
            if it["image"] not in cache:
                im = Image.open(Path(it["image"]).expanduser()).convert("RGB")
                cache[it["image"]] = codec.decode(codec.encode(im))
            img = cache[it["image"]]
            cands = it["fillers"]
            prompt = it["prompt"] + ANSWER_FORMAT.format(categories=", ".join(cands))
            res = {lbl: lpn for (lbl, _lp, lpn, _n) in sc.score_categories(img, prompt, cands)}
            a, b = it["pair"]
            w.writerow([it["meta"]["cell_key"], sut, res[a], res[b], res[a] - res[b],
                        max(res, key=res.get)])
    return out


# ------------------------------------------------------------------------- select
def select(pool_path: Path, k: int, tau: float, seed: int, out: Path) -> None:
    items = yaml.safe_load(pool_path.read_text())["items"]
    score_files = sorted(pool_path.parent.glob(pool_path.stem + ".scores.*.csv"))
    if not score_files:
        raise SystemExit("no score files: run the score step for every SUT first")
    ok: dict[str, list[float]] = defaultdict(list)
    for f in score_files:
        for r in csv.DictReader(open(f)):
            ok[r["cell_key"]].append(float(r["g"]))
    n_sut = len(score_files)
    keep = [it for it in items
            if len(ok[it["meta"]["cell_key"]]) == n_sut and min(ok[it["meta"]["cell_key"]]) >= tau]
    strata: dict[str, list[dict]] = defaultdict(list)
    for it in keep:
        m = it["meta"]
        strata[m.get("gt") or m["anchor_class_concrete"]].append(it)
    rng = random.Random(seed)
    for v in strata.values():
        rng.shuffle(v)
    chosen, keys = [], sorted(strata)
    while len(chosen) < k and any(strata[s] for s in keys):
        for s in keys:
            if strata[s] and len(chosen) < k:
                chosen.append(strata[s].pop())
    if len(chosen) < k:
        raise SystemExit(f"only {len(chosen)} items pass tau={tau} on all {n_sut} SUTs (need {k})")
    for i, it in enumerate(chosen):
        it["meta"]["cell_idx"] = i
        it["meta"]["anchor_g_by_sut"] = {
            Path(f).name.split(".scores.")[1][:-4]: g
            for f, g in zip(score_files, ok[it["meta"]["cell_key"]])}
    out.write_text(yaml.safe_dump({"seeds": {"mode": "slot_items", "slot_items": {"items": chosen}}},
                                  sort_keys=False, allow_unicode=True))
    print(f"{len(keep)}/{len(items)} pass tau={tau} on {n_sut} SUTs; wrote {k} cells to {out}")
    print({s: sum(1 for c in chosen if (c["meta"].get("gt") or c["meta"]["anchor_class_concrete"]) == s)
           for s in keys})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("task", choices=["imagenet", "pope"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-class", type=int, default=4)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261120)
    s = sub.add_parser("score")
    s.add_argument("pool", type=Path)
    s.add_argument("--sut", required=True)
    s.add_argument("--int8", action="store_true")
    s.add_argument("--device", default="cuda")
    q = sub.add_parser("select")
    q.add_argument("pool", type=Path)
    q.add_argument("--k", type=int, required=True)
    q.add_argument("--tau", type=float, default=1.0)
    q.add_argument("--seed", type=int, default=20261120)
    q.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "pool":
        items = pool_imagenet(a.per_class) if a.task == "imagenet" else pool_pope(a.n, a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(yaml.safe_dump({"items": items}, sort_keys=False, allow_unicode=True))
        print(f"wrote {len(items)} pool items to {a.out}")
    elif a.cmd == "score":
        print(score(a.pool, a.sut, a.int8, a.device))
    else:
        select(a.pool, a.k, a.tau, a.seed, a.out)


if __name__ == "__main__":
    main()
