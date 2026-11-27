#!/usr/bin/env python
"""Held-out calibration cells: seed inputs from cells/<task>_pool.yaml that pass the same eligibility rule as the test
cells (every scored SUT answers the anchor with margin >= tau) but are NOT test cells and do not share a test cell's
image. ImageNet: distinct anchor classes; POPE: half gt=yes, half gt=no. Protected words as in the test cells.

    python scripts/build_calib_cells.py --k 4 --seed 20261127
writes cells/imagenet_calib.yaml and cells/pope_calib.yaml.
"""
import argparse, csv, random
from collections import defaultdict
from pathlib import Path

import yaml


def scores(pool: Path) -> tuple[dict, list]:
    files = sorted(pool.parent.glob(pool.stem + ".scores.*.csv"))
    g = defaultdict(list)
    for f in files:
        for r in csv.DictReader(open(f)):
            g[r["cell_key"]].append(float(r["g"]))
    return g, files


def build(task: str, k: int, tau: float, seed: int, name: str = "calib", exclude_sets=()) -> Path:
    pool = Path(f"cells/{task}_pool.yaml")
    items = yaml.safe_load(pool.read_text())["items"]
    test = yaml.safe_load(Path(f"cells/{task}_cells.yaml").read_text())["seeds"]["slot_items"]["items"]
    test_keys = {t["meta"]["cell_key"] for t in test}
    test_imgs = {t["image"] for t in test}
    for prev in exclude_sets:                       # earlier calibration sets: disjoint as well
        pp = Path(f"cells/{task}_{prev}.yaml")
        if pp.exists():
            for t in yaml.safe_load(pp.read_text())["seeds"]["slot_items"]["items"]:
                test_keys.add(t["meta"]["cell_key"]); test_imgs.add(t["image"])
    test_protect = {t["meta"]["cell_key"]: t["meta"].get("protect_words") for t in test}
    g, files = scores(pool)
    keep = [it for it in items if it["meta"]["cell_key"] not in test_keys and it["image"] not in test_imgs
            and len(g[it["meta"]["cell_key"]]) == len(files) and min(g[it["meta"]["cell_key"]]) >= tau]
    rng = random.Random(seed)
    rng.shuffle(keep)
    chosen, seen = [], set()
    if task == "imagenet":
        for it in keep:
            c = it["meta"]["anchor_class_concrete"]
            if c not in seen and len(chosen) < k:
                chosen.append(it); seen.add(c)
    else:
        for want in ("yes", "no"):
            for it in keep:
                if it["meta"]["gt"] == want and it["image"] not in seen and sum(c["meta"]["gt"] == want for c in chosen) < k // 2:
                    chosen.append(it); seen.add(it["image"])
    if len(chosen) < k:
        raise SystemExit(f"{task}: only {len(chosen)} held-out items")
    for i, it in enumerate(chosen):
        m = it["meta"]
        m["cell_idx"] = i
        m["anchor_g_by_sut"] = {Path(f).name.split(".scores.")[1][:-4]: v for f, v in zip(files, g[m["cell_key"]])}
        m["protect_words"] = ["subject", "image"] if task == "imagenet" else [m["object"], "image"]
        m["calibration_holdout"] = True
    out = Path(f"cells/{task}_{name}.yaml")
    out.write_text(yaml.safe_dump({"seeds": {"mode": "slot_items", "slot_items": {"items": chosen}}},
                                  sort_keys=False, allow_unicode=True))
    print(task, f"{len(keep)} held-out eligible; chose", [it["meta"]["cell_key"] for it in chosen])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--name", default="calib", help="output cells/<task>_<name>.yaml")
    ap.add_argument("--exclude", nargs="*", default=[], help="earlier calibration sets to keep disjoint (e.g. calib)")
    a = ap.parse_args()
    for t in ("imagenet", "pope"):
        build(t, a.k, a.tau, a.seed, a.name, a.exclude)


if __name__ == "__main__":
    main()
