"""Expand the balanced experiment matrix into one runnable YAML per
(task, SUT, arm, rng seed). Each YAML runs all K cells of its task, so
every combination has the same cells, seeds and budget (METHODOLOGY.md s4).

  python scripts/make_matrix.py --exp E1 --tasks imagenet --suts llava \
      --arms all --rng-seeds 0 1 2 --out configs/generated
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_arm_configs import ARMS, merge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SUTS = {
    "llava": {"model_id": "llava-hf/llava-v1.6-mistral-7b-hf", "backend": "torch", "load_in_4bit": True},
    "qwen": {"model_id": "Qwen/Qwen3.5-4B", "backend": "torch"},
    "internvl": {"model_id": "OpenGVLab/InternVL3_5-4B-HF", "backend": "torch"},
    "ministral": {"model_id": "mistralai/Ministral-3-3B-Instruct-2512", "backend": "torch"},
}
for _s in SUTS.values():
    _s.update({"batched_scoring": False})   # canonical scoring (C2 decision 2026-11-28, docs/DETERMINISM_2026-11-28.md)
_UNUSED = {
}

TASKS = {
    "imagenet": {"base": "configs/paper/e1/base.yaml", "cells": "cells/imagenet_cells.yaml"},
    "pope": {"base": "configs/paper/e1/base.yaml", "cells": "cells/pope_cells.yaml"},
}


def build(exp: str, task: str, sut: str, arm: str, seed: int) -> dict:
    t = TASKS[task]
    cfg = yaml.safe_load((ROOT / t["base"]).read_text())
    cells = yaml.safe_load((ROOT / t["cells"]).read_text())
    cfg = merge(cfg, ARMS[arm])
    cfg["seeds"] = cells["seeds"]
    cfg["sut"] = copy.deepcopy(SUTS[sut])
    cfg["seed"] = seed
    cfg["name"] = f"{exp}_{task}_{sut}_{arm}_s{seed}"
    cfg["save_dir"] = f"runs/{exp}/{task}/{sut}/{arm}/s{seed}"
    cfg["tags"] = {"exp_id": exp, "task": task, "sut": sut, "arm": arm}
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--suts", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--rng-seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "generated")
    a = ap.parse_args()
    arms = list(ARMS) if a.arms == ["all"] else a.arms
    out = a.out / a.exp
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    # order: seed-major so a partial queue still yields complete paired blocks
    for seed in a.rng_seeds:
        for task in a.tasks:
            for sut in a.suts:
                for arm in arms:
                    cfg = build(a.exp, task, sut, arm, seed)
                    p = out / f"{seed:02d}__{task}__{sut}__{arm}.yaml"
                    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
                    n += 1
    print(f"wrote {n} configs to {out}")


if __name__ == "__main__":
    main()
