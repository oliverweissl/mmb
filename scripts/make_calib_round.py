#!/usr/bin/env python
"""Calibration round configs (continuous design, 2026-11-28).

One YAML per (task, SUT, arm) on the HELD-OUT calibration cells only (cells/<task>_calib.yaml + <task>_calib2.yaml,
never the MAIN cells), rng seed 0, full budget, canonical scoring: everything as configs/paper/e1/base.yaml
(+ the arm overrides of scripts/make_arm_configs.py) except the two weights:

    image.latent.w_img                          (image weight)
    text.composite.overrides.question_pool.w_txt (text weight)

    python scripts/make_calib_round.py --w-img 1.0 --w-txt 1.0
        -> configs/generated/CALIB/w1.0_1.0/<task>__<sut>__<arm>.yaml, runs in runs/CALIB/w1.0_1.0/<task>/<sut>/<arm>/s0
"""
import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_arm_configs import ARMS, merge  # noqa: E402
from make_matrix import SUTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("imagenet", "pope")
CALIB_ARMS = ("a6_text_only", "a7_image_only", "a0_directed")
CALIB_SUTS = ("qwen", "internvl", "ministral")


def round_name(w_img: float, w_txt: float) -> str:
    return f"w{w_img:g}_{w_txt:g}" if (w_img % 1 or w_txt % 1) else f"w{w_img:.1f}_{w_txt:.1f}"


def calib_cells(task: str) -> dict:
    items = []
    for f in (f"cells/{task}_calib.yaml", f"cells/{task}_calib2.yaml"):
        items += yaml.safe_load((ROOT / f).read_text())["seeds"]["slot_items"]["items"]
    for it in items:
        assert (it.get("meta") or {}).get("calibration_holdout", False), f"{task}: non-held-out cell {it['image']}"
    return {"mode": "slot_items", "slot_items": {"items": items}}


def build(task: str, sut: str, arm: str, w_img: float, w_txt: float, seed: int = 0) -> dict:
    rnd = round_name(w_img, w_txt)
    cfg = yaml.safe_load((ROOT / "configs/paper/e1/base.yaml").read_text())
    cfg = merge(cfg, ARMS[arm])
    cfg["seeds"] = calib_cells(task)
    cfg["sut"] = copy.deepcopy(SUTS[sut])
    cfg["seed"] = seed
    cfg.setdefault("image", {}).setdefault("latent", {})["w_img"] = float(w_img)
    comp = cfg.setdefault("text", {}).setdefault("composite", {})
    comp.setdefault("overrides", {}).setdefault("question_pool", {})["w_txt"] = float(w_txt)
    cfg["name"] = f"CALIB_{rnd}_{task}_{sut}_{arm}_s{seed}"
    cfg["save_dir"] = f"runs/CALIB/{rnd}/{task}/{sut}/{arm}/s{seed}"
    cfg["tags"] = {"exp_id": "CALIB", "round": rnd, "task": task, "sut": sut, "arm": arm,
                   "w_img": f"{w_img:g}", "w_txt": f"{w_txt:g}"}
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-img", type=float, required=True); ap.add_argument("--w-txt", type=float, required=True)
    ap.add_argument("--suts", nargs="+", default=list(CALIB_SUTS))
    a = ap.parse_args()
    rnd = round_name(a.w_img, a.w_txt)
    out = ROOT / "configs/generated/CALIB" / rnd
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.yaml"):
        old.unlink()
    n = 0
    for sut in a.suts:                 # SUT-major (run order = file order: scripts/run_queue.sh sorts by name)
        for task in TASKS:
            for arm in CALIB_ARMS:
                p = out / f"{n:02d}__{sut}__{task}__{arm}.yaml"
                p.write_text(yaml.safe_dump(build(task, sut, arm, a.w_img, a.w_txt), sort_keys=False,
                                            allow_unicode=True))
                n += 1
    print(f"wrote {n} configs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
