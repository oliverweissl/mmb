#!/usr/bin/env python
"""Operator calibration configs (T-calib, 2026-11-27): image-only (a7) and text-only (a6) MMBA on the held-out
calibration cells (cells/<task>_calib.yaml), 3 SUTs, rng seed 0, full budget, with the image/text operator knobs of a
settings file. Base = the MAIN a0 config per (task, SUT) (same SUT, budget, sampler, MMBA settings).

    python scripts/make_calib_configs.py configs/calib/r1.yaml        # -> configs/generated/CALIB_r1/*.yaml

settings file:
    round: r1
    image: {knn_metric: l2_live, region_block: 4, region_levels: [1, 2, 4, 8, 16, 32, 64, 128]}
    text:  {operators: [{name: synonym, severity: 0.5, k_max: 8}, ...]}
    arms: [a7, a6]          # optional
"""
import copy, sys
from pathlib import Path

import yaml

SUTS = ("qwen", "internvl", "ministral")
TASKS = ("imagenet", "pope")
ARMS = {"a7": ("a7_image_only", "image_only"), "a6": ("a6_text_only", "text_only"), "a0": ("a0_directed", None)}


def main() -> int:
    s = yaml.safe_load(Path(sys.argv[1]).read_text())
    rnd = s["round"]
    out = Path(f"configs/generated/CALIB_{rnd}")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for task in TASKS:
        cells = yaml.safe_load(Path(f"cells/{task}_{s.get('cells', 'calib')}.yaml").read_text())["seeds"]["slot_items"]["items"]
        for sut in SUTS:
            base = yaml.safe_load(Path(f"configs/generated/MAIN/A0__s0__{task}__{sut}__a0_directed.yaml").read_text())
            for key in s.get("arms", ["a7", "a6", "a0"]):
                arm, modality = ARMS[key]
                c = copy.deepcopy(base)
                c["name"] = f"CALIB_{rnd}_{task}_{sut}_{arm}_s0"
                c["save_dir"] = f"runs/CALIB/{rnd}/{task}/{sut}/{arm}/s0"
                c["seeds"]["slot_items"]["items"] = copy.deepcopy(cells)
                if modality:
                    c["modality"] = modality
                else:
                    c.pop("modality", None)               # full MMBA: both channels
                c["image"].update(s.get("image", {}))
                if modality == "text_only":
                    c["image"]["region_block"] = 0
                if s.get("text"):
                    c["text"]["composite"] = {"operators": s["text"]["operators"]}
                c["tags"].update(exp_id=f"CALIB_{rnd}", arm=arm, task=task, sut=sut)
                p = out / f"{rnd}__{key}__{task}__{sut}.yaml"
                p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
                n += 1
    print(f"wrote {n} configs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
