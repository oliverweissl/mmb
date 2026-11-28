"""Expand a base YAML config into the E1/E2 comparison arms (one YAML per arm).

    python scripts/make_arm_configs.py configs/templates/continuous_template.yaml configs/paper/arms
Arms differ only in ``optimizer.*`` / ``modality``; seeds, budget, sampler and variation operators stay identical.
Paper arm set (C2, 2026-11-28): MMBA a0, MIMICRY-style a5, DeepJanus-style a9, ablations text-only a6,
image-only a7, unguided MMBA x1 (x2 patch-edit ablation dropped 2026-11-28; random/GA/NSGA-II/hill-climber arms removed).
"""
import copy
import sys
from pathlib import Path

import yaml

ARMS = {
    "a0_directed": {"optimizer": {"algorithm": "mmba", "proximity_only": True}},
    # MIMICRY-style: AGE-MOEA-II with the softmax-balance objective (+ magnitude objectives)
    "a5_mimicry_objective": {"optimizer": {"algorithm": "agemoea2", "boundary_objective": "softmax_balance",
                                           "proximity_only": False}},
    "a9_deepjanus": {"optimizer": {"algorithm": "deepjanus", "proximity_only": True}},
    # MMBA component ablations
    "a6_text_only": {"modality": "text_only", "optimizer": {"algorithm": "mmba", "proximity_only": True}},
    "a7_image_only": {"modality": "image_only", "optimizer": {"algorithm": "mmba", "proximity_only": True}},
    "x1_mmba_unguided": {"optimizer": {"algorithm": "mmba_unguided", "proximity_only": True}},
}


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


if __name__ == "__main__":
    base = yaml.safe_load(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    for name, ov in ARMS.items():
        cfg = merge(base, ov)
        cfg["name"] = f"{base.get('name', 'exp')}_{name}"
        (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {len(ARMS)} arms to {out}")
