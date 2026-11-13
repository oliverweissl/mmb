"""Expand a base YAML config into the E1/E2 comparison arms (one YAML per arm).

    python scripts/make_arm_configs.py configs/templates/evolutionary_template.yaml configs/paper/arms
Arms differ only in ``optimizer.*`` / ``modality``; seeds and budget stay identical.
"""
import copy
import sys
from pathlib import Path

import yaml

ARMS = {
    "a0_directed": {},
    "a1_random_tiered": {"optimizer": {"algorithm": "random"}},
    "a2_random_uniform": {"optimizer": {"algorithm": "random", "sampling": {"mode": "uniform"}}},
    "a3_ga_single": {"optimizer": {"algorithm": "ga", "proximity_only": True}},
    "a4_nsga2": {"optimizer": {"algorithm": "nsga2"}},
    "a5_mimicry_objective": {"optimizer": {"boundary_objective": "softmax_balance"}},
    "a6_text_only": {"modality": "text_only"},
    "a7_image_only": {"modality": "image_only"},
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
