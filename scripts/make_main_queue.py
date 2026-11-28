"""MAIN queue (2026-11-28): the paper arm set on every (task, SUT), two rng seeds.

Arms: MMBA a0, MIMICRY-style a5, DeepJanus-style a9; ablations text-only a6, image-only a7, unguided MMBA x1.
Layers pair two arms so a partial queue still yields complete paired blocks; the primary comparison (a0 vs a5,
a0 vs a9) comes first. SUTs qwen / internvl / ministral (LLaVA-NeXT dropped 2026-11-25: balanced design).
x2 (patch-edit ablation) was dropped 2026-11-28.
    python scripts/make_main_queue.py           # -> configs/generated/MAIN/L{layer}__s{seed}__{task}__{sut}__{arm}.yaml
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "scripts")
from make_matrix import build  # noqa: E402

out = Path("configs/generated/MAIN"); out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.yaml"):
    f.unlink()
GRID_SUTS = ["qwen", "internvl", "ministral"]
TASKS = ["imagenet", "pope"]
PAIRS = [["a0_directed", "a5_mimicry_objective"], ["a9_deepjanus", "x1_mmba_unguided"],
         ["a6_text_only", "a7_image_only"]]


def emit(layer: int, seed: int, task: str, sut: str, arm: str) -> None:
    cfg = build("MAIN", task, sut, arm, seed)
    (out / f"L{layer}__s{seed}__{task}__{sut}__{arm}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


layer = n = 0
for seed in (0, 1):
    for arms in PAIRS:
        layer += 1
        for task in TASKS:
            for sut in GRID_SUTS:
                for arm in arms:
                    emit(layer, seed, task, sut, arm); n += 1
print(n, "configs")
