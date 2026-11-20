
import sys, yaml
from pathlib import Path
sys.path.insert(0, "scripts")
from make_matrix import build
out = Path("configs/generated/MAIN"); out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.yaml"): f.unlink()
GRID_SUTS = ["qwen", "internvl", "ministral"]; TASKS = ["imagenet", "pope"]
PAIRS = [["a0_directed", "a1_random_tiered"], ["a8_hillclimb", "a5_mimicry_objective"], ["a3_ga_single", "a4_nsga2"]]
layer = 0; n = 0
for seed in (0, 1):
    for arms in PAIRS:
        layer += 1
        for task in TASKS:
            for sut in GRID_SUTS:
                for arm in arms:
                    cfg = build("MAIN", task, sut, arm, seed)
                    (out / f"L{layer}__s{seed}__{task}__{sut}__{arm}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)); n += 1
for seed in (0, 1):
    layer += 1
    for arm in ["a0_directed", "a1_random_tiered"]:
        cfg = build("MAIN", "imagenet", "llava", arm, seed)
        (out / f"L{layer}__s{seed}__imagenet__llava__{arm}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)); n += 1
print(n, "configs")
