# EXP_SETUP — how every number in the paper is produced

Status: 2026-11-20, draft. Design decisions and their reasons live in METHODOLOGY.md; this file is
the operational recipe (hardware, software, data, cells, matrix, commands, outputs). Anything marked
TBD is fixed after the speed benchmark (step 3) and then frozen before the first counted run.

## 1. Hardware and software
- 1x NVIDIA RTX 5080 16 GB (Blackwell, sm_120), 32 CPU threads, 62 GB RAM. All GPU jobs run
  strictly serially through `scripts/run_queue.sh` (one process at a time).
- Python 3.12 venv; torch 2.11.0+cu128, transformers 5.23.0, pymoo 0.6.1.6, bitsandbytes (INT8).
  Exact versions: `pip freeze > results/env.txt` at campaign start (committed).

## 2. Data and models
| item | source | notes |
|---|---|---|
| ImageNet-1k validation (50k) | HF `Tsomaros/Imagenet-1k_validation` (ungated mirror of ILSVRC-2012 val, same label indices) | local parquet snapshot; `MMB_IMAGENET_DIR` |
| POPE | HF `lmms-lab/POPE` (test, adversarial split; COCO val2014 images) | 200-item pool, balanced yes/no |
| VQGAN f8-16384 | latent-diffusion `vq-f8` checkpoint | image operator (codebook patch swaps) |
| SUT llava | `llava-hf/llava-v1.6-mistral-7b-hf`, bitsandbytes INT8 | thesis main SUT (thesis: OpenVINO INT8) |
| SUT qwen | `Qwen/Qwen3.5-4B`, bf16 | thesis replication SUT |
| SUT internvl | `OpenGVLab/InternVL3_5-4B-HF`, bf16 | new family |

## 3. Speed benchmark (before freezing K and S)
`mmb-evolve <cfg> --device cuda --preflight` on one cell per SUT; record s/eval in
results/speed.csv. Budget rule: total GPU time of the frozen matrix <= ~80 h.

## 4. Cells (identical for every SUT)
```
python scripts/build_cells.py pool imagenet --out cells/imagenet_pool.yaml --per-class 4   # 240 cells x 4 photos = 960
python scripts/build_cells.py pool pope --out cells/pope_pool.yaml --n 200
for s in llava qwen internvl; do python scripts/build_cells.py score cells/<task>_pool.yaml --sut <model_id> [--int8]; done
python scripts/build_cells.py select cells/<task>_pool.yaml --k K --tau 1.0 --out cells/<task>_cells.yaml
```
Selection: an item is eligible iff every SUT answers the anchor on the VQGAN reconstruction with
g = l(a) - l(b) >= tau = 1.0 nats; K items are drawn stratified (ImageNet: anchor class; POPE:
ground-truth answer) with seed 20261120. K = TBD (same K for both tasks).

## 5. Matrix
| exp | tasks | SUTs | arms | rng seeds | runs |
|---|---|---|---|---|---|
| E1 | imagenet | llava | a0-a7 | S | K x 8 x S |
| E3 | imagenet | llava, qwen, internvl | a0, a1 (+a5) | S | K x 3 x 2(3) x S |
| E4 | imagenet, pope | llava, qwen, internvl | a0, a1 | S | K x 2 x 3 x 2 x S |
E3/E4 reuse the E1 llava runs where the configuration is identical (same cells, seeds, arm).
Budget per run: pop 30 x 50 generations = 1,500 proposals; early stop off.
```
python scripts/make_matrix.py --exp E1 --tasks imagenet --suts llava --arms all --rng-seeds 0 1 2
scripts/run_queue.sh configs/generated/E1
```

## 6. Outputs
- runs/<exp>/<task>/<sut>/<arm>/s<seed>/<name>_seed_<cell>_<ts>/: trace.parquet, convergence.parquet,
  stats.json (run_meta, anchor_g, config_snapshot), first_cross.png/.json, pareto_*, origin.png.
- results/<exp>.csv: `python -m mmb.analysis.collect runs/<exp> --out results/<exp>.csv`
- results/tables/*.tex, figures: `scripts/analysis/run_analysis.sh runs/<exp> <exp> a0_directed`
- results/validity_<exp>.csv: `python -m mmb.analysis... (E7, mmb.validity.score)`

## 7. Campaign log
(date, what ran, GPU hours, failures and re-runs)
