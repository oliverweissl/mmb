# EXP_SETUP — how every number in the paper is produced

Status: 2026-11-20 18:18, FROZEN for the MAIN campaign (queue started 18:17). Design decisions and their reasons live in METHODOLOGY.md; this file is
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
| SUT llava | `llava-hf/llava-v1.6-mistral-7b-hf`, bitsandbytes NF4 weight-only, bf16 compute | thesis main SUT; continuity block only (thesis: OpenVINO INT8) |
| SUT qwen | `Qwen/Qwen3.5-4B`, bf16 | thesis replication SUT |
| SUT internvl | `OpenGVLab/InternVL3_5-4B-HF`, bf16 | new family |
| SUT ministral | `mistralai/Ministral-3-3B-Instruct-2512`, FP8 checkpoint dequantised to bf16 | new family (Mistral) |

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

## 8. Frozen MAIN design (2026-11-20)
- Cells: K = 18 per task (cells/imagenet_cells.yaml: 3 per anchor class; cells/pope_cells.yaml: 9 yes / 9 no),
  tau = 1.0 nats, eligible iff all 4 SUTs (llava, qwen, internvl, ministral) answer the anchor on the VQGAN
  reconstruction with g >= tau (ImageNet 858/960, POPE 97/200 eligible). Selection seed 20261120.
- rng seeds S = 2 (0, 1). Budget 30 x 50 = 1,500 proposals, early stop off. Image operator: VQGAN f8-16384
  KNN codebook swaps, cone filter OFF for both tasks (same operator everywhere).
- Grid (balanced): tasks {imagenet, pope} x SUTs {qwen, internvl, ministral} x arms {a0, a1, a3, a4, a5, a8}
  x K x S = 2 x 3 x 6 x 36 = 1,296 runs. Continuity block: imagenet x llava x {a0, a1} x 36 = 72 runs.
- Scoring: batched full-sequence teacher forcing (sut.batched_scoring, chunk 15). Numerical tolerance of
  batched vs single-item scoring (scripts/check_batched_scoring.py, n=30): max |dg| qwen 0.021, internvl 0.11,
  ministral 0.10, llava-nf4 0.06; sign agreement 100% for all. Consequence: the |g| <= 1e-2 band is below the
  bf16 batch noise; crossings are re-verified by single-item re-scoring (agreement_replay.py).
- Queue: scripts/make_main_queue.py -> configs/generated/MAIN (76 configs, 8 layers; each layer is a complete
  balanced block: L1 a0+a1 s0, L2 a8+a5 s0, L3 a3+a4 s0, L4-L6 same for s1, L7-L8 llava a0+a1 s0/s1).
  `PY=.venv/bin/python scripts/run_queue.sh configs/generated/MAIN runs/logs/MAIN`
- Measured speed (s per generation of 30, batched): qwen ~2.8, internvl ~3.7, ministral ~5.5, llava-nf4 ~16.
  Expected GPU time: grid ~73 h, llava block ~16 h.

## 9. Scoring defect found and fixed (2026-11-20)
The inherited KV-cache scorer reused ONE mutable cache across candidates: every candidate after the
first was scored with the earlier candidates' tokens in context (order-dependent; up to ~3 nats on
multi-token labels, e.g. 'reptile' -5.49 vs -2.23 nats on Qwen3.5-4B). Fixed in e3c86f6. All MAIN numbers
use the fixed scorer. The thesis' OpenVINO path issues the same calls and is very likely affected; this
could not be verified here (no OpenVINO stack). Thesis-era numbers are therefore not re-used as evidence.

## 10. Campaign log
- 2026-11-20 18:17: MAIN queue started (session-bound). Died ~19:40 when the session ended; 30 runs done.
- 2026-11-21 16:32: restarted as user systemd unit `mmb-queue` (stop: `systemctl --user stop mmb-queue`), with --clean-partials.
- 2026-11-21 16:53: L1 ministral a0 crashed 3x (SIGSEGV/SIGFPE in llvmlite while pymoo JIT-compiled AGE-MOEA-II helpers after triton's LLVM was loaded by the Ministral stack). Fix: compile pymoo's numba helpers before the SUT loads (run_boundary_test._warm_up_numba). Same arithmetic, so no effect on results. Unit `mmb-queue-retry` re-runs failed configs after the main pass (up to 3 passes).
- 2026-11-21 18:24: L1 qwen a0/a1 (ImageNet) crashed 3x each (SIGSEGV in libnvrtc during kernel JIT). Likely cause: transformers' hub kernels (the `kernels` package, installed for Ministral's FP8 checkpoint) replacing Qwen3.5 layers. Fix: the runner disables hub kernels for every SUT except Ministral (reference implementations, same maths). Failed configs are re-run by `mmb-queue-retry`.
- 2026-11-21 23:29-00:08: hub-kernel fix confirmed (Qwen ran). Intermittent SIGSEGV inside libcuda (Ministral a1 POPE, Qwen a0 POPE), handled by retry + --resume. One retry hung at 100% CPU before writing any output (37 min); killed by hand. Added unit `mmb-watchdog` (scripts/queue_watchdog.sh): kills a run whose logs are silent for 20 min, so the queue retries it.
- 2026-11-22 00:28-00:30: burst of crashes in unrelated native code (SIGILL; SIGSEGV in llvmlite during the numba warm-up before any model load, in libcuda, and at bare addresses). The host CPU is an i9-13900KF (Raptor Lake; microcode 0x133). This pattern matches the known instability of degraded 13th/14th-gen Intel CPUs. Mitigation without root: unit `mmb-pin` (scripts/pin_ecores.sh) pins every run to the E-cores (CPUs 16-31); the workload is GPU-bound. Crashed runs are retried with --resume, so results are unaffected (each cell is either complete or re-run from scratch).
- 2026-11-22 03:10: POPE object-word protection (METHODOLOGY change 2026-11-22). Archived 8 POPE configs' runs; POPE configs regenerated; retry pass re-runs L1 POPE.
- 2026-11-22 00:38-03:10: no crashes after E-core pinning.
- 2026-11-22 08:10: archived L1 imagenet/internvl/a0 s0 (run before the diagnostic magnitude columns existed); the retry pass re-runs it so every arm has diag_* magnitudes.
- 2026-11-22 10:34: queue restarted as `mmb-queue2` (scripts/run_queue_passes.sh, 3 passes, from L1) so the missing seed-0 directed/random configs run first; the DeepJanus layer was renamed L10 -> L9z so it sorts after L9. Order: L1 remainder, L2-L8, L9, L9z.
- 2026-11-22 12:07: E-core-only pinning slowed runs ~1.8x (InternVL 5.7 vs 3.1 s/gen). Re-pinned to CPUs 0-3,8-31 (all except the two favoured 5.8 GHz P-cores, CPUs 4-7): 3.9 s/gen. Revert to 16-31 if crashes return (PIN_CPUS env of unit mmb-pin).
- 2026-11-22 12:40: MAIN v2 (minimal text profile + protected key words, METHODOLOGY 12:40). v1 runs archived; configs regenerated (100); queue restarted as `mmb-queue3` from L1.
- 2026-11-22 16:37: pending directed (a0) and image-only (a7) configs renamed with prefix Z so they run after all baselines; the directed search is under redesign (pending decision), and baseline runs remain valid whatever the redesign is.
- 2026-11-22 21:40: MMBA redesign frozen (METHODOLOGY 21:40). a0 configs renamed A* (run first), a7 configs Z* (run last); old v2 a0/a7 runs archived. Pilot runs in runs/PILOT (held-out cells).
- 2026-11-23 01:10: MAIN v3 (region edits; METHODOLOGY 01:10). v2 runs archived to runs/archive/MAIN_v2_patch; configs regenerated (100, prefixes A/B/C give the order); queue restarted as `mmb-queue8`.

## 2026-11-23 — Process-independent text rendering; replays from trace texts
- Text operators seeded their per-word RNG with Python `hash()`, which is salted per process. Searches are unaffected (one process per run, trace stores the scored text), but post-hoc re-rendering (replays) could produce different text.
- Fix: `stable_hash` (CRC32) in mmb/manipulator/text/operators (base, character_noise, saliency, fragmentation).
- Repair: scripts/fix_text_from_trace.py sets min_cross texts from trace `decoded_text` (81 fixed); neighbours differing by an image gene keep the crossing text; others re-scored by scripts/partner_replay.py (scored text kept for image-gene partners).
- Re-run: scripts/replay_maintenance.sh (pauses queue; partner, agreement, transfer replays; restarts queue).

## 2026-11-23 — Confirmation + replays
- scripts/confirm_crossings.py run for qwen/internvl/ministral (search output kept in min_cross_search.json). Must be re-run after each grid block (it skips confirmed runs).
- Agreement (greedy answer == teacher-forced argmax) at the crossing: qwen 84%, internvl 76%, ministral 79%; anchors 100/84/100%.
- agreement/transfer replays now warm up numba and disable hub kernels (qwen segfault); maintenance script pins CPUs.

## 2026-11-23 — LLaVA on both tasks
- Every SUT must cover every task: added C0/C1 LLaVA-NeXT (NF4) POPE configs (a0, a1; seeds 0, 1), same cells/budget as the other SUTs. make_main_queue.py updated accordingly (do NOT re-run it: it deletes the current queue names).

## 2026-11-24 — Seed-1 block done; baseline blocks queued
- A1 (seed 1, MMBA + random, 3 SUTs x 2 tasks) finished 18:10; after_block A1 runs confirm/agreement/transfer replays, then restarts the queue at B0.
- after_block hooks installed for B0, B1, C1 (units mmb-after-B0/B1/C1): each confirms crossings when its block is complete.
- Remaining: B0/B1 (GA, NSGA-II, MIMICRY-style, image-only, hill climber, DeepJanus-style; seeds 0/1; 72 runs), C0/C1 (LLaVA POPE; 8 runs). Estimated ~95 GPU-h.
- 2026-11-24 19:46: added block D1x = the RQ4 ablations (x1 unguided, x2 patch edits, a6 text-only) on rng seed 1 (18 configs, copies of A0x with seed 1), so RQ4 has the same two random seeds as RQ1/RQ2. Runs after C1 (~18 GPU-h); unit mmb-after-D1x confirms crossings.
