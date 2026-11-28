# Determinism of boundary pairs (CStud2, 2026-11-28)

Question (C2): is the smoke's ON/OFF pair (|g_ON - g_OFF| about 0.27-0.38 nats over a lam step of 7.5e-4) a
boundary or scoring noise? Scripts: `scripts/check_pair_determinism.py` (one pair, many scoring conditions),
`scripts/repro_generation.py` (re-run one generation exactly as the tester scored it). Results:
`results/determinism/*.json`.

## Findings
1. **VQGAN load race (fixed).** transformers switches torch's process-global default dtype while a SUT loads in a
   parallel thread; a VQGAN built in that window got bf16 parameters, which rounded the fp32 checkpoint on load.
   Renders then depended on thread timing (up to 52 grey levels; anchor g 3.26 vs 3.37 on the same seed).
   Fix: `load_vqgan` casts to fp32 before loading the weights (`mmb/manipulator/image/loading.py`).
2. **A generation is reproducible exactly** when re-rendered and re-scored with the same batch:
   max |g_trace - g_rebatch| = 0.0 (SmolVLM, pope cell, generation 9).
3. **Single path is deterministic**: 5 repeats identical; a batch of one equals the single path.
4. **Batch composition changes g** (batched bf16 teacher forcing, plus 1-3 grey levels from batched VQGAN decode):

| SUT | pair | spread of g per side over all conditions | single-path g (ON / OFF) | trace g (ON / OFF) |
|---|---|---|---|---|
| SmolVLM-256M | pope cell, MMBA Phase-2 pair | ON 0.375, OFF 0.344 | +0.031 / +0.344 | -0.156 / +0.219 |
| SmolVLM-256M | same C and its lam-1e-3 neighbour | ON 0.375, OFF 0.656 | +0.031 / +0.125 | -0.156 / - |
| Qwen3.5-4B (bf16) | same inputs | ON 0.016, OFF 0.031 | +3.31 / +3.34 | (not at Qwen's boundary) |

Conditions: single path x5, batch of one x5, the pair in one batch (both orders), mixed batches of 2/4/8/12 at the
first / middle / last position. For SmolVLM the spread equals the pair gap: **the smoke pair was noise**: scored
singly, ON does not cross. For Qwen the spread is one or two bf16 steps (1/64-1/32 nat).

## Remedy (implemented, shared by every arm)
Every boundary decision outside the search is made on the **canonical path**: input rendered alone and scored
alone (single SUT call); `VLMBoundaryTester._canonical_g`.
- min_cross.json keeps the in-budget smallest crossing (budget-bound metrics) and adds g_canonical / confirmed.
- The boundary pair: MMBA's Phase-2 pair is kept only if both sides confirm canonically (pair_source own_phase2);
  otherwise canonical bisection on scale(C, lam) from the smallest canonically confirmed in-budget crossing
  (up to 8 tried; posthoc_bisection); none if nothing confirms. Extra calls are counted in nb_extra_calls.
- Re-run of the pope smoke after the fix: the in-budget crossing (g = -0.156, magnitude 0.004) has canonical
  g = +0.031, 11 extra calls, pair_source none: correctly reported as unconfirmed.

## Measurement on the paper SUTs (scripts/measure_batch_spread.py; results/determinism/spread_{qwen,internvl,ministral}.json)
Setup as in MAIN (configs/paper/e1/base.yaml + scripts/make_matrix.py SUTS: bf16, batched_scoring, batch_chunk 15;
vqgan_latent decode_batch_size 8; no inference cache). Spread inputs: the 8 calibration cells (cells/*_calib.yaml),
per cell the unedited input + 12 edits from the shared sampler (weighted magnitude 0.02-1.0), 104 inputs per SUT.
Per input: single path (rendered alone) + 3 random batch orders (rendered in a batch, as in the search) + 1 batch with
inputs rendered alone. Spread = max - min of g over these 5 scorings.

| SUT | spread median | p90 | max | max abs(batched - single) | scorer-only dev. max | s / generation of 50: batched | canonical | ratio |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | 0.016 | 0.031 | 0.077 | 0.063 | 0.047 | 2.75 | 4.41 | 1.61 |
| InternVL3.5-4B | 0.125 | 0.338 | 0.541 | 0.417 | 0.541 | 5.06 | 6.01 | 1.19 |
| Ministral-3-3B | 0.125 | 0.311 | 0.567 | 0.552 | 0.525 | 7.83 | 6.60 | 0.84 |

- The spread does not depend on the edit magnitude (same median in the bins 0-0.25, 0.25-0.6, >0.6) and is present
  already on the unedited input (up to 0.25 InternVL, 0.43 Ministral).
- The scorer alone (inputs rendered alone, only the batch differs) reaches the full spread: batched bf16 scoring is
  the main source; batched decoding adds a little.
- Canonical scoring (decode one at a time, one single-path call each) costs 1.6x on Qwen, 1.2x on InternVL and is
  FASTER on Ministral (0.84x; its batched path pads images). Per 1500-call cell run: Qwen 82 s -> 132 s,
  InternVL 152 s -> 180 s, Ministral 235 s -> 198 s.

### Cells with |anchor g| below the SUT's spread
Fresh canonical anchor g (codec reconstruction, original question, single call) for all fixed cells:
MAIN = cells/{imagenet,pope}_cells.yaml (36), calib = cells/*_calib*.yaml (16).

| SUT | below median | below p90 | below max | min abs(anchor g) MAIN / calib | median abs(anchor g) MAIN |
|---|---|---|---|---|---|
| Qwen | 0 / 0 | 0 / 0 | 0 / 0 | 1.30 / 2.10 | 3.58 |
| InternVL | 0 / 0 | 0 / 0 | 0 / 0 | 2.13 / 1.13 | 7.79 |
| Ministral | 0 / 0 | 0 / 0 | 0 / 0 | 2.25 / 3.75 | 6.15 |

Fresh vs stored anchor g (cells meta anchor_g_by_sut): median difference 0, max 0.25 / 0.50 / 0.77, no sign flips.
(The SmolVLM smoke cell, anchor g 0.28, is not a paper cell.)

## Decision (C2, 2026-11-28): canonical search scoring for all arms
Numbers behind it: the table above. Canonical scoring removes the batch noise at the source (InternVL / Ministral up
to ~0.55 nats) at +14 % total time (Qwen x1.61, InternVL x1.19, Ministral x0.84); a per-SUT tau of ~0.55 would have
biased what counts as a crossing differently per SUT.

Implemented:
- defaults: `SUTConfig.batched_scoring = False`, `ImageConfig.decode_batch_size = 1`; also set explicitly in
  configs/paper/e1/base.yaml, configs/templates/continuous_template.yaml and scripts/make_matrix.py SUTS; MAIN (72)
  and the arm configs regenerated (all canonical, all 1500 calls).
- the tester logs a loud `!!! NON-CANONICAL SEARCH SCORING !!!` warning if either is changed; stats.json records
  `search_scoring` (canonical | batched) and `image_decode_batch_size`.
- **bit-identical renders**: the latent edit is computed row by row (a batched GPU matmul differed from the
  single-row result in the last bit, which flipped grey levels after decoding and moved SmolVLM's g by up to 0.5:
  found by the consistency flag below). Checked on the GPU: batch vs single latents bit-identical, pixels identical.
- the canonical pair confirmation stays; under canonical search it must reproduce the trace: boundary_pair.json
  `canonical_consistent` (true / false; null for batched search) and an ERROR log line if the in-budget crossing
  re-scores differently or MMBA's Phase-2 pair does not confirm.

Canonical smoke (SmolVLM-256M, POPE calib cell 2, MMBA 10 x 10; runs/SMOKE/continuous/pope/smolvlm/mmba/…/
smoke_montage.png): first crossing at eval 26 (g -0.062); in-budget min crossing g -0.094 = canonical -0.094
(confirmed); MMBA's own Phase-2 pair confirmed (pair_source own_phase2, gap 7.5e-4 in lam, OFF g +0.062, ON g
-0.219, same text: continuous), canonical_consistent true, 3 extra calls. Runtime 28.7 s for 100 calls.
Note: SmolVLM's g moves by 0.28 over a lam step of 7.5e-4 (an invisible change, |ON - OFF| x 20 is black): a
steep but deterministic boundary of a tiny bf16 model, not noise any more.

## Earlier proposals (superseded by the decision)
- Measure the batch spread for every paper SUT (qwen, internvl, ministral, llava NF4) with
  check_pair_determinism.py and set optimizer.mmba.tau (crossing margin in the search) to about that spread
  (SmolVLM about 0.5, Qwen about 0.05), so the search does not chase batch artefacts.
- Or score the search itself canonically (sut.batched_scoring: false, image.decode_batch_size: 1); measure the
  time cost per SUT first.
- Cells with |anchor g| below the SUT's batch spread are not testable on that SUT.
