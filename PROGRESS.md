# Progress (code workstream)

## Phase 1 — Minimise & restructure
- [x] Import thesis code as package `mmb` (was `src`), drop slides/notebooks/samples/archives/299 configs/HS-01 app
- [x] Vendor SMOO base classes into `mmb/core.py` (smoo now only optional, lazy, for StyleGAN loading)
- [x] StyleGAN interpolation reimplemented as plain W+ lerp (no smoo MixCandidate)
- [x] Dead code removal: vulture + unused modules (position_scoring, seed_matrix, pair_resolver) → 591 tests pass (+1 known flaky: test_trace_row_count)
- [ ] ImageBackend protocol unification (vqgan / stylegan / diffusion) — see image_backend.py, image_factory.py
- [ ] Split >700-line modules (config.py, scorer.py, pdq/runner.py, boundary_pair/runner.py, vlm_boundary_tester.py)
- [ ] README + docs/ (short), ruff clean, CI-less `make test`

## Phase 1.5 — Reviewer-driven (from GAPS.md / EXPERIMENTS-WANTED.md, read 2026-11-13)
- [x] E1 arms: `optimizer.algorithm` (agemoea2|nsga2|ga|random), `optimizer.boundary_objective` (logprob_gap|softmax_balance = MIMICRY-style), `optimizer.proximity_only`; `scripts/make_arm_configs.py` -> configs/paper/arms (8 arms load + tested end-to-end through the tester). Still wanted: MAP-Elites arm, HopSkipJump+TextFooler channel-attack arm (BASELINES.md SHOULD tier)
- [ ] E2 ablation switches (objectives subset, init, text-operator subsets, calibration)
- [ ] E7 automatic validity scorer (DINOv3/CLIP sim, LPIPS, text-embedding sim) on stored stimuli
- [ ] Seed/replicate plumbing: distinct seed_image_id per rng seed, common CSV columns `C`
- [ ] API (label-only) SUT adapter with cache/rate-limit (G8)

## Phase 2 — Backends
- [ ] Diffusion image backend (wrap smoo diffusion manipulators / diffusers) + stub test
- [ ] Generative text backend `LLMParaphrase` with gene-k = k-th candidate contract + stub test
- [ ] Backend comparison config + stub-SUT dry-run harness; runbook for GPU box

## Phase 3 — Tasks & SUTs
- [ ] `Task` protocol; VQA, captioning, POPE, OCR/chart, refusal tasks
- [ ] Additional SUTs (PaliGemma, InternVL, SmolVLM, CLIP/DINOv3 zero-shot)
- [ ] Align with ../6af72b4ebb77bd75e414baf7/GAPS.md + EXPERIMENTS-WANTED.md
