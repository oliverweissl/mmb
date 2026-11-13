# Progress (code workstream)

## Phase 1 — Minimise & restructure
- [x] Import thesis code as package `mmb` (was `src`), drop slides/notebooks/samples/archives/299 configs/HS-01 app
- [x] Vendor SMOO base classes into `mmb/core.py` (smoo now only optional, lazy, for StyleGAN loading)
- [x] StyleGAN interpolation reimplemented as plain W+ lerp (no smoo MixCandidate)
- [x] Dead code removal: vulture + unused modules (position_scoring, seed_matrix, pair_resolver) → 591 tests pass (+1 known flaky: test_trace_row_count)
- [ ] ImageBackend protocol unification (vqgan / stylegan / diffusion) — see image_backend.py, image_factory.py
- [ ] Split >700-line modules (config.py, scorer.py, pdq/runner.py, boundary_pair/runner.py, vlm_boundary_tester.py)
- [ ] README + docs/ (short), ruff clean, CI-less `make test`

## Phase 2 — Backends
- [ ] Diffusion image backend (wrap smoo diffusion manipulators / diffusers) + stub test
- [ ] Generative text backend `LLMParaphrase` with gene-k = k-th candidate contract + stub test
- [ ] Backend comparison config + stub-SUT dry-run harness; runbook for GPU box

## Phase 3 — Tasks & SUTs
- [ ] `Task` protocol; VQA, captioning, POPE, OCR/chart, refusal tasks
- [ ] Additional SUTs (PaliGemma, InternVL, SmolVLM, CLIP/DINOv3 zero-shot)
- [ ] Align with ../6af72b4ebb77bd75e414baf7/GAPS.md + EXPERIMENTS-WANTED.md
