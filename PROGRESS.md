# Progress (code workstream)

## Current state (2026-11-28, continuous design)
- [x] Real-valued genotype `[c in [-1,1]^16 | t, u]`, shared real-coded variation and sampler for every method
      (docs/SEARCH_REAL_CODED.md); MMBA Phase 1 + magnitude-minimising Phase 2 with ON/OFF boundary pair.
- [x] Image block: continuous VQGAN-latent displacement, DCT basis, no re-quantisation; fp32 codec load (determinism fix).
- [x] Text block: question pool of validated whole questions (`configs/text/question_pool.json`, review sheet
      `docs/figs/text_pool/review.html`); rejected candidates in `question_pool_rejected.json.gz`.
- [x] Canonical scoring path for every decision outside the search (docs/DETERMINISM_2026-11-28.md).
- [x] Analysis migrated to the new run outputs (budget-bound metrics, smallest-crossing magnitude, boundary pairs,
      modality composition); primary comparison MMBA vs MIMICRY-style / DeepJanus-style; replays (agreement, transfer,
      modality swap) on the canonical path; weight calibration module; suite green.
- [ ] Calibration of (w_img, w_txt) (round w1.0_1.0 running; adopt the weakest balanced step).
- [ ] MAIN campaign with the adopted weights; replays; export to the paper.
- [ ] User review of the question pool's extra meaning rules (docs/figs/text_pool/review.html).
- [ ] Human validity study on the smallest crossings (`mmb.validity.rating_set`).

## Phase 1 — Minimise & restructure
- [x] Import thesis code as package `mmb` (was `src`), drop slides/notebooks/samples/archives/299 configs/HS-01 app
- [x] Vendor SMOO base classes into `mmb/core.py` (smoo now only optional, lazy, for StyleGAN loading)
- [x] StyleGAN interpolation reimplemented as plain W+ lerp (no smoo MixCandidate) — later removed with StyleGAN
- [x] Dead code removal: vulture + unused modules (position_scoring, seed_matrix, pair_resolver)
- [x] ImageBackend protocol unification — superseded: one continuous latent backend (`mmb/manipulator/image/latent.py`)
- [ ] Split >700-line modules (config.py, scorer.py, vlm_boundary_tester.py) — deferred, not needed for the paper
- [x] README + docs/ (short)

## Phase 1.5 — Reviewer-driven (from GAPS.md / EXPERIMENTS-WANTED.md, read 2026-11-13)
- [x] E1 arms — now a0 MMBA, a5 MIMICRY-style, a9 DeepJanus-style, a6/a7 single modality, x1 w/o guidance
      (`scripts/make_arm_configs.py`); random/GA/NSGA-II/hill-climber dropped; MAP-Elites / HopSkipJump not pursued
- [x] E2 ablations — superseded by a6 / a7 / x1 under the balanced design
- [x] E7 automatic validity scorer (`mmb.validity`: LPIPS, CLIP/DINO, SSIM, MiniLM, word edit distance)
- [x] Seed/replicate plumbing: fixed cell lists (`cells/`), rng seeds 0 and 1, balanced-design guard in the export
- [ ] API (label-only) SUT adapter — out of scope for this paper

## Phase 2 — Backends — superseded
- [x] Diffusion image backend / LLM-paraphrase text backend — replaced by the continuous latent block and the
      question pool (T5 paraphrases + word substitutions, validated offline)
- [x] Backend comparison harness — replaced by the stub-SUT tests and the smoke configs (`configs/generated/SMOKE_continuous`)

## Phase 3 — Tasks & SUTs — narrowed
- [x] Tasks: ImageNet class pairs, POPE yes/no (other tasks out of scope)
- [x] SUTs: Qwen3.5-4B, InternVL3.5-4B, Ministral-3-3B (LLaVA dropped; SmolVLM for smoke runs only)
