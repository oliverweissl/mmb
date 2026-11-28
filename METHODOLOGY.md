# METHODOLOGY (decision log, continuous design)

Current as of 2026-11-28 (commit a50514a and later). The integer-genotype decision log (2026-11-20 .. 11-27) is in git
history. Design notes: [docs/METHODOLOGY_REVIEW_2026-11-28.md](docs/METHODOLOGY_REVIEW_2026-11-28.md) (why the
genotype became continuous), [docs/SEARCH_REAL_CODED.md](docs/SEARCH_REAL_CODED.md) (search, MMBA phases),
[docs/DETERMINISM_2026-11-28.md](docs/DETERMINISM_2026-11-28.md) (canonical scoring). Operational recipe:
[EXP_SETUP.md](EXP_SETUP.md). Every definition below is implemented in `mmb/analysis/collect.py` (tests with
hand-computed cases).

## 1. Definitions
| id | decision | reason |
|---|---|---|
| D1 | **Contrast** `g = logp[class_a] - logp[class_b]` (length-normalised teacher-forced log-likelihoods; pair positions `target_classes`). | A signed contrast makes a crossing a sign change. |
| D2 | **Crossing**: `sign(g) * anchor_sign < 0` (strict; tau = 0). | Ties are not flips. |
| D3 | **Anchor**: sign of `stats.anchor_g`, a reference evaluation of the unedited input outside the budget. | Independent of the sampler. |
| D4 | **Budget-bound metrics** (crossed, first_cross_eval, floor, label flip, the smallest crossing) use the first `eval_budget` evaluations only; never-crossing runs are censored at the budget. | Equal budget for every method. |
| D5 | **Smallest crossing** (`min_cross.json`): the in-budget crossing of smallest magnitude = L2(w_img * RMS(c), w_txt * t). Confirmed by re-scoring on the canonical path (`min_confirmed`). | Minimal change, one unit across modalities (weights calibrated, section 4). |
| D6 | **Boundary pair** (`boundary_pair.json`): ON (crosses) / OFF (does not) on one line `scale(C, lam)`, gap <= eps = 1e-3 in lam, both sides decided on the canonical path. Source: MMBA's own Phase 2 when it confirms, else post-hoc bisection (same procedure for every method). Extra calls are counted (`nb_extra_calls`) and stay outside every budget metric. kind = text_step when the question differs across the pair, else continuous. | A true ON/OFF pair at a continuous distance. |
| D7 | **Canonical path**: one input rendered and scored per call (`VLMBoundaryTester._canonical_g`); batched bf16 scoring shifts g with batch composition. | DETERMINISM_2026-11-28. |
| D8 | **Modality of a smallest crossing**: image (image edited, question unchanged), text, joint. Attribution by swapping (each block alone, `scripts/modality_swap.py`) is reported separately. | |

## 2. Balanced design
Every task x SUT x method uses the same seed inputs, rng seeds (0, 1) and budget (30 generations x 50 = 1,500 calls);
the same initial sampler (magnitude-stratified), SBX and block-wise polynomial mutation for every method; early stop off.
A method with an incomplete block is held back by the export (`complete_blocks`, `check_balance`).

## 3. Methods
| arm | what | question |
|---|---|---|
| a0 MMBA | margin-guided search to the first crossing, then magnitude minimisation on global and block lines (Phase 2), guided restarts below the incumbent | the method |
| a5 MIMICRY-style | softmax-balance objective | baseline |
| a9 DeepJanus-style | DeepJanus archive search | baseline |
| a6 text-only, a7 image-only | one block only | modality ablation, calibration |
| x1 w/o guidance | MMBA without margin guidance | ablation |

Primary comparison: a0 vs a5 and a0 vs a9 per (task, SUT): crossing share (exact McNemar, Holm across strata);
secondary: calls to the first crossing, smallest-crossing magnitude (Wilcoxon on pairs where both crossed).

## 4. Validity and calibration
- Text: the question pool holds only whole questions validated as wholes (NLI mutual entailment >= 0.8, CoLA rule,
  protected words, dictionary words, meaning rules; `scripts/build_question_pool.py`, review sheet
  `docs/figs/text_pool/review.html`). Image: continuous latent edits, no re-quantisation.
- Weights (w_img, w_txt) start at (1, 1) and are calibrated on held-out items so that text-only and image-only cross
  about equally often (EXP_SETUP.md section 5; `mmb.analysis.calibration`).
- Validity proxies (`mmb.validity`): LPIPS, CLIP/DINO cosine, SSIM, MiniLM distance and word edit distance of the
  question; descriptive until calibrated against the human study.
